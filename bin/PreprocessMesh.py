#!/usr/bin/env python3
"""Trimesh-based equivalent of the C++ PreprocessMesh binary.

Goal
----
Replace the Pangolin/OpenGL dependency in `bin/PreprocessMesh` with a pure-Python
implementation based on `trimesh` (+ `scipy` for KD-tree), while preserving:
- CLI shape (same flags and defaults where sensible)
- Output format: `.npz` with `pos` and `neg` arrays shaped (N, 4)
- Sampling logic (surface-near Gaussian perturbations + uniform cube samples)
- SDF sign voting rule ("all or nothing" consistency across k nearest normals)

Notes
-----
The C++ code normalizes the mesh into the unit sphere (with 1.03 buffer) before
sampling. This script replicates that behavior.

This file is intentionally implemented function-by-function to mirror the C++
structure.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import trimesh

from utils_pp import *
from sampling_pp import *

# ----------------------------
# Writers
# ----------------------------

def write_sdf_to_npz(xyz: np.ndarray, sdfs: np.ndarray, filename: str, print_num: bool = False, quiet: bool = False) -> None:
    """Write NPZ with `pos` and `neg` arrays shaped (N, 4)."""
    if len(xyz) != len(sdfs):
        raise ValueError("xyz and sdfs length mismatch")

    pos_mask = sdfs > 0
    neg_mask = ~pos_mask

    pos = np.concatenate([xyz[pos_mask], sdfs[pos_mask, None]], axis=1).astype(np.float32)
    neg = np.concatenate([xyz[neg_mask], sdfs[neg_mask, None]], axis=1).astype(np.float32)

    np.savez(filename, pos=pos, neg=neg)

    if print_num and not quiet:
        print(f"pos num: {len(pos)}")
        print(f"neg num: {len(neg)}")


def write_sdf_to_npy(xyz: np.ndarray, sdfs: np.ndarray, filename: str) -> None:
    data = np.concatenate([xyz, sdfs[:, None]], axis=1).astype(np.float32)
    np.save(filename, data)


def write_sdf_to_ply(
    xyz: np.ndarray,
    sdfs: np.ndarray,
    filename: str,
    neg_only: bool = True,
    pos_only: bool = False,
) -> None:
    """Write colored PLY similar to the C++ helper."""
    if len(xyz) != len(sdfs):
        raise ValueError("xyz and sdfs length mismatch")

    if neg_only:
        keep = sdfs <= 0
    elif pos_only:
        keep = sdfs >= 0
    else:
        keep = np.ones(len(sdfs), dtype=bool)

    pts = xyz[keep]
    sdf_keep = sdfs[keep]

    # Colour encoding mirrors the C++ writeSDFToPLY:
    #   positive SDF → blue channel  (r=0,  g=0,  b=intensity)
    #   negative SDF → red  channel  (r=intensity, g=0, b=0)
    # Points at exactly sdf=0 go to the neg group (same as C++ `sdf <= 0`).
    sdf_i = np.clip((np.abs(sdf_keep) * 255).astype(int), 0, 255)

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, sdf, si in zip(pts, sdf_keep, sdf_i):
            if sdf >= 0:   # positive → blue
                f.write(f"{p[0]} {p[1]} {p[2]} 0 0 {si}\n")
            else:          # negative → red
                f.write(f"{p[0]} {p[1]} {p[2]} {si} 0 0\n")


# ----------------------------
# Main / CLI
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="PreprocessMesh")
    parser.add_argument("-m", "--mesh", dest="mesh_file", required=True, help="input mesh file")
    parser.add_argument("-o", "--output", dest="output_file", required=True, help="output filename (npy or npz)")
    parser.add_argument("-v", "--visualize", action="store_true", help="visualize the mesh (no-op in headless mode)")
    parser.add_argument("--ply", dest="ply_path", default="", help="ply point cloud filename")
    parser.add_argument("-s", "--num_sample", dest="num_sample", type=int, default=500000, help="number of samples")
    parser.add_argument("--var", dest="variance", type=float, default=0.005, help="variance near surface")
    parser.add_argument("--sply", "--save_ply", dest="save_ply", action="store_true", help="save ply (writes sdf_samples.ply)")
    parser.add_argument("-t", "--test", dest="test_mode", action="store_true", help="test mode flag")
    parser.add_argument("-n", "--spatial", dest="spatial_sample_npz", default="", help="spatial sample npz file (ignored)")
    parser.add_argument("--check-manifold", dest="check_manifold", action="store_true", help="Check mesh manifoldness")
    parser.add_argument(
        "--strict-manifold",
        dest="strict_manifold",
        action="store_true",
        help="Exit with error if mesh is non-manifold (only with --check-manifold)",
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Suppress per-mesh output",
    )
    parser.add_argument(
        "--sdf-method",
        dest="sdf_method",
        type=str,
        default="igl",  # Changed from 'knn' to 'igl' - libigl works correctly for thin-plate structures
        choices=["knn", "igl", "raycast"],
        help="SDF computation method: 'knn' (original KD-tree + voting), 'igl' (libigl accurate SDF), "
             "or 'raycast' (trimesh proximity + ray casting). "
             "Default: igl (libigl works correctly for thin-plate structures with complex topology like ABC/0001). "
             "See docs/SDF_Analysis_Report_ABC_0001.md for analysis details.",
    )
    parser.add_argument(
        "--normal-offset",
        dest="normal_offset",
        action="store_true",
        help="For --sdf-method=igl or raycast: offset samples along surface normal instead of isotropic. "
             "This provides controlled interior/exterior sampling.",
    )
    parser.add_argument(
        "--on-surface-ratio",
        dest="on_surface_ratio",
        type=float,
        default=0.0,
        help="Initial estimate fraction of total samples for on-surface triplet generation. "
             "Range [0.0, 0.5]. Default: 0.0 (disabled). "
             "Each triplet produces P0 (on-surface) plus verified P+ and P- offsets. "
             "Due to verification failures the final count may vary.",
    )
    parser.add_argument(
        "--triplet-epsilon",
        dest="triplet_epsilon",
        type=float,
        default=0.02,
        help="Relative offset for triplet P+ and P- from P0 along surface normal. "
             "Expressed as a fraction of the mesh's smallest AABB dimension "
             "(e.g. 0.02 = 2%% of the thinnest axis). "
             "If verification fails, the absolute epsilon is halved once before abandoning the point. "
             "Default: 0.02.",
    )
    parser.add_argument(
        "--anisotropic-bias",
        dest="anisotropic_bias",
        action="store_true",
        help="Enable AABB-aware anisotropic near-surface noise for thin shapes. "
             "Scales Gaussian perturbations by the mesh AABB aspect ratio "
             "(thin axes get smaller perturbations) so more samples remain inside "
             "the interior slab. Uniform sampling still uses the full unit cube. "
             "This increases interior sample density for thin-walled objects.",
    )
    parser.add_argument(
        "--bounding-cube",
        dest="bounding_cube",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("A", "B", "C"),
        help="Per-axis extents of the bounding box for uniform SDF sampling. "
             "Uniform samples are drawn from [-A/2, A/2] x [-B/2, B/2] x [-C/2, C/2]. "
             "Default: 1.0 1.0 1.0 (i.e. [-0.5, 0.5]^3). "
             "Increase an axis to extend uniform coverage beyond the unit sphere "
             "along that direction; decrease to concentrate samples closer to the origin.",
    )
    parser.add_argument(
        "--uniform-ratio",
        dest="uniform_ratio",
        type=float,
        default=0.06,
        help="Fraction of non-triplet samples allocated to uniform random sampling "
             "(the rest go to near-surface perturbation). Default: 0.06 (3/50), "
             "matching the original DeepSDF C++ code. Increase for thin/flat shapes "
             "to improve background coverage at the cost of near-surface density.",
    )

    args = parser.parse_args(argv)

    # Check required dependencies based on method
    if args.sdf_method == "knn" and cKDTree is None and _PyKDTree is None:
        print("ERROR: scipy (cKDTree) or pykdtree is required for --sdf-method=knn", file=sys.stderr)
        return 2

    if args.sdf_method == "igl":
        try:
            import igl
        except ImportError:
            print("ERROR: libigl is required for --sdf-method=igl. Install with: pip install libigl", file=sys.stderr)
            return 2

    # raycast uses trimesh.proximity which is always available with trimesh

    if args.normal_offset and args.anisotropic_bias:
        print("WARNING: --normal-offset overrides --anisotropic-bias "
              "(normal offset already prevents cross-wall contamination by construction).",
              file=sys.stderr)

    if args.test_mode:
        args.variance = 0.05
        args.num_sample = 250000

    second_variance = args.variance / 10.0
    if args.test_mode:
        second_variance = args.variance / 100.0

    # Load mesh (no processing to stay close to input)
    mesh = trimesh.load(args.mesh_file, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"ERROR: could not load mesh from {args.mesh_file}", file=sys.stderr)
        return 2

    # Merge duplicate vertices - CRITICAL for correct SDF computation
    # Without this, meshes with split vertices (common in OBJ files) will
    # have incorrect topology, leading to wrong SDF signs
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    # Normalize into unit sphere (same as C++)
    radius = bounding_cube_normalization(mesh, fit_to_unit_sphere=True)

    # Compute AABB metrics for optional anisotropic interior-biased sampling
    used_idx_norm = np.unique(mesh.faces.reshape(-1))
    used_norm = mesh.vertices[used_idx_norm]
    aabb_lo = used_norm.min(axis=0)
    aabb_hi = used_norm.max(axis=0)
    aabb_dims = aabb_hi - aabb_lo
    aniso_scale = aabb_dims / (aabb_dims.max() + 1e-12)
    aniso_scale = np.clip(aniso_scale, 0.1, 1.0)

    if args.anisotropic_bias and not args.quiet:
        print(f"Anisotropic bias enabled:")
        print(f"  Aniso scale: {aniso_scale}")

    bounding_cube_extents = np.array(args.bounding_cube, dtype=np.float32) * float(radius)
    if not args.quiet:
        e = bounding_cube_extents
        print(f"Bounding cube extents: [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}] "
              f"(ranges: [{-e[0]/2:.4f}, {e[0]/2:.4f}] x [{-e[1]/2:.4f}, {e[1]/2:.4f}] x [{-e[2]/2:.4f}, {e[2]/2:.4f}])")

    # Optional manifoldness check
    if args.check_manifold:
        report = check_mesh_manifoldness(mesh)
        if not report.is_manifold:
            print("Warning: mesh is not manifold!", file=sys.stderr)
            print(f"  Non-manifold edges: {report.non_manifold_edges}", file=sys.stderr)
            print(f"  Boundary edges: {report.boundary_edges}", file=sys.stderr)
            print(f"  Non-manifold vertices: {report.non_manifold_vertices}", file=sys.stderr)
            if args.strict_manifold:
                print("Error: exiting due to --strict-manifold flag", file=sys.stderr)
                return 1

    # Orientation consistency (fix mixed winding)
    faces_flipped = ensure_orientation_consistency(mesh)
    if not args.quiet:
        if faces_flipped > 0:
            print(f"Orientation consistency: fixed {faces_flipped} faces")
        else:
            print("Orientation consistency: OK")

    # Calculate sample counts with triplet allocation
    on_surface_ratio = getattr(args, 'on_surface_ratio', 0.0)
    if on_surface_ratio < 0.0 or on_surface_ratio > 0.5:
        print(f"WARNING: --on-surface-ratio {on_surface_ratio} outside [0.0, 0.5], clamping",
              file=sys.stderr)
        on_surface_ratio = max(0.0, min(0.5, on_surface_ratio))

    total_samples = args.num_sample
    target_triplets = int(total_samples * on_surface_ratio)
    remaining_samples = total_samples - target_triplets
    uniform_ratio = max(0.0, min(1.0, args.uniform_ratio))
    n_uniform = int(remaining_samples * uniform_ratio)
    n_surf = remaining_samples - n_uniform

    triplet_epsilon_rel = getattr(args, 'triplet_epsilon', 0.02)
    min_dim = float(aabb_dims.min())
    triplet_epsilon = triplet_epsilon_rel * min_dim

    rng = np.random.default_rng()

    if not args.quiet:
        print(f"Sample allocation:")
        print(f"  Target triplets:    {target_triplets}")
        print(f"  Triplet epsilon:    {triplet_epsilon:.6f}  (rel={triplet_epsilon_rel:.3f} * min_dim={min_dim:.6f})")
        print(f"  Near-surface base:  {n_surf}")
        print(f"  Uniform samples:    {n_uniform}  (ratio={uniform_ratio:.2%})")

    # Generate triplet samples (P0, P+, P-)
    if target_triplets > 0:
        if args.sdf_method in ["igl", "knn"]:
            import igl
            V_igl = np.asarray(mesh.vertices, dtype=np.float64)
            F_igl = np.asarray(mesh.faces, dtype=np.int64)
            P0, Pp, Pm, S0, Sp, Sm = sample_sdf_triplets_libigl(
                V_igl, F_igl, target_triplets, triplet_epsilon, rng
            )
        else:
            P0, Pp, Pm, S0, Sp, Sm = sample_sdf_triplets_trimesh(
                mesh, target_triplets, triplet_epsilon, rng
            )

        parts_xyz = [P0]
        parts_sdf = [S0]
        if len(Pp) > 0:
            parts_xyz.append(Pp)
            parts_sdf.append(Sp)
        if len(Pm) > 0:
            parts_xyz.append(Pm)
            parts_sdf.append(Sm)
        xyz_triplet = np.vstack(parts_xyz)
        sdfs_triplet = np.concatenate(parts_sdf)

        if not args.quiet:
            print(f"Triplet results: {len(P0)} P0, {len(Pp)} P+, {len(Pm)} P-")
    else:
        xyz_triplet = np.empty((0, 3), dtype=np.float32)
        sdfs_triplet = np.empty((0,), dtype=np.float32)

    # Branch based on SDF method
    if args.sdf_method == "knn":
        # Original KD-tree + k-NN voting method
        xyz_surf, _ = sample_from_surface(mesh, n_surf)
        if not args.quiet:
            print(f"num_samp_near_surf: {len(xyz_surf)}")

        used_vertices, used_normals = compute_used_vertex_normals(mesh)
        if len(used_vertices) == 0:
            print("Error: no vertices referenced by faces; cannot build KD-tree", file=sys.stderr)
            return 1

        kd_tree = _KDTree(used_vertices)
        backend = getattr(kd_tree, "_backend", "unknown")
        if not args.quiet:
            print(f"KD-tree backend: {backend}")
            print(f"SDF method: knn (KD-tree + voting)")

        num_votes = 11
        # Use remaining_samples instead of args.num_sample for correct allocation
        num_rand_samples = int(remaining_samples - len(xyz_surf))

        xyz, sdfs = sample_sdf_near_surface(
            kd_tree=kd_tree,
            used_vertices=used_vertices,
            used_normals=used_normals,
            xyz_surf=xyz_surf,
            num_rand_samples=num_rand_samples,
            variance=float(args.variance),
            second_variance=float(second_variance),
            bounding_cube_extents=bounding_cube_extents,
            num_votes=num_votes,
            rng=rng,
            aniso_scale=aniso_scale if (args.anisotropic_bias and not args.normal_offset) else None,
        )
    elif args.sdf_method == "igl":
        # libigl accurate SDF method
        import igl

        # For igl method, we need face indices to get proper normals
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_surf)
        xyz_surf = pts.astype(np.float32)
        if not args.quiet:
            print(f"num_samp_near_surf: {len(xyz_surf)}")

        # Convert mesh to libigl format (float64, int64 required)
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)

        # Get surface normals for offset direction
        if args.normal_offset:
            surf_normals = mesh.face_normals[face_idx].astype(np.float64)
        else:
            # For isotropic perturbation, normals not used but need dummy array
            surf_normals = np.zeros_like(xyz_surf, dtype=np.float64)

        if not args.quiet:
            print(f"SDF method: igl (libigl pseudonormal)")
            print(f"  normal_offset: {args.normal_offset}")
            # Verify mesh watertightness (required for pseudonormal)
            is_edge_manifold = igl.is_edge_manifold(F)
            boundary = igl.boundary_facets(F)
            print(f"  Edge manifold: {is_edge_manifold}")
            print(f"  Boundary edges: {len(boundary)}")
            if not is_edge_manifold or len(boundary) > 0:
                print("  WARNING: Mesh not watertight - pseudonormal may produce errors")

        # Use remaining_samples instead of args.num_sample for correct allocation
        num_rand_samples = int(remaining_samples - len(xyz_surf))

        xyz, sdfs = sample_sdf_near_surface_libigl_hybrid(
            V=V,
            F=F,
            xyz_surf=xyz_surf,
            surf_normals=surf_normals,
            num_rand_samples=num_rand_samples,
            variance=float(args.variance),
            second_variance=float(second_variance),
            bounding_cube_extents=bounding_cube_extents,
            normal_offset=args.normal_offset,
            rng=rng,
            aniso_scale=aniso_scale if (args.anisotropic_bias and not args.normal_offset) else None,
        )
    else:  # args.sdf_method == "raycast"
        # trimesh proximity + ray casting method
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_surf)
        xyz_surf = pts.astype(np.float32)
        if not args.quiet:
            print(f"num_samp_near_surf: {len(xyz_surf)}")

        if not args.quiet:
            print(f"SDF method: raycast (trimesh proximity + ray casting)")
            print(f"  normal_offset: {args.normal_offset}")
            # Check watertight status for ray casting
            is_watertight = mesh.is_watertight
            print(f"  Watertight: {is_watertight}")
            if not is_watertight:
                print("  WARNING: Mesh not watertight - ray casting may produce errors near boundaries")

        # Use remaining_samples instead of args.num_sample for correct allocation
        num_rand_samples = int(remaining_samples - len(xyz_surf))

        xyz, sdfs = sample_sdf_near_surface_trimesh(
            mesh=mesh,
            xyz_surf=xyz_surf,
            num_rand_samples=num_rand_samples,
            variance=float(args.variance),
            second_variance=float(second_variance),
            bounding_cube_extents=bounding_cube_extents,
            normal_offset=args.normal_offset,
            rng=rng,
            aniso_scale=aniso_scale if (args.anisotropic_bias and not args.normal_offset) else None,
        )

    # Combine near-surface/uniform samples with triplet samples
    if len(xyz_triplet) > 0:
        xyz = np.concatenate([xyz, xyz_triplet], axis=0)
        sdfs = np.concatenate([sdfs, sdfs_triplet], axis=0)

        # Shuffle to mix triplet points with other samples (important for training)
        shuffle_idx = rng.permutation(len(xyz))
        xyz = xyz[shuffle_idx]
        sdfs = sdfs[shuffle_idx]

        if not args.quiet:
            n_pos = np.sum(sdfs > 0)
            n_neg = np.sum(sdfs < 0)
            n_zero = np.sum(sdfs == 0)
            print(f"Final sample counts:")
            print(f"  Positive (outside): {n_pos}")
            print(f"  Negative (inside):  {n_neg}")
            print(f"  Zero (on-surface):  {n_zero}")

    # Optional PLY dumps
    if args.save_ply:
        write_sdf_to_ply(xyz, sdfs, "sdf_samples.ply")
    if args.ply_path:
        write_sdf_to_ply(xyz, sdfs, args.ply_path, neg_only=False, pos_only=False)

    # Write output
    out = args.output_file
    if out.lower().endswith(".npz"):
        write_sdf_to_npz(xyz, sdfs, out, print_num=True, quiet=args.quiet)
    elif out.lower().endswith(".npy"):
        write_sdf_to_npy(xyz, sdfs, out)
    else:
        # default to npz, matching pipeline expectation
        write_sdf_to_npz(xyz, sdfs, out, print_num=True, quiet=args.quiet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
