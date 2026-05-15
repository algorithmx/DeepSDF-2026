#!/usr/bin/env python3
"""
SampleMeshSurface.py — pure-Python/trimesh drop-in replacement for the
compiled SampleVisibleMeshSurface binary.

For watertight meshes every surface point is reachable from the outside, so
the OpenGL multi-viewpoint visibility pass used by the C++ binary is
completely redundant.  This script performs the same work with trimesh:

  1. Load the input mesh.
  2. Area-weighted random sample `num_sample` points from the mesh surface.
  3. Write the samples to a PLY file in the same format as the C++ binary
     (vertex list + face list grouping every 3 consecutive points as a
      triangle), so downstream code that loads the PLY as a mesh still works.
  4. Optionally write normalization parameters (offset, scale) to an NPZ file
     using the same convention as ComputeNormalizationParameters() in Utils.cpp:
       offset = -bbox_centre            (add to raw coords to centre them)
       scale  = 1 / (max_dist * 1.03)  (multiply centred coords to fit sphere)

CLI is intentionally identical to the compiled binary:
  -m <mesh_file>   input mesh  (any format trimesh can load)
  -o <ply_file>    output PLY surface samples
  -n <npz_file>    (optional) output NPZ normalization parameters
  -s <int>         number of samples  (default 30 000)
"""

import argparse
import io
import sys
import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_points_to_ply(verts: np.ndarray, filepath: str) -> None:
    """
    Write *verts* to *filepath* as an ASCII PLY file.

    Replicates the C++ SavePointsToPLY() layout exactly:
      - element vertex  N
      - element face    N//3   (every 3 consecutive vertices form a triangle)

    This matches what the downstream evaluate.py / Chamfer-distance code
    expects when it loads the PLY as a triangle mesh.
    """
    n = len(verts)
    n_faces = n // 3

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_index\n"
        "end_header\n"
    )

    # Vectorised vertex block: one C-level savetxt call instead of N Python
    # f-string + write calls.  fmt="%.8g" is byte-for-byte equivalent to
    # the original f"{v[0]:.8g} {v[1]:.8g} {v[2]:.8g}" format.
    vert_buf = io.BytesIO()
    if n > 0:
        np.savetxt(vert_buf, verts, fmt="%.8g")
    vertex_block = vert_buf.getvalue()

    # Vectorised face block: build the [3, i, i+1, i+2] rows with numpy
    # then write them in one savetxt call.
    face_buf = io.BytesIO()
    if n_faces > 0:
        base = np.arange(n_faces, dtype=np.int32) * 3
        face_arr = np.column_stack(
            [np.full(n_faces, 3, dtype=np.int32), base, base + 1, base + 2]
        )
        np.savetxt(face_buf, face_arr, fmt="%d")
    face_block = face_buf.getvalue()

    with open(filepath, "wb") as f:
        f.write(header.encode())
        f.write(vertex_block)
        f.write(face_block)


def save_normalization_params(offset: np.ndarray, scale: float, filepath: str) -> None:
    """
    Write normalization parameters to an NPZ file.

    Replicates the C++ SaveNormalizationParamsToNPZ():
      offset : float32 array of shape (3,)
      scale  : float32 scalar stored as shape (1,)
    """
    np.savez(
        filepath,
        offset=np.array(offset, dtype=np.float32),
        scale=np.array([scale], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

from utils_pp import compute_normalization_parameters
from sampling_pp import area_weighted_sampling_from_mesh_surface

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SampleMeshSurface: trimesh-based surface sampling for watertight meshes. "
            "Drop-in replacement for the compiled SampleVisibleMeshSurface binary."
        )
    )
    parser.add_argument(
        "-m", required=True, dest="mesh_file",
        help="Input mesh file (OBJ, STL, OFF, PLY, …)",
    )
    parser.add_argument(
        "-o", required=True, dest="ply_out",
        help="Output PLY file with surface samples",
    )
    parser.add_argument(
        "-n", dest="norm_out", default=None,
        help="(Optional) Output NPZ file with normalization parameters",
    )
    parser.add_argument(
        "-s", dest="num_sample", type=int, default=30_000,
        help="Number of surface samples (default: 30000)",
    )
    parser.add_argument(
        "--quiet", dest="quiet", action="store_true",
        help="Suppress output",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load mesh
    # ------------------------------------------------------------------
    mesh = trimesh.load(args.mesh_file, force="mesh", process=False)

    # Merge duplicate vertices - ensures consistent vertex counts and
    # accurate normalization parameters (matches PreprocessMesh.py behavior)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    if not isinstance(mesh, trimesh.Trimesh):
        print(
            f"ERROR: could not load a single mesh from '{args.mesh_file}'",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.quiet:
        print(f"1 objects")   # mimic the C++ stdout so any log parser still works

    # ------------------------------------------------------------------
    # Sample surface
    # ------------------------------------------------------------------
    if len(mesh.faces) == 0:
        print(
            f"Warning: no faces found in '{args.mesh_file}' – writing empty PLY.",
            file=sys.stderr,
        )
        save_points_to_ply(np.empty((0, 3), dtype=np.float32), args.ply_out)
    else:
        points = area_weighted_sampling_from_mesh_surface(mesh, args.num_sample)
        save_points_to_ply(points, args.ply_out)

    # ------------------------------------------------------------------
    # Normalization parameters
    # ------------------------------------------------------------------
    if args.norm_out:
        offset, scale = compute_normalization_parameters(mesh)
        save_normalization_params(offset, float(scale), args.norm_out)

    if not args.quiet:
        print("ended correctly")


if __name__ == "__main__":
    main()
