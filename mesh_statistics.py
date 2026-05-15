#!/usr/bin/env python3
"""Compute per-axis mesh extent statistics after unit-sphere normalization.

Loads every mesh referenced by a split file, applies the same normalization
as PreprocessMesh.py (merge vertices, bounding_cube_normalization with 1.03
buffer), and records the per-axis extent of used vertices in the normalized
frame.

Output: percentile table of per-axis extents across the dataset, plus
recommended --bounding-cube values for several coverage targets.
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

from bin.PreprocessMesh import bounding_cube_normalization
import deep_sdf.data


def collect_extents(source_dir, source_name, split):
    """Yield (per_axis_min, per_axis_max) arrays for each normalised mesh."""
    class_directories = split[source_name]

    for class_dir, instance_dirs in class_directories.items():
        class_path = os.path.join(source_dir, class_dir)

        for instance_dir in instance_dirs:
            shape_dir = os.path.join(class_path, instance_dir)
            try:
                mesh_filename = deep_sdf.data.find_mesh_in_directory(shape_dir)
            except (deep_sdf.data.NoMeshFileError, deep_sdf.data.MultipleMeshFileError):
                continue

            mesh_path = os.path.join(shape_dir, mesh_filename)
            try:
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
            except Exception:
                continue

            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                continue

            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            bounding_cube_normalization(mesh, fit_to_unit_sphere=True)

            used_idx = np.unique(mesh.faces.reshape(-1))
            used = mesh.vertices[used_idx]

            yield used.min(axis=0), used.max(axis=0)


def compute_statistics(extents_lo, extents_hi):
    """Return summary statistics from collected extents.

    Returns dict with keys:
        per_axis_min, per_axis_max : (3,) overall min/max across dataset
        extents    : (N, 3) per-axis extent (hi - lo) for each mesh
        coverage_table : list of (fraction, a, b, c) for coverage targets
    """
    lo = np.array(extents_lo)
    hi = np.array(extents_hi)
    extents = hi - lo  # (N, 3)

    overall_min = lo.min(axis=0)
    overall_max = hi.max(axis=0)

    # For each candidate bounding cube (a, b, c), compute fraction of meshes
    # whose surface is fully contained within [-a/2, a/2] x [-b/2, b/2] x [-c/2, c/2].
    coverage_targets = [0.80, 0.90, 0.95, 0.99, 1.00]

    # Per-axis: find the extent that covers each target fraction of meshes.
    # "Covered" means the mesh's full axis range fits within [-a/2, a/2],
    # i.e. the mesh extent <= a.
    coverage_table = []
    for frac in coverage_targets:
        # Per-axis extent at this percentile
        axis_extents = np.percentile(extents, frac * 100, axis=0)  # (3,)
        coverage_table.append((frac, *axis_extents.tolist()))

    return {
        "overall_min": overall_min,
        "overall_max": overall_max,
        "extents": extents,
        "coverage_table": coverage_table,
    }


def print_report(stats, num_meshes):
    overall_min = stats["overall_min"]
    overall_max = stats["overall_max"]
    extents = stats["extents"]

    print(f"Meshes analysed: {num_meshes}")
    print()
    print("Per-axis coordinate range (normalised frame):")
    print(f"  X: [{overall_min[0]:+.4f}, {overall_max[0]:+.4f}]  extent: {overall_max[0] - overall_min[0]:.4f}")
    print(f"  Y: [{overall_min[1]:+.4f}, {overall_max[1]:+.4f}]  extent: {overall_max[1] - overall_min[1]:.4f}")
    print(f"  Z: [{overall_min[2]:+.4f}, {overall_max[2]:+.4f}]  extent: {overall_max[2] - overall_min[2]:.4f}")
    print()

    percentiles = [5, 10, 25, 50, 75, 90, 95, 99, 100]
    print("Per-axis extent distribution (extent = max - min of used vertices):")
    header = f"{'Pctile':>8s}  {'X':>8s}  {'Y':>8s}  {'Z':>8s}"
    print(header)
    print("-" * len(header))
    for p in percentiles:
        vals = np.percentile(extents, p, axis=0)
        print(f"  {p:5d}%  {vals[0]:8.4f}  {vals[1]:8.4f}  {vals[2]:8.4f}")
    print()

    # Aspect ratio statistics
    if len(extents) > 0:
        max_extents = extents.max(axis=1, keepdims=True)
        max_extents = np.maximum(max_extents, 1e-12)
        aspect_ratios = extents / max_extents
        median_aspects = np.median(aspect_ratios, axis=0)
        print("Median aspect ratio (relative to longest axis):")
        print(f"  X: {median_aspects[0]:.4f}  Y: {median_aspects[1]:.4f}  Z: {median_aspects[2]:.4f}")
        print()

    print("--bounding-cube recommendations (extent values for coverage target):")
    print(f"  {'Coverage':>10s}  {'A':>8s}  {'B':>8s}  {'C':>8s}")
    print("  " + "-" * 42)
    for row in stats["coverage_table"]:
        frac = row[0]
        a, b, c = row[1], row[2], row[3]
        label = f"{frac * 100:.0f}%"
        print(f"  {label:>10s}  {a:8.4f}  {b:8.4f}  {c:8.4f}")

    print()
    print("Example usage:")
    for row in stats["coverage_table"]:
        frac = row[0]
        if frac in (0.90, 0.95):
            a, b, c = row[1], row[2], row[3]
            print(f"  python bin/PreprocessMesh.py ... --bounding-cube {a:.4f} {b:.4f} {c:.4f}  # {frac*100:.0f}% coverage")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-axis mesh extent statistics after normalisation. "
        "Outputs recommended --bounding-cube values for PreprocessMesh.py."
    )
    parser.add_argument(
        "--source", "-s", dest="source_dir", required=True,
        help="Root directory of source meshes (same as preprocess_data.py --source).",
    )
    parser.add_argument(
        "--name", "-n", dest="source_name", default=None,
        help="Dataset name in the split file. Defaults to the source directory basename.",
    )
    parser.add_argument(
        "--split", dest="split_filename", required=True,
        help="Split JSON file (same format as preprocess_data.py --split).",
    )
    parser.add_argument(
        "--threads", dest="num_threads", type=int, default=1,
        help="Number of parallel workers (default: 1, sequential).",
    )
    parser.add_argument(
        "--max-meshes", dest="max_meshes", type=int, default=None,
        help="Limit number of meshes to process (for quick testing).",
    )

    args = parser.parse_args()

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.source_name not in split:
        print(f"Error: '{args.source_name}' not found in split file. "
              f"Available: {list(split.keys())}", file=sys.stderr)
        sys.exit(1)

    extents_lo = []
    extents_hi = []
    count = 0

    for lo, hi in collect_extents(args.source_dir, args.source_name, split):
        extents_lo.append(lo)
        extents_hi.append(hi)
        count += 1

        if args.max_meshes and count >= args.max_meshes:
            print(f"(Limited to {args.max_meshes} meshes)", file=sys.stderr)
            break

        if count % 100 == 0:
            print(f"  processed {count} meshes...", file=sys.stderr)

    if count == 0:
        print("Error: no meshes found.", file=sys.stderr)
        sys.exit(1)

    stats = compute_statistics(extents_lo, extents_hi)
    print_report(stats, count)


if __name__ == "__main__":
    main()
