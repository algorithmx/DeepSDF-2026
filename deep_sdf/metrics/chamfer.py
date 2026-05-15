#!/usr/bin/env python3
"""Chamfer distance computation for DeepSDF evaluation.

All public functions denormalize reconstructed points to raw world coordinates
before computing CD, ensuring consistent units regardless of whether
NormalizationParameters files exist.
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

from deep_sdf.metrics.normalization import load_or_compute, denormalize_points


def compute_chamfer(
    gt_mesh: trimesh.Trimesh,
    recon_mesh: trimesh.Trimesh,
    data_dir: str = None,
    dataset: str = None,
    class_name: str = None,
    shape_id: str = None,
    num_samples: int = 30000,
):
    """Compute symmetric chamfer distance between GT mesh and reconstructed mesh.

    Handles normalization automatically:
      - Loads .npz normalization params if available
      - Falls back to computing normalization from GT mesh on-the-fly
      - Denormalizes recon points to raw coordinates before comparison
      - Samples surface points from both meshes for fair comparison

    Returns CD in raw world-coordinate units² (symmetric: both directions summed).
    """
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
    recon_pts, _ = trimesh.sample.sample_surface(recon_mesh, num_samples)

    offset, scale = load_or_compute(
        data_dir, dataset, class_name, shape_id, gt_mesh=gt_mesh
    )
    recon_pts_raw = denormalize_points(recon_pts, offset, scale)

    gt_tree = KDTree(gt_pts)
    recon_tree = KDTree(recon_pts_raw)

    d_gt_to_recon, _ = recon_tree.query(gt_pts)
    d_recon_to_gt, _ = gt_tree.query(recon_pts_raw)

    return float(np.mean(np.square(d_gt_to_recon)) + np.mean(np.square(d_recon_to_gt)))


def compute_trimesh_chamfer(
    gt_points, gen_mesh, offset, scale, num_mesh_samples=30000
):
    """Legacy API: compute CD given pre-loaded offset and scale.

    Kept for backward compatibility with scripts that already have normalization params.
    """
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
    gen_points_sampled = denormalize_points(gen_points_sampled, offset, scale)

    gt_points_np = gt_points.vertices

    gen_tree = KDTree(gen_points_sampled)
    gt_tree = KDTree(gt_points_np)

    d1, _ = gen_tree.query(gt_points_np)
    d2, _ = gt_tree.query(gen_points_sampled)

    return float(np.mean(np.square(d1)) + np.mean(np.square(d2)))
