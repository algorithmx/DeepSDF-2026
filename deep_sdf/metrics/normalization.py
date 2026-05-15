"""Single source of truth for mesh normalization parameters.

Normalization pipeline (mirrors bin/utils_pp.py compute_normalization_parameters):
  1. centre = midpoint of AABB of vertices used by faces
  2. max_dist = max distance from centre to any used vertex, times buffer (1.03)
  3. offset = -centre
  4. scale = 1.0 / max_dist

Forward (preprocessing):  normalized = (raw - centre) / max_dist = (raw + offset) * scale
Inverse (denormalization): raw = normalized / scale - offset
"""

import os
import numpy as np
import trimesh

BUFFER = 1.03


def compute_from_mesh(mesh: trimesh.Trimesh, buffer: float = BUFFER):
    """Compute (offset, scale) from a mesh, using only vertices referenced by faces."""
    used_idx = np.unique(mesh.faces)
    used_verts = mesh.vertices[used_idx]

    lo = used_verts.min(axis=0)
    hi = used_verts.max(axis=0)
    centre = (lo + hi) / 2.0

    max_dist = float(np.linalg.norm(used_verts - centre, axis=1).max()) * buffer

    offset = (-centre).astype(np.float32)
    scale = np.float32(1.0 / max_dist)
    return offset, scale


def load_or_compute(data_dir, dataset, class_name, shape_id, gt_mesh=None):
    """Load normalization params from .npz file if it exists, otherwise compute on-the-fly.

    Returns (offset: np.ndarray shape (3,), scale: np.float32).
    """
    npz_path = os.path.join(
        data_dir, "NormalizationParameters", dataset, class_name, shape_id + ".npz"
    )
    if os.path.exists(npz_path):
        params = np.load(npz_path)
        return params["offset"], params["scale"]

    if gt_mesh is None:
        raise FileNotFoundError(
            f"No NormalizationParameters file at {npz_path} and no gt_mesh provided"
        )
    return compute_from_mesh(gt_mesh)


def denormalize_points(normalized_pts, offset, scale):
    """Convert normalized [-1,1] coordinates back to raw world coordinates.

    raw = normalized / scale - offset
    """
    return normalized_pts / scale - offset
