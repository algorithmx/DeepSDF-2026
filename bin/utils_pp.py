from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import trimesh


from scipy.spatial import cKDTree

# Fast KD-tree (pykdtree is typically 3-10× faster than scipy for
# float32 batch queries; falls back to scipy cKDTree transparently).
from pykdtree.kdtree import KDTree as _PyKDTree


class _KDTree:
    """Thin wrapper normalising cKDTree / pykdtree query interfaces.

    Both backends are fed float32 data.  The wrapper exposes a single
    ``query(pts, k)`` method that returns ``(dists, idxs)`` in the same
    shape as ``scipy.cKDTree.query``.
    """

    def __init__(self, pts: np.ndarray) -> None:
        pts = np.asarray(pts, dtype=np.float32)
        if _PyKDTree is not None:
            self._tree = _PyKDTree(pts)
            self._backend = "pykdtree"
        elif cKDTree is not None:
            self._tree = cKDTree(pts)
            self._backend = "scipy"
        else:
            raise RuntimeError("Neither pykdtree nor scipy is available.")

    def query(self, pts: np.ndarray, k: int):
        pts = np.asarray(pts, dtype=np.float32)
        if self._backend == "pykdtree":
            # pykdtree returns squared distances; we sqrt for API parity
            dists_sq, idxs = self._tree.query(pts, k=k)
            return np.sqrt(dists_sq), idxs
        else:
            return self._tree.query(pts, k=k, workers=-1)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ManifoldnessReport:
    is_manifold: bool
    non_manifold_edges: int
    boundary_edges: int
    non_manifold_vertices: int
    total_faces: int
    total_vertices: int
    total_edges: int


# ----------------------------
# Normalization
# ----------------------------

def bounding_cube_normalization(
    mesh: trimesh.Trimesh,
    fit_to_unit_sphere: bool,
    buffer: float = 1.03,
) -> float:
    """Normalise mesh into the unit sphere (with buffer).

    Delegates to :func:`compute_normalization_parameters` for the actual
    centre / max-distance computation, then applies the transform in-place.

    Returns
    -------
    bounding_cube_dim : float
        1.0 when ``fit_to_unit_sphere=True``, else the unscaled max distance.
    """
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return 1.0

    offset, scale = compute_normalization_parameters(mesh, buffer)
    centre = -offset
    max_dist = 1.0 / float(scale)

    verts = mesh.vertices.copy()
    verts -= centre

    if max_dist <= 0:
        max_dist = 1.0

    if fit_to_unit_sphere:
        verts /= max_dist
        mesh.vertices = verts
        return 1.0

    mesh.vertices = verts
    return max_dist


def compute_normalization_parameters(
    mesh: trimesh.Trimesh,
    buffer: float = 1.03,
):
    """
    Replicate ComputeNormalizationParameters() from Utils.cpp.

    Logic (uses only *used* vertices, i.e. those that appear in faces):
      centre    = midpoint of the axis-aligned bounding box of used vertices
      max_dist  = maximum distance from centre to any used vertex   (× buffer)
      offset    = -centre
      scale     = 1.0 / max_dist

    Returns
    -------
    offset : np.ndarray  shape (3,), dtype float32
    scale  : float32
    """
    # Gather only vertices that appear in at least one face (mirrors C++ logic)
    used_idx = np.unique(mesh.faces)
    used_verts = mesh.vertices[used_idx]

    lo = used_verts.min(axis=0)
    hi = used_verts.max(axis=0)
    centre = (lo + hi) / 2.0

    max_dist = float(np.linalg.norm(used_verts - centre, axis=1).max()) * buffer

    offset = (-centre).astype(np.float32)
    scale = np.float32(1.0 / max_dist)

    return offset, scale

# ----------------------------
# Orientation consistency
# ----------------------------

def ensure_orientation_consistency(mesh: trimesh.Trimesh) -> int:
    """Flood-fill orientation consistency, mirroring `EnsureOrientationConsistency`.

    Returns number of faces flipped.

    Implementation overview:
    - Build undirected edge → list of incident (face_index, forward_direction)
      using numpy sort + groupby instead of a Python per-face loop.
    - Build face adjacency with toggle rule:
        if two faces traverse a shared undirected edge in the same direction,
        they must have opposite flip parity.
    - BFS (FIFO, matching C++ std::queue) each component to assign flip parity.
    - Vectorised: compute component signed volume; flip entire component if < 0.
    - Vectorised: apply flips by swapping columns 1 and 2.
    """
    faces = mesh.faces
    if len(faces) == 0:
        return 0

    F = len(faces)

    # ------------------------------------------------------------------
    # 1. Vectorised edge construction  (replaces O(3F) Python dict loop)
    # ------------------------------------------------------------------
    i0 = faces[:, 0].astype(np.int64)
    i1 = faces[:, 1].astype(np.int64)
    i2 = faces[:, 2].astype(np.int64)
    face_ids = np.arange(F, dtype=np.int32)

    # Three directed edges per face: (i0→i1), (i1→i2), (i2→i0)
    a_all = np.concatenate([i0, i1, i2])           # (3F,)
    b_all = np.concatenate([i1, i2, i0])           # (3F,)
    f_all = np.tile(face_ids, 3)                   # (3F,)

    # Drop degenerate edges (same vertex at both ends)
    valid_e = a_all != b_all
    a_all = a_all[valid_e]
    b_all = b_all[valid_e]
    f_all = f_all[valid_e]

    # Undirected key: (min, max); `fwd` records whether a == min-end
    mi  = np.minimum(a_all, b_all)
    ma  = np.maximum(a_all, b_all)
    fwd = (a_all == mi).astype(np.uint8)

    # Pack into int64 for a stable sort
    V_max  = int(faces.max()) + 2
    packed = mi * V_max + ma

    sort_order = np.argsort(packed, kind="stable")
    packed_s   = packed[sort_order]
    f_s        = f_all[sort_order]
    fwd_s      = fwd[sort_order]

    # Group edges by packed key
    split_pts  = np.flatnonzero(np.diff(packed_s)) + 1
    groups_f   = np.split(f_s,   split_pts)
    groups_fwd = np.split(fwd_s, split_pts)

    # ------------------------------------------------------------------
    # 2. Build adjacency list  (O(E) Python — E unique edges, not 3F)
    # ------------------------------------------------------------------
    adjacency: List[List[Tuple[int, bool]]] = [[] for _ in range(F)]
    for gf, gfwd in zip(groups_f, groups_fwd):
        ng = len(gf)
        if ng < 2:
            continue
        for ii in range(ng):
            fi    = int(gf[ii])
            fwd_i = bool(gfwd[ii])
            for jj in range(ii + 1, ng):
                fj       = int(gf[jj])
                fwd_j    = bool(gfwd[jj])
                same_dir = fwd_i == fwd_j
                adjacency[fi].append((fj, same_dir))
                adjacency[fj].append((fi, same_dir))

    # ------------------------------------------------------------------
    # 3. BFS — sequential, FIFO to match C++ std::queue traversal order
    #    Volume computation is intentionally removed from here and done
    #    vectorised below; the BFS only assigns flip parity + component id.
    # ------------------------------------------------------------------
    visited      = np.zeros(F, dtype=np.uint8)
    flip         = np.zeros(F, dtype=np.uint8)
    component_id = np.full(F, -1, dtype=np.int32)
    num_components = 0

    for start in range(F):
        if visited[start]:
            continue
        queue: deque[int] = deque()
        queue.append(start)
        visited[start]      = 1
        flip[start]         = 0
        component_id[start] = num_components

        while queue:
            cur = queue.popleft()           # FIFO — matches C++ std::queue
            for nb, toggle in adjacency[cur]:
                desired = flip[cur] ^ (1 if toggle else 0)
                if not visited[nb]:
                    visited[nb]      = 1
                    flip[nb]         = desired
                    component_id[nb] = num_components
                    queue.append(nb)
                # else: keep first assignment (same as C++ heuristic)

        num_components += 1

    # ------------------------------------------------------------------
    # 4. Vectorised signed-volume computation per component
    #
    #    For face fi with flip[fi]==0: use winding (v0, v1, v2)
    #    For face fi with flip[fi]==1: use winding (v0, v2, v1)
    #    signed_volume = dot(v0, cross(v1, v2)) / 6
    # ------------------------------------------------------------------
    verts      = mesh.vertices
    is_flipped = flip.astype(bool)

    f1_idx = np.where(is_flipped, faces[:, 2], faces[:, 1])  # (F,)
    f2_idx = np.where(is_flipped, faces[:, 1], faces[:, 2])  # (F,)

    v0_arr = verts[faces[:, 0]]   # (F, 3)
    v1_arr = verts[f1_idx]         # (F, 3)
    v2_arr = verts[f2_idx]         # (F, 3)

    face_vols = np.einsum("fi,fi->f", v0_arr, np.cross(v1_arr, v2_arr)) / 6.0

    comp_vols = np.zeros(num_components, dtype=np.float64)
    np.add.at(comp_vols, component_id, face_vols)
    flip_component = (comp_vols < 0.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # 5. Vectorised flip application — swap columns 1 and 2
    # ------------------------------------------------------------------
    do_flip = (
        (flip.astype(np.int32) ^ flip_component[component_id].astype(np.int32)) != 0
    )
    faces_new = faces.copy()
    tmp                  = faces_new[do_flip, 1].copy()
    faces_new[do_flip, 1] = faces_new[do_flip, 2]
    faces_new[do_flip, 2] = tmp
    faces_flipped = int(do_flip.sum())

    mesh.faces = faces_new
    return faces_flipped


# ----------------------------
# Manifoldness check (optional)
# ----------------------------

def check_mesh_manifoldness(mesh: trimesh.Trimesh) -> ManifoldnessReport:
    """Approximate-equivalent of `CheckMeshManifoldness` (edges + vertex fans).

    - Non-manifold edges: edges with >2 incident faces
    - Boundary edges: edges with exactly 1 incident face
    - Non-manifold vertices: vertices whose incident faces form >1 disconnected fan

    This is used only when `--check-manifold` is passed.
    """
    v = len(mesh.vertices)
    f = len(mesh.faces)

    if f == 0:
        return ManifoldnessReport(
            is_manifold=False,
            non_manifold_edges=0,
            boundary_edges=0,
            non_manifold_vertices=0,
            total_faces=0,
            total_vertices=v,
            total_edges=0,
        )

    # Edge counts
    edges = mesh.edges_sorted
    # Each face contributes 3 edges; group identical edges
    edges_view = edges.view([("a", edges.dtype), ("b", edges.dtype)])
    uniq, counts = np.unique(edges_view, return_counts=True)

    boundary_edges = int(np.sum(counts == 1))
    non_manifold_edges = int(np.sum(counts > 2))
    total_edges = int(len(uniq))

    # Vertex fan check
    # Build edge->faces map using mesh.edges_sorted (already sorted).
    edge_occ          = mesh.edges_sorted
    edge_occ_view     = edge_occ.view([("a", edge_occ.dtype), ("b", edge_occ.dtype)])
    face_for_edge_occ = mesh.edges_face

    sort_idx        = np.argsort(edge_occ_view, axis=0, order=("a", "b"))
    edge_occ_sorted = edge_occ[sort_idx]
    face_sorted     = face_for_edge_occ[sort_idx]

    edge_sorted_view = edge_occ_sorted.view(
        [("a", edge_occ_sorted.dtype), ("b", edge_occ_sorted.dtype)]
    )
    uniq2, start_idx, counts2 = np.unique(
        edge_sorted_view, return_index=True, return_counts=True
    )

    # -------------------------------------------------------------------
    # Build vertex_fan_adj: vertex → list[(fi, fj)] of face-pair adjacencies
    # that are connected in that vertex's fan.  For each edge (a, b) with
    # incident faces [fi, fj, ...], endpoints a and b both gain an adjacency
    # between each pair of those faces.
    #
    # Manifold edges (count == 2) are handled vectorised; non-manifold (> 2)
    # fall back to a Python loop (rare).
    # -------------------------------------------------------------------
    man_mask   = counts2 == 2
    man_starts = start_idx[man_mask]

    ea_uniq = uniq2[man_mask]
    ea      = np.array([int(u["a"]) for u in ea_uniq], dtype=np.int32)  # (E_man,)
    eb      = np.array([int(u["b"]) for u in ea_uniq], dtype=np.int32)  # (E_man,)
    fi_m    = face_sorted[man_starts]                                    # (E_man,)
    fj_m    = face_sorted[man_starts + 1]                               # (E_man,)

    # Pre-build lookup: vertex → [(fi, fj), ...]
    vertex_fan_adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    ea_list = ea.tolist()
    eb_list = eb.tolist()
    fi_list = fi_m.tolist()
    fj_list = fj_m.tolist()
    for k in range(len(ea_list)):
        pair = (fi_list[k], fj_list[k])
        vertex_fan_adj[ea_list[k]].append(pair)
        vertex_fan_adj[eb_list[k]].append(pair)

    # Non-manifold edges (count > 2) — Python fallback (rare)
    nm_s = start_idx[counts2 > 2].tolist()
    nm_c = counts2[counts2 > 2].tolist()
    for s, c in zip(nm_s, nm_c):
        flist  = face_sorted[s : s + c]
        edge_a = int(edge_occ_sorted[s, 0])
        edge_b = int(edge_occ_sorted[s, 1])
        for ii in range(c):
            for jj in range(ii + 1, c):
                pair = (int(flist[ii]), int(flist[jj]))
                vertex_fan_adj[edge_a].append(pair)
                vertex_fan_adj[edge_b].append(pair)

    # Per-vertex fan connectivity check using pre-built vertex_fan_adj.
    # Avoids per-face edge scanning + per-edge dict lookup that the original
    # implementation performed inside the per-vertex loop.
    vertex_faces_arr  = mesh.vertex_faces  # (V, max_degree), padded with -1
    non_manifold_vertices = 0

    for vi in range(v):
        inc = vertex_faces_arr[vi]
        inc = inc[inc >= 0]
        if len(inc) <= 1:
            continue

        inc_set = set(int(x) for x in inc)

        # Build tiny face adjacency for this vertex's fan
        face_adj_v: Dict[int, List[int]] = {fi: [] for fi in inc_set}
        for fi_p, fj_p in vertex_fan_adj.get(vi, []):
            if fi_p in inc_set and fj_p in inc_set:
                face_adj_v[fi_p].append(fj_p)
                face_adj_v[fj_p].append(fi_p)

        # Count connected components (DFS)
        unvisited  = set(inc_set)
        components = 0
        while unvisited:
            seed  = next(iter(unvisited))
            stack = [seed]
            unvisited.discard(seed)
            while stack:
                cur = stack.pop()
                for nb in face_adj_v.get(cur, []):
                    if nb in unvisited:
                        unvisited.discard(nb)
                        stack.append(nb)
            components += 1
            if components > 1:
                break

        if components > 1:
            non_manifold_vertices += 1

    is_manifold = (non_manifold_edges == 0 and non_manifold_vertices == 0)
    return ManifoldnessReport(
        is_manifold=is_manifold,
        non_manifold_edges=non_manifold_edges,
        boundary_edges=boundary_edges,
        non_manifold_vertices=non_manifold_vertices,
        total_faces=f,
        total_vertices=v,
        total_edges=total_edges,
    )
