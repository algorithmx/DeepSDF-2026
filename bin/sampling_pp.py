from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import trimesh

# ----------------------------
# Surface sampling
# ----------------------------

def sample_from_surface(
    mesh: trimesh.Trimesh,
    num_sample: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weighted triangle sampling + per-sample face normal.

    Mirrors `SampleFromSurface(geom, xyz_surf, xyz_surf_normals, ...)`.

    Returns
    -------
    xyz_surf : (N, 3) float32
    n_surf   : (N, 3) float32 (face normal)
    """
    if len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    pts, face_idx = trimesh.sample.sample_surface(mesh, int(num_sample))
    pts = pts.astype(np.float32)

    # trimesh face_normals uses current winding
    fn = mesh.face_normals[face_idx]
    fn = fn.astype(np.float32)

    return pts, fn


def sample_on_surface_libigl(
    V: np.ndarray,
    F: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points exactly on the mesh surface (SDF=0) using libigl.

    Uses area-weighted random sampling to distribute points proportionally
    to triangle areas. This provides exact on-surface points without any
    perturbation, ideal for explicit SDF=0 supervision.

    Parameters
    ----------
    V : (|V|, 3) float64
        Mesh vertices in libigl format.
    F : (|F|, 3) int64
        Face indices in libigl format.
    num_samples : int
        Number of on-surface samples to generate.
    rng : np.random.Generator
        Random number generator (for reproducibility via seeding).

    Returns
    -------
    xyz_on_surf : (num_samples, 3) float32
        Surface points with exact SDF=0.
    sdfs_on_surf : (num_samples,) float32
        All zeros (by definition for on-surface points).

    Notes
    -----
    - Uses igl.random_points_on_mesh which performs area-weighted sampling
    - More stable than trimesh for edge cases (thin features, degenerate faces)
    - Returns barycentric coordinates internally, converted to 3D positions
    """
    import igl

    if num_samples <= 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # libigl random_points_on_mesh returns:
    #   B: (n, 3) barycentric coordinates
    #   FI: (n,) face indices
    #   X: (n, 3) 3D positions (already computed from barycentric + vertices)
    B, FI, X = igl.random_points_on_mesh(num_samples, V, F)

    # Convert to float32 for consistency with rest of pipeline
    xyz_on_surf = X.astype(np.float32)
    sdfs_on_surf = np.zeros(len(xyz_on_surf), dtype=np.float32)

    return xyz_on_surf, sdfs_on_surf


def sample_sdf_triplets_libigl(
    V: np.ndarray,
    F: np.ndarray,
    num_triplets: int,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate SDF triplets (P0, P+, P-) using libigl.

    P0 is on-surface (SDF=0). P+ = P0 + eps*n, P- = P0 - eps*n.
    Verifies that P+ has positive SDF and P- has negative SDF using
    igl.signed_distance. On failure, epsilon is halved once; if still
    failing, only the failed offset point is abandoned (P0 is always kept).

    Returns
    -------
    P0, P+, P-, S0, S+, S-
    """
    import igl

    if num_triplets <= 0:
        empty = np.empty((0, 3), dtype=np.float32)
        empty1d = np.empty((0,), dtype=np.float32)
        return empty, empty, empty, empty1d, empty1d, empty1d

    B, FI, P0 = igl.random_points_on_mesh(num_triplets, V, F)
    P0 = P0.astype(np.float32)

    # Face normals for offset direction (robust, no smooth-normal flips)
    FN = igl.per_face_normals(V, F, np.array([0.0, 1.0, 0.0]))
    n = FN[FI]
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    n = n.astype(np.float32)

    P0_f64 = P0.astype(np.float64)
    n_f64 = n.astype(np.float64)
    eps = float(epsilon)
    min_eps = 1e-6

    # ---- P+ verification ----
    Pp_all = P0_f64 + eps * n_f64
    S_all, _, _, _ = igl.signed_distance(
        Pp_all, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
    Sp_all = S_all.astype(np.float32)

    plus_valid = Sp_all > 0
    plus_retry = ~plus_valid

    Pp_valid = Pp_all[plus_valid].astype(np.float32)
    Sp_valid = Sp_all[plus_valid]

    if np.any(plus_retry) and eps / 2.0 >= min_eps:
        Pp_retry = P0_f64[plus_retry] + (eps / 2.0) * n_f64[plus_retry]
        S_retry, _, _, _ = igl.signed_distance(
            Pp_retry, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )
        Sp_retry = S_retry.astype(np.float32)
        retry_valid = Sp_retry > 0

        if np.any(retry_valid):
            Pp_valid = np.vstack([Pp_valid, Pp_retry[retry_valid].astype(np.float32)])
            Sp_valid = np.concatenate([Sp_valid, Sp_retry[retry_valid]])

    # ---- P- verification ----
    Pm_all = P0_f64 - eps * n_f64
    S_all, _, _, _ = igl.signed_distance(
        Pm_all, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
    Sm_all = S_all.astype(np.float32)

    minus_valid = Sm_all < 0
    minus_retry = ~minus_valid

    Pm_valid = Pm_all[minus_valid].astype(np.float32)
    Sm_valid = Sm_all[minus_valid]

    if np.any(minus_retry) and eps / 2.0 >= min_eps:
        Pm_retry = P0_f64[minus_retry] - (eps / 2.0) * n_f64[minus_retry]
        S_retry, _, _, _ = igl.signed_distance(
            Pm_retry, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )
        Sm_retry = S_retry.astype(np.float32)
        retry_valid = Sm_retry < 0

        if np.any(retry_valid):
            Pm_valid = np.vstack([Pm_valid, Pm_retry[retry_valid].astype(np.float32)])
            Sm_valid = np.concatenate([Sm_valid, Sm_retry[retry_valid]])

    S0 = np.zeros(len(P0), dtype=np.float32)
    return P0, Pp_valid, Pm_valid, S0, Sp_valid, Sm_valid


def sample_sdf_triplets_trimesh(
    mesh: trimesh.Trimesh,
    num_triplets: int,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate SDF triplets (P0, P+, P-) using trimesh.

    Same verification and retry logic as the libigl version.
    """
    if num_triplets <= 0:
        empty = np.empty((0, 3), dtype=np.float32)
        empty1d = np.empty((0,), dtype=np.float32)
        return empty, empty, empty, empty1d, empty1d, empty1d

    pts, face_idx = trimesh.sample.sample_surface(mesh, num_triplets)
    P0 = pts.astype(np.float32)

    fn = mesh.face_normals[face_idx].astype(np.float32)
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

    eps = float(epsilon)
    min_eps = 1e-6

    # ---- P+ verification ----
    Pp_all = P0 + eps * fn
    Sp_all = _compute_trimesh_signed_distance(mesh, Pp_all.astype(np.float32))
    Sp_all = Sp_all.astype(np.float32)

    plus_valid = Sp_all > 0
    plus_retry = ~plus_valid

    Pp_valid = Pp_all[plus_valid].astype(np.float32)
    Sp_valid = Sp_all[plus_valid]

    if np.any(plus_retry) and eps / 2.0 >= min_eps:
        Pp_retry = P0[plus_retry] + (eps / 2.0) * fn[plus_retry]
        Sp_retry = _compute_trimesh_signed_distance(mesh, Pp_retry.astype(np.float32))
        Sp_retry = Sp_retry.astype(np.float32)
        retry_valid = Sp_retry > 0

        if np.any(retry_valid):
            Pp_valid = np.vstack([Pp_valid, Pp_retry[retry_valid].astype(np.float32)])
            Sp_valid = np.concatenate([Sp_valid, Sp_retry[retry_valid]])

    # ---- P- verification ----
    Pm_all = P0 - eps * fn
    Sm_all = _compute_trimesh_signed_distance(mesh, Pm_all.astype(np.float32))
    Sm_all = Sm_all.astype(np.float32)

    minus_valid = Sm_all < 0
    minus_retry = ~minus_valid

    Pm_valid = Pm_all[minus_valid].astype(np.float32)
    Sm_valid = Sm_all[minus_valid]

    if np.any(minus_retry) and eps / 2.0 >= min_eps:
        Pm_retry = P0[minus_retry] - (eps / 2.0) * fn[minus_retry]
        Sm_retry = _compute_trimesh_signed_distance(mesh, Pm_retry.astype(np.float32))
        Sm_retry = Sm_retry.astype(np.float32)
        retry_valid = Sm_retry < 0

        if np.any(retry_valid):
            Pm_valid = np.vstack([Pm_valid, Pm_retry[retry_valid].astype(np.float32)])
            Sm_valid = np.concatenate([Sm_valid, Sm_retry[retry_valid]])

    S0 = np.zeros(len(P0), dtype=np.float32)
    return P0, Pp_valid, Pm_valid, S0, Sp_valid, Sm_valid


def compute_used_vertex_normals(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-vertex normals by averaging incident face normals.

    Mirrors the C++ logic in PreprocessMesh.cpp exactly:
    - Computes the raw cross product (edge1 x edge2) for each face
    - Skips degenerate faces (cross product norm <= 1e-12, same threshold as C++)
    - Accumulates the normalised face normals per vertex
    - Normalises the summed vertex normals; default (0,0,1) for isolated vertices

    Uses vectorised numpy (np.bincount) for speed.

    Returns
    -------
    used_vertices : (M, 3) float32
    used_normals  : (M, 3) float32 aligned with used_vertices
    """
    if len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    faces = mesh.faces          # (F, 3)
    verts = mesh.vertices        # (V, 3)
    V = len(verts)
    F = len(faces)

    v0 = verts[faces[:, 0]]     # (F, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Raw cross products, same as C++ edge1.cross(edge2) with threshold 1e-12
    cross = np.cross(v1 - v0, v2 - v0)          # (F, 3)
    norms = np.linalg.norm(cross, axis=1)        # (F,)
    valid = norms > 1e-12                        # matches C++ `> 1e-12f`

    face_n = np.zeros((F, 3), dtype=np.float64)
    face_n[valid] = cross[valid] / norms[valid, None]  # normalise only valid

    # Accumulate per-vertex using np.bincount (compiled sort-based reduction;
    # 3-5× faster than np.add.at for large meshes; numerically equivalent).
    #
    # Layout: faces.reshape(-1) = [f0v0, f0v1, f0v2, f1v0, f1v1, f1v2, ...]
    # np.repeat(face_contrib, 3, axis=0) repeats each row 3 times, so row k
    # aligns with faces.reshape(-1)[k] — each face's contribution goes to
    # all three of its vertices.
    all_indices  = faces.reshape(-1)                    # (3F,)
    face_contrib = face_n * valid[:, None]              # (F, 3)
    all_contrib  = np.repeat(face_contrib, 3, axis=0)   # (3F, 3)

    vn = np.empty((V, 3), dtype=np.float64)
    for dim in range(3):
        vn[:, dim] = np.bincount(all_indices, weights=all_contrib[:, dim], minlength=V)

    cnt = np.bincount(
        all_indices,
        weights=np.repeat(valid.astype(np.float64), 3),
        minlength=V,
    ).astype(np.int32)

    # Normalise per-vertex average
    out = np.zeros((V, 3), dtype=np.float64)
    out[:, 2] = 1.0   # default (0,0,1) for isolated / degenerate vertices
    has_n = cnt > 0
    avg = np.where(has_n[:, None], vn / np.maximum(cnt[:, None], 1), 0.0)
    avg_norm = np.linalg.norm(avg, axis=1)
    good = has_n & (avg_norm > 1e-12)
    out[good] = avg[good] / avg_norm[good, None]

    used_idx = np.unique(faces.reshape(-1))
    return verts[used_idx].astype(np.float32), out[used_idx].astype(np.float32)


# ----------------------------
# SDF sampling
# ----------------------------

def area_weighted_sampling_from_mesh_surface(mesh: trimesh.Trimesh, num_sample: int) -> np.ndarray:
    """
    Area-weighted random sampling from the mesh surface.

    Direct replacement for the C++ SampleFromSurfaceInside() with
    delta ≈ 0 (which is the correct regime for watertight meshes — every
    randomly sampled surface point is within epsilon of the visible shell).

    Returns
    -------
    points : np.ndarray  shape (num_sample, 3), dtype float32
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_sample)
    return points.astype(np.float32)


def sample_sdf_near_surface(
    kd_tree: "_KDTree",
    used_vertices: np.ndarray,
    used_normals: np.ndarray,
    xyz_surf: np.ndarray,
    num_rand_samples: int,
    variance: float,
    second_variance: float,
    bounding_cube_extents: np.ndarray,
    num_votes: int,
    rng: np.random.Generator,
    aniso_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirrors `SampleSDFNearSurface` logic.

    Parameters
    ----------
    bounding_cube_extents : (3,) array
        Per-axis extents [a, b, c].  Uniform samples are drawn from
        [-a/2, a/2] x [-b/2, b/2] x [-c/2, c/2].  Default [1,1,1] reproduces
        the original [-0.5, 0.5]^3 cube.

    Returns
    -------
    xyz_used : (K, 3) float32
    sdfs     : (K,) float32
    """
    stdv = float(np.sqrt(variance))
    stdv2 = float(np.sqrt(second_variance))

    # Near-surface perturbations (two per surface point)
    if len(xyz_surf) > 0:
        noise1 = rng.normal(0.0, stdv, size=(len(xyz_surf), 3)).astype(np.float32)
        noise2 = rng.normal(0.0, stdv2, size=(len(xyz_surf), 3)).astype(np.float32)
        if aniso_scale is not None:
            noise1 *= aniso_scale.astype(np.float32)
            noise2 *= aniso_scale.astype(np.float32)
        samp1 = xyz_surf + noise1
        samp2 = xyz_surf + noise2
        xyz = np.concatenate([samp1, samp2], axis=0)
    else:
        xyz = np.empty((0, 3), dtype=np.float32)

    # Uniform samples in bounding box (per-axis extents)
    if num_rand_samples > 0:
        extents = np.asarray(bounding_cube_extents, dtype=np.float32)  # (3,)
        half = extents / np.float32(2.0)
        uni = rng.random((int(num_rand_samples), 3)).astype(np.float32)
        uni = uni * extents - half
        xyz = np.concatenate([xyz, uni], axis=0)

    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Query kNN for all points  (_KDTree wrapper handles parallelism internally)
    dists, idxs = kd_tree.query(xyz, k=int(num_votes))
    # For k=1, scipy returns 1D; standardize
    if num_votes == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    # -----------------------------------------------------------------------
    # Fully vectorised SDF computation — mirrors the C++ loop exactly.
    #
    # For each query point p:
    #   cl_verts[p, k]   = used_vertices[idxs[p, k]]          (N, K, 3)
    #   ray_vec[p, k]    = p - cl_verts[p, k]                  (N, K, 3)
    #   ray_len[p, k]    = ||ray_vec[p, k]||                   (N, K)
    #
    # SDF magnitude (vote 0 / nearest neighbour only):
    #   if ray_len[0] < stdv:  |dot(normal_0, ray_vec_0)|   (point-to-plane)
    #   else:                  ray_len[0]                   (Euclidean)
    #
    # Sign vote (all K neighbours, same as C++):
    #   count votes where dot(normal_k, ray_vec_k / ray_len_k) > 0
    #   and ray_len_k > 1e-6 (avoid zero-division on degenerate queries)
    #
    # Accept only if num_pos == 0 or num_pos == num_votes  (all-or-nothing)
    # Sign:  negative  if num_pos <= num_votes // 2  (matches C++ integer division)
    # -----------------------------------------------------------------------

    N = len(xyz)
    K = int(num_votes)

    # Gather nearest-vertex positions and normals  (N, K, 3)
    cl_verts   = used_vertices[idxs]               # (N, K, 3)
    cl_normals = used_normals[idxs]                # (N, K, 3)

    ray_vecs = xyz[:, None, :] - cl_verts          # (N, K, 3)
    ray_lens = np.linalg.norm(ray_vecs, axis=2)    # (N, K)  — Euclidean dist

    # --- SDF magnitude from nearest neighbour (vote 0) ---
    ray_vec_0 = ray_vecs[:, 0, :]                  # (N, 3)
    ray_len_0 = ray_lens[:, 0]                     # (N,)
    normal_0  = cl_normals[:, 0, :]                # (N, 3)
    point_to_plane_0 = np.abs(np.einsum('ni,ni->n', normal_0, ray_vec_0))  # (N,)
    sdf_mag = np.where(ray_len_0 < stdv, point_to_plane_0, ray_len_0)      # (N,)

    # --- Sign votes across all K neighbours ---
    valid_len = ray_lens > np.float32(1e-6)                          # (N, K)  avoids 0/0
    # Use np.float32(1.0) as fill — a Python-float 1.0 would silently upcast
    # the entire ray_lens_safe (and everything downstream) from float32 to float64.
    ray_lens_safe = np.where(valid_len, ray_lens, np.float32(1.0))   # (N, K) float32
    ray_vecs_unit = ray_vecs / ray_lens_safe[..., None]              # (N, K, 3) float32
    # dot(normal_k, (p - cl_vert_k) / |p - cl_vert_k|)
    dot_sign = np.einsum('nki,nki->nk', cl_normals, ray_vecs_unit)  # (N, K)
    # Only count votes where ray_len > 1e-6 (same guard as C++)
    num_pos = np.sum((dot_sign > 0) & valid_len, axis=1)  # (N,)

    # --- All-or-nothing filter ---
    accept = (num_pos == 0) | (num_pos == K)

    # --- Final signed SDF ---
    # num_pos <= num_votes // 2  → negative  (matches C++ integer division)
    sdf_signed = np.where(num_pos <= (K // 2), -sdf_mag, sdf_mag)  # (N,)

    xyz_out  = xyz[accept].astype(np.float32)
    sdfs_out = sdf_signed[accept].astype(np.float32)

    if len(xyz_out) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return xyz_out, sdfs_out


def sample_sdf_near_surface_libigl_hybrid(
    V: np.ndarray,
    F: np.ndarray,
    xyz_surf: np.ndarray,
    surf_normals: np.ndarray,
    num_rand_samples: int,
    variance: float,
    second_variance: float,
    bounding_cube_extents: np.ndarray,
    normal_offset: bool,
    rng: np.random.Generator,
    aniso_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid: C++-style near-surface sampling + libigl accurate SDF.

    Mimics the C++ sampling pattern (concentrated near surface) but uses
    libigl's accurate signed distance instead of k-NN voting.

    Parameters
    ----------
    V : (|V|, 3) float64
        Mesh vertices in libigl format.
    F : (|F|, 3) int64
        Face indices in libigl format.
    xyz_surf : (N, 3) float32 or float64
        Surface sample points (e.g., from area-weighted sampling).
    surf_normals : (N, 3) float32 or float64
        Normals at surface sample points (for offset direction).
    num_rand_samples : int
        Number of uniform random samples in bounding box.
    variance : float
        Variance for primary Gaussian perturbation.
    second_variance : float
        Variance for secondary Gaussian perturbation.
    bounding_cube_extents : (3,) array
        Per-axis extents [a, b, c].  Uniform samples are drawn from
        [-a/2, a/2] x [-b/2, b/2] x [-c/2, c/2].
    normal_offset : bool
        If True, offset along surface normal (controlled interior/exterior).
        If False, isotropic perturbation (like C++).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    xyz_out : (K, 3) float32
        Accepted query points.
    sdfs_out : (K,) float32
        Signed distances (negative inside, positive outside).
    """
    import igl

    stdv = float(np.sqrt(variance))
    stdv2 = float(np.sqrt(second_variance))

    # ------------------------------------------------------------------
    # 1. Near-surface perturbations (two per surface point, like C++)
    # ------------------------------------------------------------------
    if len(xyz_surf) > 0:
        xyz_surf = np.asarray(xyz_surf, dtype=np.float64)
        surf_normals = np.asarray(surf_normals, dtype=np.float64)

        if normal_offset:
            # Controlled: offset along normal direction
            # Positive noise = exterior, negative noise = interior
            noise1 = rng.normal(0.0, stdv, size=(len(xyz_surf), 1))
            noise2 = rng.normal(0.0, stdv2, size=(len(xyz_surf), 1))
            samp1 = xyz_surf + surf_normals * noise1
            samp2 = xyz_surf + surf_normals * noise2
        else:
            # C++ style: isotropic Gaussian perturbation
            noise1 = rng.normal(0.0, stdv, size=(len(xyz_surf), 3))
            noise2 = rng.normal(0.0, stdv2, size=(len(xyz_surf), 3))
            if aniso_scale is not None:
                noise1 *= aniso_scale.astype(np.float64)
                noise2 *= aniso_scale.astype(np.float64)
            samp1 = xyz_surf + noise1
            samp2 = xyz_surf + noise2

        xyz = np.vstack([samp1, samp2]).astype(np.float64)
    else:
        xyz = np.empty((0, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Uniform samples in bounding box (per-axis extents)
    # ------------------------------------------------------------------
    if num_rand_samples > 0:
        extents = np.asarray(bounding_cube_extents, dtype=np.float64)  # (3,)
        half = extents / 2.0
        uni = rng.random((int(num_rand_samples), 3))
        xyz_uni = (uni * extents - half).astype(np.float64)
        xyz = np.vstack([xyz, xyz_uni]) if len(xyz) > 0 else xyz_uni

    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Accurate SDF evaluation with libigl
    # ------------------------------------------------------------------
    S, I, C, N = igl.signed_distance(xyz, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)

    # ------------------------------------------------------------------
    # 4. Consistency filtering (similar to C++ all-or-nothing)
    # ------------------------------------------------------------------
    # C++ accepts only when all votes agree (num_pos == 0 or num_pos == k)
    # libigl: verify pseudonormal sign consistency
    displacement = xyz - C
    dot_products = np.sum(displacement * N, axis=1)

    confidence_threshold = 1e-8 * float(np.max(bounding_cube_extents))
    valid_mask = np.abs(dot_products) > confidence_threshold

    xyz_out = xyz[valid_mask].astype(np.float32)
    sdfs_out = S[valid_mask].astype(np.float32)

    if len(xyz_out) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return xyz_out, sdfs_out

def _compute_trimesh_signed_distance(mesh: trimesh.Trimesh, xyz_query: np.ndarray) -> np.ndarray:
    """Compute signed distance using trimesh.

    Returns DeepSDF convention: negative = inside, positive = outside.
    On failure returns NaN array so callers can filter out invalid points.
    """
    try:
        return -trimesh.proximity.signed_distance(mesh, xyz_query)
    except (ValueError, IndexError) as exc:
        warnings.warn(
            f"trimesh.proximity.signed_distance failed ({type(exc).__name__}: {exc}). "
            f"Returning NaN for {len(xyz_query)} query points.",
            stacklevel=2,
        )
        return np.full(len(xyz_query), np.nan, dtype=np.float64)


def sample_sdf_near_surface_trimesh(
    mesh: trimesh.Trimesh,
    xyz_surf: np.ndarray,
    num_rand_samples: int,
    variance: float,
    second_variance: float,
    bounding_cube_extents: np.ndarray,
    normal_offset: bool,
    rng: np.random.Generator,
    aniso_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample SDF using trimesh built-in proximity queries (ray casting).

    This replaces the k-NN vertex-based approach with proper closest-point
    queries using trimesh's AABB tree and ray-casting sign determination,
    while keeping the exact same sampling pattern as the original C++ code.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh (already normalized).
    xyz_surf : (N, 3) float32 or float64
        Surface sample points (e.g., from area-weighted sampling).
    num_rand_samples : int
        Number of uniform random samples in bounding box.
    variance : float
        Variance for primary Gaussian perturbation.
    second_variance : float
        Variance for secondary Gaussian perturbation.
    bounding_cube_extents : (3,) array
        Per-axis extents [a, b, c].  Uniform samples are drawn from
        [-a/2, a/2] x [-b/2, b/2] x [-c/2, c/2].
    normal_offset : bool
        If True, offset along surface normal (controlled interior/exterior).
        If False, isotropic perturbation (like C++).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    xyz_out : (K, 3) float32
        Accepted query points.
    sdfs_out : (K,) float32
        Signed distances (negative inside, positive outside).
    """
    stdv = float(np.sqrt(variance))
    stdv2 = float(np.sqrt(second_variance))

    # ------------------------------------------------------------------
    # 1. Near-surface perturbations (two per surface point, like C++)
    # ------------------------------------------------------------------
    if len(xyz_surf) > 0:
        xyz_surf = np.asarray(xyz_surf, dtype=np.float64)

        if normal_offset:
            # Get face normals for offset direction
            # Need to find which face each surface point belongs to
            # For area-weighted samples from trimesh, we have face indices
            # Here we approximate using closest face normal
            closest_points, _, face_indices = trimesh.proximity.closest_point(
                mesh, xyz_surf.astype(np.float32)
            )
            surf_normals = mesh.face_normals[face_indices].astype(np.float64)

            # Controlled: offset along normal direction
            noise1 = rng.normal(0.0, stdv, size=(len(xyz_surf), 1))
            noise2 = rng.normal(0.0, stdv2, size=(len(xyz_surf), 1))
            samp1 = xyz_surf + surf_normals * noise1
            samp2 = xyz_surf + surf_normals * noise2
        else:
            # C++ style: isotropic Gaussian perturbation
            noise1 = rng.normal(0.0, stdv, size=(len(xyz_surf), 3))
            noise2 = rng.normal(0.0, stdv2, size=(len(xyz_surf), 3))
            if aniso_scale is not None:
                noise1 *= aniso_scale.astype(np.float64)
                noise2 *= aniso_scale.astype(np.float64)
            samp1 = xyz_surf + noise1
            samp2 = xyz_surf + noise2

        xyz = np.vstack([samp1, samp2]).astype(np.float64)
    else:
        xyz = np.empty((0, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Uniform samples in bounding box (per-axis extents)
    # ------------------------------------------------------------------
    if num_rand_samples > 0:
        extents = np.asarray(bounding_cube_extents, dtype=np.float64)  # (3,)
        half = extents / 2.0
        uni = rng.random((int(num_rand_samples), 3))
        xyz_uni = (uni * extents - half).astype(np.float64)
        xyz = np.vstack([xyz, xyz_uni]) if len(xyz) > 0 else xyz_uni

    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. SDF computation using trimesh proximity queries (ray casting)
    # ------------------------------------------------------------------
    # Convert to float32 for trimesh (its internal tree uses float32)
    xyz_query = xyz.astype(np.float32)

    signed_distances = _compute_trimesh_signed_distance(mesh, xyz_query)

    # ------------------------------------------------------------------
    # 4. Filter ambiguous points (similar to C++ all-or-nothing voting)
    # ------------------------------------------------------------------
    # C++ accepts only when all votes agree (num_pos == 0 or num_pos == k)
    # Here we filter points where the sign is ambiguous (very close to surface)
    # or where trimesh couldn't determine a sign (returns NaN)
    valid_mask = ~np.isnan(signed_distances)

    # Additional filter: points too close to surface may have sign ambiguity
    # This mimics the C++ consistency requirement
    confidence_threshold = 1e-8 * float(np.max(bounding_cube_extents))
    valid_mask &= np.abs(signed_distances) > confidence_threshold

    xyz_out = xyz[valid_mask].astype(np.float32)
    sdfs_out = signed_distances[valid_mask].astype(np.float32)

    if len(xyz_out) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return xyz_out, sdfs_out
