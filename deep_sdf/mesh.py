#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deep_sdf.utils


def _decoder_device(decoder):
    return next(
        (t.device for t in decoder.parameters()),
        next(
            (b.device for b in decoder.buffers()),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ),
    )


def _eval_sdf_on_coords(decoder, latent_vec, coords_np, max_batch):
    """Evaluate decoder on an (M, 3) numpy array of world-space coords.

    Returns a 1-D numpy float32 array of length M.
    """
    device = _decoder_device(decoder)
    M = coords_np.shape[0]
    out = np.empty(M, dtype=np.float32)
    coords_t = torch.from_numpy(coords_np.astype(np.float32, copy=False))
    with torch.no_grad():
        head = 0
        while head < M:
            tail = min(head + max_batch, M)
            sub = coords_t[head:tail].to(device, non_blocking=True)
            sdf = deep_sdf.utils.decode_sdf(decoder, latent_vec, sub).squeeze(1)
            out[head:tail] = sdf.detach().cpu().numpy()
            head = tail
    return out


def _eval_sdf_dense_grid(decoder, latent_vec, N, max_batch):
    """Dense evaluation of decoder on an N^3 grid spanning [-1, 1]^3.

    Returns sdf[i, j, k] = SDF(x = -1 + i*vs, y = -1 + j*vs, z = -1 + k*vs),
    as a torch CPU FloatTensor of shape (N, N, N).
    """
    decoder.eval()
    voxel_size = 2.0 / (N - 1)
    device = _decoder_device(decoder)
    sdf_flat = torch.empty(N ** 3, dtype=torch.float32)
    axis = torch.arange(N, device=device, dtype=torch.float32) * voxel_size - 1.0
    num_samples = N ** 3
    with torch.no_grad():
        head = 0
        while head < num_samples:
            tail = min(head + max_batch, num_samples)
            idx = torch.arange(head, tail, device=device, dtype=torch.long)
            i = torch.div(idx, N * N, rounding_mode="floor")
            j = torch.div(idx, N, rounding_mode="floor") % N
            k = idx % N
            xyz = torch.stack([axis[i], axis[j], axis[k]], dim=1)
            sdf_chunk = deep_sdf.utils.decode_sdf(decoder, latent_vec, xyz).squeeze(1)
            sdf_flat[head:tail] = sdf_chunk.detach().cpu()
            head = tail
    return sdf_flat.view(N, N, N)


def _nonzero_sign(values, dtype=np.int8):
    """Return +/-1 signs with zeros treated as outside (+1)."""
    return np.where(values < 0, -1, 1).astype(dtype, copy=False)


def _dense_nearest_neighbor_upsample(grid, mapped_idx):
    """Upsample a dense cubic grid with separable nearest-neighbour gathers."""
    upsampled = np.take(grid, mapped_idx, axis=0)
    upsampled = np.take(upsampled, mapped_idx, axis=1)
    return np.take(upsampled, mapped_idx, axis=2)


def _upsample_active_cells_to_corner_mask(active_cells, ratio):
    """Expand active coarse cells to the fine-grid corners they touch."""
    fine_cell_mask = np.repeat(
        np.repeat(np.repeat(active_cells, ratio, axis=0), ratio, axis=1),
        ratio,
        axis=2,
    )
    N = fine_cell_mask.shape[0] + 1
    corner_mask = np.zeros((N, N, N), dtype=bool)
    fcm = fine_cell_mask
    corner_mask[:-1, :-1, :-1] |= fcm
    corner_mask[:-1, :-1, 1:] |= fcm
    corner_mask[:-1, 1:, :-1] |= fcm
    corner_mask[:-1, 1:, 1:] |= fcm
    corner_mask[1:, :-1, :-1] |= fcm
    corner_mask[1:, :-1, 1:] |= fcm
    corner_mask[1:, 1:, :-1] |= fcm
    corner_mask[1:, 1:, 1:] |= fcm
    return corner_mask


def _active_cells_to_corner_coords(active_cells, ratio):
    """Return unique fine-grid corner coordinates touched by active cells.

    Uses packed integer deduplication when the sparse working set is smaller than
    the old dense-mask expansion, and falls back to the dense path otherwise.
    """
    active_coords = np.argwhere(active_cells)
    if active_coords.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    fine_shape = np.asarray(active_cells.shape, dtype=np.int64) * ratio + 1
    dense_work = int(np.prod(fine_shape) + np.prod(fine_shape - 1))
    block_corners = (ratio + 1) ** 3
    packed_work = int(active_coords.shape[0] * block_corners * 8)
    if packed_work >= dense_work:
        return np.argwhere(_upsample_active_cells_to_corner_mask(active_cells, ratio))

    active_coords = active_coords.astype(np.int64, copy=False)
    base_coords = active_coords * np.int64(ratio)
    offsets = np.stack(
        np.meshgrid(
            np.arange(ratio + 1, dtype=np.int64),
            np.arange(ratio + 1, dtype=np.int64),
            np.arange(ratio + 1, dtype=np.int64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    stride_y = fine_shape[1] * fine_shape[2]
    stride_z = fine_shape[2]
    packed = np.empty(active_coords.shape[0] * offsets.shape[0], dtype=np.int64)
    write_head = 0
    for offset in offsets:
        corners = base_coords + offset
        tail = write_head + active_coords.shape[0]
        packed[write_head:tail] = (
            corners[:, 0] * stride_y + corners[:, 1] * stride_z + corners[:, 2]
        )
        write_head = tail

    packed = np.unique(packed)
    x = packed // stride_y
    yz = packed % stride_y
    y = yz // stride_z
    z = yz % stride_z
    return np.column_stack((x, y, z))


def _map_child_to_parent_coords(child_coords, ratio, parent_size):
    """Map child-grid corner coordinates to nearest parent-grid coordinates."""
    if child_coords.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    mapped = np.rint(child_coords / ratio).astype(np.int64, copy=False)
    return np.clip(mapped, 0, parent_size - 1)


def _gather_dense_grid_values(grid, coords):
    """Gather values from a dense cubic grid at integer coordinates."""
    if coords.size == 0:
        return np.empty((0,), dtype=grid.dtype)
    return grid[coords[:, 0], coords[:, 1], coords[:, 2]]


def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    """Extract an iso-surface mesh from a DeepSDF decoder on a uniform N^3 grid.

    Semantics-preserving rewrite of the original Facebook DeepSDF implementation:
      - `voxel_origin` is the (bottom, left, down) corner of the cube [-1, 1]^3.
      - The SDF array layout is sdf[i, j, k] = SDF(x = -1 + i * vs,
                                                  y = -1 + j * vs,
                                                  z = -1 + k * vs),
        where vs = 2 / (N - 1).

    Modernization vs. the original:
      - Build sample coordinates on GPU using vectorized index math
        (avoids `LongTensor / N` which is no longer integer division).
      - Stream coordinates through the decoder without round-tripping a
        big (N^3, 4) CPU tensor.
      - Use `torch.no_grad()` so no graph is built for inference.
    """
    start = time.time()
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1.0, -1.0, -1.0]
    voxel_size = 2.0 / (N - 1)

    sdf_values = _eval_sdf_dense_grid(decoder, latent_vec, N, max_batch)

    print("sampling takes: %f" % (time.time() - start))

    convert_sdf_samples_to_ply(
        sdf_values,
        voxel_origin,
        voxel_size,
        filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # Check if SDF values contain zero crossing (required for marching cubes)
    if numpy_3d_sdf_tensor.min() > 0 or numpy_3d_sdf_tensor.max() < 0:
        logging.warning(f"Skipping {ply_filename_out}: no zero crossing in SDF values (min={numpy_3d_sdf_tensor.min():.4f}, max={numpy_3d_sdf_tensor.max():.4f})")
        return

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except (ValueError, RuntimeError) as e:
        logging.warning(f"Skipping {ply_filename_out}: marching cubes failed - {e}")
        return

    # Apply origin, scale, and offset in one vectorized pass.
    mesh_points = verts + np.asarray(voxel_grid_origin, dtype=verts.dtype)
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # Vectorized structured-array construction (avoids per-vertex Python loops).
    mesh_points = mesh_points.astype(np.float32, copy=False)
    verts_tuple = np.empty(mesh_points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts_tuple["x"] = mesh_points[:, 0]
    verts_tuple["y"] = mesh_points[:, 1]
    verts_tuple["z"] = mesh_points[:, 2]

    faces_tuple = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    faces_tuple["vertex_indices"] = faces.astype(np.int32, copy=False)

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def _save_polys_as_ply(verts, faces, ply_filename_out):
    """Write (verts, faces) to PLY using the same vectorized layout as
    `convert_sdf_samples_to_ply`. `verts` is (V, 3) float32, `faces` is
    (F, 3) int32.
    """
    verts = verts.astype(np.float32, copy=False)
    verts_tuple = np.empty(verts.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts_tuple["x"] = verts[:, 0]
    verts_tuple["y"] = verts[:, 1]
    verts_tuple["z"] = verts[:, 2]

    faces_tuple = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    faces_tuple["vertex_indices"] = faces.astype(np.int32, copy=False)

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
    plyfile.PlyData([el_verts, el_faces]).write(ply_filename_out)


def create_mesh_multigrid(
    decoder,
    latent_vec,
    filename,
    N_coarse=128,
    N_fine=512,
    K=3.0,
    dilation=1,
    bg=1.0,
    adaptivity=0.0,
    max_batch=32 ** 3,
    offset=None,
    scale=None,
):
    """Multigrid mesh extraction using OpenVDB for iso-surface meshing.

    Algorithm:
      1. Evaluate decoder densely on a coarse N_coarse^3 grid.
      2. Mark coarse cells "active" if min over 8 corners of |sdf| <= K * vs_coarse.
      3. Binary-dilate the active mask by `dilation` cells for safety.
      4. Determine the set of fine-grid corner indices covered by active cells.
      5. Evaluate decoder only at those active fine corners.
      6. Fill inactive fine corners with +/- bg using the nearest-neighbour sign
         from the coarse grid (creates a closed band).
      7. Copy dense fine SDF to a vdb.FloatGrid and call convertToPolygons.
      8. Translate vertices from VDB index/world space to [-1, 1]^3 and write PLY.

    Args:
      N_coarse: coarse grid resolution (e.g. 128).
      N_fine:   fine grid resolution (e.g. 512). Must satisfy
                (N_fine - 1) % (N_coarse - 1) == 0.
      K:        active-cell band width in coarse voxels. Effective Lipschitz
                tolerance ~ K + sqrt(3)/2; with default K=3 + 1-cell dilation
                this covers decoder Lipschitz up to ~4.4.
      dilation: number of cells to dilate the active mask in each direction.
      bg:       magnitude of the "definitely outside/inside" value written into
                inactive fine corners (must exceed all active-band |sdf|).
      adaptivity: OpenVDB mesh adaptivity (0.0 = uniform, 1.0 = max adaptation).
                  Higher values reduce polygon count (12-43%) with slight CD
                  increase (0.5-2.5%). Default 0.0 for backward compatibility.
    """
    import openvdb as vdb

    start = time.time()
    assert (N_fine - 1) % (N_coarse - 1) == 0, (
        f"N_fine-1 ({N_fine - 1}) must be a multiple of N_coarse-1 ({N_coarse - 1})"
    )
    r = (N_fine - 1) // (N_coarse - 1)
    vs_coarse = 2.0 / (N_coarse - 1)
    vs_fine = 2.0 / (N_fine - 1)
    origin = -1.0
    decoder.eval()

    # --- 1. Coarse pass --------------------------------------------------
    t0 = time.time()
    coarse_sdf = _eval_sdf_dense_grid(decoder, latent_vec, N_coarse, max_batch).numpy()
    t_coarse = time.time() - t0

    # --- 2. Active coarse-cell mask --------------------------------------
    # Cell (i,j,k) covers corners [i:i+2, j:j+2, k:k+2]. Use min-pool of |sdf|.
    abs_sdf = np.abs(coarse_sdf)
    corner_min = np.minimum.reduce([
        abs_sdf[:-1, :-1, :-1], abs_sdf[:-1, :-1, 1:],
        abs_sdf[:-1, 1:, :-1], abs_sdf[:-1, 1:, 1:],
        abs_sdf[1:, :-1, :-1], abs_sdf[1:, :-1, 1:],
        abs_sdf[1:, 1:, :-1], abs_sdf[1:, 1:, 1:],
    ])
    active_cells = corner_min <= (K * vs_coarse)  # (N_c-1)^3

    # --- 3. Dilate active mask -------------------------------------------
    if dilation > 0:
        # Cheap 3D dilation via cumulative max over a (2d+1)^3 neighbourhood.
        # Implemented with three 1-D max-pool passes.
        from scipy.ndimage import binary_dilation
        active_cells = binary_dilation(active_cells, iterations=dilation)

    n_active = int(active_cells.sum())
    if n_active == 0:
        logging.warning(f"Skipping {filename}.ply: no active coarse cells (K={K})")
        return

    # --- 4. Fine-corner mask ---------------------------------------------
    # Coarse cell (i,j,k) covers fine corner indices [i*r .. (i+1)*r] inclusive
    # in each axis. Build a fine corner mask by upsampling cells then taking
    # corners of the upsampled cell-mask.
    # Equivalently: a fine corner (a,b,c) is active iff any of the up-to-8
    # coarse cells touching that corner is active.
    active_idx = _active_cells_to_corner_coords(active_cells, r)
    n_fine_active = int(active_idx.shape[0])

    # --- 5. Initialize fine SDF with sign from coarse ---------------------
    # Nearest-neighbour upsample of sign(coarse_sdf) to fine grid.
    # Map fine index a in [0, N_f-1] -> coarse index round(a / r).
    fine_idx = np.arange(N_fine)
    coarse_idx_of_fine = np.clip(np.rint(fine_idx / r).astype(np.int64), 0, N_coarse - 1)
    coarse_sign = _nonzero_sign(coarse_sdf)
    sign_up = _dense_nearest_neighbor_upsample(coarse_sign, coarse_idx_of_fine)
    fine_sdf = sign_up * np.float32(bg)
    del coarse_sign, sign_up

    # --- 6. Decoder eval at active fine corners --------------------------
    t0 = time.time()
    coords = active_idx.astype(np.float32) * np.float32(vs_fine) + np.float32(origin)
    sdfs = _eval_sdf_on_coords(decoder, latent_vec, coords, max_batch)
    if n_fine_active:
        fine_sdf[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]] = sdfs
    t_fine = time.time() - t0

    # --- 7. Mesh extraction via OpenVDB ----------------------------------
    if fine_sdf.min() > 0 or fine_sdf.max() < 0:
        logging.warning(
            f"Skipping {filename}.ply: no zero crossing in fine SDF "
            f"(min={fine_sdf.min():.4f}, max={fine_sdf.max():.4f})"
        )
        return

    t0 = time.time()
    g = vdb.FloatGrid(background=float(bg))
    g.transform = vdb.createLinearTransform(voxelSize=vs_fine)
    # copyFromArray expects contiguous float32; np.ascontiguousarray is a no-op
    # if already contiguous.
    g.copyFromArray(np.ascontiguousarray(fine_sdf, dtype=np.float32))
    verts, tris, quads = g.convertToPolygons(isovalue=0.0, adaptivity=adaptivity)
    t_mesh = time.time() - t0

    # --- 8. Triangulate quads, translate verts, write PLY -----------------
    if quads.size:
        # Quad (a,b,c,d) -> tris (a,b,c) and (a,c,d).
        quad_tris = np.empty((quads.shape[0] * 2, 3), dtype=quads.dtype)
        quad_tris[0::2, 0] = quads[:, 0]
        quad_tris[0::2, 1] = quads[:, 1]
        quad_tris[0::2, 2] = quads[:, 2]
        quad_tris[1::2, 0] = quads[:, 0]
        quad_tris[1::2, 1] = quads[:, 2]
        quad_tris[1::2, 2] = quads[:, 3]
        faces = np.concatenate([tris, quad_tris], axis=0) if tris.size else quad_tris
    else:
        faces = tris

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        logging.warning(f"Skipping {filename}.ply: empty mesh after extraction")
        return

    # VDB returns vertices in world space defined by `voxelSize=vs_fine` with
    # zero translation, so world = vs_fine * index_coord. Shift to [-1, 1]^3.
    mesh_points = verts + np.float32(origin)
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    _save_polys_as_ply(mesh_points, faces, filename + ".ply")

    total = time.time() - start
    print(
        "multigrid: coarse %.2fs (%d^3) | fine eval %.2fs (%d/%d corners = %.2f%%) "
        "| mesh %.2fs | total %.2fs"
        % (
            t_coarse, N_coarse,
            t_fine, n_fine_active, N_fine ** 3,
            100.0 * n_fine_active / (N_fine ** 3),
            t_mesh, total,
        )
    )


def create_mesh_multigrid_hierarchical(
    decoder,
    latent_vec,
    filename,
    resolutions=(33, 65, 129, 257, 513),
    K=3.0,
    dilation=1,
    bg=1.0,
    adaptivity=0.0,
    max_batch=32 ** 3,
    offset=None,
    scale=None,
):
    """Hierarchical multigrid mesh extraction with progressive refinement.

    Evaluates the decoder at multiple grid resolutions, pruning inactive cells
    at each level. This reduces total decoder evaluations compared to the
    2-level approach by progressively narrowing the active band.

    Algorithm:
      1. Evaluate decoder on the coarsest grid (level 0).
      2. Mark cells active if min|sdf| at 8 corners <= K * vs_level.
      3. For each subsequent level:
         a. Map active cells from previous level to current level corners.
         b. Evaluate decoder only at active corners.
         c. Mark active cells using current level's threshold.
      4. At the finest level, fill inactive corners with signed background.
      5. Extract mesh via OpenVDB.

    Args:
      resolutions: Grid resolutions from coarsest to finest (default: (33, 65, 129, 257, 513)).
                   Must satisfy (resolutions[i+1] - 1) % (resolutions[i] - 1) == 0.
      K: Active-cell band width in voxels (default: 3.0).
      dilation: Mask dilation iterations at each level (default: 1).
      bg: Background SDF magnitude (default: 1.0).
      adaptivity: OpenVDB mesh adaptivity (0.0 = uniform, 1.0 = max, default: 0.0).
    """
    import openvdb as vdb
    from scipy.ndimage import binary_dilation

    start = time.time()
    decoder.eval()

    resolutions = list(resolutions)
    n_levels = len(resolutions)

    if isinstance(K, (int, float)):
        K_per_level = [float(K)] * n_levels
    else:
        K_per_level = list(K)
        assert len(K_per_level) == n_levels, (
            f"K list length ({len(K_per_level)}) must match resolutions length ({n_levels})"
        )

    for i in range(n_levels - 1):
        assert (resolutions[i + 1] - 1) % (resolutions[i] - 1) == 0, (
            f"Level {i+1} ({resolutions[i+1]}) must be compatible with "
            f"level {i} ({resolutions[i]}): (N-1) must divide evenly"
        )

    origin = -1.0
    level_stats = []
    total_decoder_evals = 0

    N0 = resolutions[0]
    vs0 = 2.0 / (N0 - 1)
    t0 = time.time()
    sdf_0 = _eval_sdf_dense_grid(decoder, latent_vec, N0, max_batch).numpy()
    t_eval = time.time() - t0
    total_decoder_evals += N0 ** 3

    abs_sdf = np.abs(sdf_0)
    corner_min = np.minimum.reduce([
        abs_sdf[:-1, :-1, :-1], abs_sdf[:-1, :-1, 1:],
        abs_sdf[:-1, 1:, :-1], abs_sdf[:-1, 1:, 1:],
        abs_sdf[1:, :-1, :-1], abs_sdf[1:, :-1, 1:],
        abs_sdf[1:, 1:, :-1], abs_sdf[1:, 1:, 1:],
    ])
    active_cells = corner_min <= (K_per_level[0] * vs0)
    if dilation > 0:
        active_cells = binary_dilation(active_cells, iterations=dilation)

    n_active_0 = int(active_cells.sum())
    n_total_0 = (N0 - 1) ** 3
    level_stats.append({
        "N": N0, "evals": N0 ** 3, "active_cells": n_active_0,
        "frac": n_active_0 / n_total_0, "time": t_eval,
    })

    prev_sdf = sdf_0
    prev_sign = _nonzero_sign(sdf_0)
    prev_active_cells = active_cells
    prev_N = N0

    for level_idx in range(1, n_levels):
        N = resolutions[level_idx]
        vs = 2.0 / (N - 1)
        r = (N - 1) // (prev_N - 1)

        t0 = time.time()
        active_idx = _active_cells_to_corner_coords(prev_active_cells, r)
        t_upsample = time.time() - t0
        t_corner = 0.0

        t0 = time.time()
        parent_idx = _map_child_to_parent_coords(active_idx, r, prev_N)
        parent_sdf = _gather_dense_grid_values(prev_sdf, parent_idx)
        t_sdfup = time.time() - t0

        t0 = time.time()
        keep_mask = np.abs(parent_sdf) <= (K_per_level[level_idx] * vs * 2)
        active_idx = active_idx[keep_mask]
        n_pruned = int(active_idx.shape[0])
        t_prune = time.time() - t0

        t0 = time.time()
        prev_idx = np.arange(N)
        prev_mapped = np.clip(np.rint(prev_idx / r).astype(np.int64), 0, prev_N - 1)
        sign_level = _dense_nearest_neighbor_upsample(prev_sign, prev_mapped)
        sdf_level = sign_level * np.float32(bg)
        t_sign = time.time() - t0

        t0 = time.time()
        coords = active_idx.astype(np.float32) * np.float32(vs) + np.float32(origin)
        sdfs = _eval_sdf_on_coords(decoder, latent_vec, coords, max_batch)
        if n_pruned:
            sdf_level[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]] = sdfs
            sign_level[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]] = _nonzero_sign(sdfs)
        t_eval = time.time() - t0
        total_decoder_evals += n_pruned

        t0 = time.time()
        active_cells = _active_cells_from_sparse_corners(
            active_idx,
            sdfs,
            K_per_level[level_idx],
            vs,
            dilation,
            N,
        )
        t_detect = time.time() - t0

        n_active = int(active_cells.sum())
        n_total = (N - 1) ** 3
        level_stats.append({
            "N": N, "evals": n_pruned, "active_cells": n_active,
            "frac": n_active / n_total, "time": t_eval,
            "t_upsample": t_upsample, "t_corner": t_corner,
            "t_sdfup": t_sdfup, "t_prune": t_prune,
            "t_sign": t_sign, "t_detect": t_detect,
        })

        prev_sdf = sdf_level
        prev_sign = sign_level
        prev_active_cells = active_cells
        prev_N = N

    fine_sdf = prev_sdf
    N_fine = resolutions[-1]
    vs_fine = 2.0 / (N_fine - 1)

    if fine_sdf.min() > 0 or fine_sdf.max() < 0:
        logging.warning(
            f"Skipping {filename}.ply: no zero crossing in fine SDF "
            f"(min={fine_sdf.min():.4f}, max={fine_sdf.max():.4f})"
        )
        return

    t0 = time.time()
    g = vdb.FloatGrid(background=float(bg))
    g.transform = vdb.createLinearTransform(voxelSize=vs_fine)
    g.copyFromArray(np.ascontiguousarray(fine_sdf, dtype=np.float32))
    verts, tris, quads = g.convertToPolygons(isovalue=0.0, adaptivity=adaptivity)
    t_mesh = time.time() - t0

    if quads.size:
        quad_tris = np.empty((quads.shape[0] * 2, 3), dtype=quads.dtype)
        quad_tris[0::2, 0] = quads[:, 0]
        quad_tris[0::2, 1] = quads[:, 1]
        quad_tris[0::2, 2] = quads[:, 2]
        quad_tris[1::2, 0] = quads[:, 0]
        quad_tris[1::2, 1] = quads[:, 2]
        quad_tris[1::2, 2] = quads[:, 3]
        faces = np.concatenate([tris, quad_tris], axis=0) if tris.size else quad_tris
    else:
        faces = tris

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        logging.warning(f"Skipping {filename}.ply: empty mesh after extraction")
        return

    mesh_points = verts + np.float32(origin)
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    _save_polys_as_ply(mesh_points, faces, filename + ".ply")

    total = time.time() - start
    dense_evals = resolutions[-1] ** 3
    print(
        "hierarchical_multigrid: levels=%s | total_decoder_evals=%d "
        "(vs dense %d = %.1f%%) | mesh %.2fs | total %.2fs"
        % (
            "→".join(str(r) for r in resolutions),
            total_decoder_evals, dense_evals,
            100.0 * total_decoder_evals / dense_evals,
            t_mesh, total,
        )
    )
    for i, s in enumerate(level_stats):
        if i == 0:
            print(
                "  L%d (%d³): evals=%d active_cells=%d (%.1f%%) time=%.2fs"
                % (i, s["N"], s["evals"], s["active_cells"], 100.0 * s["frac"], s["time"])
            )
        else:
            print(
                "  L%d (%d³): evals=%d active_cells=%d (%.1f%%) time=%.2fs"
                " | up=%.3f corner=%.3f sdfup=%.3f prune=%.3f sign=%.3f detect=%.3f"
                % (i, s["N"], s["evals"], s["active_cells"], 100.0 * s["frac"], s["time"],
                   s["t_upsample"], s["t_corner"], s["t_sdfup"],
                   s["t_prune"], s["t_sign"], s["t_detect"])
            )


def _active_cells_from_sparse_corners(coords, sdf_values, K, vs, dilation, N):
    """Mark active cells from evaluated corner coordinates whose |sdf| is small.

    This matches the dense min-pool path when inactive corners are filled with a
    background magnitude greater than `K * vs`.
    """
    threshold = K * vs
    active_cells = np.zeros((N - 1, N - 1, N - 1), dtype=bool)
    if coords.size == 0 or sdf_values.size == 0:
        return active_cells

    near_surface = coords[np.abs(sdf_values) <= threshold]
    if near_surface.size == 0:
        return active_cells

    packed = np.empty(near_surface.shape[0] * 8, dtype=np.int64)
    stride_y = (N - 1) * (N - 1)
    stride_z = (N - 1)
    write_head = 0

    for dx in (0, -1):
        for dy in (0, -1):
            for dz in (0, -1):
                cells = near_surface + np.array([dx, dy, dz], dtype=np.int64)
                valid = (
                    (cells[:, 0] >= 0) & (cells[:, 0] < N - 1) &
                    (cells[:, 1] >= 0) & (cells[:, 1] < N - 1) &
                    (cells[:, 2] >= 0) & (cells[:, 2] < N - 1)
                )
                if not np.any(valid):
                    continue
                cells = cells[valid]
                tail = write_head + cells.shape[0]
                packed[write_head:tail] = (
                    cells[:, 0] * stride_y + cells[:, 1] * stride_z + cells[:, 2]
                )
                write_head = tail

    if write_head == 0:
        return active_cells

    packed = np.unique(packed[:write_head])
    x = packed // stride_y
    yz = packed % stride_y
    y = yz // stride_z
    z = yz % stride_z
    active_cells[x, y, z] = True

    if dilation > 0:
        from scipy.ndimage import binary_dilation

        active_cells = binary_dilation(active_cells, iterations=dilation)

    return active_cells
