#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import time
import torch
import torch.utils.data
from typing import Optional

import deep_sdf.workspace as ws


def adaptive_pos_sample_weights(
    pos_sdf: torch.Tensor,
    threshold: float = 0.1,
    decay_exp: float = 1.0,
) -> torch.Tensor:
    """Compute per-sample importance weights for positive (outside-surface) SDF samples.

    Samples with SDF <= *threshold* receive uniform weight.  Samples whose
    SDF exceeds *threshold* are down-weighted by an inverse power law so that
    far-outside points are sampled less frequently.

    The weighting function is::

        w_i = 1.0                         if sdf_i <= threshold
        w_i = (threshold / sdf_i)^k       if sdf_i >  threshold

    where *k* = ``decay_exp``.

    Args:
        pos_sdf: 1-D tensor of positive SDF values (shape ``(N,)``).
        threshold: SDF value above which decay begins.  Typically matches
            the training ``ClampingDistance`` (default 0.1).
        decay_exp: Power-law exponent *k*.  ``k = 0`` disables decay
            (uniform sampling, backward-compatible).  ``k = 1`` gives inverse
            decay; higher values cull far points more aggressively.

    Returns:
        Normalised probability tensor of shape ``(N,)`` that sums to 1.
    """
    if decay_exp <= 0.0 or pos_sdf.numel() == 0:
        return torch.ones_like(pos_sdf) / pos_sdf.numel()

    weights = torch.ones_like(pos_sdf)
    far_mask = pos_sdf > threshold
    if far_mask.any():
        weights[far_mask] = (threshold / pos_sdf[far_mask]).pow(decay_exp)

    weights = weights.clamp(min=1e-12)
    return weights / weights.sum()


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    # 返回文件名，而不是完整路径，避免后续 os.path.join 时路径重复
    return os.path.basename(mesh_filenames[0])


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def _remove_nans_np(arr: np.ndarray) -> np.ndarray:
    """Filter NaNs in the SDF column (index 3) in numpy."""
    if arr.size == 0:
        return arr
    # Expect shape [N,4]. Be defensive.
    if arr.ndim != 2 or arr.shape[1] < 4:
        return arr
    mask = ~np.isnan(arr[:, 3])
    return arr[mask]


def _npy_cache_paths_from_npz(npz_path: str):
    """Return cache paths for a given .npz.

    Preferred layout:
      <DATA_ROOT>/<SdfSamples>/(.../instance.npz)
      <DATA_ROOT>/<SdfSamples_npy>/(.../instance_pos.npy, instance_neg.npy)

    If the input path doesn't appear to be under <SdfSamples>/, fall back to
    writing next to the .npz.
    """

    npz_path = os.path.abspath(npz_path)
    marker = os.sep + ws.sdf_samples_subdir + os.sep
    if marker in npz_path:
        prefix, rel = npz_path.split(marker, 1)
        rel_base, _ext = os.path.splitext(rel)
        cache_root = prefix + os.sep + (ws.sdf_samples_subdir + "_npy")
        pos_path = os.path.join(cache_root, rel_base + "_pos.npy")
        neg_path = os.path.join(cache_root, rel_base + "_neg.npy")
        lock_path = os.path.join(cache_root, rel_base + "_npy.lock")
        return pos_path, neg_path, lock_path

    # Fallback: same directory as .npz
    # Put cache at the same level as the "source" directory containing .npz
    # files by using a sibling directory named '<npz_dir>_npy'.
    npz_dir = os.path.dirname(npz_path)
    parent = os.path.dirname(npz_dir)
    cache_root = os.path.join(parent, os.path.basename(npz_dir) + "_npy")
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    pos_path = os.path.join(cache_root, stem + "_pos.npy")
    neg_path = os.path.join(cache_root, stem + "_neg.npy")
    lock_path = os.path.join(cache_root, stem + "_npy.lock")
    return pos_path, neg_path, lock_path


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    """Atomically save a .npy by writing to a temp file then os.replace()."""
    directory = os.path.dirname(path) or "."
    # np.save() appends ".npy" if the path doesn't end with it.
    # Ensure our temp path ends with ".npy" so os.replace() targets the real file.
    tmp_path = f"{path}.tmp.{os.getpid()}.npy"
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def _cleanup_orphan_npy_tmps(final_path: str) -> None:
    """Remove orphaned temp files like '<final>.tmp.<pid>.npy' (best-effort)."""
    try:
        for p in glob.glob(final_path + ".tmp.*.npy"):
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def _ensure_npy_cache_from_npz(
    npz_path: str,
    *,
    wait_s: float = 30.0,
    stale_lock_s: float = 300.0,
) -> tuple[str, str]:
    """Ensure cached *_pos.npy and *_neg.npy exist for the given .npz.

    Safe under DataLoader multi-worker concurrency using a simple lock file.
    Falls back by raising on unrecoverable errors.
    """
    pos_path, neg_path, lock_path = _npy_cache_paths_from_npz(npz_path)

    # Ensure cache directory exists before we try to create a lock file inside it.
    cache_dir = os.path.dirname(pos_path) or "."
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        # If we cannot even create the cache dir, let fallback (.npz) handle it.
        raise

    if os.path.isfile(pos_path) and os.path.isfile(neg_path):
        _cleanup_orphan_npy_tmps(pos_path)
        _cleanup_orphan_npy_tmps(neg_path)
        return pos_path, neg_path

    # Fast permission check: if we cannot write in directory, bail early.
    out_dir = os.path.dirname(pos_path) or "."
    check_dir = out_dir if os.path.isdir(out_dir) else (os.path.dirname(out_dir) or ".")
    if not os.access(check_dir, os.W_OK):
        raise PermissionError(f"No write permission for cache dir: {out_dir}")

    deadline = time.time() + float(wait_s)
    while True:
        # If another worker finished while we waited.
        if os.path.isfile(pos_path) and os.path.isfile(neg_path):
            _cleanup_orphan_npy_tmps(pos_path)
            _cleanup_orphan_npy_tmps(neg_path)
            return pos_path, neg_path

        # Try to acquire the lock.
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
            finally:
                os.close(fd)

            # We have the lock.
            try:
                # Re-check after lock acquisition.
                if os.path.isfile(pos_path) and os.path.isfile(neg_path):
                    return pos_path, neg_path

                # Ensure cache directory exists.
                os.makedirs(os.path.dirname(pos_path) or ".", exist_ok=True)
                os.makedirs(os.path.dirname(neg_path) or ".", exist_ok=True)

                with np.load(npz_path) as npz:
                    pos = np.asarray(npz["pos"], dtype=np.float32)
                    neg = np.asarray(npz["neg"], dtype=np.float32)

                pos = _remove_nans_np(pos)
                neg = _remove_nans_np(neg)

                _atomic_save_npy(pos_path, pos)
                _atomic_save_npy(neg_path, neg)

                # Best-effort cleanup of orphan temp files.
                _cleanup_orphan_npy_tmps(pos_path)
                _cleanup_orphan_npy_tmps(neg_path)
                return pos_path, neg_path
            finally:
                try:
                    os.remove(lock_path)
                except Exception:
                    pass

        except FileExistsError:
            # Another worker is converting. Wait for it, or break stale locks.
            try:
                st = os.stat(lock_path)
                if (time.time() - st.st_mtime) > float(stale_lock_s):
                    # Best-effort stale lock cleanup.
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
            except FileNotFoundError:
                pass

            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(0.05)


def _load_pos_neg_from_npy(pos_path: str, neg_path: str):
    # Load normally (not memmap) to avoid non-writable-array warnings in torch.
    pos = np.load(pos_path)
    neg = np.load(neg_path)
    return pos, neg


def unpack_sdf_samples_from_arrays(
    pos_arr: np.ndarray,
    neg_arr: np.ndarray,
    subsample=None,
    pos_decay_threshold: float = 0.0,
    pos_decay_exp: float = 1.0,
):
    if subsample is None:
        return {"pos": pos_arr, "neg": neg_arr}

    pos_tensor = torch.from_numpy(np.asarray(pos_arr))
    neg_tensor = torch.from_numpy(np.asarray(neg_arr))

    half = int(subsample / 2)

    if pos_decay_threshold > 0.0 and pos_decay_exp > 0.0:
        pos_sdf = pos_tensor[:, 3]
        weights = adaptive_pos_sample_weights(pos_sdf, pos_decay_threshold, pos_decay_exp)
        random_pos = torch.multinomial(weights, half, replacement=True)
    else:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()

    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    samples = torch.cat([sample_pos, sample_neg], 0)
    return samples


def read_sdf_samples_into_ram(filename):
    # This function is primarily used by RAM-loading paths.
    # If a cached .npy exists (or can be created), prefer it for speed.
    try:
        pos_path, neg_path = _ensure_npy_cache_from_npz(filename)
        pos_np, neg_np = _load_pos_neg_from_npy(pos_path, neg_path)
        pos_tensor = torch.from_numpy(np.asarray(pos_np))
        neg_tensor = torch.from_numpy(np.asarray(neg_np))
    except Exception:
        with np.load(filename) as npz:
            pos_tensor = torch.from_numpy(npz["pos"])
            neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(
    filename,
    subsample=None,
    pos_decay_threshold: float = 0.0,
    pos_decay_exp: float = 1.0,
):
    try:
        pos_path, neg_path = _ensure_npy_cache_from_npz(filename)
        pos_np, neg_np = _load_pos_neg_from_npy(pos_path, neg_path)
        return unpack_sdf_samples_from_arrays(
            pos_np, neg_np, subsample,
            pos_decay_threshold=pos_decay_threshold,
            pos_decay_exp=pos_decay_exp,
        )
    except Exception:
        with np.load(filename) as npz:
            if subsample is None:
                return npz
            pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
            neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

        half = int(subsample / 2)

        if pos_decay_threshold > 0.0 and pos_decay_exp > 0.0:
            pos_sdf = pos_tensor[:, 3]
            weights = adaptive_pos_sample_weights(pos_sdf, pos_decay_threshold, pos_decay_exp)
            random_pos = torch.multinomial(weights, half, replacement=True)
        else:
            random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()

        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples


def unpack_sdf_samples_from_ram(
    data,
    subsample=None,
    pos_decay_threshold: float = 0.0,
    pos_decay_exp: float = 1.0,
):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    if pos_decay_threshold > 0.0 and pos_decay_exp > 0.0 and pos_size > half:
        pos_sdf = pos_tensor[:, 3]
        weights = adaptive_pos_sample_weights(pos_sdf, pos_decay_threshold, pos_decay_exp)
        random_pos_indices = torch.multinomial(weights, half, replacement=True)
        sample_pos = torch.index_select(pos_tensor, 0, random_pos_indices)
    elif pos_size > half:
        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]
    else:
        random_pos_indices = (torch.rand(half) * pos_size).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos_indices)

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        pos_decay_threshold: float = 0.0,
        pos_decay_exp: float = 1.0,
    ):
        self.subsample = subsample
        self.pos_decay_threshold = pos_decay_threshold
        self.pos_decay_exp = pos_decay_exp

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(
                    self.loaded_data[idx],
                    self.subsample,
                    pos_decay_threshold=self.pos_decay_threshold,
                    pos_decay_exp=self.pos_decay_exp,
                ),
                idx,
            )
        else:
            return unpack_sdf_samples(
                filename,
                self.subsample,
                pos_decay_threshold=self.pos_decay_threshold,
                pos_decay_exp=self.pos_decay_exp,
            ), idx
