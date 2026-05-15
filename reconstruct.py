#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import trimesh

import deep_sdf
import deep_sdf.workspace as ws


def compute_eikonal_loss(decoder, latent, xyz):
    xyz_grad = xyz.clone().detach().requires_grad_(True)
    inputs = torch.cat([latent.expand(len(xyz_grad), -1), xyz_grad], 1)
    pred = decoder(inputs)
    grad_outputs = torch.ones_like(pred)
    spatial_grad = torch.autograd.grad(
        pred, xyz_grad, grad_outputs=grad_outputs, create_graph=True
    )[0]
    return (spatial_grad.norm(dim=1) - 1.0).abs().mean()


def sample_gt_surface_norm(mesh_path, n_points):
    mesh = trimesh.load(mesh_path, process=False)
    from deep_sdf.metrics.normalization import compute_from_mesh

    offset, scale = compute_from_mesh(mesh)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts_norm = (pts.astype(np.float32) + offset) * scale
    return pts_norm, offset, scale


def extract_sdf_features(npz_path):
    data = np.load(npz_path)
    neg = data["neg"][:, 3]
    pos = data["pos"][:, 3]
    n_total = len(neg) + len(pos)
    max_depth = float(abs(neg.min())) if len(neg) > 0 else 0.0
    mean_depth = float(abs(neg.mean())) if len(neg) > 0 else 0.0
    deep_p = float((neg < -0.01).mean()) if len(neg) > 0 else 0.0
    eff_thick = float(abs(neg[neg < -0.001]).mean()) if (neg < -0.001).any() else 0.0
    neg_ratio = len(neg) / n_total if n_total > 0 else 0.0
    near_surf = float((np.abs(np.concatenate([pos, neg])) < 0.001).mean())
    return np.array([max_depth, mean_depth, deep_p, eff_thick, neg_ratio, near_surf],
                    dtype=np.float32)


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    loss_type="l1",
    sign_penalty_lambda=10.0,
    surface_pts=None,
    eikonal_lambda=0.0,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    use_surface_pull = loss_type in ("surface_pull", "combined") and surface_pts is not None
    use_sign_penalty = loss_type in ("sign_penalty", "combined")
    use_eikonal = eikonal_lambda > 0 and surface_pts is not None

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    elif isinstance(stat, torch.Tensor):
        latent = stat.clone().detach().cuda()
        if latent.dim() == 2:
            latent = latent[0:1]
        elif latent.dim() == 1:
            latent = latent.unsqueeze(0)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    if surface_pts is not None:
        surface_pts = torch.from_numpy(surface_pts).float().cuda()
        n_surf_pull = min(5000, len(surface_pts))
        n_eik = min(2000, len(surface_pts))

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        xyz = deep_sdf.utils.feat_eng(xyz)

        # Defensive: verify feature engineering output dimension matches decoder config
        expected_xyz_dim = deep_sdf.utils.get_xyz_dim()
        assert xyz.shape[1] == expected_xyz_dim, (
            f"feat_eng output dim {xyz.shape[1]} != decoder xyz_dim {expected_xyz_dim}. "
            f"Did you forget to call deep_sdf.utils.set_xyz_dim()?"
        )

        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        # Defensive: verify concatenated input matches decoder's first-layer width
        expected_input_dim = latent_size + expected_xyz_dim
        assert inputs.shape[1] == expected_input_dim, (
            f"decoder input dim {inputs.shape[1]} != expected {expected_input_dim} "
            f"(latent={latent_size}, xyz={expected_xyz_dim})"
        )

        pred_sdf = decoder(inputs)

        # Defensive: verify scalar SDF output
        assert pred_sdf.shape[1] == 1, f"pred_sdf shape {pred_sdf.shape} != (N, 1)"

        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)

        if use_sign_penalty:
            sign_penalty = sign_penalty_lambda * torch.relu(pred_sdf) * (sdf_gt < 0).float()
            loss = loss + sign_penalty.mean()

        if use_surface_pull and e % 5 == 0:
            idx = torch.randperm(len(surface_pts))[:n_surf_pull]
            surf_xyz = surface_pts[idx]
            surf_xyz = deep_sdf.utils.feat_eng(surf_xyz)
            surf_inputs = torch.cat([latent.expand(len(surf_xyz), -1), surf_xyz], 1)
            surf_pred = decoder(surf_inputs)
            surf_loss = surf_pred.abs().mean()
            loss = loss + surf_loss

        if use_eikonal and e % 5 == 0:
            idx = torch.randperm(len(surface_pts))[:n_eik]
            eik_xyz = surface_pts[idx]
            eik_xyz = deep_sdf.utils.feat_eng(eik_xyz)
            eik_loss = compute_eikonal_loss(decoder, latent, eik_xyz)
            loss = loss + eikonal_lambda * eik_loss

        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        default=True,
        help="Skip meshes which have already been reconstructed (default: True, "
        + "use --no-skip to disable).",
    )
    arg_parser.add_argument(
        "--init-method",
        dest="init_method",
        choices=["gaussian", "empirical", "nearest_neighbor"],
        default="gaussian",
        help="Latent code initialization method: 'gaussian' for N(0, sigma), "
        + "'empirical' for training set statistics, "
        + "'nearest_neighbor' for closest training shape's latent code.",
    )
    arg_parser.add_argument(
        "--init-sigma",
        dest="init_sigma",
        type=float,
        default=0.01,
        help="Standard deviation for Gaussian initialization (default: 0.01)",
    )
    arg_parser.add_argument(
        "--loss",
        dest="loss_type",
        choices=["l1", "sign_penalty", "surface_pull", "combined"],
        default="l1",
        help="Loss function for latent code optimization: 'l1' (original), "
        + "'sign_penalty' (L1 + penalty on interior sign errors), "
        + "'surface_pull' (pull SDF→0 at GT surface points), "
        + "'combined' (sign_penalty + surface_pull + optional eikonal)",
    )
    arg_parser.add_argument(
        "--sign-penalty-lambda",
        dest="sign_penalty_lambda",
        type=float,
        default=10.0,
        help="Weight for sign error penalty (default: 10.0)",
    )
    arg_parser.add_argument(
        "--surface-samples",
        dest="n_surface_samples",
        type=int,
        default=10000,
        help="Number of GT surface points to sample for surface pulling (default: 10000)",
    )
    arg_parser.add_argument(
        "--eikonal-lambda",
        dest="eikonal_lambda",
        type=float,
        default=0.0,
        help="Weight for eikonal regularization |∇SDF|→1 (default: 0.0, disabled)",
    )
    arg_parser.add_argument(
        "--mesh-N",
        dest="mesh_N",
        type=int,
        default=256,
        help="Marching-cubes grid resolution for dense mesh extraction (default: 256). "
             "Ignored when --multigrid is set.",
    )
    arg_parser.add_argument(
        "--multigrid",
        dest="multigrid",
        action="store_true",
        help="Use create_mesh_multigrid (OpenVDB) instead of dense create_mesh.",
    )
    arg_parser.add_argument(
        "--mg-coarse",
        dest="mg_coarse",
        type=int,
        default=65,
        help="Multigrid coarse resolution (default: 65). Requires (mg_fine-1) %% (mg_coarse-1) == 0.",
    )
    arg_parser.add_argument(
        "--mg-fine",
        dest="mg_fine",
        type=int,
        default=513,
        help="Multigrid fine resolution (default: 513).",
    )
    arg_parser.add_argument(
        "--mg-K",
        dest="mg_K",
        type=float,
        default=3.0,
        help="Multigrid active-band factor in coarse voxels (default: 3.0).",
    )
    arg_parser.add_argument(
        "--mg-dilation",
        dest="mg_dilation",
        type=int,
        default=1,
        help="Multigrid mask dilation in coarse cells (default: 1).",
    )
    arg_parser.add_argument(
        "--mg-adaptivity",
        dest="mg_adaptivity",
        type=float,
        default=0.0,
        help="OpenVDB mesh adaptivity (0.0=uniform, 1.0=max). "
             "Higher values reduce polygons 12-43%% with slight CD increase (default: 0.0).",
    )
    arg_parser.add_argument(
        "--mg-hierarchical",
        dest="mg_hierarchical",
        action="store_true",
        help="Use hierarchical multigrid (5-level) instead of 2-level. "
             "Reduces decoder evaluations ~10x by progressive refinement.",
    )
    arg_parser.add_argument(
        "--mg-resolutions",
        dest="mg_resolutions",
        type=int,
        nargs="+",
        default=[33, 65, 129, 257, 513],
        help="Grid resolutions for hierarchical multigrid (default: 33 65 129 257 513).",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    if args.skip:
        logging.info("Resume mode: skipping already-reconstructed shapes")
    else:
        logging.info("Reconstructing all shapes from scratch")

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    deep_sdf.utils.set_xyz_dim(specs["NetworkSpecs"].get("xyz_dim", 3))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    # Handle state dict key mismatch (module. prefix)
    state_dict = saved_model_state["model_state_dict"]
    # Check if keys have module. prefix
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        # Add module. prefix to all keys
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    decoder.load_state_dict(state_dict)

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    if args.init_method == "empirical":
        logging.info("Using empirical initialization from training set")
        latent_codes_file = os.path.join(
            args.experiment_directory, ws.latent_codes_subdir, args.checkpoint + ".pth"
        )
        if not os.path.isfile(latent_codes_file):
            raise Exception(
                'Latent codes file "{}" does not exist'.format(latent_codes_file)
            )
        latent_codes_data = torch.load(latent_codes_file)
        if "latent_codes" not in latent_codes_data:
            raise Exception(
                'Latent codes file does not contain "latent_codes" key'
            )
        latent_codes = latent_codes_data["latent_codes"]
        if isinstance(latent_codes, torch.Tensor):
            emp_mean = latent_codes.mean(dim=0)
            emp_var = latent_codes.var(dim=0)
        elif isinstance(latent_codes, dict) and "weight" in latent_codes:
            emp_mean = latent_codes["weight"].mean(dim=0)
            emp_var = latent_codes["weight"].var(dim=0)
        else:
            raise Exception("Unexpected latent codes format")
        logging.info(
            "Empirical stats: mean={:.4f}, std={:.4f}".format(
                emp_mean.mean().item(), emp_var.sqrt().mean().item()
            )
        )
        init_stat = (emp_mean, emp_var.sqrt())
        per_shape_init = None
    elif args.init_method == "nearest_neighbor":
        logging.info("Using nearest-neighbor latent code initialization")
        latent_codes_file = os.path.join(
            args.experiment_directory, ws.latent_codes_subdir, args.checkpoint + ".pth"
        )
        if not os.path.isfile(latent_codes_file):
            raise Exception(
                'Latent codes file "{}" does not exist'.format(latent_codes_file)
            )
        latent_codes_data = torch.load(latent_codes_file)
        if "latent_codes" not in latent_codes_data:
            raise Exception('Latent codes file does not contain "latent_codes" key')
        lc = latent_codes_data["latent_codes"]
        if isinstance(lc, dict) and "weight" in lc:
            train_latents = lc["weight"]
        elif isinstance(lc, torch.Tensor):
            train_latents = lc
        else:
            raise Exception("Unexpected latent codes format")

        train_split_file = specs.get("TrainSplit")
        if not train_split_file:
            raise Exception("specs.json missing TrainSplit")
        with open(train_split_file, "r") as f:
            train_split = json.load(f)
        train_ids = [s for d in train_split.values() for c in d.values() for s in c]

        logging.info("Extracting SDF features for {} training shapes...".format(len(train_ids)))
        train_features = []
        for sid in train_ids:
            npz_path = os.path.join(args.data_source, ws.sdf_samples_subdir,
                                    "models", sid + ".npz")
            train_features.append(extract_sdf_features(npz_path))
        train_features = np.stack(train_features)
        feat_mean = train_features.mean(axis=0, keepdims=True)
        feat_std = train_features.std(axis=0, keepdims=True) + 1e-8
        train_features_norm = (train_features - feat_mean) / feat_std

        per_shape_init = {}
        for sid in [s for d in split.values() for c in d.values() for s in c]:
            npz_path = os.path.join(args.data_source, ws.sdf_samples_subdir,
                                    "models", sid + ".npz")
            if not os.path.exists(npz_path):
                continue
            test_feat = extract_sdf_features(npz_path)
            test_feat_norm = (test_feat - feat_mean) / feat_std
            dists = np.linalg.norm(train_features_norm - test_feat_norm, axis=1)
            nn_idx = int(np.argmin(dists))
            per_shape_init[sid] = train_latents[nn_idx:nn_idx+1].clone()
            logging.debug("  {} -> nearest train {} (dist={:.3f})".format(
                sid, train_ids[nn_idx], dists[nn_idx]))

        init_stat = args.init_sigma
        logging.info("Nearest-neighbor init ready for {} shapes".format(len(per_shape_init)))
    else:
        logging.info("Using Gaussian initialization with sigma={}".format(args.init_sigma))
        init_stat = args.init_sigma
        per_shape_init = None

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            surface_pts_norm = None
            if args.loss_type in ("surface_pull", "combined"):
                mesh_path = os.path.join(args.data_source, npz[:-4], "mesh.obj")
                if os.path.exists(mesh_path):
                    try:
                        surface_pts_norm, _, _ = sample_gt_surface_norm(
                            mesh_path, args.n_surface_samples
                        )
                        logging.debug(
                            "sampled {} surface points from {}".format(
                                len(surface_pts_norm), mesh_path
                            )
                        )
                    except Exception as exc:
                        logging.warning(
                            "failed to sample surface from {}: {}".format(mesh_path, exc)
                        )
                else:
                    logging.warning("GT mesh not found: {}".format(mesh_path))

            start = time.time()
            shape_init_stat = init_stat
            if per_shape_init is not None:
                sid = os.path.splitext(os.path.basename(npz))[0]
                if sid in per_shape_init:
                    shape_init_stat = per_shape_init[sid]
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                shape_init_stat,
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                loss_type=args.loss_type,
                sign_penalty_lambda=args.sign_penalty_lambda,
                surface_pts=surface_pts_norm,
                eikonal_lambda=args.eikonal_lambda,
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    if args.multigrid:
                        if args.mg_hierarchical:
                            deep_sdf.mesh.create_mesh_multigrid_hierarchical(
                                decoder, latent, mesh_filename,
                                resolutions=args.mg_resolutions,
                                K=args.mg_K,
                                dilation=args.mg_dilation,
                                adaptivity=args.mg_adaptivity,
                                max_batch=int(2 ** 18),
                            )
                        else:
                            deep_sdf.mesh.create_mesh_multigrid(
                                decoder, latent, mesh_filename,
                                N_coarse=args.mg_coarse,
                                N_fine=args.mg_fine,
                                K=args.mg_K,
                                dilation=args.mg_dilation,
                                adaptivity=args.mg_adaptivity,
                                max_batch=int(2 ** 18),
                            )
                    else:
                        deep_sdf.mesh.create_mesh(
                            decoder, latent, mesh_filename,
                            N=args.mesh_N, max_batch=int(2 ** 18),
                        )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
