#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import multiprocessing as mp
from pathlib import Path

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.async_loader import AsyncPrefetchLoader
from deep_sdf.watchdog import SystemMonitor


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def initialize_latent_vectors(
    num_scenes,
    latent_size,
    code_bound,
    code_init_std_dev,
    scene_categories=None,
):
    """Create and initialize the latent code embedding.

    When ``scene_categories`` is supplied, the first ``num_categories``
    dimensions are reserved for category-aware *initialization* via one-hot
    spikes.  **All dimensions remain learnable** after this init.

    Checkpoint loading is intentionally kept outside this boundary.
    """
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        code_init_std_dev / math.sqrt(latent_size),
    )
    if scene_categories is not None:
        num_categories = int(scene_categories.max().item()) + 1
        if num_categories > latent_size:
            raise ValueError(
                "latent_size ({}) must be >= num_categories ({}).".format(
                    latent_size, num_categories
                )
            )
        one_hot = torch.zeros(
            num_scenes, num_categories, device=lat_vecs.weight.device
        )
        one_hot.scatter_(
            1, scene_categories.unsqueeze(1).to(lat_vecs.weight.device), 1.0
        )
        lat_vecs.weight.data[:, :num_categories] = one_hot
    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )
    return lat_vecs


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split, validate_data=True):

    logging.debug("running " + experiment_directory)

    # Validate data before training (if requested)
    if validate_data:
        logging.info("Running pre-training data validation...")
        try:
            # Import here to avoid circular imports
            import subprocess
            import sys
            validate_script = Path(__file__).parent / "validate_data.py"
            result = subprocess.run(
                [sys.executable, str(validate_script), "-e", experiment_directory],
                capture_output=False,
                text=True
            )
            if result.returncode != 0:
                logging.error("=" * 60)
                logging.error("Data validation FAILED!")
                logging.error("Please fix the data issues above before training.")
                logging.error("You can regenerate experiment splits with create_experiment.py")
                logging.error("Or skip validation with --no-validate-data (not recommended)")
                logging.error("=" * 60)
                sys.exit(1)
        except Exception as e:
            logging.warning(f"Could not run data validation: {e}")
            logging.warning("Continuing without validation...")

    specs = ws.load_experiment_specifications(experiment_directory)

    deep_sdf.utils.set_xyz_dim(specs["NetworkSpecs"].get("xyz_dim", 3))

    description = specs["Description"] if isinstance(specs["Description"], str) else "\n".join(specs["Description"])
    logging.info("Experiment description: \n" + description)

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    stop_requested = False
    cleanup_done = False

    def cleanup_resources(immediate=False):
        """Cleanup resources on exit - GPU first priority, then data loaders."""
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True

        logging.info("Cleaning up resources...")

        # 0. Stop watchdog
        try:
            watchdog.stop()
        except Exception as e:
            logging.warning(f"Error stopping watchdog: {e}")

        # 1. FIRST PRIORITY: Stop async data loader (terminates background processes)
        try:
            sdf_loader.stop(immediate=immediate)
            logging.info("AsyncPrefetchLoader stopped")
        except Exception as e:
            logging.warning(f"Error stopping loader: {e}")

        # 2. SECOND PRIORITY: Release GPU memory
        try:
            # Move model back to CPU to release GPU memory
            decoder.cpu()
            logging.info("Decoder moved to CPU")
        except Exception as e:
            logging.warning(f"Error moving decoder to CPU: {e}")

        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logging.info("GPU cache cleared and synchronized")
        except Exception as e:
            logging.warning(f"Error clearing GPU cache: {e}")

        # Force garbage collection
        import gc
        gc.collect()

    def signal_handler(sig, frame):
        nonlocal stop_requested
        if stop_requested:
            # Second Ctrl+C - force exit immediately without full cleanup
            logging.info("Force exiting immediately...")
            # Try minimal cleanup
            try:
                sdf_loader.stop(immediate=True)
            except Exception as e:
                logging.warning(f"Error stopping loader during force exit: {e}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.warning(f"Error clearing GPU cache during force exit: {e}")
            sys.exit(1)
        stop_requested = True
        logging.info("Stopping early (Ctrl+C again to force exit)...")

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    pos_decay_threshold = get_spec_with_default(specs, "PosDecayThreshold", clamp_dist)
    pos_decay_exp = get_spec_with_default(specs, "PosDecayExp", 1.0)

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    gpu_count = torch.cuda.device_count()
    logging.info("training with {} GPU(s)".format(gpu_count))

    if gpu_count > 1:
        decoder = torch.nn.DataParallel(decoder)

    decoder_device = next(decoder.parameters()).device

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False,
        pos_decay_threshold=pos_decay_threshold,
        pos_decay_exp=pos_decay_exp,
    )

    # Multi-producer configuration for AsyncPrefetchLoader
    num_producers = get_spec_with_default(specs, "NumProducers", 4)
    workers_per_producer = get_spec_with_default(specs, "WorkersPerProducer", 4)

    # Validate total process count to prevent system overload
    MAX_SAFE_PROCESSES = 64
    total_processes = num_producers * (1 + workers_per_producer)  # producers + their workers
    if total_processes > MAX_SAFE_PROCESSES:
        logging.warning(
            "Configured {} producers x {} workers = {} total processes exceeds safe limit ({}). "
            "Reducing to safe defaults: 4 producers x 4 workers.".format(
                num_producers, workers_per_producer, total_processes, MAX_SAFE_PROCESSES
            )
        )
        num_producers = 4
        workers_per_producer = 4
        total_processes = num_producers * (1 + workers_per_producer)

    logging.debug(
        "loading data with {} producer(s), {} workers each ({} total processes)".format(
            num_producers, workers_per_producer, total_processes
        )
    )

    # Use AsyncPrefetchLoader for non-blocking data loading
    sdf_loader = AsyncPrefetchLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        num_producers=num_producers,
        workers_per_producer=workers_per_producer,
        queue_size=512,  # Pre-load 512 batches in background
        mp_context=mp.get_context("spawn"),
        auto_restart=True,
        max_restart_attempts=3,
    )
    sdf_loader.start()
    logging.info(
        "Started AsyncPrefetchLoader: {} producers x {} workers, queue size 512".format(
            num_producers, workers_per_producer
        )
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)
    batches_per_epoch = num_scenes // scene_per_batch
    scene_names = [
        os.path.splitext(os.path.basename(f))[0] for f in sdf_dataset.npyfiles
    ]
    expected_scenes = set(scene_names[: batches_per_epoch * scene_per_batch])

    logging.info("There are {} scenes".format(num_scenes))

    if pos_decay_threshold > 0.0:
        logging.info(
            "Positive SDF decay enabled: threshold={:.4f}, exp={:.2f}".format(
                pos_decay_threshold, pos_decay_exp
            )
        )

    logging.debug(decoder)

    # Load optional scene category mapping for one-hot initialization.
    # Backward-compatible: if specs.json does not mention SceneCategories,
    # the latent code is initialized with the old random-only method.
    scene_categories_file = get_spec_with_default(specs, "SceneCategories", None)
    scene_categories = None
    if scene_categories_file is not None:
        if not os.path.isabs(scene_categories_file):
            scene_categories_file = os.path.join(data_source, scene_categories_file)
        if os.path.isfile(scene_categories_file):
            with open(scene_categories_file, "r") as f:
                cat_map = json.load(f)
            cat_list = []
            for npyfile in sdf_dataset.npyfiles:
                basename = os.path.splitext(os.path.basename(npyfile))[0]
                if basename not in cat_map:
                    raise KeyError(
                        "Scene {} not found in {}".format(
                            basename, scene_categories_file
                        )
                    )
                cat_list.append(cat_map[basename])
            scene_categories = torch.LongTensor(cat_list).to(decoder_device)
        else:
            raise FileNotFoundError(
                'Scene categories file "{}" does not exist'.format(
                    scene_categories_file
                )
            )

    code_init_std_dev = get_spec_with_default(specs, "CodeInitStdDev", 1.0)
    lat_vecs = initialize_latent_vectors(
        num_scenes, latent_size, code_bound, code_init_std_dev,
        scene_categories=scene_categories,
    ).to(decoder_device)

    num_categories = None
    if scene_categories is not None:
        num_categories = int(scene_categories.max().item()) + 1

    # unmask-at-epoch is the single source of truth for the one-hot training
    # schedule.  It lives in specs.json (default 0 = never masked).
    unmask_at_epoch = get_spec_with_default(specs, "unmask-at-epoch", 0)

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    watchdog = SystemMonitor(
        interval=0.5,
        log_path=os.path.join(experiment_directory, "watchdog.log"),
    )
    watchdog.start()

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            if stop_requested:
                break

            start = time.time()

            logging.info("epoch {}...".format(epoch))

            decoder.train()

            adjust_learning_rate(lr_schedules, optimizer_all, epoch)

            batch_times = []  # Track batch processing times
            consumed_batch_ids = set()
            for sdf_data, indices in sdf_loader:
                if stop_requested:
                    break

                batch_start = time.time()

                # Log queue fill level periodically
                queue_fill = sdf_loader.get_queue_fill()
                if queue_fill < 10:
                    logging.debug(f"Queue fill: {queue_fill} (low!)")

                # Process the input data
                sdf_data = sdf_data.reshape(-1, 4).to(decoder_device, non_blocking=True)

                num_sdf_samples = sdf_data.shape[0]

                sdf_data.requires_grad = False

                xyz = sdf_data[:, 0:3]
                xyz = deep_sdf.utils.feat_eng(xyz)

                # Defensive: verify feature engineering output dimension matches decoder config
                expected_xyz_dim = specs["NetworkSpecs"].get("xyz_dim", 3)
                assert xyz.shape[1] == expected_xyz_dim, (
                    f"feat_eng output dim {xyz.shape[1]} != decoder xyz_dim {expected_xyz_dim}. "
                    f"Did you forget to call deep_sdf.utils.set_xyz_dim()?"
                )

                sdf_gt = sdf_data[:, 3].unsqueeze(1)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                xyz = torch.chunk(xyz, batch_split)
                # Indices are moved to GPU so the embedding lookup and
                # gradient update both happen on device, avoiding a
                # per-batch host→device DMA of the full latent vectors.
                indices_gpu = torch.chunk(
                    indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1).to(decoder_device, non_blocking=True),
                    batch_split,
                )

                sdf_gt = torch.chunk(sdf_gt, batch_split)

                # Accumulate chunk losses as a GPU tensor so .item() (which
                # forces a CPU/GPU sync) can be deferred until after step().
                # This keeps the GPU busy across the batch_split inner loop.
                batch_loss_tensor = torch.zeros(1, device=decoder_device)

                optimizer_all.zero_grad()

                for i in range(batch_split):

                    # non_blocking=True submits the host→device copy to the
                    # CUDA DMA engine and lets the CPU continue immediately.
                    # The consuming op (torch.cat below) will not execute on
                    # the GPU until the DMA completes — correctness is
                    # guaranteed by CUDA stream ordering.
                    # lat_vecs lives on GPU, so the lookup and all gradient
                    # accumulation happen entirely on device — no host→device
                    # transfer of embedding vectors per batch.
                    batch_vecs = lat_vecs(indices_gpu[i])

                    input = torch.cat([batch_vecs, xyz[i]], dim=1)

                    # Defensive: verify concatenated input matches decoder's first-layer width
                    expected_input_dim = latent_size + expected_xyz_dim
                    assert input.shape[1] == expected_input_dim, (
                        f"decoder input dim {input.shape[1]} != expected {expected_input_dim} "
                        f"(latent={latent_size}, xyz={expected_xyz_dim})"
                    )

                    # NN optimization
                    pred_sdf = decoder(input)

                    # Defensive: verify scalar SDF output
                    assert pred_sdf.shape[1] == 1, f"pred_sdf shape {pred_sdf.shape} != (N, 1)"

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                    # Loss normalization note (keep the comments below!):
                    #
                    # ``num_sdf_samples`` is computed BEFORE chunking as
                    #     num_sdf_samples = ScenesPerBatch * SamplesPerScene
                    # i.e. the total number of SDF sample rows in the FULL
                    # batch, NOT in this chunk. Dividing each chunk's L1 by
                    # this full-batch count makes the chunk losses sum to the
                    # per-row mean L1 over the whole batch — that is the
                    # quantity appended to ``loss_log`` below.
                    chunk_loss = loss_l1(pred_sdf, sdf_gt[i]) / num_sdf_samples

                    if do_code_regularization:
                        # (keep the comments below)
                        # ``batch_vecs`` has one row per sample (the scene's
                        # latent is repeated SamplesPerScene times via
                        # ``indices_gpu``), so
                        #     sum_j ||batch_vecs_j|| ≈ chunk_rows * mean||z||
                        # Dividing again by ``num_sdf_samples`` (full batch)
                        # means the regularizer summed across all chunks
                        # equals
                        #     λ · warmup · mean(||z||) over scenes in batch
                        # — the SamplesPerScene factor cancels out. The logged
                        # batch loss is therefore
                        #     mean_L1_per_row  +  λ · warmup · mean_||z||
                        # (no SamplesPerScene division on the reg term).
                        l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                        reg_loss = (
                            code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                        ) / num_sdf_samples

                        chunk_loss = chunk_loss + reg_loss

                    chunk_loss.backward()

                    # Accumulate on GPU — no CPU sync yet.
                    # .detach() ensures the compute graph is released immediately
                    # rather than kept alive until .item() is called.
                    batch_loss_tensor = batch_loss_tensor + chunk_loss.detach()

                # Apply gradient mask on the category one-hot slice when the
                # current epoch is still before unmask_at_epoch.
                if num_categories is not None and epoch < unmask_at_epoch:
                    if lat_vecs.weight.grad is not None:
                        lat_vecs.weight.grad[:, :num_categories] = 0.0

                if grad_clip is not None:

                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

                optimizer_all.step()

                # .item() is deferred until after all backward() calls and step()
                # are complete. This reduces CPU-GPU sync points from batch_split
                # times per batch to once per batch, keeping the GPU better
                # utilized. The logged value is identical since .backward() was
                # already called for all chunks before this sync point.
                batch_loss = batch_loss_tensor.item()

                logging.debug("loss = {}".format(batch_loss))

                loss_log.append(batch_loss)

                # Track batch processing time
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                consumed_batch_ids.add(
                    frozenset(scene_names[i] for i in indices.tolist())
                )
                if len(batch_times) % 10 == 0:
                    avg_time = sum(batch_times[-10:]) / 10
                    logging.debug(
                        f"batch {len(batch_times)}: avg={avg_time:.3f}s, last={batch_time:.3f}s, queue={queue_fill}"
                    )

            # EPOCH BARRIER ----------------------------------------------------
            # Drain any batches that leaked from future epochs.  The fence in
            # AsyncPrefetchLoader limits producers to at most 1 epoch ahead,
            # but the queue may still contain stale prefetched batches when a
            # fast producer finished its partition before a slow one.
            sdf_loader.flush_epoch()
            # ------------------------------------------------------------------

            # Verify scene-level invariants (detects batch loss, duplication, producer failures)
            sdf_loader._verify_batch_invariants(
                consumed_batch_ids,
                expected_scenes,
                epoch,
            )

            end = time.time()

            seconds_elapsed = end - start
            timing_log.append(seconds_elapsed)

            # Epoch summary
            num_batches = len(batch_times)
            avg_batch_time = sum(batch_times) / num_batches if num_batches > 0 else 0
            recent_losses = loss_log[-num_batches:] if num_batches > 0 else []
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            logging.info(
                "epoch {} complete: {} batches, {:.1f}s total, {:.3f}s/batch, avg_loss={:.6f}".format(
                    epoch, num_batches, seconds_elapsed, avg_batch_time, avg_loss
                )
            )

            lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

            lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

            append_parameter_magnitudes(param_mag_log, decoder)

            if epoch in checkpoints:
                save_checkpoints(epoch)

            if epoch % log_frequency == 0:

                save_latest(epoch)
                save_logs(
                    experiment_directory,
                    loss_log,
                    lr_log,
                    timing_log,
                    lat_mag_log,
                    param_mag_log,
                    epoch,
                )
    except KeyboardInterrupt:
        stop_requested = True
        logging.info("Interrupted by user")
    finally:
        # Cleanup: stop loader first, then release GPU
        cleanup_resources(immediate=stop_requested)

        # Force exit if interrupted (prevents hanging from zombie processes/threads)
        if stop_requested:
            logging.info("Exiting...")
            # Kill any remaining child processes (optional, if psutil available)
            try:
                import psutil
                parent = psutil.Process()
                for child in parent.children(recursive=True):
                    try:
                        child.terminate()
                    except Exception as e:
                        logging.warning(f"Failed to terminate child process {child.pid}: {e}")
            except ImportError:
                pass  # psutil not available, skip
            except Exception as e:
                logging.warning(f"Error during child process cleanup: {e}")
            # os._exit is immediate and doesn't wait for threads to finish
            os._exit(1)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument(
        "--no-validate-data",
        dest="validate_data",
        action="store_false",
        default=True,
        help="Skip pre-training data validation. Not recommended - validation "
        + "catches missing data files and misconfigured experiments before "
        + "wasting compute on failed training runs.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split), args.validate_data)
