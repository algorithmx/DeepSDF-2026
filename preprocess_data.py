#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import threading

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.utils import resolve_ml_env_python

# Python interpreter to use when spawning the preprocessor scripts.
# Using the conda ml_env interpreter avoids bus-errors that occur when the
# ambient Python (e.g. base conda) is missing compiled extensions or uses
# incompatible native libraries (trimesh, scipy, numpy with native routines).
_SCRIPT_PYTHON = resolve_ml_env_python()


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:

        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh(mesh_filepath, target_filepath, executable, additional_args, quiet=False):
    if not quiet:
        logging.info(mesh_filepath + " --> " + target_filepath)

    # Python scripts need a dedicated interpreter prefix.
    # We use _SCRIPT_PYTHON (the ml_env conda interpreter) so that native
    # libraries (trimesh, scipy, numpy AVX routines) are loaded from the
    # correct environment, preventing SIGBUS crashes on mismatched builds.
    if executable.endswith(".py"):
        command = [_SCRIPT_PYTHON, executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args
    else:
        command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    if quiet:
        command.append("--quiet")

    subproc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, stderr_data = subproc.communicate()
    if stderr_data:
        msg = stderr_data.decode(errors="replace").strip()
        if msg:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
    if subproc.returncode != 0:
        raise RuntimeError(
            f"Preprocessor exited with code {subproc.returncode}: {mesh_filepath}"
        )


def append_data_source_map(data_source_map_filename, name, source):

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to "
        + "a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the "
        + "directory name.",
    )

    ds_group = arg_parser.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--datasource_map",
        dest="datasource_map",
        default=None,
        help=(
            "Path to the datasource map JSON file. Defaults to <data_dir>/.datasources.json. "
            "Useful when you want full CLI control over bookkeeping paths."
        ),
    )
    ds_group.add_argument(
        "--no_datasource_map",
        dest="no_datasource_map",
        default=False,
        action="store_true",
        help=(
            "Disable reading/writing the datasource map entirely. "
            "This bypasses the name->source consistency check (use with care)."
        ),
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
        + "Otherwise, the script will produce SDF samples for training.",
    )
    arg_parser.add_argument(
        "--preprocess-args",
        dest="preprocess_args",
        default="",
        help="Extra arguments forwarded verbatim to the preprocessor script "
        "(PreprocessMesh.py or SampleMeshSurface.py). "
        "Example: --preprocess-args '--anisotropic-bias --on-surface-ratio 0.15'",
    )
    arg_parser.add_argument(
        "--uniform-ratio",
        dest="uniform_ratio",
        type=float,
        default=None,
        help="Fraction of non-triplet samples allocated to uniform random sampling "
        "(forwarded to PreprocessMesh.py). Default: 0.06 (original DeepSDF split).",
    )
    arg_parser.add_argument(
        "--bounding-cube",
        dest="bounding_cube",
        type=float,
        nargs=3,
        default=None,
        metavar=("A", "B", "C"),
        help="Per-axis extents for uniform sampling box (forwarded to PreprocessMesh.py). "
        "Default: 1.0 1.0 1.0 (i.e. [-0.5, 0.5]^3).",
    )
    arg_parser.add_argument(
        "--sdf-method",
        dest="sdf_method",
        type=str,
        default=None,
        choices=["knn", "igl", "raycast"],
        help="SDF computation backend (forwarded to PreprocessMesh.py). Default: igl.",
    )
    arg_parser.add_argument(
        "--normal-offset",
        dest="normal_offset",
        action="store_true",
        default=False,
        help="Offset near-surface samples along surface normal (forwarded to PreprocessMesh.py). Default: off.",
    )
    arg_parser.add_argument(
        "--triplet-epsilon",
        dest="triplet_epsilon",
        type=float,
        default=None,
        help="Relative offset for on-surface triplets (forwarded to PreprocessMesh.py). Default: 0.02.",
    )
    arg_parser.add_argument(
        "--num-sample",
        dest="num_sample",
        type=int,
        default=None,
        help="Total SDF samples per mesh (forwarded to PreprocessMesh.py). Default: 500000.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    additional_general_args = []

    deepsdf_dir = os.path.dirname(os.path.abspath(__file__))
    if args.surface_sampling:
        executable = os.path.join(deepsdf_dir, "bin/SampleMeshSurface.py")
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = os.path.join(deepsdf_dir, "bin/PreprocessMesh.py")
        subdir = ws.sdf_samples_subdir
        extension = ".npz"

        if args.test_sampling:
            additional_general_args += ["-t"]

    if args.preprocess_args:
        import shlex
        additional_general_args += shlex.split(args.preprocess_args)

    if args.uniform_ratio is not None:
        additional_general_args += ["--uniform-ratio", str(args.uniform_ratio)]

    if args.bounding_cube is not None:
        additional_general_args += ["--bounding-cube"] + [str(v) for v in args.bounding_cube]

    if args.sdf_method is not None:
        additional_general_args += ["--sdf-method", args.sdf_method]

    if args.normal_offset:
        additional_general_args += ["--normal-offset"]

    if args.triplet_epsilon is not None:
        additional_general_args += ["--triplet-epsilon", str(args.triplet_epsilon)]

    if args.num_sample is not None:
        additional_general_args += ["--num_sample", str(args.num_sample)]

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    dest_dir = os.path.join(args.data_dir, subdir, args.source_name)

    logging.info(
        "Preprocessing data from "
        + args.source_dir
        + " and placing the results in "
        + dest_dir
    )

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    if args.surface_sampling:
        normalization_param_dir = os.path.join(
            args.data_dir, ws.normalization_param_subdir, args.source_name
        )
        if not os.path.isdir(normalization_param_dir):
            os.makedirs(normalization_param_dir)

    if not args.no_datasource_map:
        datasource_map_filename = (
            args.datasource_map
            if args.datasource_map is not None
            else ws.get_data_source_map_filename(args.data_dir)
        )
        append_data_source_map(datasource_map_filename, args.source_name, args.source_dir)

    class_directories = split[args.source_name]

    meshes_targets_and_specific_args = []

    for class_dir in class_directories:
        class_path = os.path.join(args.source_dir, class_dir)
        instance_dirs = class_directories[class_dir]

        logging.debug(
            "Processing " + str(len(instance_dirs)) + " instances of class " + class_dir
        )

        target_dir = os.path.join(dest_dir, class_dir)

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        for instance_dir in instance_dirs:

            shape_dir = os.path.join(class_path, instance_dir)

            processed_filepath = os.path.join(target_dir, instance_dir + extension)
            if args.skip and os.path.isfile(processed_filepath):
                logging.debug("skipping " + processed_filepath)
                continue

            try:
                mesh_filename = deep_sdf.data.find_mesh_in_directory(shape_dir)

                specific_args = []

                if args.surface_sampling:
                    normalization_param_target_dir = os.path.join(
                        normalization_param_dir, class_dir
                    )

                    if not os.path.isdir(normalization_param_target_dir):
                        os.mkdir(normalization_param_target_dir)

                    normalization_param_filename = os.path.join(
                        normalization_param_target_dir, instance_dir + ".npz"
                    )
                    specific_args = ["-n", normalization_param_filename]

                meshes_targets_and_specific_args.append(
                    (
                        os.path.join(shape_dir, mesh_filename),
                        processed_filepath,
                        specific_args,
                    )
                )

            except deep_sdf.data.NoMeshFileError:
                logging.warning("No mesh found for instance " + instance_dir)
            except deep_sdf.data.MultipleMeshFileError:
                logging.warning("Multiple meshes found for instance " + instance_dir)

    # Progress tracking for quiet mode
    total_meshes = len(meshes_targets_and_specific_args)
    # Report ~10 times across the run; for small batches report every mesh.
    progress_interval = max(1, total_meshes // 10)
    # Use lists to allow mutation from nested function (nonlocal doesn't work at module level)
    completed_count = [0]
    failed_count = [0]
    progress_lock = threading.Lock()

    def process_mesh_with_progress(mesh_filepath, target_filepath, executable, args_list):
        try:
            process_mesh(mesh_filepath, target_filepath, executable, args_list, quiet=args.quiet)
            success = True
        except RuntimeError as exc:
            logging.warning(str(exc))
            success = False
        if args.quiet:
            with progress_lock:
                if success:
                    completed_count[0] += 1
                else:
                    failed_count[0] += 1
                done = completed_count[0] + failed_count[0]
                if done % progress_interval == 0 or done == total_meshes:
                    sys.stdout.write(
                        f"\rProgress: {completed_count[0]}/{total_meshes} meshes"
                        + (f" ({failed_count[0]} failed)" if failed_count[0] else "")
                    )
                    sys.stdout.flush()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:
        futures = [
            executor.submit(
                process_mesh_with_progress,
                mesh_filepath,
                target_filepath,
                executable,
                specific_args + additional_general_args,
            )
            for mesh_filepath, target_filepath, specific_args
            in meshes_targets_and_specific_args
        ]
    # executor.__exit__ has already called shutdown(wait=True) here,
    # so all futures are complete; re-raise any worker exceptions.
    try:
        for future in futures:
            future.result()
    finally:
        if args.quiet and total_meshes > 0:
            print()  # guarantee newline after \r progress line
