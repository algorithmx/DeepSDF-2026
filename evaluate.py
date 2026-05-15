#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import os
import trimesh

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.metrics.chamfer import compute_chamfer


def evaluate(experiment_directory, checkpoint, data_dir, split_filename, output_filename="chamfer.csv"):

    with open(split_filename, "r") as f:
        split = json.load(f)

    specs = deep_sdf.workspace.load_experiment_specifications(experiment_directory)
    datasource = specs.get("DataSource", data_dir)

    chamfer_results = []
    skipped = []

    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                logging.debug(
                    "evaluating " + os.path.join(dataset, class_name, instance_name)
                )

                reconstructed_mesh_filename = ws.get_reconstructed_mesh_filename(
                    experiment_directory, checkpoint, dataset, class_name, instance_name
                )

                logging.debug(
                    'reconstructed mesh is "' + reconstructed_mesh_filename + '"'
                )

                if not os.path.exists(reconstructed_mesh_filename):
                    logging.warning(
                        "skipping {} — reconstruction not found: {}".format(
                            instance_name, reconstructed_mesh_filename
                        )
                    )
                    skipped.append(instance_name)
                    continue

                gt_mesh_filename = os.path.join(
                    datasource,
                    dataset,
                    class_name,
                    instance_name,
                    "mesh.obj",
                )

                if not os.path.exists(gt_mesh_filename):
                    gt_mesh_filename = os.path.join(
                        data_dir,
                        dataset,
                        class_name,
                        instance_name,
                        "mesh.obj",
                    )

                logging.debug("ground truth mesh is " + gt_mesh_filename)

                gt_mesh = trimesh.load(gt_mesh_filename)
                reconstruction = trimesh.load(reconstructed_mesh_filename)

                chamfer_dist = compute_chamfer(
                    gt_mesh,
                    reconstruction,
                    data_dir=datasource,
                    dataset=dataset,
                    class_name=class_name,
                    shape_id=instance_name,
                )

                logging.debug("chamfer distance: " + str(chamfer_dist))

                chamfer_results.append(
                    (os.path.join(dataset, class_name, instance_name), chamfer_dist)
                )

    eval_dir = ws.get_evaluation_dir(experiment_directory, checkpoint, True)
    with open(
        os.path.join(eval_dir, output_filename),
        "w",
    ) as f:
        f.write("shape,chamfer_dist\n")
        for result in chamfer_results:
            f.write("{},{}\n".format(result[0], result[1]))

    if skipped:
        logging.warning(
            "Skipped {}/{} shapes (no reconstruction found)".format(
                len(skipped), len(skipped) + len(chamfer_results)
            )
        )

    return chamfer_results, skipped


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
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
        help="The split to evaluate.",
    )
    arg_parser.add_argument(
        "--output",
        "-o",
        dest="output_filename",
        default="chamfer.csv",
        help="Output CSV filename (default: chamfer.csv). Use 'chamfer_all.csv' for full dataset evaluation.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    evaluate(
        args.experiment_directory,
        args.checkpoint,
        args.data_source,
        args.split_filename,
        args.output_filename,
    )
