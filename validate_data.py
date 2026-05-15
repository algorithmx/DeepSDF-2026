#!/usr/bin/env python3
"""
Data Validation Script for DeepSDF Training

This script validates that all data required for training can actually be loaded
using the same code path as train_deep_sdf.py. It catches data errors before
starting a long training run.

Key Principle: This validation is INDEPENDENT of data preparation and uses the
EXACT same data loading code as training to ensure accuracy.

Usage:
    python validate_data.py -e experiments/abc_100_trial
    python validate_data.py -e experiments/abc_100_trial --quick

Exit codes:
    0 - All validation passed
    1 - Validation failed
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import deep_sdf
import deep_sdf.workspace as ws


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_split(experiment_dir: Path, split_name: str, quick: bool = False) -> Tuple[bool, int, List[str]]:
    specs = ws.load_experiment_specifications(experiment_dir)
    data_source = specs["DataSource"]
    split_file = specs["TrainSplit"] if split_name == "train" else specs["TestSplit"]
    num_samp_per_scene = specs.get("SamplesPerScene", 16384)
    
    logging.info(f"Loading {split_name} split from: {split_file}")
    
    try:
        with open(split_file, "r") as f:
            split = json.load(f)
    except Exception as e:
        return False, 0, [f"Cannot load {split_name} split file: {e}"]
    
    # Create dataset using the EXACT same code as training
    try:
        sdf_dataset = deep_sdf.data.SDFSamples(
            data_source, split, num_samp_per_scene, load_ram=False
        )
    except Exception as e:
        return False, 0, [f"Cannot create SDFSamples dataset: {e}"]
    
    num_scenes = len(sdf_dataset)
    logging.info(f"Dataset contains {num_scenes} scenes")
    
    if num_scenes == 0:
        return False, 0, ["Dataset is empty - no scenes found"]
    
    if quick:
        indices_to_check = [0]
        if num_scenes > 1:
            indices_to_check.append(num_scenes - 1)
        if num_scenes > 10:
            indices_to_check.extend([num_scenes // 4, num_scenes // 2, 3 * num_scenes // 4])
        indices_to_check = sorted(set(indices_to_check))
        logging.info(f"Quick mode: checking {len(indices_to_check)} scenes")
    else:
        indices_to_check = range(num_scenes)
        logging.info(f"Checking all {num_scenes} scenes...")
    
    errors = []
    for idx in indices_to_check:
        try:
            sdf_data, scene_idx = sdf_dataset[idx]
            if sdf_data is None:
                errors.append(f"Scene {idx}: returned None")
                continue
            if hasattr(sdf_data, 'shape'):
                if sdf_data.shape[0] == 0:
                    errors.append(f"Scene {idx}: empty tensor")
                elif sdf_data.shape[1] < 4:
                    errors.append(f"Scene {idx}: unexpected shape {sdf_data.shape}")
        except FileNotFoundError as e:
            fname = getattr(e, 'filename', 'unknown')
            errors.append(f"Scene index {idx}: File not found - {fname}")
        except Exception as e:
            errors.append(f"Scene index {idx}: {type(e).__name__}: {e}")
    
    return len(errors) == 0, num_scenes, errors


def validate_experiment(experiment_dir: Path, quick: bool = False, verbose: bool = False) -> bool:
    setup_logging(verbose)
    
    logging.info("=" * 60)
    logging.info("DeepSDF Data Validation")
    logging.info("=" * 60)
    logging.info(f"Experiment: {experiment_dir}")
    logging.info(f"Mode: {'Quick' if quick else 'Full'}")
    
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    
    if not experiment_dir.exists():
        logging.error(f"Experiment directory not found: {experiment_dir}")
        return False
    
    specs_path = experiment_dir / "specs.json"
    if not specs_path.exists():
        logging.error(f"specs.json not found: {specs_path}")
        return False
    
    try:
        specs = ws.load_experiment_specifications(experiment_dir)
        logging.info(f"Data source: {specs['DataSource']}")
    except Exception as e:
        logging.error(f"Cannot load specs.json: {e}")
        return False
    
    # Validate train split
    logging.info("")
    logging.info("-" * 60)
    logging.info("Validating TRAIN split")
    logging.info("-" * 60)
    
    train_success, train_scenes, train_errors = validate_split(experiment_dir, "train", quick)
    
    if train_errors:
        logging.error(f"Train split FAILED with {len(train_errors)} errors:")
        for i, err in enumerate(train_errors[:10]):
            logging.error(f"  {i+1}. {err}")
        if len(train_errors) > 10:
            logging.error(f"  ... and {len(train_errors) - 10} more")
    else:
        logging.info(f"✓ Train split PASSED ({train_scenes} scenes)")
    
    # Validate test split
    logging.info("")
    logging.info("-" * 60)
    logging.info("Validating TEST split")
    logging.info("-" * 60)
    
    test_success, test_scenes, test_errors = validate_split(experiment_dir, "test", quick)
    
    if test_errors:
        logging.error(f"Test split FAILED with {len(test_errors)} errors:")
        for i, err in enumerate(test_errors[:10]):
            logging.error(f"  {i+1}. {err}")
        if len(test_errors) > 10:
            logging.error(f"  ... and {len(test_errors) - 10} more")
    else:
        logging.info(f"✓ Test split PASSED ({test_scenes} scenes)")
    
    # Summary
    logging.info("")
    logging.info("=" * 60)
    
    if train_success and test_success:
        logging.info("VALIDATION PASSED")
        logging.info("=" * 60)
        logging.info(f"Train scenes: {train_scenes}")
        logging.info(f"Test scenes:  {test_scenes}")
        logging.info("Ready to train!")
        return True
    else:
        logging.error("VALIDATION FAILED")
        logging.info("=" * 60)
        logging.error("Please fix data issues before training.")
        logging.error("Regenerate splits with: python create_experiment.py ...")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate DeepSDF training data using training code paths."
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        required=True,
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check - only validate a few samples per split"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    success = validate_experiment(
        Path(args.experiment),
        quick=args.quick,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
