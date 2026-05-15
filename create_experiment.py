#!/usr/bin/env python3
"""
Create DeepSDF Experiment Configuration

This script creates experiment configurations (train/test splits and specs.json)
for prepared datasets (ABC, WHUCAD, etc.). It can be run multiple times with
different hyperparameters on the same prepared data.

Usage:
    # Create experiment with default settings
    python create_experiment.py \\
        --data_dir /path/to/data/ABC \\
        --experiment_dir /path/to/experiments/abc_baseline

    # Create experiment with custom hyperparameters (limited GPU memory)
    python create_experiment.py \\
        --data_dir /path/to/data/ABC \\
        --experiment_dir /path/to/experiments/abc_small \\
        --code-length 128 \\
        --scenes-per-batch 32

    # Create another experiment with same data but different settings
    python create_experiment.py \\
        --data_dir /path/to/data/ABC \\
        --experiment_dir /path/to/experiments/abc_large \\
        --code-length 512 \\
        --scenes-per-batch 128

    # For WHUCAD dataset
    python create_experiment.py \\
        --data_dir /path/to/data/WHUCAD \\
        --experiment_dir /path/to/experiments/whucad_baseline \\
        --dataset-name WHUCAD

The script will:
1. Discover all models in the data directory
2. Create train/test splits
3. Write specs.json with configurable hyperparameters
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Import experiment_config directly to avoid deep_sdf package dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    'experiment_config', 
    Path(__file__).parent / 'deep_sdf' / 'experiment_config.py'
)
experiment_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_config_module)

ExperimentConfigBuilder = experiment_config_module.ExperimentConfigBuilder
Colors = experiment_config_module.Colors
print_header = experiment_config_module.print_header
print_success = experiment_config_module.print_success
print_warning = experiment_config_module.print_warning
print_error = experiment_config_module.print_error


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def discover_models(data_dir: Path) -> List[str]:
    """
    Discover all model IDs in the prepared data directory.
    
    Looks for models in <data_dir>/models/*/mesh.obj
    
    Args:
        data_dir: Path to prepared dataset directory
        
    Returns:
        List of model IDs (directory names)
    """
    models_dir = data_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}\n"
            "Ensure the data directory points to a prepared dataset "
            "(run data preparation script first)."
        )
    
    model_ids = []
    for model_dir in sorted(models_dir.iterdir()):
        if model_dir.is_dir() and list(model_dir.glob("*.obj")):
            model_ids.append(model_dir.name)
    
    if not model_ids:
        raise ValueError(
            f"No models found in {models_dir}. "
            "Ensure the dataset has been properly prepared."
        )
    
    return model_ids


def validate_data_directory(data_dir: Path) -> Tuple[bool, str]:
    """
    Validate that the data directory contains a properly prepared dataset.
    
    Args:
        data_dir: Path to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data_dir.exists():
        return False, f"Data directory does not exist: {data_dir}"
    
    # Check for required subdirectories
    models_dir = data_dir / "models"
    sdf_dir = data_dir / "SdfSamples" / "models"
    
    if not models_dir.exists():
        return False, f"Missing models/ directory: {models_dir}"
    
    if not sdf_dir.exists():
        return False, f"Missing SdfSamples/models/ directory: {sdf_dir}"
    
    # Check that we have models
    model_count = len(list(models_dir.iterdir()))
    if model_count == 0:
        return False, "No models found in models/ directory"
    
    # Check that we have SDF samples
    sdf_count = len(list(sdf_dir.glob("*.npz")))
    if sdf_count == 0:
        return False, "No SDF samples found in SdfSamples/models/"
    
    return True, f"Found {model_count} models and {sdf_count} SDF samples"


def create_experiment(
    data_dir: Path,
    experiment_dir: Path,
    dataset_name: Optional[str] = None,
    num_samples: Optional[int] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    # GPU-related configurable hyperparameters
    code_length: int = ExperimentConfigBuilder.DEFAULT_CODE_LENGTH,
    scenes_per_batch: int = ExperimentConfigBuilder.DEFAULT_SCENES_PER_BATCH,
    samples_per_scene: int = ExperimentConfigBuilder.DEFAULT_SAMPLES_PER_SCENE,
    dataloader_threads: int = ExperimentConfigBuilder.DEFAULT_DATALOADER_THREADS,
    num_epochs: int = ExperimentConfigBuilder.DEFAULT_NUM_EPOCHS,
    clamping_distance: float = ExperimentConfigBuilder.DEFAULT_CLAMPING_DISTANCE,
    pos_decay_threshold: float = ExperimentConfigBuilder.DEFAULT_POS_DECAY_THRESHOLD,
    pos_decay_exp: float = ExperimentConfigBuilder.DEFAULT_POS_DECAY_EXP,
    description: Optional[str] = None,
    verbose: bool = False,
    # Network architecture overrides
    dims: Optional[List[int]] = None,
    nerf_features: bool = False,
    # Category-aware latent code initialization
    scene_categories: Optional[str] = None,
    unmask_at_epoch: int = 0,
) -> Tuple[Path, Path, Path]:
    """
    Create a complete experiment configuration.
    
    Args:
        data_dir: Path to prepared dataset directory
        experiment_dir: Directory for experiment files (created if needed)
        dataset_name: Name of dataset (auto-detected if not provided)
        num_samples: Number of models to sample (default: all available)
        train_ratio: Train/test split ratio
        seed: Random seed for reproducibility
        code_length: Dimension of latent codes
        scenes_per_batch: Batch size for training
        samples_per_scene: Samples per scene
        dataloader_threads: Number of data loader workers
        num_epochs: Total training epochs
        clamping_distance: SDF clamping delta
        description: Optional custom description
        verbose: Enable verbose logging
        scene_categories: Path to scene_categories.json for one-hot category
            initialization. If None, specs.json omits SceneCategories (old random init).
        unmask_at_epoch: Epoch after which one-hot dimensions become trainable.
            Only used when scene_categories is set. Default 0 (never masked).
        
    Returns:
        Tuple of (train_split_path, test_split_path, specs_path)
    """
    setup_logging(verbose)
    
    # Validate data directory
    print_header("Validating Data Directory")
    is_valid, message = validate_data_directory(data_dir)
    if not is_valid:
        print_error(message)
        raise ValueError(message)
    print_success(message)
    
    # Auto-detect dataset name if not provided
    if dataset_name is None:
        dataset_name = data_dir.name.upper()
        print(f"Auto-detected dataset name: {dataset_name}")
    
    # Discover models
    print_header("Discovering Models")
    model_ids = discover_models(data_dir)
    print_success(f"Found {len(model_ids)} models")
    
    # Determine number of samples
    if num_samples is None:
        num_samples = len(model_ids)
        print(f"Using all {num_samples} models")
    elif num_samples > len(model_ids):
        print_warning(
            f"Requested {num_samples} samples but only {len(model_ids)} available. "
            f"Using all {len(model_ids)} models."
        )
        num_samples = len(model_ids)
    
    # Create description if not provided
    if description is None:
        description = f"DeepSDF experiment on {dataset_name} dataset."
    
    # Create experiment builder
    builder = ExperimentConfigBuilder(
        experiment_dir=experiment_dir,
        dataset_name=dataset_name,
        data_source=data_dir,
        description=description,
        verbose=True,
    )
    
    # Create splits
    train_split, test_split = builder.create_splits(
        model_ids=model_ids,
        num_samples=num_samples,
        train_ratio=train_ratio,
        seed=seed,
        step_name="Creating Train/Test Splits",
    )
    
    # Create specs
    specs_path = builder.create_specs(
        train_split=train_split,
        test_split=test_split,
        code_length=code_length,
        scenes_per_batch=scenes_per_batch,
        samples_per_scene=samples_per_scene,
        dataloader_threads=dataloader_threads,
        num_epochs=num_epochs,
        clamping_distance=clamping_distance,
        pos_decay_threshold=pos_decay_threshold,
        pos_decay_exp=pos_decay_exp,
        scene_categories=scene_categories,
        unmask_at_epoch=unmask_at_epoch,
    )

    # Apply network architecture overrides if requested
    if dims is not None or nerf_features:
        with open(specs_path, "r") as f:
            specs = json.load(f)
        if dims is not None:
            specs["NetworkSpecs"]["dims"] = dims
        if nerf_features:
            specs["NetworkSpecs"]["xyz_dim"] = 51
        with open(specs_path, "w") as f:
            json.dump(specs, f, indent=2)
        if verbose:
            print_success("Patched NetworkSpecs in specs.json")

    # Print final instructions
    print_header("Experiment Setup Complete!")
    print(f"""
{Colors.BOLD}Experiment directory:{Colors.ENDC}
    {experiment_dir}

{Colors.BOLD}Configuration:{Colors.ENDC}
    Dataset: {dataset_name}
    Models: {num_samples} ({int(num_samples * train_ratio)} train, {num_samples - int(num_samples * train_ratio)} test)
    CodeLength: {code_length}
    ScenesPerBatch: {scenes_per_batch}
    SamplesPerScene: {samples_per_scene}
    DataLoaderThreads: {dataloader_threads}

{Colors.OKCYAN}To start training:{Colors.ENDC}
    conda activate ml_env
    python train_deep_sdf.py -e {experiment_dir}

{Colors.OKCYAN}To monitor with tensorboard:{Colors.ENDC}
    tensorboard --logdir {experiment_dir}/logs

{Colors.OKCYAN}To reconstruct test shapes:{Colors.ENDC}
    conda activate ml_env
    python reconstruct.py -e {experiment_dir} -c 2000 \\
            -d {data_dir} --split {test_split}
""")
    
    return train_split, test_split, specs_path


def main():
    parser = argparse.ArgumentParser(
        description="Create DeepSDF experiment configuration for prepared datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create experiment with defaults (uses all models)
    python create_experiment.py \\
        --data_dir /path/to/data/ABC \\
        --experiment_dir experiments/abc_baseline

    # Create with specific sample count and custom hyperparameters
    python create_experiment.py \\
        --data_dir /path/to/data/ABC \\
        --experiment_dir experiments/abc_small \\
        --num_samples 500 \\
        --code-length 128 \\
        --scenes-per-batch 32

    # High-capacity experiment for powerful GPU
    python create_experiment.py \\
        --data_dir /path/to/data/WHUCAD \\
        --experiment_dir experiments/whucad_large \\
        --dataset-name WHUCAD \\
        --code-length 512 \\
        --scenes-per-batch 128

    # Create multiple experiments from same data
    python create_experiment.py --data_dir data/ABC --exp_dir experiments/abc_v1
    python create_experiment.py --data_dir data/ABC --exp_dir experiments/abc_v2 --code-length 128
    python create_experiment.py --data_dir data/ABC --exp_dir experiments/abc_v3 --scenes-per-batch 128
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to prepared dataset directory (e.g., data/ABC or data/WHUCAD)"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Directory for experiment files (specs.json, splits, logs, checkpoints)"
    )
    
    # Dataset identification
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (default: auto-detect from data_dir name)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Custom description for specs.json (default: auto-generated)"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of models to sample (default: use all available)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio for training split (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # GPU-related configurable hyperparameters
    parser.add_argument(
        "--code-length",
        type=int,
        default=ExperimentConfigBuilder.DEFAULT_CODE_LENGTH,
        help=f"Dimension of latent code vectors (default: {ExperimentConfigBuilder.DEFAULT_CODE_LENGTH}). "
             "Higher values = more expressive models but more GPU memory."
    )
    parser.add_argument(
        "--scenes-per-batch",
        type=int,
        default=ExperimentConfigBuilder.DEFAULT_SCENES_PER_BATCH,
        help=f"Number of scenes per batch for training (default: {ExperimentConfigBuilder.DEFAULT_SCENES_PER_BATCH}). "
             "Primary GPU memory control. Reduce for limited VRAM, increase for better utilization."
    )
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=ExperimentConfigBuilder.DEFAULT_SAMPLES_PER_SCENE,
        help=f"Number of SDF samples per scene (default: {ExperimentConfigBuilder.DEFAULT_SAMPLES_PER_SCENE})"
    )
    parser.add_argument(
        "--dataloader-threads",
        type=int,
        default=ExperimentConfigBuilder.DEFAULT_DATALOADER_THREADS,
        help=f"Number of data loader worker threads (default: {ExperimentConfigBuilder.DEFAULT_DATALOADER_THREADS})"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=ExperimentConfigBuilder.DEFAULT_NUM_EPOCHS,
        help=f"Total training epochs (default: {ExperimentConfigBuilder.DEFAULT_NUM_EPOCHS})"
    )
    parser.add_argument(
        "--clamping-distance",
        type=float,
        default=ExperimentConfigBuilder.DEFAULT_CLAMPING_DISTANCE,
        help=f"SDF clamping distance delta (default: {ExperimentConfigBuilder.DEFAULT_CLAMPING_DISTANCE})"
    )
    parser.add_argument(
        "--pos-decay-threshold",
        type=float,
        default=ExperimentConfigBuilder.DEFAULT_POS_DECAY_THRESHOLD,
        help="SDF threshold above which positive samples are down-weighted. "
             f"0 disables decay (default: {ExperimentConfigBuilder.DEFAULT_POS_DECAY_THRESHOLD})"
    )
    parser.add_argument(
        "--pos-decay-exp",
        type=float,
        default=ExperimentConfigBuilder.DEFAULT_POS_DECAY_EXP,
        help="Exponent k for inverse-power decay: w=(threshold/sdf)^k for sdf>threshold. "
             f"Higher = more aggressive culling (default: {ExperimentConfigBuilder.DEFAULT_POS_DECAY_EXP})"
    )
    
    # Category-aware latent code initialization
    parser.add_argument(
        "--scene-categories",
        type=str,
        default=None,
        help="Path to scene_categories.json for one-hot category initialization. "
             "If omitted, specs.json will not mention SceneCategories (old random init). "
             "Relative paths are resolved against --data_dir."
    )
    parser.add_argument(
        "--unmask-epoch",
        type=int,
        default=0,
        help="Epoch after which category one-hot dimensions become trainable. "
             "Only has effect when --scene-categories is set. Default 0 (never masked)."
    )
    
    # Network architecture options
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="Override the hidden layer widths of the decoder MLP (e.g. --dims 768 512 512 512 512 512 512 512). "
             "By default the standard [512, 512, 512, 512, 512, 512, 512, 512] is used."
    )
    parser.add_argument(
        "--nerf-features",
        action="store_true",
        default=False,
        help="Enable NeRF-style positional encoding (xyz_dim=51). "
             "By default the decoder uses raw 3D coordinates (xyz_dim=3)."
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = Path(args.data_dir).expanduser().resolve()
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    
    # Resolve scene-categories path if relative
    scene_categories = args.scene_categories
    if scene_categories is not None and not Path(scene_categories).is_absolute():
        scene_categories = str(data_dir / scene_categories)

    try:
        create_experiment(
            data_dir=data_dir,
            experiment_dir=experiment_dir,
            dataset_name=args.dataset_name,
            num_samples=args.num_samples,
            train_ratio=args.train_ratio,
            seed=args.seed,
            code_length=args.code_length,
            scenes_per_batch=args.scenes_per_batch,
            samples_per_scene=args.samples_per_scene,
            dataloader_threads=args.dataloader_threads,
            num_epochs=args.num_epochs,
            clamping_distance=args.clamping_distance,
            pos_decay_threshold=args.pos_decay_threshold,
            pos_decay_exp=args.pos_decay_exp,
            description=args.description,
            verbose=args.verbose,
            dims=args.dims,
            nerf_features=args.nerf_features,
            scene_categories=scene_categories,
            unmask_at_epoch=args.unmask_epoch,
        )
        return 0
    except Exception as e:
        print_error(f"Failed to create experiment: {e}")
        logging.exception("Error details:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
