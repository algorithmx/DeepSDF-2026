#!/usr/bin/env python3
"""
Experiment configuration module for DeepSDF.

This module provides shared functionality for creating experiment configurations
(specs.json and train/test splits) that works with any prepared dataset (ABC, WHUCAD, etc.).

Usage:
    # From preparation scripts
    from deep_sdf.experiment_config import ExperimentConfigBuilder
    
    builder = ExperimentConfigBuilder(
        experiment_dir=Path("experiments/my_exp"),
        dataset_name="ABC",
        data_source=Path("data/ABC"),
        description="Randomly sampled watertight CAD models from ABC dataset.",
    )
    train_split, test_split = builder.create_splits(
        model_ids=passing_ids,
        num_samples=1000,
        train_ratio=0.8,
        seed=42,
    )
    specs_path = builder.create_specs(
        train_split=train_split,
        test_split=test_split,
        code_length=256,  # Configurable!
        scenes_per_batch=64,  # Configurable!
    )
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class Colors:
    """Terminal color codes for pretty printing."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


class ExperimentConfigBuilder:
    """
    Builder for creating DeepSDF experiment configurations.
    
    This class handles creation of:
    1. Train/test split JSON files
    2. specs.json with configurable hyperparameters
    
    The hyperparameters (CodeLength, ScenesPerBatch, etc.) can be customized
    to adapt to different GPU capabilities.
    """
    
    # Default hyperparameters matching the original DeepSDF paper
    DEFAULT_CODE_LENGTH = 256
    DEFAULT_SCENES_PER_BATCH = 64
    DEFAULT_SAMPLES_PER_SCENE = 16384
    DEFAULT_DATALOADER_THREADS = 16
    DEFAULT_NUM_EPOCHS = 2001
    DEFAULT_CLAMPING_DISTANCE = 0.1
    DEFAULT_POS_DECAY_THRESHOLD = 0.0
    DEFAULT_POS_DECAY_EXP = 1.0
    DEFAULT_CODE_REGULARIZATION_LAMBDA = 1e-4
    DEFAULT_CODE_BOUND = 1.0
    DEFAULT_SNAPSHOT_FREQUENCY = 1000
    DEFAULT_ADDITIONAL_SNAPSHOTS = [100, 500]

    # AsyncPrefetchLoader multi-producer defaults
    DEFAULT_NUM_PRODUCERS = 4
    DEFAULT_WORKERS_PER_PRODUCER = 4
    
    # Network architecture (fixed per paper - Section 3)
    NETWORK_ARCH = "deep_sdf_decoder"
    NETWORK_SPECS = {
        "dims": [512, 512, 512, 512, 512, 512, 512, 512],
        "dropout": [0, 1, 2, 3, 4, 5, 6, 7],
        "dropout_prob": 0.2,
        "norm_layers": [0, 1, 2, 3, 4, 5, 6, 7],
        "latent_in": [4],
        "xyz_in_all": False,
        "use_tanh": False,
        "latent_dropout": False,
        "weight_norm": True,
        "xyz_dim": 3
    }
    
    # Default learning rate schedule
    DEFAULT_LEARNING_RATE_SCHEDULE = [
        {"Type": "Step", "Initial": 0.0005, "Interval": 500, "Factor": 0.5},
        {"Type": "Step", "Initial": 0.001, "Interval": 500, "Factor": 0.5}
    ]
    
    def __init__(
        self,
        experiment_dir: Path,
        dataset_name: str,
        data_source: Path,
        description: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the experiment config builder.
        
        Args:
            experiment_dir: Directory where experiment files will be written
            dataset_name: Name of the dataset (e.g., "ABC", "WHUCAD")
            data_source: Path to the prepared dataset directory
            description: Optional description for specs.json. If None, a default
                        description will be generated.
            verbose: Whether to print progress messages
        """
        self.experiment_dir = Path(experiment_dir)
        self.dataset_name = dataset_name
        self.data_source = Path(data_source)
        self.description = description or f"DeepSDF trained on {dataset_name} dataset."
        self.verbose = verbose
        
        # Ensure experiment directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def create_splits(
        self,
        model_ids: List[str],
        num_samples: int,
        train_ratio: float = 0.8,
        seed: int = 42,
        step_name: str = "STEP 7: Finalizing Train/Test Splits",
    ) -> Tuple[Path, Path]:
        """
        Create train/test split JSON files.
        
        Args:
            model_ids: List of validated model IDs to sample from
            num_samples: Number of models to select for the experiment
            train_ratio: Ratio of samples for training (default: 0.8)
            seed: Random seed for reproducibility
            step_name: Header text to print (can be customized for different datasets)
            
        Returns:
            Tuple of (train_split_path, test_split_path)
            
        Raises:
            RuntimeError: If not enough models passed validation
        """
        if self.verbose:
            print_header(step_name)
        
        if len(model_ids) < num_samples:
            raise RuntimeError(
                f"Only {len(model_ids)} models available, need {num_samples}. "
                f"Consider increasing oversample factor or using more data."
            )
        
        # Random shuffle and select
        rng = random.Random(seed)
        selected = list(model_ids)
        rng.shuffle(selected)
        selected = selected[:num_samples]
        
        # Split train/test
        split_idx = int(num_samples * train_ratio)
        train_ids = selected[:split_idx]
        test_ids = selected[split_idx:]
        
        # Write split files
        train_split_path = self.experiment_dir / f"{self.dataset_name.lower()}_train.json"
        test_split_path = self.experiment_dir / f"{self.dataset_name.lower()}_test.json"
        
        with open(train_split_path, "w") as f:
            json.dump({"": {"models": train_ids}}, f, indent=2)
        with open(test_split_path, "w") as f:
            json.dump({"": {"models": test_ids}}, f, indent=2)
        
        if self.verbose:
            discarded = len(model_ids) - num_samples
            print_success(
                f"Splits written: {len(train_ids)} train + {len(test_ids)} test "
                f"(selected from {len(model_ids)} validated models, "
                f"{discarded} discarded)"
            )
        
        return train_split_path, test_split_path
    
    def create_specs(
        self,
        train_split: Path,
        test_split: Path,
        code_length: int = DEFAULT_CODE_LENGTH,
        scenes_per_batch: int = DEFAULT_SCENES_PER_BATCH,
        samples_per_scene: int = DEFAULT_SAMPLES_PER_SCENE,
        dataloader_threads: int = DEFAULT_DATALOADER_THREADS,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        clamping_distance: float = DEFAULT_CLAMPING_DISTANCE,
        pos_decay_threshold: float = DEFAULT_POS_DECAY_THRESHOLD,
        pos_decay_exp: float = DEFAULT_POS_DECAY_EXP,
        code_regularization_lambda: float = DEFAULT_CODE_REGULARIZATION_LAMBDA,
        code_bound: float = DEFAULT_CODE_BOUND,
        snapshot_frequency: int = DEFAULT_SNAPSHOT_FREQUENCY,
        additional_snapshots: Optional[List[int]] = None,
        learning_rate_schedule: Optional[List[Dict[str, Any]]] = None,
        code_regularization: bool = True,
        num_producers: int = DEFAULT_NUM_PRODUCERS,
        workers_per_producer: int = DEFAULT_WORKERS_PER_PRODUCER,
        scene_categories: Optional[str] = None,
        unmask_at_epoch: int = 0,
    ) -> Path:
        """
        Create the specs.json configuration file.

        All hyperparameters related to GPU utilization are configurable here:
        - code_length: Dimensionality of latent codes (affects memory and model size)
        - scenes_per_batch: Number of scenes per batch (primary GPU memory control)
        - samples_per_scene: Samples per scene (affects total batch memory)
        - dataloader_threads: Number of data loading workers (legacy, use num_producers instead)
        - num_producers: Number of producer processes for AsyncPrefetchLoader
        - workers_per_producer: Workers per producer process

        Args:
            train_split: Path to train split JSON file
            test_split: Path to test split JSON file
            code_length: Dimension of latent code vectors (default: 256)
            scenes_per_batch: Batch size for training (default: 64)
            samples_per_scene: Number of samples per scene (default: 16384)
            dataloader_threads: Number of data loader workers (default: 16)
            num_epochs: Total training epochs (default: 2001)
            clamping_distance: SDF clamping distance δ (default: 0.1)
            code_regularization_lambda: Regularization strength (default: 1e-4)
            code_bound: Max norm for latent codes (default: 1.0)
            snapshot_frequency: Epoch interval for saving checkpoints (default: 1000)
            additional_snapshots: Additional epochs to save (default: [100, 500])
            learning_rate_schedule: LR schedule config (uses default if None)
            code_regularization: Whether to apply code regularization (default: True)
            num_producers: Number of producer processes (default: 4)
            workers_per_producer: Workers per producer (default: 4)
            scene_categories: Path to scene_categories.json for one-hot category
                initialization. If None, SceneCategories is omitted from specs.json.
            unmask_at_epoch: Epoch after which one-hot dims become trainable.
                Only used when scene_categories is set. Default 0 (never masked).

        Returns:
            Path to the created specs.json file
        """
        if additional_snapshots is None:
            additional_snapshots = self.DEFAULT_ADDITIONAL_SNAPSHOTS.copy()
        if learning_rate_schedule is None:
            learning_rate_schedule = self.DEFAULT_LEARNING_RATE_SCHEDULE.copy()

        specs = {
            "Description": [
                f"DeepSDF trained on {self.dataset_name} dataset.",
                self.description
            ],
            "DataSource": str(self.data_source),
            "TrainSplit": str(train_split),
            "TestSplit": str(test_split),
            "NetworkArch": self.NETWORK_ARCH,
            "NetworkSpecs": self.NETWORK_SPECS.copy(),
            "CodeLength": code_length,
            "NumEpochs": num_epochs,
            "SnapshotFrequency": snapshot_frequency,
            "AdditionalSnapshots": additional_snapshots,
            "LearningRateSchedule": learning_rate_schedule,
            "SamplesPerScene": samples_per_scene,
            "ScenesPerBatch": scenes_per_batch,
            "DataLoaderThreads": dataloader_threads,
            "NumProducers": num_producers,
            "WorkersPerProducer": workers_per_producer,
            "ClampingDistance": clamping_distance,
            "PosDecayThreshold": pos_decay_threshold,
            "PosDecayExp": pos_decay_exp,
            "CodeRegularization": code_regularization,
            "CodeRegularizationLambda": code_regularization_lambda,
            "CodeBound": code_bound,
        }

        # Category-aware latent code initialization (optional, backward-compatible)
        if scene_categories is not None:
            specs["SceneCategories"] = scene_categories
            specs["unmask-at-epoch"] = unmask_at_epoch

        specs_path = self.experiment_dir / "specs.json"
        with open(specs_path, 'w') as f:
            json.dump(specs, f, indent=2)

        if self.verbose:
            print_success(f"Created experiment config: {specs_path}")
            print(f"  - CodeLength: {code_length}")
            print(f"  - ScenesPerBatch: {scenes_per_batch}")
            print(f"  - SamplesPerScene: {samples_per_scene}")
            print(f"  - NumProducers: {num_producers}")
            print(f"  - WorkersPerProducer: {workers_per_producer}")
            effective_threads = num_producers * workers_per_producer
            if dataloader_threads != self.DEFAULT_DATALOADER_THREADS:
                print_warning(
                    f"DataLoaderThreads={dataloader_threads} is ignored by AsyncPrefetchLoader. "
                    f"Effective thread count: {effective_threads} "
                    f"(NumProducers × WorkersPerProducer = {num_producers} × {workers_per_producer})"
                )
            else:
                print(f"  - DataLoaderThreads: {dataloader_threads} (legacy, ignored by AsyncPrefetchLoader)")
            if scene_categories is not None:
                print(f"  - SceneCategories: {scene_categories}")
                print(f"  - unmask-at-epoch: {unmask_at_epoch}")

        return specs_path
    
    def print_final_instructions(
        self,
        data_dir: Path,
        test_split: Path,
        cache_dir: Optional[Path] = None,
        custom_instructions: Optional[str] = None,
    ):
        """
        Print final setup instructions.
        
        Args:
            data_dir: Path to the dataset directory
            test_split: Path to test split file
            cache_dir: Optional cache directory to mention in cleanup instructions
            custom_instructions: Optional custom instructions string
        """
        if custom_instructions:
            print(custom_instructions)
            return
        
        print_header("SETUP COMPLETE!")
        
        cache_cleanup = ""
        if cache_dir:
            cache_cleanup = f"""
{Colors.OKCYAN}Cache location (for reuse):{Colors.ENDC}
    {cache_dir}

{Colors.WARNING}Cache cleanup:{Colors.ENDC}
    python prepare_data.py --data_dir {data_dir.parent} --clean_cache"""
        
        print(f"""
{Colors.BOLD}Your {self.dataset_name} dataset is ready for training!{Colors.ENDC}

{Colors.OKCYAN}To start training:{Colors.ENDC}
    conda activate ml_env
    python train_deep_sdf.py -e {self.experiment_dir}

{Colors.OKCYAN}To monitor with tensorboard:{Colors.ENDC}
    tensorboard --logdir {self.experiment_dir}/logs

{Colors.OKCYAN}To reconstruct test shapes:{Colors.ENDC}
    conda activate ml_env
    python reconstruct.py -e {self.experiment_dir} -c 2000 \\
            -d {data_dir} --split {test_split}

{Colors.OKCYAN}To generate training meshes:{Colors.ENDC}
    conda activate ml_env
    python generate_training_meshes.py -e {self.experiment_dir} -c latest

{Colors.OKCYAN}Dataset location:{Colors.ENDC}
    Meshes: {data_dir}/models/
    SDF Samples: {data_dir}/SdfSamples/
{cache_cleanup}
""")


def create_experiment(
    experiment_dir: Path,
    dataset_name: str,
    data_source: Path,
    model_ids: List[str],
    num_samples: int,
    description: Optional[str] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    # GPU-related configurable hyperparameters
    code_length: int = ExperimentConfigBuilder.DEFAULT_CODE_LENGTH,
    scenes_per_batch: int = ExperimentConfigBuilder.DEFAULT_SCENES_PER_BATCH,
    samples_per_scene: int = ExperimentConfigBuilder.DEFAULT_SAMPLES_PER_SCENE,
    dataloader_threads: int = ExperimentConfigBuilder.DEFAULT_DATALOADER_THREADS,
    # Other hyperparameters
    num_epochs: int = ExperimentConfigBuilder.DEFAULT_NUM_EPOCHS,
    clamping_distance: float = ExperimentConfigBuilder.DEFAULT_CLAMPING_DISTANCE,
    verbose: bool = True,
    step_name: str = "STEP 7: Finalizing Train/Test Splits",
    # Category-aware latent code initialization
    scene_categories: Optional[str] = None,
    unmask_at_epoch: int = 0,
) -> Tuple[Path, Path, Path]:
    """
    Convenience function to create a complete experiment configuration.
    
    This is a one-shot function that creates both splits and specs.
    
    Args:
        experiment_dir: Directory for experiment files
        dataset_name: Name of the dataset
        data_source: Path to prepared dataset
        model_ids: List of validated model IDs
        num_samples: Number of models to select
        description: Optional dataset description
        train_ratio: Train/test split ratio
        seed: Random seed
        code_length: Dimension of latent codes (GPU memory related)
        scenes_per_batch: Batch size (GPU memory related)
        samples_per_scene: Samples per scene (GPU memory related)
        dataloader_threads: Number of data loader workers
        num_epochs: Total training epochs
        clamping_distance: SDF clamping delta
        verbose: Whether to print progress
        step_name: Name of the split creation step
        scene_categories: Path to scene_categories.json for one-hot category
            initialization. If None, SceneCategories is omitted from specs.json.
        unmask_at_epoch: Epoch after which one-hot dims become trainable.
            Only used when scene_categories is set. Default 0 (never masked).
        
    Returns:
        Tuple of (train_split_path, test_split_path, specs_path)
    """
    builder = ExperimentConfigBuilder(
        experiment_dir=experiment_dir,
        dataset_name=dataset_name,
        data_source=data_source,
        description=description,
        verbose=verbose,
    )
    
    train_split, test_split = builder.create_splits(
        model_ids=model_ids,
        num_samples=num_samples,
        train_ratio=train_ratio,
        seed=seed,
        step_name=step_name,
    )
    
    specs_path = builder.create_specs(
        train_split=train_split,
        test_split=test_split,
        code_length=code_length,
        scenes_per_batch=scenes_per_batch,
        samples_per_scene=samples_per_scene,
        dataloader_threads=dataloader_threads,
        num_epochs=num_epochs,
        clamping_distance=clamping_distance,
        scene_categories=scene_categories,
        unmask_at_epoch=unmask_at_epoch,
    )
    
    return train_split, test_split, specs_path
