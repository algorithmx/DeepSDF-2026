#!/usr/bin/env python3
"""
RF Dataset Preparation Script for DeepSDF.

Organizes a flat folder of pre-validated OBJ files into DeepSDF-compatible
directory structure, then generates SDF samples for all models in a single pass.
Splits and experiment configuration are left to create_experiment.py so you can
create multiple experiments from the same prepared data.

Folder Organization:
--------------------
```
<source_dir>/                        # Flat folder of OBJ files (read-only)
├── 000000.obj
├── 000001.obj
└── ...

<data_dir>/
└── RF/                              # Processed output
    ├── .rf_prepare_state.json       # Resume state (auto-managed)
    ├── models/                      # Organized OBJ meshes
    │   ├── 0001/mesh.obj
    │   └── 0002/mesh.obj
    ├── SdfSamples/                  # SDF samples (training data)
    │   └── models/
    │       ├── 0001.npz
    │       └── 0002.npz
    └── SurfaceSamples/              # Surface samples (optional, --surface)
        └── models/
            ├── 0001.ply
            └── 0002.ply
```

Usage:
------
# Prepare all RF models with SDF samples
python rf_prepare.py \\
    --source_dir /opt/data/DeepSDF/data/RF/obj \\
    --data_dir /opt/data/DeepSDF/data

# Also generate surface samples for evaluation
python rf_prepare.py \\
    --source_dir /opt/data/DeepSDF/data/RF/obj \\
    --data_dir /opt/data/DeepSDF/data \\
    --surface

# Resume interrupted run (picks up where it left off)
python rf_prepare.py \\
    --source_dir /opt/data/DeepSDF/data/RF/obj \\
    --data_dir /opt/data/DeepSDF/data \\
    --surface --resume

# Afterwards, create experiments from the prepared data:
python create_experiment.py \\
    --data_dir /opt/data/DeepSDF/data/RF \\
    --experiment_dir experiments/rf_exp1 \\
    --num-samples 1000

python create_experiment.py \\
    --data_dir /opt/data/DeepSDF/data/RF \\
    --experiment_dir experiments/rf_exp2 \\
    --num-samples 500 --code-length 128

Key Arguments:
--------------
--source_dir:   Flat directory containing pre-validated .obj files (read-only)
--data_dir:     Output root directory (creates RF/ subdirectory here)
--surface:      Also generate SurfaceSamples + NormalizationParameters (for evaluate.py)
--resume:       Resume from last completed stage (uses .rf_prepare_state.json)
--threads:      Number of parallel workers for SDF generation (default: 8)
--seed:         Random seed (default: 42, currently reserved for future use)

Requirements:
-------------
- Conda environment 'ml_env' must be available
- bin/PreprocessMesh.py and bin/SampleMeshSurface.py must be present
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# DeepSDF utilities
from deep_sdf.experiment_config import (
    Colors,
    print_header,
    print_success,
    print_warning,
    print_error,
)
from deep_sdf.utils import build_ml_env_python_cmd, ML_ENV_NAME, resolve_conda_exe

PROJECT_ROOT = Path(__file__).parent.absolute()
BIN_DIR = PROJECT_ROOT / "bin"

STATE_FILE = ".rf_prepare_state.json"


# =============================================================================
# Utility
# =============================================================================

def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def get_disk_free_bytes(path: Path) -> Optional[int]:
    try:
        usage = shutil.disk_usage(str(path))
        return int(usage.free)
    except Exception:
        return None


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# =============================================================================
# State file (for --resume)
# =============================================================================

def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state_path: Path, stage: str, model_ids: List[str]):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump({"stage": stage, "model_ids": model_ids, "model_count": len(model_ids)}, f, indent=2)


# =============================================================================
# Stage 1: Discover OBJ files
# =============================================================================

def discover_obj_files(source_dir: Path) -> List[Path]:
    """Discover all OBJ files in a flat source directory."""
    obj_files = sorted(source_dir.glob("*.obj"))
    if not obj_files:
        raise RuntimeError(f"No .obj files found in {source_dir}")
    return obj_files


# =============================================================================
# Stage 2: Copy to models/
# =============================================================================

def organize_all_models(
    obj_files: List[Path],
    dataset_dir: Path,
    state_path: Path,
) -> List[str]:
    """Copy all OBJ files into DeepSDF models/ directory structure.

    Each file gets a 4-digit ID and is copied as models/<id>/mesh.obj.
    This is the canonical layout that preprocess_data.py expects.

    Returns the list of assigned model IDs.
    """
    print_header("Stage 2: Organizing Models")

    models_dir = dataset_dir / "models"

    # Resume: check if models already exist
    existing_ids = _scan_model_ids(models_dir)
    if existing_ids:
        # Check completeness — require ALL obj_files to be present
        expected_count = len(obj_files)
        if len(existing_ids) == expected_count:
            print_success(f"All {expected_count} models already organized (skipping)")
            return existing_ids
        else:
            # Partial — clean and redo (shouldn't happen with state file, but safe)
            print_warning(
                f"Found {len(existing_ids)}/{expected_count} models — re-organizing"
            )
            for model_id in existing_ids:
                shutil.rmtree(models_dir / model_id, ignore_errors=True)

    models_dir.mkdir(parents=True, exist_ok=True)

    total = len(obj_files)
    log_interval = max(1, total // 10)
    model_ids: List[str] = []

    logging.info(f"Copying {total} models to {models_dir}...")
    for i, obj_path in enumerate(obj_files, 1):
        model_id = f"{i:04d}"
        model_subdir = models_dir / model_id
        model_subdir.mkdir(exist_ok=True)
        shutil.copy2(obj_path, model_subdir / "mesh.obj")
        model_ids.append(model_id)

        if i % log_interval == 0 or i == total:
            logging.info(f"  Copied {i}/{total} models...")

    _save_state(state_path, "organized", model_ids)
    print_success(f"Organized {total} models in {models_dir}")
    return model_ids


def _scan_model_ids(models_dir: Path) -> List[str]:
    """Return sorted list of existing model IDs in models/."""
    if not models_dir.exists():
        return []
    ids = []
    for d in sorted(models_dir.iterdir()):
        if d.is_dir() and (d / "mesh.obj").exists():
            ids.append(d.name)
    return ids


# =============================================================================
# Stage 3: Verify preprocessor scripts
# =============================================================================

def verify_python_scripts() -> None:
    """Verify that required preprocessor scripts and conda env are present."""
    print_header("Stage 3: Verifying Preprocessor Scripts")

    for script in [BIN_DIR / "PreprocessMesh.py", BIN_DIR / "SampleMeshSurface.py"]:
        if not script.exists():
            raise FileNotFoundError(
                f"Required script not found: {script}\n"
                "Make sure bin/PreprocessMesh.py and bin/SampleMeshSurface.py are present."
            )
        print_success(f"{script.name} found")

    conda_exe = resolve_conda_exe()
    if not conda_exe:
        raise RuntimeError(
            f"Conda executable not found. This pipeline requires conda env '{ML_ENV_NAME}'."
        )
    print_success(f"Using Python via: {conda_exe} run -n {ML_ENV_NAME} python")


# =============================================================================
# Stage 4: Generate SDF samples (all models, one pass)
# =============================================================================

# Default SDF sampling flags forwarded to PreprocessMesh.py via preprocess_data.py.
# Values match the PreprocessMesh.py defaults so behaviour is unchanged when
# no overrides are given at any level.
DEFAULT_SDF_PREPROCESS_ARGS = (
    "--anisotropic-bias"
    " --on-surface-ratio 0.15"
    " --sdf-method igl"
    " --triplet-epsilon 0.02"
    " --num_sample 500000"
)


def generate_sdf_for_all(
    dataset_dir: Path,
    model_ids: List[str],
    num_threads: int = 8,
    preprocess_args: str = DEFAULT_SDF_PREPROCESS_ARGS,
) -> None:
    """Generate SDF samples for every model via preprocess_data.py.

    Writes a temporary split covering all model_ids so preprocess_data.py
    processes them in a single pass.
    """
    print_header("Stage 4: Generating SDF Samples")

    if preprocess_args:
        logging.info(f"  SDF preprocessor flags: {preprocess_args}")

    # Check if already complete
    if _all_npz_exist(dataset_dir, model_ids):
        print_success(f"SDF samples already exist for all {len(model_ids)} models (skipping)")
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="rf_sdf_", delete=False
    ) as fh:
        split_path = Path(fh.name)
        json.dump({"": {"models": model_ids}}, fh, indent=2)

    try:
        cmd = build_ml_env_python_cmd(
            PROJECT_ROOT / "preprocess_data.py",
            "--data_dir", str(dataset_dir),
            "--source", str(dataset_dir),
            "--name", "",
            "--split", str(split_path),
            "--threads", str(num_threads),
            "--no_datasource_map",
            "--skip",
            "--quiet",
        )
        if preprocess_args:
            cmd += ["--preprocess-args", preprocess_args]
        subprocess.run(cmd, check=True)

        sdf_dir = dataset_dir / "SdfSamples" / "models"
        npz_count = len(list(sdf_dir.glob("*.npz")))
        if npz_count == 0:
            raise RuntimeError(f"No SDF files generated in {sdf_dir}")
        print_success(f"Generated {npz_count} SDF sample files")
    finally:
        split_path.unlink(missing_ok=True)


def _all_npz_exist(dataset_dir: Path, model_ids: List[str]) -> bool:
    """Check whether every model has a corresponding .npz in SdfSamples/models/."""
    sdf_dir = dataset_dir / "SdfSamples" / "models"
    if not sdf_dir.exists():
        return False
    for mid in model_ids:
        if not (sdf_dir / f"{mid}.npz").exists():
            return False
    return True


# =============================================================================
# Stage 5: Generate surface samples (optional)
# =============================================================================

def generate_surface_for_all(
    dataset_dir: Path,
    model_ids: List[str],
    num_threads: int = 8,
) -> None:
    """Generate surface samples for every model via preprocess_data.py --surface.

    Writes SurfaceSamples/ and NormalizationParameters/ alongside the SDF data.
    """
    print_header("Stage 5: Generating Surface Samples + Normalization Parameters")

    # Check if already complete
    if _all_ply_exist(dataset_dir, model_ids):
        print_success(f"Surface samples already exist for all {len(model_ids)} models (skipping)")
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="rf_surface_", delete=False
    ) as fh:
        split_path = Path(fh.name)
        json.dump({"": {"models": model_ids}}, fh, indent=2)

    try:
        cmd = build_ml_env_python_cmd(
            PROJECT_ROOT / "preprocess_data.py",
            "--data_dir", str(dataset_dir),
            "--source", str(dataset_dir),
            "--name", "",
            "--split", str(split_path),
            "--threads", str(num_threads),
            "--no_datasource_map",
            "--skip",
            "--quiet",
            "--surface",
        )
        subprocess.run(cmd, check=True)

        surface_dir = dataset_dir / "SurfaceSamples" / "models"
        ply_count = len(list(surface_dir.glob("*.ply")))
        print_success(f"Generated {ply_count} surface sample files")
    finally:
        split_path.unlink(missing_ok=True)


def _all_ply_exist(dataset_dir: Path, model_ids: List[str]) -> bool:
    """Check whether every model has a corresponding .ply in SurfaceSamples/models/."""
    surface_dir = dataset_dir / "SurfaceSamples" / "models"
    if not surface_dir.exists():
        return False
    for mid in model_ids:
        if not (surface_dir / f"{mid}.ply").exists():
            return False
    return True


# =============================================================================
# Print next steps
# =============================================================================

def print_next_steps(dataset_dir: Path, has_surface: bool):
    print_header("PREPARATION COMPLETE")

    lines = [
        "",
        f"{Colors.BOLD}RF dataset prepared at:{Colors.ENDC}",
        f"    {dataset_dir}",
        "",
        f"{Colors.OKCYAN}Next — create an experiment:{Colors.ENDC}",
        "    python create_experiment.py \\",
        f"        --data_dir {dataset_dir} \\",
        f"        --experiment_dir experiments/rf_exp1",
        "",
        f"{Colors.OKCYAN}Tuning hyperparameters (re-run as many times as you want):{Colors.ENDC}",
        "    python create_experiment.py \\",
        f"        --data_dir {dataset_dir} \\",
        f"        --experiment_dir experiments/rf_small \\",
        "        --num-samples 500 \\",
        "        --code-length 128 \\",
        "        --scenes-per-batch 32",
        "",
        f"{Colors.OKCYAN}Start training:{Colors.ENDC}",
        "    conda activate ml_env",
        "    python train_deep_sdf.py -e experiments/rf_exp1",
    ]

    if has_surface:
        lines += [
            "",
            f"{Colors.OKCYAN}Evaluate:{Colors.ENDC}",
            "    python evaluate.py -e experiments/rf_exp1 -c 2000 \\",
            f"        -d {dataset_dir} --split experiments/rf_exp1/rf_test.json",
        ]

    print("\n".join(lines))
    print()


def _build_sdf_preprocess_args(args) -> str:
    """Build the --preprocess-args string from explicit CLI options."""
    parts = [DEFAULT_SDF_PREPROCESS_ARGS]
    if args.uniform_ratio is not None:
        parts.append(f"--uniform-ratio {args.uniform_ratio}")
    if args.bounding_cube is not None:
        a, b, c = args.bounding_cube
        parts.append(f"--bounding-cube {a} {b} {c}")
    if args.sdf_method is not None:
        parts.append(f"--sdf-method {args.sdf_method}")
    if args.normal_offset:
        parts.append("--normal-offset")
    if args.triplet_epsilon is not None:
        parts.append(f"--triplet-epsilon {args.triplet_epsilon}")
    if args.num_sample is not None:
        parts.append(f"--num_sample {args.num_sample}")
    return " ".join(parts)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare RF dataset for DeepSDF (models + SDF, no splits)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare all RF models with SDF samples
    python rf_prepare.py \\
        --source_dir /opt/data/DeepSDF/data/RF/obj \\
        --data_dir /opt/data/DeepSDF/data

    # With surface samples + resume
    python rf_prepare.py \\
        --source_dir /opt/data/DeepSDF/data/RF/obj \\
        --data_dir /opt/data/DeepSDF/data \\
        --surface --resume

After preparation, use create_experiment.py to make train/test splits:
    python create_experiment.py \\
        --data_dir /opt/data/DeepSDF/data/RF \\
        --experiment_dir experiments/rf_exp1 \\
        --num-samples 1000
        """,
    )
    parser.add_argument(
        "--source_dir", type=str, required=True,
        help="Directory containing pre-validated .obj files (flat structure, read-only)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Output root directory (creates RF/ subdirectory here)",
    )
    parser.add_argument(
        "--surface", action="store_true",
        help="Also generate SurfaceSamples + NormalizationParameters for evaluation",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last completed stage",
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Number of parallel workers for SDF generation (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42, reserved for future use)",
    )
    parser.add_argument(
        "--uniform-ratio", type=float, default=None,
        help="Fraction of non-triplet samples for uniform random sampling "
        "(forwarded to PreprocessMesh.py). Default: 0.06.",
    )
    parser.add_argument(
        "--bounding-cube", type=float, nargs=3, default=None,
        metavar=("A", "B", "C"),
        help="Per-axis extents for uniform sampling box (forwarded to PreprocessMesh.py). "
        "Default: 1.0 1.0 1.0.",
    )
    parser.add_argument(
        "--sdf-method", type=str, default=None,
        choices=["knn", "igl", "raycast"],
        help="SDF computation backend (forwarded to PreprocessMesh.py). Default: igl.",
    )
    parser.add_argument(
        "--normal-offset", action="store_true", default=False,
        help="Offset near-surface samples along surface normal instead of isotropic "
        "(forwarded to PreprocessMesh.py). Default: off.",
    )
    parser.add_argument(
        "--triplet-epsilon", type=float, default=None,
        help="Relative offset for on-surface triplets as fraction of min AABB dim "
        "(forwarded to PreprocessMesh.py). Default: 0.02.",
    )
    parser.add_argument(
        "--num-sample", type=int, default=None,
        help="Total number of SDF samples per mesh (forwarded to PreprocessMesh.py). "
        "Default: 500000.",
    )

    args = parser.parse_args()
    setup_logging()

    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        print_error(f"Source directory not found: {source_dir}")
        return 1

    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset_dir = data_dir / "RF"
    state_path = dataset_dir / STATE_FILE

    print(f"""
{Colors.BOLD}{Colors.OKCYAN}╔══════════════════════════════════════════════════════════════════════╗
║            RF Dataset Preparation for DeepSDF                        ║
║          (models + SDF — splits via create_experiment.py)            ║
╚══════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")

    logging.info(f"Source OBJ files: {source_dir}")
    logging.info(f"Output directory: {dataset_dir}")
    logging.info(f"Surface samples:  {args.surface}")
    logging.info(f"Resume:           {args.resume}")
    print()

    # Disk space sanity check
    free_bytes = get_disk_free_bytes(data_dir)
    if free_bytes is not None:
        est_bytes_per_model = 12 * 1024 * 1024  # ~12 MB per SDF .npz
        est_total = int(1600 * est_bytes_per_model)
        if free_bytes < est_total:
            print_warning(
                f"Low disk space. Free: {bytes_to_gb(free_bytes):.1f} GB, "
                f"estimated need: ~{bytes_to_gb(est_total):.1f} GB"
            )
            print()

    # Load resume state
    state = _load_state(state_path) if args.resume else {}
    resume_stage = state.get("stage", "")
    resume_model_ids = state.get("model_ids", [])

    if args.resume and resume_stage:
        print(f"Resuming from stage: {resume_stage} ({state.get('model_count', 0)} models)")
        print()

    try:
        # --- Stage 1: Discover OBJ files ---
        if resume_stage in ("organized", "sdf", "surface", "done") and resume_model_ids:
            print_header("Stage 1: OBJ Files (skipped — using state file)")
            model_ids = resume_model_ids
            print_success(f"Using {len(model_ids)} models from previous run")
        else:
            print_header("Stage 1: Discovering OBJ Files")
            obj_files = discover_obj_files(source_dir)
            print_success(f"Found {len(obj_files)} OBJ files")

            # --- Stage 2: Organize models ---
            model_ids = organize_all_models(obj_files, dataset_dir, state_path)

        # --- Stage 3: Verify scripts ---
        verify_python_scripts()

        # --- Stage 4: Generate SDF ---
        if resume_stage in ("sdf", "surface", "done"):
            print_header("Stage 4: SDF Samples (skipped — already complete)")
            print_success(f"SDF samples for {len(model_ids)} models exist")
        else:
            sdf_extra_args = _build_sdf_preprocess_args(args)
            generate_sdf_for_all(dataset_dir, model_ids, args.threads, preprocess_args=sdf_extra_args)
            _save_state(state_path, "sdf", model_ids)

        # --- Stage 5: Surface samples (optional) ---
        if args.surface:
            if resume_stage in ("surface", "done"):
                print_header("Stage 5: Surface Samples (skipped — already complete)")
                print_success(f"Surface samples for {len(model_ids)} models exist")
            else:
                generate_surface_for_all(dataset_dir, model_ids, args.threads)
                _save_state(state_path, "surface", model_ids)
        else:
            _save_state(state_path, "done", model_ids)

        # Clean up state file on full success
        _save_state(state_path, "done", model_ids)

        print_next_steps(dataset_dir, args.surface)
        return 0

    except KeyboardInterrupt:
        print_warning(f"\nInterrupted. Resume with: --resume --source_dir {source_dir} --data_dir {data_dir}" + (" --surface" if args.surface else ""))
        return 130
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        logging.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
