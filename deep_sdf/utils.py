#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""
DeepSDF Utilities Module.

This module provides common utilities for the DeepSDF project, including:
- Conda environment resolution (single source of truth)
- Python command builders for ml_env environment
- Logging configuration
- SDF decoding helpers

Conda Environment Resolution:
-----------------------------
This module is the SINGLE SOURCE OF TRUTH for resolving the ml_env conda
environment and building Python commands. All scripts should use:

    from deep_sdf.utils import build_ml_env_python_cmd, resolve_ml_env_python

Functions:
    - resolve_conda_exe(): Find conda executable (CONDA_EXE env var or PATH)
    - resolve_ml_env_python(): Get path to ml_env Python interpreter (crashes if not found)
    - build_ml_env_python_cmd(): Build command list for conda run (crashes if not found)

Constants:
    - ML_ENV_NAME: Name of the conda environment ("ml_env")

Note: These functions crash hard with RuntimeError if conda or ml_env is not found.
There is no fallback to sys.executable - the pipeline requires a proper conda setup.
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import List, Optional

import math

import torch

# Default conda environment name for the project
ML_ENV_NAME = "ml_env"

# Module-level feature-engineering dimension.
# Set via set_xyz_dim() when an experiment is loaded so that feat_eng
# and decode_sdf behave consistently with the trained decoder.
_XYZ_DIM = 3


def set_xyz_dim(dim):
    """Set the spatial feature dimension expected by the decoder.

    Must be called after loading experiment specs and before any forward
    passes that use feat_eng() or decode_sdf().
    """
    global _XYZ_DIM
    _XYZ_DIM = dim


def get_xyz_dim():
    return _XYZ_DIM


def resolve_conda_exe() -> Optional[str]:
    """Find the conda executable.
    
    Checks in order:
    1. CONDA_EXE environment variable
    2. 'conda' executable in PATH
    
    Returns:
        Path to conda executable or None if not found.
    """
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe is None or not os.path.isfile(conda_exe):
        conda_exe = shutil.which("conda")
    return conda_exe if conda_exe and os.path.isfile(conda_exe) else None


def resolve_ml_env_python() -> str:
    """Resolve the Python interpreter for the ml_env conda environment.
    
    Returns the path to the Python executable in the ml_env environment.
    Crashes hard with RuntimeError if conda is not found or ml_env doesn't exist.
    
    Returns:
        Path to Python executable in ml_env.
        
    Raises:
        RuntimeError: If conda is not found or ml_env environment doesn't exist.
    """
    conda_exe = resolve_conda_exe()
    if conda_exe is None:
        raise RuntimeError(
            f"Conda executable not found. This pipeline requires conda env '{ML_ENV_NAME}'. "
            "Please ensure conda is installed and available in PATH or set CONDA_EXE environment variable."
        )
    
    try:
        result = subprocess.run(
            [conda_exe, "info", "--base"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            conda_base = result.stdout.strip()
            # Construct path to ml_env Python
            if os.name == "nt":  # Windows
                ml_env_python = os.path.join(conda_base, "envs", ML_ENV_NAME, "python.exe")
                # Also check ~/.conda/envs/ as fallback (common on some systems)
                fallback_python = os.path.expanduser(os.path.join("~", ".conda", "envs", ML_ENV_NAME, "python.exe"))
            else:
                ml_env_python = os.path.join(conda_base, "envs", ML_ENV_NAME, "bin", "python")
                # Also check ~/.conda/envs/ as fallback (common on some systems)
                fallback_python = os.path.expanduser(os.path.join("~", ".conda", "envs", ML_ENV_NAME, "bin", "python"))

            if os.path.isfile(ml_env_python):
                return ml_env_python
            elif os.path.isfile(fallback_python):
                return fallback_python
            else:
                raise RuntimeError(
                    f"Python interpreter not found in conda env '{ML_ENV_NAME}': {ml_env_python}\n"
                    f"Also checked fallback: {fallback_python}\n"
                    f"Please ensure the '{ML_ENV_NAME}' environment is created."
                )
        else:
            raise RuntimeError(
                f"Failed to get conda base directory. Exit code: {result.returncode}"
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(
            f"Failed to resolve conda environment '{ML_ENV_NAME}': {e}"
        ) from e


def build_ml_env_python_cmd(script_path: str, *script_args: str) -> List[str]:
    """Build a Python command that runs in the ml_env conda environment.
    
    Uses 'conda run -n ml_env python' to ensure the script runs with the
    correct environment. This is more reliable than hardcoded paths.
    
    The '--no-capture-output' flag ensures terminal output is passed through
    correctly (no buffering), which is important for seeing progress and logs.
    
    Args:
        script_path: Path to the Python script to run.
        *script_args: Additional arguments to pass to the script.
        
    Returns:
        Command list suitable for subprocess.
        
    Raises:
        RuntimeError: If conda executable is not found.
    """
    conda_exe = resolve_conda_exe()
    if conda_exe is None:
        raise RuntimeError(
            f"Conda executable not found in PATH. This pipeline requires conda env '{ML_ENV_NAME}'."
        )
    
    return [
        conda_exe, "run", "--no-capture-output", "-n", ML_ENV_NAME, "python", str(script_path), *script_args
    ]


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def feat_eng(xyz):
    """Feature engineering for 3D coordinates.

    Default behavior (xyz_dim == 3) returns the input unchanged.
    When xyz_dim > 3 (e.g. 51 for NeRF-style positional encoding),
    the function maps (x,y,z) -> (x,y,z, ext(x,8), ext(y,8), ext(z,8))
    where ext(v,8) = [sin(pi * v * 2^L), cos(pi * v * 2^L)] for L = 0..7.

    The active dimension is controlled by set_xyz_dim() and must match the
    decoder's xyz_dim argument.

    Args:
        xyz: (N, 3) tensor of 3D coordinates.

    Returns:
        (N, _XYZ_DIM) tensor of engineered features.
    """
    global _XYZ_DIM
    if _XYZ_DIM == 3:
        return xyz

    # NeRF-style positional encoding for xyz_dim == 51
    # frequencies: 2^0, 2^1, ..., 2^7  -> shape (8,)
    freq = 2.0 ** torch.arange(8, device=xyz.device, dtype=xyz.dtype)

    # (N, 3, 1) * (8,) -> (N, 3, 8)
    args = xyz.unsqueeze(-1) * freq * math.pi

    # sin and cos: each (N, 3, 8)
    encoded = torch.stack([torch.sin(args), torch.cos(args)], dim=-1)
    # (N, 3, 8, 2) -> (N, 48)
    encoded = encoded.view(xyz.shape[0], -1)

    # concatenate original coordinates: (N, 3) + (N, 48) -> (N, 51)
    return torch.cat([xyz, encoded], dim=-1)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    queries = feat_eng(queries)

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf
