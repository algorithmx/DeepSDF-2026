# Stage 2: Data Preparation
**Date:** 2026-04-29 23:37 - 23:52

## Status: COMPLETE (SDF + Experiments), Surface Samples IN PROGRESS

## Stage 2a: Dataset Preparation (Models + SDF Samples)
- Source: `/opt/data/DeepSDF/data/RF/obj` (1600 OBJ files, read-only)
- Output: `/opt/data/DeepSDF/data/RF/`
- Models organized: 1600 models in `models/<id>/mesh.obj` (0001-1600)
- SDF samples generated: 1600 `.npz` files in `SdfSamples/models/`
- SDF flags: `--anisotropic-bias --on-surface-ratio 0.15`
- Note: Had to set `CONDA_EXE=/home/vipuser/miniconda3/bin/conda` since conda isn't in PATH when calling python directly

### Spot-check Results
- `0001.npz`: pos=(578271, 4), neg=(471229, 4)
- `0800.npz`: pos=(727898, 4), neg=(321600, 4)
- `1600.npz`: pos=(588731, 4), neg=(460769, 4)

## Stage 2b: Surface Samples
- Running in background (surface generation for evaluation)
- Resumed state file from "done" to "sdf" to re-trigger surface generation

## Stage 2c: Experiments Created

### Experiment 1: rf_128_16
- Path: `/opt/data/DeepSDF/experiments/rf_128_16/`
- Files: `specs.json`, `rf_train.json`, `rf_test.json`
- Config: CodeLength=128, ScenesPerBatch=16, SamplesPerScene=16384
- Split: 1280 train + 320 test

### Experiment 2: rf_256_16
- Path: `/opt/data/DeepSDF/experiments/rf_256_16/`
- Files: `specs.json`, `rf_train.json`, `rf_test.json`
- Config: CodeLength=256, ScenesPerBatch=16, SamplesPerScene=16384
- Split: 1280 train + 320 test

## Issue Encountered
- `rf_prepare.py` couldn't find conda when run directly via `/home/vipuser/.conda/envs/ml_env/bin/python`
- Fix: Set `CONDA_EXE=/home/vipuser/miniconda3/bin/conda` environment variable
- State file had to be manually corrected from "done" to "sdf" to trigger surface sample generation
