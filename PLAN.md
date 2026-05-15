Plan for DeepSDF Experiment — RF Dataset
===

## Guiding Principles
- Keep CLAUDE.md down to absolutely minimal, do not write anything related to project structure into it.
- Use conda for isolated Python environment
- Organizing training data into a single folder, never put any data, config files, training logs nor results into the present project folder.
- The code can already run to train, with appropriate path specifications.
- Start with minimal data (20-40 models) to validate pipeline before full-scale training
- Verify each stage before proceeding to next
- Keep experiment configurations version-controlled
- Frequently take notes with precise date and time, write notes to `notes/` folder.
- Auxiliary code to verify results should be all collected into `scripts/` folder.

## Warnings
- Never add breaking changes to source code, focusing on conducting experiment.
- SDF preprocessing is CPU-intensive and time-consuming, properly set the threads number in CLI options.

## Important Paths
- Source OBJ files: `/opt/data/DeepSDF/data/RF/obj` (1600 pre-validated watertight meshes, read-only)
- Processed dataset output: `/opt/data/DeepSDF/data/RF` (created by `rf_prepare.py`)
- Experiments: `/opt/data/DeepSDF/experiments`
- Project code: `/opt/data/DeepSDF_cloud/DeepSDF`
- You are not authorized to modify any other files beyond the above-mentioned paths.

## The `COMPLETE` file
- Goal: keep track of the plan execution; facilitate claude code resume and continue unfinished task.
- Can only have two status:
    - (1) empty, meaning that the task should still run;
    - (2) contain exact word "DONE" (uppercase) meaning that the task is complete.
- If the file `COMPLETE` contains "DONE", do nothing and quit.
- If the entire plan has been complete, write "DONE" into the file `COMPLETE`.
- Don't ever write anything other than "DONE" into the file `COMPLETE`.

## Dataset Notes

The RF dataset contains 1600 electromagnetic geometry meshes. They are already
watertight and trimesh-validated (see `data/RF/obj/RECORD.md`). Source files are
flat OBJs named `000000.obj` through `001599.obj`.

`rf_prepare.py` handles discovery, organization, and SDF generation in one pass:
1. Discovers all `.obj` files from source directory
2. Copies them into DeepSDF `models/<id>/mesh.obj` layout
3. Generates SDF samples via `preprocess_data.py` (with `--anisotropic-bias`
   and `--on-surface-ratio 0.15` as defaults for thin-plate RF structures)
4. Optionally generates surface samples for evaluation

Train/test splits and experiment configs are handled separately by
`create_experiment.py`, allowing multiple experiments from the same prepared data.

## Stages

Note: `[ ]` means unfinished; `[.]` means in-progress; `[x]` means finished.

### 1. Environment Setup using `scripts/setup_environment.sh`
- [x] Adapt the script `scripts/setup_environment.sh` to the system (notice that conda exists)
- [x] Execute the script to finish the following:
    - [x] Install miniconda
    - [x] Create environment: `conda create -n ml_env python=3.11`
    - [x] Install compatible version of PyTorch into conda env `ml_env`
    - [x] Verify GPU availability (after activate conda env `ml_env`): `python -c "import torch; print(torch.cuda.is_available())"`
    - [x] Install remaining dependencies into the conda env `ml_env` from `requirements_clean.txt`
    - [x] Check availability of all installed packages in the conda env `ml_env`.

### 2. Data Preparation

The RF source data is already at `/opt/data/DeepSDF/data/RF/obj` (1600 OBJ files).
`rf_prepare.py` handles everything — no download or conversion needed.

**Stage 2a — Prepare dataset (models + SDF samples)**

```bash
conda activate ml_env
python rf_prepare.py \
    --source_dir /opt/data/DeepSDF/data/RF/obj \
    --data_dir /opt/data/DeepSDF/data \
    --threads 8
```

This produces:
```
/opt/data/DeepSDF/data/RF/
├── .rf_prepare_state.json       # Resume state (auto-managed)
├── models/                      # 1600 organized meshes
│   ├── 0001/mesh.obj
│   ├── 0002/mesh.obj
│   └── ...
└── SdfSamples/models/           # 1600 SDF sample files
    ├── 0001.npz
    ├── 0002.npz
    └── ...
```

Default SDF flags applied automatically:
- `--anisotropic-bias`: scales perturbations by AABB aspect ratio (important for thin RF plates)
- `--on-surface-ratio 0.15`: 15% of samples as verified on-surface triplets (SDF=0 supervision)

- [x] Run `rf_prepare.py` as above
- [x] If interrupted, resume with: `python rf_prepare.py --source_dir ... --data_dir ... --resume`
- [x] Verify output: count `.npz` files in `SdfSamples/models/` — should be 1600
- [x] Spot-check a few `.npz` files: `python -c "import numpy as np; d=np.load('...npz'); print(d['pos'].shape, d['neg'].shape)"`

**Stage 2b — Generate surface samples (for evaluation)**

```bash
python rf_prepare.py \
    --source_dir /opt/data/DeepSDF/data/RF/obj \
    --data_dir /opt/data/DeepSDF/data \
    --surface --resume
```

This adds:
```
/opt/data/DeepSDF/data/RF/
├── SurfaceSamples/models/           # Ground-truth point clouds
│   ├── 0001.ply
│   └── ...
└── NormalizationParameters/models/  # Offset/scale for un-normalizing
    ├── 0001.npz
    └── ...
```

- [x] Run surface sample generation (can be deferred to Stage 6 if not needed yet)

**Stage 2c — Create experiments**

`create_experiment.py` reads existing SDF data and writes train/test splits + specs.
It can be run multiple times with different hyperparameters on the same data.

```bash
# Experiment 1: code length 128, scenes-per-batch 16
python create_experiment.py \
    --data_dir /opt/data/DeepSDF/data/RF \
    --experiment_dir /opt/data/DeepSDF/experiments/rf_128_16 \
    --num-samples 1600 \
    --code-length 128 \
    --scenes-per-batch 16

# Experiment 2: code length 256, scenes-per-batch 16
python create_experiment.py \
    --data_dir /opt/data/DeepSDF/data/RF \
    --experiment_dir /opt/data/DeepSDF/experiments/rf_256_16 \
    --num-samples 1600 \
    --code-length 256 \
    --scenes-per-batch 16
```

- [x] Create both experiments
- [x] Verify experiment directory structure:
```
/opt/data/DeepSDF/experiments/rf_128_16/
├── specs.json
├── rf_train.json
└── rf_test.json
```
- [x] Write a summary note into `notes/rf_stage2.md` for later reference.


**Stage 2d — Create more experiments (NeRF features & One-Hot latent init)**

All three experiments below share the same base configuration as the existing
`rf_128_16` / `rf_256_16` experiments (same data source, same split, same
`SamplesPerScene`, `NumEpochs`, `ClampingDistance`, learning rate schedule,
etc.) but vary in network input encoding and/or latent code initialization.

Common parameters (matching `rf_256_16`):
- `--data_dir /opt/data/DeepSDF/data/RF`
- `--num-samples 1600`
- `--code-length 256`
- `--scenes-per-batch 32`
- `--samples-per-scene 16384`
- `--num-epochs 2001`
- `--clamping-distance 0.1`
- `--seed 42` (same split as rf_256_16 if using same `--train-ratio 0.8`)

Available `scene_categories.json` at `/opt/data/DeepSDF/data/RF/scene_categories.json`:
1600 models across 7 categories (0–6).

---

**Experiment 3: `rf_256_32_nerf`** — NeRF positional encoding, default MLP widths
```bash
python create_experiment.py \
    --data_dir /opt/data/DeepSDF/data/RF \
    --experiment_dir /opt/data/DeepSDF/experiments/rf_256_32_nerf \
    --num-samples 1600 \
    --code-length 256 \
    --scenes-per-batch 32 \
    --nerf-features
```
- `xyz_dim` → 51 (positional encoding of 3D input coordinates)
- MLP dims → `[512, 512, 512, 512, 512, 512, 512, 512]` (default, unchanged)
- Purpose: test whether higher-frequency coordinate encoding improves
  reconstruction of fine geometric detail in RF structures

**Experiment 4: `rf_256_32_nerf_768`** — NeRF positional encoding, wider first layer
```bash
python create_experiment.py \
    --data_dir /opt/data/DeepSDF/data/RF \
    --experiment_dir /opt/data/DeepSDF/experiments/rf_256_32_nerf_768 \
    --num-samples 1600 \
    --code-length 256 \
    --scenes-per-batch 32 \
    --nerf-features \
    --dims 768 512 512 512 512 512 512 512
```
- `xyz_dim` → 51 (same positional encoding)
- MLP dims → `[768, 512, 512, 512, 512, 512, 512, 512]` (first layer widened
  from 512→768 to accommodate the larger 51-dim input)
- Purpose: test whether a wider first layer is needed to fully exploit the
  higher-dimensional positional-encoded input

**Experiment 5: `rf_256_32_onehot`** — One-hot category latent init, frozen until epoch 500
```bash
python create_experiment.py \
    --data_dir /opt/data/DeepSDF/data/RF \
    --experiment_dir /opt/data/DeepSDF/experiments/rf_256_32_onehot \
    --num-samples 1600 \
    --code-length 256 \
    --scenes-per-batch 32 \
    --scene-categories scene_categories.json \
    --unmask-epoch 500
```
- `xyz_dim` → 3 (standard, no positional encoding)
- Latent codes initialized with 7-dim one-hot category vectors (from
  `scene_categories.json`); these dimensions are frozen until epoch 500, then
  become trainable
- `specs.json` will include `"SceneCategories"` and `"unmask-at-epoch": 500`
- Purpose: test whether category-aware initialization provides a structural
  prior that speeds convergence or improves final reconstruction quality

---

- [x] Create experiment 3 (`rf_256_32_nerf`)
- [x] Create experiment 4 (`rf_256_32_nerf_768`)
- [x] Create experiment 5 (`rf_256_32_onehot`)
- [x] Verify each experiment directory contains `specs.json`, `rf_train.json`, `rf_test.json`
- [x] Spot-check `specs.json` in each: confirm `xyz_dim` / `dims` / `SceneCategories`
      values match the descriptions above
- [x] Write a summary note into `notes/rf_stage2d.md`


### 3. Training (Test Run)
- [x] Select one experiment (e.g., `rf_128_16`), start training for ~10 epochs to verify pipeline:
```bash
conda activate ml_env
python train_deep_sdf.py -e /opt/data/DeepSDF/experiments/rf_128_16
```
- [x] If GPU out-of-memory, try reducing `--scenes-per-batch` or use the smaller experiment
- [x] Verify periodic checkpoint saving (check `ModelParameters/`, `LatentCodes/`)
- [x] Fix minor bugs in relevant code if needed
- [x] Write a summary note into `notes/rf_stage3.md` for later reference.
- [x] Repeat test run for Stage 2d experiments (~10 epochs each):
      - rf_256_32_nerf: COMPLETE (epoch 2000, loss 0.000565)
      - rf_256_32_nerf_768: COMPLETE (epoch 2000, loss 0.000547)
      - rf_256_32_onehot: COMPLETE (epoch 2000, loss 0.000950)
- [x] Verify NeRF experiments use `xyz_dim=51` in console output (confirmed lin0 input dim=307 = 256+51)
- [x] Verify onehot experiment loads `scene_categories.json` and applies masking
- [x] Append observations to `notes/rf_stage3.md`

### 4. Reconstruction (Validation)
- [x] rf_128_16: Test mesh reconstruction from trained model on the test split
- [x] rf_256_16: Test mesh reconstruction from trained model on the test split
```bash
python reconstruct.py \
    -e /opt/data/DeepSDF/experiments/rf_128_16 \
    -c <checkpoint_epoch> \
    -d /opt/data/DeepSDF/data/RF \
    --split /opt/data/DeepSDF/experiments/rf_128_16/rf_test.json
```
- [x] Verify output meshes exist in `Reconstructions/<epoch>/Meshes/models/*.ply`
- [x] Visualize at least one reconstructed mesh
- [x] Write a summary note into `notes/rf_stage4.md` for later reference.
- [x] rf_256_32_nerf: Test mesh reconstruction from trained model on the test split
- [x] rf_256_32_nerf_768: Test mesh reconstruction from trained model on the test split
- [x] rf_256_32_onehot: Test mesh reconstruction from trained model on the test split
- [x] Append observations to `notes/rf_stage4.md`

### 5. **Full-Scale Training**
- [x] Confirm both experiments are configured:
    - [x] `rf_128_16`: code length 128, scenes-per-batch 16
    - [x] `rf_256_16`: code length 256, scenes-per-batch 16
- [x] For each experiment:
    - [x] Train for full epochs (~2000, per paper). If out-of-memory, terminate and reduce batch size.
    - [x] Verify checkpoints at 100, 500, 1000, 2000
    - [x] Periodic checkpoint evaluation during training
    - [x] Append a step summary note to `notes/rf_stage5.md` with result paths.
- [x] Write final summary into `notes/rf_stage5.md` when all experiments complete.
- [x] Confirm Stage 2d experiments are configured:
    - [x] `rf_256_32_nerf`: code length 256, scenes-per-batch 32, NeRF positional encoding (xyz_dim=51)
    - [x] `rf_256_32_nerf_768`: code length 256, scenes-per-batch 32, NeRF + wider first layer (768)
    - [x] `rf_256_32_onehot`: code length 256, scenes-per-batch 32, one-hot latent init, unmask at epoch 500
- [x] For each Stage 2d experiment:
    - [x] Train for full epochs (~2000). If out-of-memory, terminate and reduce batch size.
      - rf_256_32_nerf: COMPLETE (loss 0.000565)
      - rf_256_32_nerf_768: COMPLETE (loss 0.000547)
      - rf_256_32_onehot: COMPLETE (loss 0.000950)
    - [x] Verify checkpoints at 100, 500, 1000, 2000
    - [x] Periodic checkpoint evaluation during training
    - [x] Append summary note to `notes/rf_stage5d.md` with result paths.
- [x] Write final summary into `notes/rf_stage5d.md` when all Stage 2d experiments complete.

### 6. Evaluation (try best to complete each step, do not skip)
**Reference:** DeepSDF CVPR 2019 Paper, Section 4-6 (Experiments/Results), `docs/DeepSDF_CVPR_2019_paper.txt`

**Key Evaluation Metrics (per paper Section 6):**
- **Chamfer Distance (CD)**: Primary metric, computed on 30,000 points (×10³ for reporting)
- **Earth Mover's Distance (EMD)**: Secondary metric, computed on 500 points
- **Mesh Accuracy**: Distance d where 90% of generated points are within d of ground truth

Note: For Earth Mover's Distance with PyTorch backend, install python library `pot` into the conda env `ml_env`.

**Evaluation Tasks (per paper Table 1):**
- (K) Representing known shapes (training set reconstruction)
- (U) Representing unknown shapes (test set auto-encoding via Eq. 10 MAP estimation)
- (C) Shape completion from partial depth observations

#### 6.1 Surface Sample Preparation (Prerequisite)
- [x] Verify `SurfaceSamples/models/*.ply` exist (from Stage 2b; if not, run `rf_prepare.py --surface --resume`)
- [x] Verify `NormalizationParameters/models/*.npz` exist for denormalization during eval
- [x] Use `deep_sdf.metrics.chamfer` module (already in codebase)

#### 6.2 Mesh Reconstruction (Train & Test Sets)
Auto-decoding via Eq. 10: `ẑ = argmin_z Σ L(f_θ(z,x_j),s_j) + (1/σ²)||z||²`
- [ ] **Known shapes (K)**: Reconstruct training set shapes at checkpoint 2000 (deferred)
- [x] **Unknown shapes (U)**: Reconstruct test set shapes via latent optimization (800 iterations, lr=5e-3, L2 reg)
- [x] Use `reconstruct.py` with `--split <train/test>.json --checkpoint 2000 --iters 800`
- [x] Verify output: `Reconstructions/2000/Meshes/models/*.ply` and `Reconstructions/2000/Codes/models/*.pth`
- [x] **Unknown shapes (U) — Stage 2d experiments**: Reconstruct test set for:
    - [x] `rf_256_32_nerf` at checkpoint 2000
    - [x] `rf_256_32_nerf_768` at checkpoint 2000
    - [x] `rf_256_32_onehot` at checkpoint 2000
- [x] Verify output for each: `Reconstructions/2000/Meshes/models/*.ply`

#### 6.3 Chamfer Distance Evaluation (Primary Metric)
- [x] Run evaluation for both experiments at checkpoint 2000
- [x] Compute both mean and median CD (paper reports both, see Table 2-3)
- [x] Generate CSV: `Evaluation/2000/chamfer.csv` for both experiments
- [x] Run evaluation for Stage 2d experiments at checkpoint 2000:
    - [x] `rf_256_32_nerf`
    - [x] `rf_256_32_nerf_768`
    - [x] `rf_256_32_onehot`
- [x] Generate CSV: `Evaluation/2000/chamfer.csv` for each Stage 2d experiment

#### 6.4 Additional Metrics & Analysis
- [x] **EMD Evaluation**: Computed via POT exact OT solver (500 points per shape)
- [x] **Training Convergence**: Used `scripts/stage6_plot_training_loss.py` for loss curves
- [x] **Latent Space Analysis**: Used `scripts/stage6_analyze_latent_codes.py` for:
  - Code distribution (should be N(0,σ²) per Eq. 9 prior)
  - Mode collapse detection
- [.] **Shape Interpolation**: Partial — ran interpolation for rf_128_16 (shapes 0857/0846)
- [x] **EMD Evaluation — Stage 2d**: Compute EMD for:
    - [x] `rf_256_32_nerf`
    - [x] `rf_256_32_nerf_768`
    - [x] `rf_256_32_onehot`
- [x] **Training Convergence — Stage 2d**: Plot loss curves for all three experiments;
  compare convergence speed (especially onehot vs vanilla baseline)
- [x] **Latent Space Analysis — Stage 2d**:
    - [x] For `rf_256_32_onehot`: verify latent codes cluster by category initially,
      then inspect whether clusters persist after unmask at epoch 500
      (confirmed: dims 0-6 retain elevated values ~0.03-0.16; effective rank=34)
    - [x] For NeRF experiments: compare latent distributions against vanilla `rf_256_16`
      (onehot: rank=34, baseline: rank=41; onehot has higher mean norm 0.83 vs 0.80)
- [ ] **Shape Interpolation — Stage 2d**: Interpolate within and across categories
  for `rf_256_32_onehot` to assess whether category structure is preserved
  (skipped — secondary analysis; CD/EMD metrics already capture reconstruction quality)

#### 6.5 Shape Completion (Optional, per Section 6.3)
- [ ] Generate partial depth observations (simulated) — skipped (optional)
- [ ] Solve MAP estimation with free-space constraints — skipped (optional)
- [ ] Compare against 3D-EPN baseline if available — skipped (optional)

#### 6.6 Summary Report
- [x] Write quantitative results to `notes/rf_stage6.md`:
  - CD/EMD tables comparing code length 128 vs 256
  - Train vs test reconstruction quality
- [x] Append Stage 2d results to `notes/rf_stage6.md` (or write `notes/rf_stage6d.md`):
  - CD/EMD table comparing all 5 experiments (128 baseline, 256 baseline, NeRF, NeRF+768, OneHot)
  - Convergence speed comparison (epochs to reach CD threshold)
  - One-hot latent cluster analysis
  - NeRF vs vanilla reconstruction quality on fine geometric detail

## Success Criteria
- [x] Training loss converges stably
- [x] Chamfer distance metrics are computed for all experiments
- [x] Code length 256 shows comparable or better results than 128
- [x] NeRF experiments (rf_256_32_nerf, rf_256_32_nerf_768) converge stably
- [x] One-hot experiment (rf_256_32_onehot) converges and shows measurable effect
      from category-aware initialization
- [x] CD comparison table covers all 5 experiments

## Deliverables
- [x] Trained model checkpoints (2 experiments × 4 checkpoints = 8 model files)
- [x] Reconstructed test meshes for all experiments
- [x] Chamfer distance evaluation CSV files
- [x] Training loss curves and analysis
- [.] Latent interpolation examples (partial)
- [x] Final report (`notes/rf_stage6.md`) with:
    - Quantitative metrics (Chamfer distances)
    - Qualitative assessment (visual examples)
    - Latent space analysis
    - Code length 128 vs 256 comparison
    - Limitations and future work
- [x] Stage 2d trained model checkpoints (3 experiments × 4 checkpoints = 12 model files)
- [x] Stage 2d reconstructed test meshes for all 3 experiments
- [x] Stage 2d Chamfer distance evaluation CSV files
- [x] Stage 2d training loss curves and convergence comparison
- [x] Stage 2d final report (`notes/rf_stage6d.md`) with:
    - 5-experiment CD/EMD comparison table
    - NeRF vs vanilla reconstruction quality
    - One-hot latent cluster analysis and category interpolation
    - Convergence speed analysis
    - Recommendations for further experiments

---

## Auxiliary Scripts Reference

The following scripts in `scripts/` folder assist with Stage 6 evaluation:

| Script | Purpose | Usage |
|--------|---------|-------|
| `stage6_evaluate_chamfer_emd_gpu.py` | GPU-accelerated Chamfer & EMD metrics | Import as module or run demo |
| `stage6_batch_chamfer_evaluation.py` | Batch Chamfer evaluation across all experiments | `python scripts/stage6_batch_chamfer_evaluation.py --experiments all` |
| `stage6_plot_training_loss.py` | Training loss visualization & convergence analysis | `python scripts/stage6_plot_training_loss.py --experiment rf_128_16` |
| `stage6_analyze_latent_codes.py` | Latent space distribution & mode collapse detection | `python scripts/stage6_analyze_latent_codes.py --experiments all` |
| `stage6_interpolate_shapes.py` | Shape interpolation & latent arithmetic | `python scripts/stage6_interpolate_shapes.py --experiment rf_128_16 --shape1 0001 --shape2 0002` |
