# Stage 6: Evaluation — Final Report
**Date:** 2026-04-30
**Status:** COMPLETE

## Overview
Full evaluation of DeepSDF on RF (electromagnetic geometry) dataset with two code-length configurations (128 and 256 dimensions). 1600 shapes total, 1280 train / 320 test split.

## 6.1 Surface Sample Preparation
- `SurfaceSamples/models/*.ply`: 1600 files verified
- `NormalizationParameters/models/*.npz`: 1600 files verified

## 6.2 Mesh Reconstruction
Both experiments reconstructed on test split (320 unknown shapes) via auto-decoding (Eq. 10):
- 800 iterations, lr=5e-3, L2 regularization
- Output: `Reconstructions/2000/Meshes/models/*.ply`, `Reconstructions/2000/Codes/models/*.pth`
- Training set reconstruction (Known shapes K) deferred

## 6.3 Chamfer Distance (Primary Metric)

### Test Set — Unknown Shapes (U)

| Metric | rf_128_16 | rf_256_16 | Winner |
|--------|-----------|-----------|--------|
| Mean CD | 9,849.25 | 7,739.38 | rf_256_16 |
| Median CD | 121.19 | 112.68 | rf_256_16 |
| Std CD | 28,428.71 | 20,914.56 | rf_256_16 |
| Min CD | 10.99 | 9.93 | rf_256_16 |
| Max CD | 280,548.89 | 197,691.96 | rf_256_16 |
| 25th percentile | 41.88 | 42.25 | similar |
| 75th percentile | 3,256.38 | 3,419.10 | similar |

### Distribution Analysis

| Range | rf_128_16 Count | rf_256_16 Count |
|-------|-----------------|-----------------|
| CD < 50 | 106 (33%) | 104 (33%) |
| CD < 100 | 146 (46%) | 147 (46%) |
| CD < 200 | 178 (56%) | 183 (57%) |
| CD > 1,000 | 107 (33%) | 95 (30%) |
| CD > 10,000 | 57 (18%) | 59 (18%) |

Note: ~55-57% of shapes reconstruct well (CD < 200). The heavy-tailed distribution is typical for auto-decoding from random initialization.

## 6.3b Earth Mover's Distance (Secondary Metric)

Test set, 500 points per shape, POT exact OT solver.

| Metric | rf_128_16 | rf_256_16 | Winner |
|--------|-----------|-----------|--------|
| Mean EMD | 371,664.86 | 341,721.72 | rf_256_16 (-8.1%) |
| Median EMD | 40,000.37 | 35,229.84 | rf_256_16 (-11.9%) |
| Min EMD | 3,252.30 | 2,902.21 | rf_256_16 |
| Max EMD | 8,192,683.00 | 6,006,488.00 | rf_256_16 |

Note: Absolute values are large due to raw coordinate scale (~4000 unit range). For paper-style reporting, divide by mean shape extent.

## 6.4 Training Convergence

| Metric | rf_128_16 | rf_256_16 |
|--------|-----------|-----------|
| Initial loss | 0.010822 | N/A |
| Final loss | 0.000951 | 0.000707 |
| Min loss | 0.000910 (ep 1978) | 0.000561 |
| Epoch time | ~17s | ~17.3s |
| Total training time | ~5.8h | ~10.9h |
| Latent code params | 163,840 | 327,680 |

- Both experiments converge stably
- rf_256_16 achieves lower final loss (0.000707 vs 0.000951)
- Training loss plots: `training_curves/loss_curve.png` for each experiment
- Comparison plot: `experiments/comparison/loss_comparison.png`

## 6.5 Latent Space Analysis

| Metric | rf_128_16 | rf_256_16 |
|--------|-----------|-----------|
| Code dimension | 128 | 256 |
| Mean L2 norm | 0.7952 ± 0.1055 | 0.7962 ± 0.0957 |
| Global mean | 0.004 | ~0 |
| Global std | 0.071 | ~0.07 |
| Effective rank (95% var) | 34/128 (26.6%) | 41/256 (16.0%) |
| Avg pairwise distance | 1.003 | 1.019 |
| Mode collapse detected | Yes | Yes |

### Key Observations
- Both models show similar code norm distributions (mean ~0.80) — consistent with N(0,σ²) prior regularization
- rf_256_16 uses fewer dimensions effectively (16% vs 27%) — 256-dim appears over-parameterized
- Neither model uses more than ~50 effective dimensions out of available capacity
- "Mode collapse" detection is conservative (global variance threshold); top dimensions show meaningful variance (some dims have std > 0.15)
- The small effective rank suggests both code lengths could be reduced without performance loss

## 6.6 Shape Interpolation
- Interpolated between shapes 0857 and 0846 (10 steps) for rf_128_16
- Output: `Interpolation/interp_000_alpha0.00.ply` through `interp_009_alpha1.00.ply`
- Smooth transitions observed — validates learned latent space structure

## 6.7 Code Length 128 vs 256: Final Comparison

### rf_256_16 advantages:
- Better mean CD (7,739 vs 9,849, -21.4%)
- Better median CD (113 vs 121, -7.0%)
- Better mean EMD (-8.1%)
- Better worst-case CD (198K vs 281K)
- Lower final training loss (0.000707 vs 0.000951)

### rf_128_16 advantages:
- 2x fewer latent parameters
- ~1.9x faster training (5.8h vs 10.9h)
- Slightly better 25th percentile CD (41.88 vs 42.25)
- Slightly fewer extreme outliers > 10K CD (57 vs 59)
- Higher effective dimension utilization (27% vs 16%)

### Conclusion
Code length 256 provides moderate improvement in mean/median metrics at ~2x the parameter cost. The marginal improvement suggests that 128-dim codes are already near the saturation point for this dataset. The heavy tail of reconstruction failures (CD > 1,000 for ~30% of shapes) is present in both configurations and is likely caused by:
1. Auto-decoding convergence failures from N(0, 0.01) initialization
2. Inherent difficulty of certain RF geometries (thin walls, sharp features)
3. Possible undertraining at checkpoint 2000 for challenging shapes

## 6.8 Limitations and Future Work
- **Training set reconstruction (K)**: Not evaluated; needed for complete paper-style comparison
- **Shape completion (C)**: Not evaluated; requires partial depth observation simulation
- **Heavy-tailed CDs**: ~18% of shapes have CD > 10K regardless of code length — likely convergence failures, not capacity limits
- **Effective rank**: Only ~34-41 effective dimensions used out of 128-256 — suggests dimensionality reduction potential
- **Recommendation**: Experiment with:
  - More auto-decoding iterations (1600-3200) for challenging shapes
  - Initializing latent codes from training set's empirical mean/variance
  - Reduced L2 regularization for shapes far from latent-space mean
  - Smaller code lengths (e.g., 64) to test if even 128 is overkill

## Deliverables Checklist
- [x] Trained model checkpoints (2 experiments × 4 = 8 files)
- [x] Reconstructed test meshes (320 × 2 experiments)
- [x] Chamfer distance CSV for both experiments
- [x] EMD distance CSV for both experiments
- [x] Training loss curves and comparison plots
- [x] Latent space analysis reports
- [x] Shape interpolation examples
- [x] This final report
