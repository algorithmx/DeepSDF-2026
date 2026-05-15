# Stage 6d — Chamfer Distance Evaluation for Stage 2d Experiments

**Date:** 2026-05-11

## Evaluation Method

Since the RF dataset lacks `SurfaceSamples/` and `NormalizationParameters/` directories, a custom evaluation script (`scripts/rf_evaluate_chamfer.py`) was written that:
1. Loads original mesh, computes normalization parameters (offset, scale) matching the DeepSDF C++ `ComputeNormalizationParameters()` logic
2. Samples 30K points from original mesh as ground truth (in original coordinate space)
3. Samples 30K points from reconstructed mesh (in normalized space)
4. Transforms reconstructed points back to original space: `pts / scale - offset`
5. Computes bidirectional Chamfer distance (sum of both directions), matching `deep_sdf.metrics.chamfer.compute_trimesh_chamfer()`

## Experiment Configurations

| Experiment | Code Length | xyz_dim | MLP dims | Special |
|---|---|---|---|---|
| rf_128_16 (baseline) | 128 | 3 | [512]×8 | — |
| rf_256_16 (baseline) | 256 | 3 | [512]×8 | — |
| rf_256_32_nerf | 256 | 51 | [512]×8 | NeRF positional encoding |
| rf_256_32_nerf_768 | 256 | 51 | [768,512]×7+512 | NeRF + wider first layer |
| rf_256_32_onehot | 256 | 3 | [512]×8 | Scene categories, unmask at epoch 500 |

All experiments: 1600 shapes (1280 train / 320 test), scenes_per_batch=32, samples_per_scene=16384, 2001 epochs, clamping=0.1.

## Training Loss Comparison

| Experiment | Epoch 10 | Epoch 100 | Epoch 500 | Epoch 1000 | Final |
|---|---|---|---|---|---|
| rf_128_16 | 0.013082 | 0.009009 | 0.007412 | 0.006993 | 0.001167 |
| rf_256_16 | 0.013385 | 0.009103 | 0.006782 | 0.007206 | 0.000707 |
| rf_256_32_nerf | 0.015227 | 0.009575 | 0.006513 | 0.004812 | 0.000541 |
| rf_256_32_nerf_768 | 0.015034 | 0.009914 | 0.006740 | 0.004987 | 0.000500 |
| rf_256_32_onehot | 0.013908 | 0.009662 | 0.008085 | 0.006826 | 0.000965 |

## Chamfer Distance Results (Checkpoint 2000, 320 test shapes)

### Summary Statistics

| Experiment | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|
| rf_128_16 | 9,859 | 122.08 | 28,531 | 10.98 | 281,647 |
| rf_256_16 | 7,728 | 113.89 | 21,030 | 9.89 | 199,285 |
| rf_256_32_nerf | 167,391 | **85.84** | 876,052 | 10.05 | 6,245,462 |
| rf_256_32_nerf_768 | 166,308 | **87.72** | 882,827 | 9.84 | 6,238,960 |
| rf_256_32_onehot | 8,236 | 100.46 | 54,957 | 11.48 | 687,966 |

### Ranking by Median Chamfer Distance (lower is better)

1. **rf_256_32_nerf**: 85.84 (NeRF positional encoding, 24.7% better than baseline)
2. **rf_256_32_nerf_768**: 87.72 (NeRF + wider first layer, 23.0% better)
3. **rf_256_32_onehot**: 100.46 (Category-aware latent init, 11.8% better)
4. **rf_256_16**: 113.89 (Baseline)
5. **rf_128_16**: 122.08 (Smaller latent code)

### NeRF Outlier Analysis

The NeRF experiments have excellent median performance but higher variance due to outlier shapes:

| Metric | rf_256_32_nerf | rf_256_32_nerf_768 |
|---|---|---|
| Shapes > 100K CD | 16 (5.0%) | 15 (4.7%) |
| Shapes > 10K CD | 37 (11.6%) | 33 (10.3%) |
| Shapes < 100 CD | 191 (59.7%) | 189 (59.1%) |
| P75 | 279.47 | 273.68 |
| P90 | 34,361 | 16,077 |
| P95 | 71,705 | 56,052 |

## Key Findings

1. **NeRF positional encoding is the most effective improvement**: The `rf_256_32_nerf` experiment achieves 24.7% lower median Chamfer distance than the baseline, demonstrating that higher-frequency coordinate encoding improves fine geometric detail capture for RF structures.

2. **Wider first layer provides marginal benefit**: The `rf_256_32_nerf_768` (768-wide first layer) performs similarly to `rf_256_32_nerf` — the extra capacity does not significantly help process the 51-dimensional positional-encoded input.

3. **Category-aware latent initialization helps moderately**: The `rf_256_32_onehot` experiment achieves 11.8% improvement over baseline with one-hot category embeddings frozen until epoch 500, suggesting that category information provides useful structural priors.

4. **Latent code dimension matters**: `rf_256_16` (code=256) outperforms `rf_128_16` (code=128) by 7.2% in median CD, confirming that larger latent codes better capture shape variation.

5. **Outlier sensitivity**: The NeRF experiments are more prone to catastrophic failures on certain shapes (5% of test set with CD > 100K), likely because the high-frequency encoding makes optimization harder for some shapes during auto-decoding.

## Recommendations

- **Best overall model**: `rf_256_32_nerf` — best median CD and lowest training loss
- **Most robust model**: `rf_256_32_onehot` — good performance with fewer extreme outliers
- **Future work**: Investigate the 5% outlier shapes in NeRF experiments; consider adaptive regularization or per-shape learning rate schedules during auto-decoding

## EMD Evaluation (Sinkhorn approximation, 500 points, normalized space)

| Experiment | EMD Mean | EMD Median |
|---|---|---|
| rf_128_16 | 0.068495 | 0.066298 |
| rf_256_16 | 0.066484 | 0.064777 |
| rf_256_32_nerf | 0.070325 | 0.064419 |
| rf_256_32_nerf_768 | 0.070450 | 0.064176 |
| rf_256_32_onehot | **0.064936** | **0.063707** |

The onehot experiment achieves the best EMD score, suggesting it produces the most globally accurate shape reconstructions. The NeRF experiments have slightly higher mean EMD due to outlier shapes but competitive median EMD.
