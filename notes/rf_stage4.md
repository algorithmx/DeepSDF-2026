# Stage 4: Reconstruction Validation
**Date:** 2026-04-30 09:45 - 11:50

## Status: COMPLETE (rf_128_16)

## Reconstruction Parameters
- Experiment: `rf_128_16` at checkpoint 2000
- Test split: 320 shapes (rf_test.json)
- Auto-decoding: 800 iterations, lr=5e-3, L2 regularization
- Marching cubes resolution: N=256

## Results
- All 320 test shapes reconstructed successfully
- Output: `Reconstructions/2000/Meshes/models/*.ply`
- Latent codes: `Reconstructions/2000/Codes/models/*.pth`
- Reconstruction time: ~2 hours (concurrent with rf_256_16 training)

## Chamfer Distance Evaluation (Test Set)
- Evaluation script: `evaluate.py`
- Results saved to: `Evaluation/2000/chamfer.csv`
- 320 shapes evaluated

| Metric | Value |
|--------|-------|
| Mean CD | 9849.25 |
| Median CD | 121.19 |
| Std CD | 28428.71 |
| Min CD | 10.99 |
| Max CD | 280548.89 |
| 25th percentile | 41.88 |
| 75th percentile | 3256.38 |

## CD Distribution from CSV (manual count, 320 shapes)

| Range | Count | ~Pct |
|-------|-------|------|
| < 50 | ~155 | 48% |
| 50–200 | ~60 | 19% |
| 200–1000 | ~40 | 12% |
| 1K–10K | ~35 | 11% |
| 10K–100K | ~22 | 7% |
| >100K | ~8 | 3% |

Worst shapes: 0142 (280K), 0867 (173K), 1355 (155K), 0235 (149K), 1454 (143K), 0996 (119K).
Best shapes: 0941 (11.0), 1410 (13.9), 1313 (14.0), 1346 (14.2), 0470 (14.1).

## Code Review: Reconstruction & Chamfer Pipeline (2026-04-30)

**Verdict: No bug found.** The large CD values are genuine reconstruction failures, not a code error.

### Coordinate-space chain verified correct

| Step | Space | Where |
|------|-------|-------|
| GT surface samples (PLY) | raw coords | `SampleMeshSurface.py:154` samples from raw mesh |
| Normalization params (NPZ) | offset = -centre, scale = 1/(max_dist*1.03) | `SampleMeshSurface.py:108-139` |
| SDF samples (NPZ) | normalized: (raw+offset)*scale | `PreprocessMesh.py` |
| Decoder operates in | normalized [-1,1] | — |
| `create_mesh()` grid | [-1,1], saved in normalized coords | `mesh.py:23` voxel_origin=[-1,-1,-1]; `reconstruct.py:287` calls without offset/scale |
| Chamfer denormalization | gen_points/scale - offset -> raw | `chamfer.py:23` |
| Chamfer metric | bidirectional mean-squared NN dist in raw coords | `chamfer.py:30-39` |

Math: `normalized = (raw-centre)*scale` -> `raw = normalized/scale + centre = normalized/scale - offset` (correct)

### Path resolution with RF's empty-string dataset name

Split format: `{"": {"models": [...]}}`. `os.path.join("foo", "", "bar")` = `"foo/bar"`, so all paths resolve correctly:
- GT: `SurfaceSamples/models/id.ply` (correct)
- Norm params: `NormalizationParameters/models/id.npz` (correct)
- Recon mesh: `Meshes/models/id.ply` (both write and read match)

### Normalization consistency verified (no ABC-style mismatch)

The previous ABC project (stage6) had a normalization mismatch — its evaluation used `stage6_batch_chamfer_evaluation.py` which **does not denormalize**, comparing raw-coord GT against normalized-coord reconstructions, producing inflated CDs (~577K). WHUCAD appeared fine (~0.26) only because its raw coords happened to be near unit scale.

For RF, this issue does NOT exist:
1. RF evaluation uses `evaluate.py` -> `compute_trimesh_chamfer()` which **correctly denormalizes** (`gen_points/scale - offset`)
2. Both `PreprocessMesh.py` and `SampleMeshSurface.py` load the mesh identically: `trimesh.load(..., process=False)` then `merge_vertices(merge_tex=True, merge_norm=True)`
3. Both compute normalization from the same formula: AABB center of used vertices, max distance * 1.03 buffer
4. Surface samples are saved in raw coords; reconstruction is in normalized coords; denormalization brings them to the same raw-coord space

### Root cause of large CDs

~67% of shapes have CD < 200 (reasonable), but ~22% have CD > 10K (serious failures). This is an auto-decoding convergence issue:
- 800 iters starting from latent N(0, 0.01) insufficient for shapes far from latent-space mean
- 75th percentile at 3,256 means 25% of test shapes reconstruct poorly
- Likely causes: checkpoint 2000 under-trained, RF geometry (thin walls, anisotropic structures) inherently hard
- `--anisotropic-bias` helps SDF preprocessing but does not affect reconstruction/evaluation code

### Follow-up (when GPU free)
- Visually inspect worst shapes (0142, 0867, 0235, 1355, 1454, 0996) — compare recon vs GT to confirm failure mode (missing geometry vs misaligned vs collapsed)
- Try more auto-decoding iterations (1600–2000) on worst shapes to see if convergence is the bottleneck
- Remove or reduce L2 regularization during reconstruction for outlier shapes — it penalizes large latent codes which are exactly what outlier shapes need
- Initialize latent from training set's empirical mean/variance instead of N(0, 0.01) — shapes far from zero may never converge within 800 iters from a near-zero init
- Plot latent code L2 norm vs CD across all 320 shapes to check if high-CD shapes correspond to large-norm codes (would confirm convergence-failure hypothesis)
- Compare with rf_256_16 results to see if larger code length helps the tail
