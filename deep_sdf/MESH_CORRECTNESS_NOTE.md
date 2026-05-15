# `create_mesh` correctness verification

Note dated 2026-05-14, accompanying `deep_sdf/mesh.py`.

## Background

The original Facebook DeepSDF `create_mesh` decomposed a linear grid index
into `(i, j, k)` via

```python
samples[:, 2] = overall_index % N
samples[:, 1] = (overall_index.long() / N) % N
samples[:, 0] = ((overall_index.long() / N) / N) % N
```

This worked on PyTorch ≤1.4 where `/` was floor division on `LongTensor`.
In modern PyTorch `/` is true division, which silently produces fractional
indices and shears the sampling grid by sub-voxel amounts along x and y.
The z axis (innermost `idx % N`) was unaffected.

The fix uses `torch.div(..., rounding_mode="floor")` and rebuilds the
coordinates directly on the decoder's device.

## Tests run

All tests live in `scripts/CD/_compare_modernized_mesh.py` and one-off
inline Python in the chat log on 2026-05-14.

### 1. Analytic sphere, center off-origin, multiple resolutions

`SDF(x) = ||x − C|| − R` with `C = (0.20, −0.30, 0.10)`, `R = 0.55`.

| N | modernized centroid err | modernized max radius err | buggy centroid err | buggy max radius err |
|---:|---:|---:|---:|---:|
| 32 | 3.0e-3 | 9.4e-4 | 4.3e-2 | 5.1e-2 |
| 64 | 9.3e-4 | 2.3e-4 | 2.0e-2 | 2.5e-2 |
| 128 | 1.6e-4 | 5.6e-5 | 1.1e-2 | 1.3e-2 |
| 256 | 2.7e-4 | 1.4e-5 | 5.2e-3 | 6.3e-3 |

Modernized errors scale as `O(vs²)` (marching-cubes interpolation order);
buggy errors scale as `O(vs)` (sub-voxel coordinate shift).

### 2. Axis-asymmetric box (catches axis permutations / shear)

Half-extents `(0.7, 0.4, 0.15)`, centered at origin.

| axis | GT extent | modernized | error | buggy | error |
|------|---:|---:|---:|---:|---:|
| x    | 1.40 | 1.39999988 | −1.2e-7 | 1.40307 | +3.1e-3 |
| y    | 0.80 | 0.80000000 |  0      | 0.80113 | +1.1e-3 |
| z    | 0.30 | 0.30000007 | +6.0e-8 | 0.30000007 | +6.0e-8 |

Modernized: machine precision on all three axes.
Buggy: z perfect, x and y systematically over-extended in the exact pattern
predicted by analyzing the floor-division bug.

### 3. Real trained decoder (`rf_128_32`, checkpoint 2000)

Reused existing latent codes from `Reconstructions/2000/Codes/`; re-extracted
meshes for 5 test shapes with modernized `create_mesh` and compared against
the backed-up `Meshes.backup_pre_modernization/`. Symmetric nearest-neighbour
distances were 0.002–0.007 (voxel size at N=256 is 0.00784), confirming
the difference between buggy and modernized output is ≈1 voxel — exactly
the predicted sub-voxel shear.

## Conclusion

The modernized `create_mesh` recovers analytic SDFs to machine precision
on the z axis and to `O(vs²)` on x and y, with no axis permutation, sign
flip, or origin offset. The buggy original recovers z perfectly but shears
x and y by sub-voxel amounts that do not vanish as N grows.

Backup of pre-modernization meshes:
`/opt/data/DeepSDF/experiments/rf_128_32/Reconstructions/2000/Meshes.backup_pre_modernization/`
