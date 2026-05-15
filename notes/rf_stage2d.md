# Stage 2d — Additional Experiments Created

**Date:** 2026-05-10

## Summary

Created three new experiments sharing the same RF dataset and 80/20 split (1280 train / 320 test) as the baseline `rf_256_16` experiment (seed=42).

### Experiment 3: `rf_256_32_nerf`
- **Path:** `/opt/data/DeepSDF/experiments/rf_256_32_nerf`
- **Key settings:** code_length=256, scenes_per_batch=32, xyz_dim=51 (NeRF positional encoding)
- **MLP dims:** [512, 512, 512, 512, 512, 512, 512, 512] (default)
- **Purpose:** Test whether higher-frequency coordinate encoding improves fine geometric detail

### Experiment 4: `rf_256_32_nerf_768`
- **Path:** `/opt/data/DeepSDF/experiments/rf_256_32_nerf_768`
- **Key settings:** code_length=256, scenes_per_batch=32, xyz_dim=51 (NeRF positional encoding)
- **MLP dims:** [768, 512, 512, 512, 512, 512, 512, 512] (wider first layer for 51-dim input)
- **Purpose:** Test whether a wider first layer better exploits the higher-dimensional input

### Experiment 5: `rf_256_32_onehot`
- **Path:** `/opt/data/DeepSDF/experiments/rf_256_32_onehot`
- **Key settings:** code_length=256, scenes_per_batch=32, xyz_dim=3 (standard)
- **SceneCategories:** `/opt/data/DeepSDF/data/RF/scene_categories.json` (7 categories, 0–6)
- **unmask-at-epoch:** 500 (one-hot dimensions frozen until epoch 500)
- **Purpose:** Test whether category-aware initialization provides a structural prior

## Verification
- All three directories contain: `specs.json`, `rf_train.json`, `rf_test.json`
- `rf_train.json` has 1280 entries, `rf_test.json` has 320 entries
- All specs verified: xyz_dim, dims, SceneCategories, unmask-at-epoch match plan
