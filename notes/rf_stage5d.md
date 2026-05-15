# Stage 5d — Full-Scale Training for Stage 2d Experiments

**Date:** 2026-05-10

## Experiment 1: rf_256_32_nerf (COMPLETED)

- **Path:** `/opt/data/DeepSDF/experiments/rf_256_32_nerf`
- **Architecture:** xyz_dim=51 (NeRF positional encoding), dims=[512]*8, code_length=256
- **Settings:** scenes_per_batch=32, samples_per_scene=16384, num_epochs=2001, clamping=0.1
- **GPU usage:** ~22.3GB on A800-SXM4-40GB
- **Training speed:** ~18.7s/epoch
- **Total training time:** ~10.4 hours
- **Checkpoints:** 100, 500, 1000, 2000

### Loss Progression:
| Epoch | Avg Loss |
|-------|----------|
| 10    | 0.007079 |
| 100   | 0.001763 |
| 500   | 0.000836 |
| 1000  | 0.000732 |
| 2000  | 0.000565 |

### Key Observation:
- Stable convergence throughout training
- NeRF positional encoding (xyz_dim=51) with standard 512-wide layers works well
- lin0 input dimension confirmed at 307 (256 latent + 51 xyz)

---

## Experiment 2: rf_256_32_nerf_768 (IN PROGRESS)

- **Path:** `/opt/data/DeepSDF/experiments/rf_256_32_nerf_768`
- **Architecture:** xyz_dim=51 (NeRF), dims=[768, 512, 512, 512, 512, 512, 512, 512], code_length=256
- **Settings:** scenes_per_batch=32, samples_per_scene=16384, num_epochs=2001, clamping=0.1
- **GPU usage:** ~24.5GB on A800-SXM4-40GB
- **Training speed:** ~20.7s/epoch
- **ETA:** ~11.5 hours from start (started ~19:40 UTC)
- **Checkpoints so far:** (none yet)

### Architecture Verified:
- lin0: [768, 307] — wider first layer for 51-dim positional-encoded input
- lin3: [205, 512] — standard skip connection

---

## Experiment 3: rf_256_32_onehot (PENDING)

- **Path:** `/opt/data/DeepSDF/experiments/rf_256_32_onehot`
- **Architecture:** xyz_dim=3 (standard), dims=[512]*8, code_length=256, scene_categories (7 categories)
- **Settings:** scenes_per_batch=32, samples_per_scene=16384, num_epochs=2001, clamping=0.1
- **Special:** One-hot category latent init, frozen until epoch 500 (unmask-at-epoch=500)
- **Status:** Queued after rf_256_32_nerf_768 completes
