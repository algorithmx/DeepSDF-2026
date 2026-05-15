# Stage 5: Full-Scale Training
**Date:** 2026-04-29 23:57 - 2026-04-30 09:44 (rf_128_16)
**Date:** 2026-04-30 09:45 - (rf_256_16, in progress)

## Status: BOTH COMPLETE

## rf_128_16 Training Results
- Started: 2026-04-29 23:57
- Completed: 2026-04-30 09:44 (total ~9.8 hours)
- GPU: 1x NVIDIA A100-SXM4-40GB
- Batch split: 1 (no sub-batching needed)

### Training Loss
- Initial loss (epoch 1): 0.010822
- Final loss (epoch 2000): 0.000951
- Minimum loss: 0.000910 at epoch 1978
- Convergence: Stable, loss decreased ~11.4x over training

### Checkpoints
All 4 milestone checkpoints saved:
| Checkpoint | Size | Saved At |
|-----------|------|----------|
| 100.pth | 7.4 MB (model), 657 KB (codes) | 00:26 |
| 500.pth | 7.4 MB (model), 657 KB (codes) | 02:24 |
| 1000.pth | 7.4 MB (model), 657 KB (codes) | 04:50 |
| 2000.pth | 7.4 MB (model), 657 KB (codes) | 09:43 |

### Training Speed
- Batch time: ~0.21s/batch (80 batches/epoch)
- Epoch time: ~17s/epoch
- Total: 2001 epochs in ~5.8 hours (dedicated GPU)

## rf_256_16 Training (In Progress)
- Started: 2026-04-30 09:45
- Code length: 256 (vs 128 for rf_128_16)
- Latent code parameters: 327,680 (1280 codes x 256 dim)
- Batch time: ~0.48s/batch (slower due to larger latent)
- Checkpoint 100.pth saved
- Estimated completion: ~7.3 hours from start

## rf_256_16 Training Results
- Completed: 2026-04-30 20:37 (total ~10.9 hours)
- Final loss (epoch 2000): 0.000707
- Epoch time: ~17.3s/epoch
- Latent magnitude: 0.796

### Checkpoints
All 4 milestone checkpoints saved:
| Checkpoint | Size | Saved At |
|-----------|------|----------|
| 100.pth | 7.4 MB (model), 1.3 MB (codes) | 10:44 |
| 500.pth | 7.4 MB (model), 1.3 MB (codes) | 13:15 |
| 1000.pth | 7.4 MB (model), 1.3 MB (codes) | 15:43 |
| 2000.pth | 7.4 MB (model), 1.3 MB (codes) | 20:37 |

## Notes
- rf_256_16 training is ~2.3x slower per batch due to larger latent dimension
- Both experiments share identical network architecture except latent code length
- Training loss plot saved to: `experiments/rf_128_16/training_loss.png`
