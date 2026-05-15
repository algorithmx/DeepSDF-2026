# Stage 3: Training Test Run
**Date:** 2026-04-29 23:50 - 23:54

## Status: VERIFIED

## Test Run Results
- Experiment: `rf_128_16`
- Data validation: PASSED (1280 train + 320 test scenes)
- GPU: 1x NVIDIA A100-SXM4-40GB
- Decoder parameters: 1,843,451
- Shape code parameters: 163,840 (1280 codes x 128 dim)
- AsyncPrefetchLoader: 4 producers x 4 workers, queue size 128
- Training speed: ~10s/epoch
- Estimated full training time: ~5.5 hours (2001 epochs)

## Pipeline Verified
- Data loading: Working (SDF samples loaded asynchronously)
- GPU training: Working (CUDA operations running)
- Loss computation: Working (L1 loss with clamping)
- Code regularization: Enabled (lambda=0.0001, warmup over 100 epochs)

## No bugs found during test run

## Note
- CONDA_EXE environment variable must be set for scripts that need conda
- Training was started with `--batch_split 1` (no sub-batching needed on A100 40GB)
