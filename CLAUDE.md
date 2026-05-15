# CLAUDE.md

## Long-running Tasks
- Training/data prep require patience: wait ≥300s between checks with backoff; wait until finish/crash for critical jobs.
- Properly clean up before launching training job.

## Key Principles
- Use conda env `ml_env` for Python,  available at `/home/vipuser/.conda/envs/ml_env`
- Conda: conda 24.9.2 installed at `/home/vipuser/miniconda3/`
- Keep all data/configs/logs/results outside this project folder.
- Take frequent notes (in folder `notes/`) with timestamps.
- Update PLAN.md: change `[ ]` or `[.]` to `[X]` for completed steps.
- Put auxiliary verification scripts in `scripts/`.
- Reference DeepSDF paper in `docs/`.
- When encounter running error, modify code in minimal efforts and take notes by appending to `notes/issues_encountered.md`.
