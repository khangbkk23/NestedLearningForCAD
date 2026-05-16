# Pipeline Reference

This document describes the active Meta-NATH CAD Phase 1-2 Light pipeline.
The full target architecture is specified in `docs/instruction_CAD.md`.

## Active Flow

1. Load `conf/config.yaml`.
2. Build a continual task stream with `dataset/load_dataset.py`.
3. Build `MetaNATHCore` from `models/meta_nath_core.py`.
4. Stream each task through `MetaNATHEngine.train_task()`.
5. Update CADIC only with normal samples (`anomaly == 0`).
6. Evaluate image and pixel anomaly scores with patch nearest-neighbor matching.
7. Write per-task metrics, run summary, cumulative metrics, and checkpoints.

## Main Entrypoints

- `training/run_experiment.py`
  - Main experiment runner.
  - Supports `--profile tiny|small|default`, `--max_tasks`, `--run_suffix`, `--disable_wandb`, and `--quiet`.

- `scripts/test_integration_2.py`
  - Fast integration check for TITANS, ACC, CADIC, MetaNATHCore, checkpointing, and normal-only update.

- `scripts/summarize_run.py`
  - Converts a run directory into a readable markdown summary.

- `scripts/compute_forgetting.py`
  - Computes standard forgetting from `forgetting_matrix.json` or `task_records.json` with `forgetting_eval`.

## Output Layout

Each run writes to:

```text
results/<experiment_name>_<timestamp>[_suffix]/
```

Typical files:

- `resolved_config.yaml`
- `run_summary.json`
- `task_records.json`
- `task_XX_metrics.json`
- `task_XX_checkpoint.pt`
- `final_cumulative_metrics.json`
- `forgetting_matrix.json` when `evaluation.forgetting_matrix: true`

## Current Scope

Implemented:

- Frozen backbone adapter for HuggingFace DINOv2 and future DINOv3 dict outputs.
- TITANS-style fast memory.
- ACC gating.
- CADIC unified coreset.
- Patch nearest-neighbor image/pixel scoring.
- Current and cumulative evaluation.
- Optional forgetting matrix infrastructure.

Not implemented yet:

- NSP2.
- CBP.
- Subspace Recycling.
- N2B-NC backbone evolution.
- Phase 3 cloud consolidation notebook.

## Legacy

Old ViT-CMS/trainer/evaluator code has been moved to `legacy/`. It is retained for reference but is not part of the active benchmark path.
