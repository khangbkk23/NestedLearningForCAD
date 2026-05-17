# Pipeline Reference

This document describes the active Meta-NATH CAD Phase 1-2 Light pipeline.
The full target architecture is specified in `docs/instruction_CAD.md`.

DINOv2 is the intentional stable backbone for the current implementation and
reporting path. DINOv3 remains a planned migration after the DINOv2 path is
locked and reproducible.

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

- `scripts/run_server_phase3.sh`
  - Linux server workflow for the verified Phase 3.0 path: anchor warmup,
    before eval, conservative consolidation, after eval, and acceptance report.

- `scripts/run_server_visa.sh`
  - Linux server workflow for VisA Phase 1-2 once the dataset is available.

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
- `last_checkpoint.pt` by default
- `task_XX_checkpoint.pt` only when `logging.checkpoint_policy: "all"`
- `best_image_auroc.pt` / `best_pixel_aupr.pt` when `logging.checkpoint_policy: "best_and_last"`
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
- Phase 3.0 minimal N2B-NC consolidation with balanced anchors, drift rollback,
  coreset refresh, patch-token preservation logging, and metric-gated acceptance.
- Accepted conservative 8-task Phase 3.0 configuration in
  `conf/config_phase3_conservative.yaml`.
- NSP2 projection implementation with Subspace Recycling fallback logging.
- CBP reset helper with unit/integration coverage; benchmark reset remains off
  unless an experimental config is explicitly selected.
- VisA config and server entrypoint are prepared.

Not implemented yet:

- DINOv3 production migration.
- VisA benchmark result, because local `data/visa` is not present.
- Accepted benchmark with NSP2 enabled.
- Accepted benchmark with CBP reset enabled.
- Advanced Subspace Recycling policy beyond the current fallback projection.
- Phase 3 cloud notebook is a thin orchestration notebook.

Verified results are tracked in `docs/runs.md`. Do not claim NSP2, CBP reset,
Subspace Recycling, DINOv3, or VisA results until their corresponding runs are
added there.

## Legacy

Old ViT-CMS/trainer/evaluator code has been moved to `legacy/`. It is retained for reference but is not part of the active benchmark path.
