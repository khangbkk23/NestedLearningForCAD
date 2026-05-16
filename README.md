# Meta-NATH CAD

Phase 1-2 Light prototype for continual anomaly detection on MVTec AD.

The active pipeline is:

1. Frozen backbone feature extraction (`facebook/dinov2-base` for the current prototype).
2. TITANS-style test-time memory update.
3. ACC gating.
4. CADIC unified coreset update with normal samples only.
5. Patch nearest-neighbor image and pixel anomaly scoring.

See `docs/instruction_CAD.md` for the full research target. Phase 3 now has
a smoke path for N2B-NC consolidation; NSP2 and CBP are present behind config
flags and should be enabled only after the minimal consolidation path is stable.

## Setup

```powershell
.\.pixi\envs\default\python.exe --version
```

or install from `requirements.txt` in a separate environment.

## Verify

```powershell
.\.pixi\envs\default\python.exe scripts\test_integration_2.py
.\.pixi\envs\default\python.exe scripts\01_data_preparation.py --run_verify --config .\conf\config.yaml
```

## Run

Smoke:

```powershell
.\.pixi\envs\default\python.exe training\run_experiment.py --max_tasks 4 --disable_wandb --quiet
```

Full Phase 1-2 baseline:

```powershell
.\.pixi\envs\default\python.exe training\run_experiment.py --disable_wandb --quiet --run_suffix raw_baseline_full15
```

Phase 3 anchor warmup and consolidation smoke:

```powershell
.\.pixi\envs\default\python.exe training\run_experiment.py --config .\conf\config_phase3.yaml --max_tasks 4 --disable_wandb --quiet --run_suffix phase3_anchor_warmup_4task
.\.pixi\envs\default\python.exe scripts\run_phase3_consolidation.py --config .\conf\config_phase3.yaml --checkpoint results\<warmup_run>\last_checkpoint.pt --run_suffix phase3_smoke
.\.pixi\envs\default\python.exe scripts\evaluate_checkpoint.py --config .\conf\config_phase3.yaml --checkpoint results\<phase3_run>\last_checkpoint.pt --max_tasks 4 --quiet --run_suffix phase3_after_eval_4task
```

## Inspect Results

```powershell
.\.pixi\envs\default\python.exe scripts\summarize_run.py results\<run_dir>
```

Baseline runs are tracked in `docs/runs.md`.

By default runs save `last_checkpoint.pt` only. Set
`logging.checkpoint_policy: "all"` if you need per-task checkpoints, or
`"best_and_last"` if you want best image/pixel checkpoints as well.

## Active Files

- `models/meta_nath_core.py`: frozen backbone, TITANS, ACC, CADIC orchestration.
- `models/cadic_coreset.py`: unified memory bank and patch-NN scoring.
- `training/meta_nath_engine.py`: normal-only streaming update and evaluation.
- `training/run_experiment.py`: main experiment entrypoint.
- `training/consolidation_engine.py`: Phase 3 N2B-NC consolidation.
- `dataset/load_dataset.py`: MVTec/VisA continual task stream.
- `scripts/run_phase3_consolidation.py`: Phase 3 CLI entrypoint.
- `scripts/evaluate_checkpoint.py`: evaluates a saved checkpoint without retraining.
- `scripts/summarize_run.py`: markdown run summaries.
- `scripts/compute_forgetting.py`: forgetting metric from an evaluation matrix.

## Legacy Code

Older ViT-CMS/trainer/evaluator scripts live under `legacy/`. They are kept for reference only and should not be used for Meta-NATH Phase 1-2 or Phase 3 benchmarks.
