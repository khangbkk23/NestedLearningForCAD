# Meta-NATH CAD

Phase 1-2 Light prototype for continual anomaly detection on MVTec AD.

The active pipeline is:

1. Frozen backbone feature extraction (`facebook/dinov2-base` for the current prototype).
2. TITANS-style test-time memory update.
3. ACC gating.
4. CADIC unified coreset update with normal samples only.
5. Patch nearest-neighbor image and pixel anomaly scoring.

See `docs/instruction_CAD.md` for the full research target. This repo does not yet implement Phase 3 (`NSP2`, `CBP`, `Subspace Recycling`, `N2B-NC`).

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
- `dataset/load_dataset.py`: MVTec/VisA continual task stream.
- `scripts/summarize_run.py`: markdown run summaries.
- `scripts/compute_forgetting.py`: forgetting metric from an evaluation matrix.

## Legacy Code

Older ViT-CMS/trainer/evaluator scripts live under `legacy/`. They are kept for reference only and should not be used for Meta-NATH Phase 1-2 or Phase 3 benchmarks.
