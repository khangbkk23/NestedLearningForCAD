# Pipeline Reference

## End-to-end flow

1. Load config from `conf/config.yaml`.
2. Build MVTec task stream with `dataset/load_dataset.py`.
3. Build model from config (`ViT_CMS` by default).
4. Train per task with `training/trainer.py`.
5. Evaluate per task with `training/evaluator.py`.
6. Save artifacts to `results/<run_name>/`.

## Script roles

- `training/run_experiment.py`
  - Runs one experiment end-to-end.
  - Supports `default`, `tiny`, and `small` profiles.
  - Saves checkpoints and metrics per task.

- `training/run_sweep.py`
  - Runs multiple experiments by varying selected config fields.
  - Calls `run_experiment` for each combination.
  - Writes sweep summary to `results/sweeps/`.

## Artifact layout per run

- `resolved_config.yaml`
- `run_summary.json`
- `task_XX_metrics.json`
- `task_XX_checkpoint.pt`

## Notebook usage

Notebook: `notebooks/eval_checkpoint.ipynb`

Customize in Cell 3:
- `RUN_DIR_PATTERNS` to select result folders.
- `CHECKPOINTS_TO_REEVALUATE` to evaluate checkpoints again.
- `METRICS_TO_PLOT` and `TASK_METRIC_TO_PLOT` to control charts.

Visual outputs:
- Run-level bar comparison.
- Task-level heatmap comparison.
