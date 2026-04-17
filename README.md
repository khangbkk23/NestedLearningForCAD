# MVTec Anomaly Detection (Nested Learning / CMS)

## Purpose

This repository trains and evaluates continual anomaly detection models on MVTec.
It uses a ViT backbone with CMS layers and supports two execution modes:

- Single run execution for one experiment setup.
- Sweep execution for comparing multiple configurations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Check

```bash
python dataset/download_mvtec.py
python scripts/01_data_preparation.py --run_verify --config ./conf/config.yaml
```

## Run One Experiment

Use `run_experiment.py` when you want one controlled run.

```bash
python training/run_experiment.py --config ./conf/config.yaml
```

Fast profiles:

```bash
python training/run_experiment.py --config ./conf/config.yaml --profile tiny --max_tasks 2
python training/run_experiment.py --config ./conf/config.yaml --profile small --max_tasks 4
```

## Run Multiple Experiments

Use `run_sweep.py` when you want to compare different configs automatically.

```bash
python training/run_sweep.py --config ./conf/config.yaml --profile tiny --max_tasks 2 --disable_wandb
```

## Inspect Checkpoints and Metrics

Use the notebook below to compare run summaries and task-level metrics, and optionally re-evaluate checkpoints:

- `notebooks/eval_checkpoint.ipynb`

Additional reference:

- `docs/PIPELINE.md`

## Output Layout

Each run writes to:

- `results/<experiment_name>_<timestamp>[_suffix]/`

Typical files per run:

- `resolved_config.yaml`
- `run_summary.json`
- `task_XX_metrics.json`
- `task_XX_checkpoint.pt`

## Project Layout

```
conf/  dataset/  models/  scripts/  training/  notebooks/  results/
```
