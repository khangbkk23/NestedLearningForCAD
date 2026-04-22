# NestedLearningForCAD — Comprehensive Project Documentation

## Project Summary

NestedLearningForCAD is a research codebase for continual anomaly detection experiments on the MVTec dataset using a Vision Transformer (ViT) backbone enhanced with CMS (context-modulated-sparsity) feed-forward blocks. The repository supports streaming tasks (category-by-category), configurable anomaly generators, baseline models, and utilities to run single experiments and parameter sweeps. Outputs include image-level and pixel-level anomaly detection metrics and run artifacts for reproducibility.

## Quick Highlights

- **Data stream:** Category-by-category continual tasks built from MVTec.
- **Model:** ViT backbone with CMS blocks and an anomaly decoder; CNN baselines available.
- **Outputs:** Per-task metrics, run summaries, resolved configs, and checkpoints.
- **Entrypoints:** `training/run_experiment.py`, `training/run_sweep.py`, `scripts/01_data_preparation.py`.

## Setup (Quick)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Optional: download MVTec (one-time):

```bash
python dataset/download_mvtec.py
```

3. Verify dataset and preparation:

```bash
python scripts/01_data_preparation.py --run_verify --config conf/config.yaml
```

## Main Commands (examples)

Run a configured experiment (default config path):

```bash
python training/run_experiment.py --config conf/config.yaml
```

Compact smoke test (1 task, 1 epoch):

```bash
python training/run_experiment.py --config conf/config.yaml --max_tasks 1 --epochs 1 --profile tiny
```

Parameter sweep example:

```bash
python training/run_sweep.py --config conf/config.yaml --max_tasks 2 --epochs 1 --seeds 42 123
```

## Output artifacts and locations

- Single-run outputs: `results/runs/<run_id>/` — contains `resolved_config.yaml`, `task_metrics.json`, `run_summary.json`, and `checkpoints/` (if enabled).
- Sweep outputs: `results/sweeps/` — contains per-sweep directories and `sweep_summary_*.json`.
- EDA and intermediate artifacts: `results/eda/`.

## Configuration

Primary defaults live in `conf/config.yaml`. Key sections:

- `dataset`: `root_dir`, `image_size`, `anomaly_generator`, `class_order`, streaming options.
- `model`: backbone choice (ViT), CMS block parameters, freeze/backbone flags.
- `training`: `device`, learning rate, `epochs`, anomaly loss and replay options.
- `logging`: experiment name, output root, Weights & Biases options.

Refer to `conf/config.py` for helpers that load and resolve the YAML configuration.

## Project layout (folder-by-folder summary)

Below is a concise description of each top-level folder and important files contained within the repository.

- **LICENSE**: Project license text.
- **README.md**: High-level README and quick start (this file expands on it).
- **requirements.txt**: Python package dependencies required to run experiments.

- **conf/**
  - `config.yaml`: Primary experiment defaults and configuration schema.
  - `config.py`: Utilities to load, validate, and resolve configuration values.

- **data/**
  - `cifar-10-batches-py/`: CIFAR-10 batch files (used for legacy/verification scripts).
  - `mvtec/`: Local copy of MVTec dataset (if downloaded). Within each category: `test/`, `train/`, and `ground_truth/` directories and per-category `readme.txt` and `license.txt`.

- **dataset/**
  - `__init__.py`: package initializer.
  - `download_mvtec.py`: Script to download MVTec dataset automatically.
  - `load_dataset.py`: Code to create PyTorch datasets and streaming task loaders (category streaming, transforms, splits).
  - `noise.py`: Utilities to add synthetic noise-based anomalies or corruptions.
  - `anomaly_generators/`: Implementations of anomaly generation strategies used in experiments:
    - `base.py`: Abstract base classes and utility helpers for anomaly generators.
    - `destseg.py`: Destination segmentation-based anomaly generation.
    - `mixed.py`: Compositions / mixing of multiple anomaly types.
    - `perlin.py`: Perlin-noise-based anomalies.
    - `realnet.py`: Real network or dataset-based anomaly insertion.
    - `superpixel.py`: Superpixel-based patch-level anomalies.

- **legacy/**
  - `dataset/cifar_dataloader.py`: Legacy CIFAR dataloader used for older experiments or verification.
  - `scripts/02_check_cifar_loader.py`: Script to validate legacy CIFAR loader correctness.

- **models/**
  - `__init__.py`: model package initialization.
  - `cms.py`: Implementation of CMS-enhanced feed-forward blocks and any custom layers specific to the CMS approach.
  - `cnn_baseline.py`: CNN baseline model(s) used for ablation comparisons.
  - `vit_cms.py`: ViT model definitions integrated with CMS blocks and the anomaly decoder head.

- **papers/**
  - Notes, references, or PDF copies of papers referenced while implementing or designing the method.

- **results/**
  - `eda/`: exploratory data analysis outputs.
  - `runs/`: Single-run outputs (one directory per experiment run).
  - `sweeps/`: Parameter sweep outputs.
  - Example run directories in this repository (pre-existing experimental outputs) show naming patterns like `vit_cms_hybrid_cad_YYYYMMDD_hhmmss_<profile>_.../`.

- **scripts/**
  - `01_data_preparation.py`: Dataset verification and initial preprocessing script used to check MVTec layout and required files.

- **training/**
  - `__init__.py`.
  - `cms_optim.py`: Optimizer wrappers or parameter-group helpers tailored for CMS-specific layers or schedules.
  - `evaluator.py`: Metric computation and evaluation logic (image-level and pixel-level AUC, F1, IoU where applicable), plus utilities for saving `task_metrics.json`.
  - `run_experiment.py`: Single-run entrypoint. Parses CLI args (for `--config`, `--max_tasks`, `--epochs`, `--profile`, etc.), instantiates dataset, model, trainer, and evaluator, and writes run artifacts.
  - `run_sweep.py`: Orchestrates parameter sweeps across seeds or hyperparameters; outputs to `results/sweeps/` and produces `sweep_summary_*.json`.
  - `trainer.py`: Training loop, checkpointing, replay buffer or continual learning logic, logging hooks, and scheduler steps.

- **utils/**
  - `__init__.py`.
  - `get_mvtec_meta.py`: Utilities to read and expose MVTec metadata and ground-truth mask formats.
  - `global_seed.py`: Set and manage random seeds across Python, NumPy, and PyTorch for reproducible runs.

## File-level notes and expectations

- `training/run_experiment.py` is the simplest single-run entrypoint and the place to start when reproducing results. It exports a `resolved_config.yaml` describing the exact config used for the run.
- `conf/config.yaml` contains the canonical defaults; prefer editing a copy or use CLI overrides for reproducibility.
- `dataset/anomaly_generators` contains the code that controls synthetic anomaly types used in ablations (Perlin noise, superpixel patches, mixed strategies). If you modify or add generators, update `conf/config.yaml` to reference them.

## Typical output files explained

- `resolved_config.yaml`: Fully-resolved config for the run (useful for reproduction).
- `task_metrics.json`: JSON object with per-task (per-category) metrics like image-level AUC and pixel-level performance.
- `run_summary.json`: High-level summary of the run (averages, runtime, seed, hardware used).
- `checkpoints/`: Saved model weights (typically saved per-epoch or best model by validation metric).

## Troubleshooting and tips

- If CUDA is unavailable, set `training.device` to `cpu` in `conf/config.yaml` or pass `--device cpu` on the CLI.
- If MVTec files are missing or paths are incorrect, run `python scripts/01_data_preparation.py --run_verify --config conf/config.yaml` and check `dataset.root_dir` in `conf/config.yaml`.
- If import errors occur after dependency changes, reinstall with `pip install -r requirements.txt`.
- For quick smoke tests, use the `--profile tiny` or `--max_tasks 1 --epochs 1` flags to run fast checks.

## Recommended next actions (for contributors)

1. Read `conf/config.yaml` and run the smoke test command to validate environment.
2. Inspect `training/run_experiment.py` to follow how dataset and model are instantiated.
3. Add unit tests around new anomaly generators or model modules to keep regressions low.

## Contact / Attribution

If you need clarification about any module, open an issue or contact the repository maintainers listed in `README.md`.

---

_This document was generated to provide a single, navigable overview of the codebase. For code-level questions, consult the specific module source files referenced above._
