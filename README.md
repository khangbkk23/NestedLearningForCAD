# NestedLearningForCAD

Continual anomaly detection experiments on MVTec using a ViT backbone with CMS-enhanced feed-forward blocks.

## What this repository runs

- Data stream: category-by-category continual tasks from MVTec.
- Model: ViT backbone with CMS blocks and anomaly decoder.
- Outputs: image-level and pixel-level anomaly metrics.
- Experiment flow: one command for a single run, one command for sweeps.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional one-time data download:

```bash
python dataset/download_mvtec.py
```

Quick dataset verification:

```bash
python scripts/01_data_preparation.py --run_verify --config conf/config.yaml
```

## Main entrypoints

Single configured experiment:

```bash
python training/run_experiment.py --config conf/config.yaml
```

Compact smoke test (1 task, 1 epoch):

```bash
python training/run_experiment.py --config conf/config.yaml --max_tasks 1 --epochs 1 --profile tiny
```

Parameter sweep:

```bash
python training/run_sweep.py --config conf/config.yaml --max_tasks 2 --epochs 1 --seeds 42 123
```

## Output structure

- Single runs are written under `results/runs/`.
- Sweeps are written under `results/sweeps/`.
- Generated artifacts in both folders are ignored by git.

Typical single-run artifacts:

- `resolved_config.yaml`
- `task_metrics.json`
- `run_summary.json`
- `checkpoints/` (if enabled)

## Configuration notes

Main defaults live in `conf/config.yaml`.

Important sections:

- `dataset`: root path, image size, anomaly generator, class order
- `model`: backbone, CMS params, freeze flags
- `training`: device, LR, epochs, anomaly loss/replay options
- `logging`: experiment name, output root, W&B options

## Troubleshooting

- If CUDA is unavailable, set `training.device` to `cpu` or pass `--device cpu`.
- If import errors occur after dependency changes, reinstall with `pip install -r requirements.txt`.
- If MVTec path is wrong, update `dataset.root_dir` in `conf/config.yaml`.

## Project layout

```text
conf/        # configs
dataset/     # streaming and anomaly generators
models/      # ViT/CMS and baselines
training/    # trainer, evaluator, run scripts
scripts/     # utility scripts
results/     # generated experiment outputs
```
