# Continual Learning with Nested Learning (CMS)

Implementation of Continuum Memory System (CMS) for continual learning, based on Google's Nested Learning paper.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Models

1. **ViT-CMS**: Vision Transformer with CMS (nested MLPs replacing standard MLPs)
2. **ViT-Simple**: Standard ViT with linear head (baseline)
3. **CNN-Replay**: CNN with experience replay buffer

## Experiment Setup

**Dataset**: CIFAR-10 with pairwise binary task setup
- Task 0: class 0 vs 1
- Task 1: class 2 vs 3
- Task 2: class 4 vs 5
- Task 3: class 6 vs 7
- Task 4: class 8 vs 9

**Metrics**: Accuracy, F1, Forgetting (performance drop on previous tasks)

## Run Scripts

### Validate CIFAR-10 task loader (pairwise tasks)
```bash
python scripts/02_check_cifar_loader.py --download --max_tasks 5 --batch_size 16
```

Fast offline check (no dataset download):
```bash
python scripts/02_check_cifar_loader.py --use_fake_data --max_tasks 5 --batch_size 16
```

This script confirms:
- Task splits are `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, `(8,9)`
- Each task loader only yields labels from its assigned pair
- `task_classes` is attached to each task dataset for trainer/evaluator masking

### Legacy anomaly-data pipeline script
```bash
python scripts/01_data_preparation.py
```

Note: smoke-test execution was removed from this pipeline entrypoint.

## Key Parameters (Loader Check)

- `--data_root`: CIFAR-10 storage location (default: `./data`)
- `--batch_size`: Batch size for validation pass (default: `16`)
- `--num_workers`: DataLoader workers (default: `0`)
- `--max_tasks`: Number of tasks to validate (default: `5`)
- `--download`: Download CIFAR-10 if not available locally

## CMS Implementation

The Continuum Memory System (CMS) replaces standard MLP blocks with nested MLPs:

```python
# Standard MLP: x → MLP → output

# CMS (Nested MLPs):
for i, level in enumerate(levels):
    if step % (k^i) == 0:  # Level i updates every k^i steps
        x = x + level(x)    # Residual connection
```

- **Level 0**: Updates every step (k^0 = 1) - fast processing
- **Level 1**: Updates every k steps - medium speed
- **Level 2**: Updates every k^2 steps - slow, long-term memory

This creates a hierarchy of processing speeds for continual learning.

## Papers

- `papers/ReplayCAD_Generative_Diffusion_Replay_for_Continual_Anomaly_Detection.pdf`
- `papers/CADIC_Continual_Anomaly_Detection_Based_on_Incremental_Coreset.pdf`
- `papers/Nested_Learning_The_Illusion_of_Deep_Learning_Architectures.pdf`

## Structure

```
conf/                 # YAML and config loader
dataset/              # MVTec pipeline + CIFAR task dataloader
models/               # CMS, ViT/CNN model definitions
scripts/              # Utility scripts (data prep, CIFAR loader check)
training/             # Trainer and evaluator
papers/               # Reference papers
```
