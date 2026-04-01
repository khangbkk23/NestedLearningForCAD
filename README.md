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

# Test setup
python test_setup.py
```

## Models

1. **ViT-CMS**: Vision Transformer with CMS (nested MLPs replacing standard MLPs)
2. **ViT-Simple**: Standard ViT with linear head (baseline)
3. **CNN-Replay**: CNN with experience replay buffer

## Experiment Setup

**Dataset**: CIFAR-10 with task-as-class setup
- Task 0: class 0 vs others (binary classification)
- Task 1: class 1 vs others
- ... up to Task 9

**Metrics**: Accuracy, F1, Forgetting (performance drop on previous tasks)

## Running Experiments

### Quick Test
```bash
python run_experiment.py --model vit_cms --num_tasks 2 --epochs 2
```

### ViT-CMS (with nested levels)
```bash
python run_experiment.py \
    --model vit_cms \
    --cms_levels 3 \
    --k 2 \
    --num_tasks 5 \
    --epochs 10 \
    --batch_size 32
```

### ViT-Simple (baseline)
```bash
python run_experiment.py \
    --model vit_simple \
    --num_tasks 5 \
    --epochs 10
```

### CNN with Replay
```bash
python run_experiment.py \
    --model cnn_replay \
    --buffer_size 1000 \
    --num_tasks 5 \
    --epochs 10
```

### Compare All Models
```bash
python run_comparison.py
```

## Key Parameters

- `--model`: vit_cms | vit_simple | cnn_replay
- `--cms_levels`: Number of nested levels (default: 3)
- `--k`: Speed multiplier, level i updates every k^i steps (default: 2)
- `--num_tasks`: Number of tasks to train (default: 5, max: 10)
- `--epochs`: Epochs per task (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--freeze_backbone`: Freeze backbone weights
- `--cpu`: Use CPU instead of GPU

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

## Results

Results are saved to `./results/<experiment_name>/`:
- `results.json`: Complete metrics
- `config.json`: Experiment configuration

## Structure

```
models/          # CMS, ViT-CMS, ViT-Simple, CNN-Replay
datasets/        # Task-as-class CIFAR-10 loaders
training/        # Trainer and Evaluator
run_experiment.py    # Main experiment runner
test_setup.py        # Component tests
```
