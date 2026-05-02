## Tree for 
```
├── .gitignore
├── conf/
│   ├── config.py
│   └── config.yaml
├── dataset/
│   ├── anomaly_generators/
│   │   ├── base.py
│   │   ├── destseg.py
│   │   ├── mixed.py
│   │   ├── perlin.py
│   │   ├── realnet.py
│   │   ├── superpixel.py
│   │   └── __init__.py
│   ├── download_mvtec.py
│   ├── load_dataset.py
│   ├── noise.py
│   └── __init__.py
├── docs/
│   └── PROJECT_SUMMARY.md
├── legacy/
│   ├── dataset/
│   │   └── cifar_dataloader.py
│   └── scripts/
│       └── 02_check_cifar_loader.py
├── LICENSE
├── models/
│   ├── cms.py
│   ├── cnn_baseline.py
│   ├── dino_nsp2.py
│   ├── titans_memory.py
│   ├── vit_cms.py
│   └── __init__.py
├── papers/
│   ├── CADIC_Continual_Anomaly_Detection_Based_on_Incremental_Coreset.pdf
│   ├── CAD_Fundamental.pdf
│   ├── Nested_Learning_The_Illusion_of_Deep_Learning_Architectures.pdf
│   ├── ReplayCAD_Generative_Diffusion_Replay_for_Continual_Anomaly_Detection.pdf
│   └── Visual Prompt Tuning in Null Space for Continual Learning.pdf
├── README.md
├── requirements.txt
├── results/
│   └── eda/
│       ├── pipeline_verify_hazelnut.png
│       ├── pipeline_verify_screw.png
│       ├── pipeline_verify_toothbrush.png
│       ├── pipeline_verify_transistor.png
│       ├── pipeline_verify_wood.png
│       └── pipeline_verify_zipper.png
├── scripts/
│   └── 01_data_preparation.py
├── training/
│   ├── cms_optim.py
│   ├── evaluator.py
│   ├── memory_buffer.py
│   ├── nsp2_optim.py
│   ├── run_experiment.py
│   ├── run_sweep.py
│   ├── trainer.py
│   └── __init__.py
└── utils/
    ├── get_mvtec_meta.py
    ├── global_seed.py
    └── __init__.py
```

## File: .gitignore
```
# Virtual environments
venv/
env/
ENV/
.venv/

# Python cache and build artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
*.pyd
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Dataset archives and local data
data/
datasets/
*.tar.gz
*.zip

# Generated experiment artifacts
checkpoints/
logs/
results/runs/
results/sweeps/
*.pth
*.pt
*.ckpt

# Notebook checkpoints
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
tempCodeRunnerFile.py

# OS
.DS_Store
Thumbs.db

# Distribution / packaging
dist/
build/
*.egg-info/
.eggs/

# Secrets
kaggle.json
```
## File: LICENSE
```
MIT License

Copyright (c) 2026 Khang Bui Tran Duy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## File: README.md
```markdown
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
```
## File: requirements.txt
```
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
tqdm>=4.65.0
scikit-learn>=1.4.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
Pillow>=9.5.0
opencv-contrib-python>=4.10.0

# Optional experiment tracking
wandb>=0.17.0
```
## File: conf\config.py
```python
import yaml
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config file at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print(f"Config loading successfully {config_path}")
    return config
```
## File: conf\config.yaml
```yaml
dataset:
  name: "mvtec"
  root_dir: "data/mvtec"
  img_size: 256
  batch_size: 32
  num_workers: 4
  split_ratio: 0.8
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  use_dtd: true
  dtd_dir: "data/dtd/images"
  anomaly_generator: "superpixel"   # or "perlin", "destseg", "realnet", "mixed"
  class_order: [
    "bottle", "cable", "capsule", "carpet", "grid", 
    "hazelnut", "leather", "metal_nut", "pill", "screw", 
    "tile", "toothbrush", "transistor", "wood", "zipper"
  ]

model:
  backbone: "vit_small_patch14_dinov2.lvd142m"
  pretrained: true
  freeze_backbone: true
  prompt_length: 8
  prompt_layers: 4
  prompt_dropout: 0.05
  gating_hidden_dim: 256
  gating_dropout: 0.1
  gating_threshold: 0.55
  use_torchhub_fallback: true
  torchhub_model: "dinov2_vits14"

memory:
  coreset_sampling_ratio: 0.01
  nearest_neighbors: 9
  distance_metric: "cosine" 

training:
  optimizer: "adamw"
  scheduler: "cosine" 
  learning_rate: 0.0001
  weight_decay: 0.00001
  epochs_per_task: 10
  task_type: "anomaly"
  pixel_loss_weight: 0.2
  acc_loss_weight: 0.5
  proxy_loss_weight: 0.5
  ln_loss_weight: 0.1
  gradient_clip_norm: 1.0

  titans_bank_size: 8192
  titans_k_neighbors: 8
  slow_memory_size: 2048

  nsp2_svd_tol: 0.000001
  nsp2_svd_rel_tol: 0.0001

  cbp_patience: 50
  cbp_activation_threshold: 0.00001

  use_replay: false
  replay_batch_size: 32
  
  device: "cuda"
  seed: 42

logging:
  experiment_name: "vit_cms_hybrid_cad"
  results_dir: "results"
  runs_subdir: "runs"
  save_models: true
  log_frequency: 10
  use_wandb: false
  wandb_project: "nested-learning-for-cad"
  wandb_entity: null
```
## File: dataset\download_mvtec.py
```python
import argparse
import os
import shutil
import tarfile
import urllib.error
import urllib.request


MVTec_URLS = [
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz",
    "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection.tar.xz",
]

EXPECTED_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def _dataset_complete(root_dir):
    return all(os.path.isdir(os.path.join(root_dir, cls_name)) for cls_name in EXPECTED_CLASSES)


def _flatten_nested_root(extract_dir):
    nested_root = os.path.join(extract_dir, "mvtec_anomaly_detection")
    if not os.path.isdir(nested_root):
        return

    if _dataset_complete(extract_dir):
        return

    if _dataset_complete(nested_root):
        for entry in os.listdir(nested_root):
            src = os.path.join(nested_root, entry)
            dst = os.path.join(extract_dir, entry)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested_root, ignore_errors=True)


def _safe_extract_tar(tar_path, extract_dir):
    extract_dir_abs = os.path.abspath(extract_dir)
    with tarfile.open(tar_path, "r:xz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extract_dir, member.name))
            if not member_path.startswith(extract_dir_abs + os.sep) and member_path != extract_dir_abs:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(path=extract_dir)


def _download_with_fallbacks(urls, output_path):
    last_error = None
    for idx, url in enumerate(urls, start=1):
        try:
            print(f"Attempt {idx}/{len(urls)}: downloading from {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response, open(output_path, "wb") as target:
                shutil.copyfileobj(response, target)
            print(f"Download completed: {output_path}")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Download failed from {url}: {exc}")

    raise RuntimeError(
        "All download URLs failed. Please download mvtec_anomaly_detection.tar.xz manually "
        f"into {os.path.dirname(output_path)}. Last error: {last_error}"
    )


def download_and_extract_mvtec(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "mvtec_anomaly_detection.tar.xz")
    extract_dir = os.path.join(data_dir, "mvtec")
    os.makedirs(extract_dir, exist_ok=True)

    _flatten_nested_root(extract_dir)
    if _dataset_complete(extract_dir):
        print(f"MVTec dataset already prepared at: {extract_dir}. Skipping download/extract.")
        return extract_dir

    if not os.path.exists(tar_path):
        _download_with_fallbacks(MVTec_URLS, tar_path)
    else:
        print(f"Using existing archive: {tar_path}")

    print("Extracting archive...")
    _safe_extract_tar(tar_path, extract_dir)
    _flatten_nested_root(extract_dir)

    if not _dataset_complete(extract_dir):
        raise RuntimeError(
            "Extraction finished but dataset structure is incomplete. "
            "Expected MVTec class folders were not found."
        )

    print(f"MVTec dataset is ready at: {extract_dir}")
    return extract_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract MVTec AD dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_extract_mvtec(args.data_dir)
```
## File: dataset\load_dataset.py
```python
import os
import sys
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import logging

from conf.config import load_config
from dataset.anomaly_generators import build_anomaly_generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def default_image_loader(path):
    """Top-level image loader (picklable for multiprocessing)."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_mask_loader(path):
    """Top-level mask loader (picklable for multiprocessing)."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class ContinualAnomalyDataset(Dataset):
    """
    Train mode
    ----------
    Loads only *normal* images (per CLAD framework).
    With 50 % probability, applies synthetic anomaly generation via the
    configured generator (superpixel | perlin | destseg | realnet | mixed).
    Switch the generator in config.yaml:  dataset.anomaly_generator: "perlin"

    Test mode
    ---------
    Loads normal + all defect images with their real ground-truth masks.
    """

    def __init__(self, cfg, category, is_train=True, transform=None, target_transform=None):
        self.root_dir     = cfg['root_dir']
        self.dataset_name = cfg['name'].lower()
        self.category     = category
        self.split_ratio  = cfg.get('split_ratio', 0.8)
        self.is_train     = is_train

        self.transform        = transform
        self.target_transform = target_transform

        self.loader        = default_image_loader
        self.loader_target = default_mask_loader

        if self.is_train:
            self.anomaly_generator = build_anomaly_generator(cfg)
            gen_name = cfg.get('anomaly_generator', 'superpixel')
            logger.info(f"[{category.upper()}] Anomaly generator: '{gen_name}'")

        self.data_all = []
        self._build_dataset_index()

    def _build_dataset_index(self):
        category_path = os.path.join(self.root_dir, self.category)
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Cannot found: {category_path}")

        if self.dataset_name == "mvtec":
            self._parse_mvtec(category_path)
        elif self.dataset_name == "visa":
            self._parse_visa(category_path)
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
            
        logger.info(f"[{self.category.upper()}] Loaded {len(self.data_all)} samples (Train={self.is_train})")

    def _parse_mvtec(self, category_path):
        if self.is_train:
            img_dir = os.path.join(category_path, 'train', 'good')
            for img_path in sorted(glob.glob(os.path.join(img_dir, '*.png'))):
                self.data_all.append({
                    'img_path': img_path, 'mask_path': '', 
                    'cls_name': self.category, 'specie_name': 'good', 'anomaly': 0
                })
        else:
            test_dir = os.path.join(category_path, 'test')
            defect_types = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
            
            for defect in defect_types:
                defect_dir = os.path.join(test_dir, defect)
                for img_path in sorted(glob.glob(os.path.join(defect_dir, '*.png'))):
                    if defect == 'good':
                        self.data_all.append({
                            'img_path': img_path, 'mask_path': '', 
                            'cls_name': self.category, 'specie_name': 'good', 'anomaly': 0
                        })
                    else:
                        mask_name = os.path.basename(img_path).replace('.png', '_mask.png')
                        mask_path = os.path.join(category_path, 'ground_truth', defect, mask_name)
                        self.data_all.append({
                            'img_path': img_path, 'mask_path': mask_path, 
                            'cls_name': self.category, 'specie_name': defect, 'anomaly': 1
                        })

    def _parse_visa(self, category_path):
        normal_dir = os.path.join(category_path, 'Data', 'Images', 'Normal')
        anomaly_dir = os.path.join(category_path, 'Data', 'Images', 'Anomaly')
        mask_dir = os.path.join(category_path, 'Data', 'Masks', 'Anomaly')
        
        normal_imgs = sorted([img for img in glob.glob(os.path.join(normal_dir, '*.*')) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
        split_idx = int(len(normal_imgs) * self.split_ratio)
        
        if self.is_train:
            for img_path in normal_imgs[:split_idx]:
                self.data_all.append({'img_path': img_path, 'mask_path': '', 'cls_name': self.category, 'specie_name': 'normal', 'anomaly': 0})
        else:
            for img_path in normal_imgs[split_idx:]:
                self.data_all.append({'img_path': img_path, 'mask_path': '', 'cls_name': self.category, 'specie_name': 'normal', 'anomaly': 0})
            
            if os.path.exists(anomaly_dir):
                anomaly_imgs = sorted([img for img in glob.glob(os.path.join(anomaly_dir, '*.*')) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for img_path in anomaly_imgs:
                    mask_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.png'
                    mask_path = os.path.join(mask_dir, mask_name)
                    self.data_all.append({'img_path': img_path, 'mask_path': mask_path, 'cls_name': self.category, 'specie_name': 'anomaly', 'anomaly': 1})

    def __len__(self):
        return len(self.data_all)
    
    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path = data['img_path'], data['mask_path']
        cls_name, specie_name, anomaly = data['cls_name'], data['specie_name'], data['anomaly']

        img = self.loader(img_path)
        img_w, img_h = img.size   # (width, height)

        if self.is_train:
            if np.random.rand() > 0.5:
                img_np = np.array(img).astype(np.float32)   # (H, W, 3)
                result_np, mask_np, has_anomaly = self.anomaly_generator.generate(
                    img_np, self.category
                )
                if has_anomaly:
                    img      = Image.fromarray(result_np.astype(np.uint8))
                    img_mask = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
                    anomaly  = 1
                else:
                    img_mask = Image.fromarray(np.zeros((img_h, img_w), dtype=np.uint8), mode='L')
                    anomaly  = 0
            else:
                # Normal sample
                img_mask = Image.fromarray(np.zeros((img_h, img_w), dtype=np.uint8), mode='L')
                anomaly  = 0

        else:
            if anomaly == 0 or mask_path == '':
                img_mask = Image.fromarray(np.zeros((img_h, img_w), dtype=np.uint8), mode='L')
            else:
                mask_arr = np.array(self.loader_target(mask_path)) > 0
                img_mask = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode='L')

        img      = self.transform(img)             if self.transform        is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return {
            'img':        img,
            'img_mask':   img_mask,
            'cls_name':   cls_name,
            'specie_name': specie_name,
            'anomaly':    anomaly,
            'img_path':   img_path,
        }

class ContinualStreamingManager:

    def __init__(self, config):
        self.dataset_cfg = config['dataset']
        
        self.dataset_name = self.dataset_cfg['name']
        self.root_dir = self.dataset_cfg['root_dir']
        self.batch_size = self.dataset_cfg['batch_size']
        self.num_workers = self.dataset_cfg.get('num_workers', 4)
        self.split_ratio = self.dataset_cfg.get('split_ratio', 0.8)
        self.img_size = self.dataset_cfg['img_size']
        
        self.data_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.dataset_cfg.get('mean', [0.485, 0.456, 0.406]), 
                                 std=self.dataset_cfg.get('std', [0.229, 0.224, 0.225]))
        ])
        
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        self.categories = self._get_categories()
        self.current_task_idx = 0
        self.test_datasets_history = []

    def _get_categories(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root dir does not exist: {self.root_dir}")
        
        predefined_order = self.dataset_cfg.get('class_order', [])
        if predefined_order:
            categories = [c for c in predefined_order if os.path.isdir(os.path.join(self.root_dir, c))]
        else:
            categories = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        return categories

    def get_next_task(self):
        if self.current_task_idx >= len(self.categories):
            logger.info("Data loading completed for all tasks.")
            return None, None, None
            
        current_category = self.categories[self.current_task_idx]
        logger.info(f"Dataset loading for: {self.current_task_idx}: {current_category.upper()}")
        
        train_dataset = ContinualAnomalyDataset(
            cfg=self.dataset_cfg, 
            category=current_category, 
            is_train=True, 
            transform=self.data_transforms, 
            target_transform=self.gt_transforms
        )
        
        current_test_dataset = ContinualAnomalyDataset(
            cfg=self.dataset_cfg, 
            category=current_category, 
            is_train=False, 
            transform=self.data_transforms, 
            target_transform=self.gt_transforms
        )
        
        self.test_datasets_history.append(current_test_dataset)
        concat_test_dataset = ConcatDataset(self.test_datasets_history)
        
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True if torch.cuda.is_available() else False,
            'persistent_workers': True if self.num_workers > 0 else False
        }
        
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
        test_loader = DataLoader(concat_test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
        
        task_info = {
            'task_id': self.current_task_idx,
            'category': current_category,
        }
        
        self.current_task_idx += 1
        return train_loader, test_loader, task_info

if __name__ == "__main__":
    default_config = os.path.join(PROJECT_ROOT, "conf", "config.yaml")
    config = load_config(default_config)
    manager = ContinualStreamingManager(config)
    
    train_loader, test_loader, info = manager.get_next_task()
    if train_loader:
        batch = next(iter(train_loader))
        print(f"Task info: {info}")
        print(f"Keys in batch: {batch.keys()}")
        print(f"Image batch shape: {batch['img'].shape}")
        print(f"Mask batch shape: {batch['img_mask'].shape}")
```
## File: dataset\noise.py
```python
from ctypes import c_int64
from math import floor

import numpy as np
import random

try:
    from numba import njit, prange
except ImportError:
    # Fallback keeps functionality when numba is not installed.
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range


class Simplex_CLASS:

    def __init__(self):
        self.newSeed()

    def newSeed(self, seed=None):
        # treat 0 as a valid seed; only generate when seed is None
        if seed is None:
            # use Python's random to avoid numpy int32 bounds issues
            seed = random.randint(-(2**63), 2**63 - 1)
        self._perm, self._perm_grad_index3 = _init(seed)


    def noise2(self, x, y):
        return _noise2(x, y, self._perm)

    def noise2array(self, x, y):
        return _noise2a(x, y, self._perm)

    def noise3(self, x, y, z):
        return _noise3(x, y, z, self._perm, self._perm_grad_index3)

    def noise3array(self, x, y, z):
        return _noise3a(x, y, z, self._perm, self._perm_grad_index3)

    def rand_3d_octaves(self, shape, octaves=1, persistence=0.5, frequency=32):
        """
            Returns a layered fractal noise in 3D
        :param shape: Shape of 3D tensor output
        :param octaves: Number of levels of fractal noise
        :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases
        :param frequency: Frequency of initial octave of noise
        :return: Fractal noise sample with n lots of 2D images
        """
        assert len(shape) == 3
        noise = np.zeros(shape)
        z, y, x = [np.arange(0, end) for end in shape]
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.noise3array(x / frequency, y / frequency, z / frequency)
            frequency /= 2
            amplitude *= persistence
        return noise

    def rand_2d_octaves(self, shape, octaves=1, persistence=0.5, frequency=32):
        """
            Returns a layered fractal noise in 2D
        :param shape: Shape of 2D tensor output
        :param octaves: Number of levels of fractal noise
        :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases
        :param frequency: Frequency of initial octave of noise
        :return: Fractal noise sample with n lots of 2D images
        """
        assert len(shape) == 2
        noise = np.zeros(shape)
        y, x = [np.arange(0, end) for end in shape]
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.noise2array(x / frequency, y / frequency)
            frequency /= 2
            amplitude *= persistence
        return noise

    def rand_3d_fixed_T_octaves(self, shape, T, octaves=1, persistence=0.5, frequency=32):
        """
        Returns a layered fractal noise in 3D
        :param shape: Shape of 3D tensor output
        :param octaves: Number of levels of fractal noise
        :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases
        :param frequency: Frequency of initial octave of noise
        :return: Fractal noise sample with n lots of 2D images
        """
        assert len(shape) == 2
        noise = np.zeros((1, *shape))
        y, x = [np.arange(0, end) for end in shape]
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.noise3array(x / frequency, y / frequency, T / frequency)
            frequency /= 2
            amplitude *= persistence
        return noise

DEFAULT_SEED = 3

# Gradients for 2D. They approximate the directions to the
# vertices of an octagon from the center.
GRADIENTS2 = np.array(
        [
            5, 2, 2, 5,
            -5, 2, -2, 5,
            5, -2, 2, -5,
            -5, -2, -2, -5,
            ], dtype=np.int64
        )

# Gradients for 3D. They approximate the directions to the
# vertices of a rhombicuboctahedron from the center, skewed so
# that the triangular and square facets can be inscribed inside
# circles of the same radius.
GRADIENTS3 = np.array(
        [
            -11, 4, 4, -4, 11, 4, -4, 4, 11,
            11, 4, 4, 4, 11, 4, 4, 4, 11,
            -11, -4, 4, -4, -11, 4, -4, -4, 11,
            11, -4, 4, 4, -11, 4, 4, -4, 11,
            -11, 4, -4, -4, 11, -4, -4, 4, -11,
            11, 4, -4, 4, 11, -4, 4, 4, -11,
            -11, -4, -4, -4, -11, -4, -4, -4, -11,
            11, -4, -4, 4, -11, -4, 4, -4, -11,
            ], dtype=np.int64
        )

# Gradients for 4D. They approximate the directions to the
# vertices of a disprismatotesseractihexadecachoron from the center,
# skewed so that the tetrahedral and cubic facets can be inscribed inside
# spheres of the same radius.
GRADIENTS4 = np.array(
        [
            3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3,
            -3, 1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 1, -1, 1, 1, 3,
            3, -1, 1, 1, 1, -3, 1, 1, 1, -1, 3, 1, 1, -1, 1, 3,
            -3, -1, 1, 1, -1, -3, 1, 1, -1, -1, 3, 1, -1, -1, 1, 3,
            3, 1, -1, 1, 1, 3, -1, 1, 1, 1, -3, 1, 1, 1, -1, 3,
            -3, 1, -1, 1, -1, 3, -1, 1, -1, 1, -3, 1, -1, 1, -1, 3,
            3, -1, -1, 1, 1, -3, -1, 1, 1, -1, -3, 1, 1, -1, -1, 3,
            -3, -1, -1, 1, -1, -3, -1, 1, -1, -1, -3, 1, -1, -1, -1, 3,
            3, 1, 1, -1, 1, 3, 1, -1, 1, 1, 3, -1, 1, 1, 1, -3,
            -3, 1, 1, -1, -1, 3, 1, -1, -1, 1, 3, -1, -1, 1, 1, -3,
            3, -1, 1, -1, 1, -3, 1, -1, 1, -1, 3, -1, 1, -1, 1, -3,
            -3, -1, 1, -1, -1, -3, 1, -1, -1, -1, 3, -1, -1, -1, 1, -3,
            3, 1, -1, -1, 1, 3, -1, -1, 1, 1, -3, -1, 1, 1, -1, -3,
            -3, 1, -1, -1, -1, 3, -1, -1, -1, 1, -3, -1, -1, 1, -1, -3,
            3, -1, -1, -1, 1, -3, -1, -1, 1, -1, -3, -1, 1, -1, -1, -3,
            -3, -1, -1, -1, -1, -3, -1, -1, -1, -1, -3, -1, -1, -1, -1, -3,
            ], dtype=np.int64
        )

STRETCH_CONSTANT2 = -0.211324865405187  # (1/Math.sqrt(2+1)-1)/2
SQUISH_CONSTANT2 = 0.366025403784439  # (Math.sqrt(2+1)-1)/2
STRETCH_CONSTANT3 = -1.0 / 6  # (1/Math.sqrt(3+1)-1)/3
SQUISH_CONSTANT3 = 1.0 / 3  # (Math.sqrt(3+1)-1)/3
STRETCH_CONSTANT4 = -0.138196601125011  # (1/Math.sqrt(4+1)-1)/4
SQUISH_CONSTANT4 = 0.309016994374947  # (Math.sqrt(4+1)-1)/4

NORM_CONSTANT2 = 47
NORM_CONSTANT3 = 103
NORM_CONSTANT4 = 30


def overflow(x):
    # Since normal python ints and longs can be quite humongous we have to use
    # self hack to make them be able to overflow.
    # Using a np.int64 won't work either, as it will still complain with:
    # "OverflowError: int too big to convert"
    return c_int64(x).value

def _init(seed=DEFAULT_SEED):
    # Have to zero fill so we can properly loop over it later
    perm = np.zeros(256, dtype=np.int64)
    perm_grad_index3 = np.zeros(256, dtype=np.int64)
    source = np.arange(256)
    # Generates a proper permutation (i.e. doesn't merely perform N
    # successive pair swaps on a base array)
    seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
    seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
    seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
    for i in range(255, -1, -1):
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        r = int((seed + 31) % (i + 1))
        if r < 0:
            r += i + 1
        perm[i] = source[r]
        perm_grad_index3[i] = int((perm[i] % (len(GRADIENTS3) / 3)) * 3)
        source[r] = source[i]
    return perm, perm_grad_index3

@njit(cache=True)
def _extrapolate2(perm, xsb, ysb, dx, dy):
    index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E
    g1, g2 = GRADIENTS2[index:index + 2]
    return g1 * dx + g2 * dy

@njit(cache=True)
def _extrapolate3(perm, perm_grad_index3, xsb, ysb, zsb, dx, dy, dz):
    index = perm_grad_index3[
        (perm[(perm[xsb & 0xFF] + ysb) & 0xFF] + zsb) & 0xFF
        ]
    g1, g2, g3 = GRADIENTS3[index:index + 3]
    return g1 * dx + g2 * dy + g3 * dz

@njit(cache=True)
def _noise2(x, y, perm):
    # Place input coordinates onto grid.
    stretch_offset = (x + y) * STRETCH_CONSTANT2
    xs = x + stretch_offset
    ys = y + stretch_offset

    # Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
    xsb = floor(xs)
    ysb = floor(ys)

    # Skew out to get actual coordinates of rhombus origin. We'll need these later.
    squish_offset = (xsb + ysb) * SQUISH_CONSTANT2
    xb = xsb + squish_offset
    yb = ysb + squish_offset

    # Compute grid coordinates relative to rhombus origin.
    xins = xs - xsb
    yins = ys - ysb

    # Sum those together to get a value that determines which region we're in.
    in_sum = xins + yins

    # Positions relative to origin point.
    dx0 = x - xb
    dy0 = y - yb

    value = 0

    # Contribution (1,0)
    dx1 = dx0 - 1 - SQUISH_CONSTANT2
    dy1 = dy0 - 0 - SQUISH_CONSTANT2
    attn1 = 2 - dx1 * dx1 - dy1 * dy1
    if attn1 > 0:
        attn1 *= attn1
        value += attn1 * attn1 * _extrapolate2(perm, xsb + 1, ysb + 0, dx1, dy1)

    # Contribution (0,1)
    dx2 = dx0 - 0 - SQUISH_CONSTANT2
    dy2 = dy0 - 1 - SQUISH_CONSTANT2
    attn2 = 2 - dx2 * dx2 - dy2 * dy2
    if attn2 > 0:
        attn2 *= attn2
        value += attn2 * attn2 * _extrapolate2(perm, xsb + 0, ysb + 1, dx2, dy2)

    if in_sum <= 1:  # We're inside the triangle (2-Simplex) at (0,0)
        zins = 1 - in_sum
        if zins > xins or zins > yins:  # (0,0) is one of the closest two triangular vertices
            if xins > yins:
                xsv_ext = xsb + 1
                ysv_ext = ysb - 1
                dx_ext = dx0 - 1
                dy_ext = dy0 + 1
            else:
                xsv_ext = xsb - 1
                ysv_ext = ysb + 1
                dx_ext = dx0 + 1
                dy_ext = dy0 - 1
        else:  # (1,0) and (0,1) are the closest two vertices.
            xsv_ext = xsb + 1
            ysv_ext = ysb + 1
            dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT2
            dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT2
    else:  # We're inside the triangle (2-Simplex) at (1,1)
        zins = 2 - in_sum
        if zins < xins or zins < yins:  # (0,0) is one of the closest two triangular vertices
            if xins > yins:
                xsv_ext = xsb + 2
                ysv_ext = ysb + 0
                dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT2
                dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT2
            else:
                xsv_ext = xsb + 0
                ysv_ext = ysb + 2
                dx_ext = dx0 + 0 - 2 * SQUISH_CONSTANT2
                dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT2
        else:  # (1,0) and (0,1) are the closest two vertices.
            dx_ext = dx0
            dy_ext = dy0
            xsv_ext = xsb
            ysv_ext = ysb
        xsb += 1
        ysb += 1
        dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT2
        dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT2

    # Contribution (0,0) or (1,1)
    attn0 = 2 - dx0 * dx0 - dy0 * dy0
    if attn0 > 0:
        attn0 *= attn0
        value += attn0 * attn0 * _extrapolate2(perm, xsb, ysb, dx0, dy0)

    # Extra Vertex
    attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
    if attn_ext > 0:
        attn_ext *= attn_ext
        value += attn_ext * attn_ext * _extrapolate2(perm, xsv_ext, ysv_ext, dx_ext, dy_ext)

    return value / NORM_CONSTANT2

@njit(cache=True, parallel=True)
def _noise2a(x, y, perm):
    noise = np.zeros(x.size * y.size, dtype=np.double)
    for i in prange(y.size):
        for j in prange(x.size):
            noise[i * y.size + j] = _noise2(x[j], y[i], perm)
    return noise.reshape((x.size, y.size))

@njit(cache=True)
def _noise3(x, y, z, perm, perm_grad_index3):
    # Place input coordinates on simplectic honeycomb.
    stretch_offset = (x + y + z) * STRETCH_CONSTANT3
    xs = x + stretch_offset
    ys = y + stretch_offset
    zs = z + stretch_offset

    # Floor to get simplectic honeycomb coordinates of rhombohedron (stretched cube) super-cell origin.
    xsb = floor(xs)
    ysb = floor(ys)
    zsb = floor(zs)

    # Skew out to get actual coordinates of rhombohedron origin. We'll need these later.
    squish_offset = (xsb + ysb + zsb) * SQUISH_CONSTANT3
    xb = xsb + squish_offset
    yb = ysb + squish_offset
    zb = zsb + squish_offset

    # Compute simplectic honeycomb coordinates relative to rhombohedral origin.
    xins = xs - xsb
    yins = ys - ysb
    zins = zs - zsb

    # Sum those together to get a value that determines which region we're in.
    in_sum = xins + yins + zins

    # Positions relative to origin point.
    dx0 = x - xb
    dy0 = y - yb
    dz0 = z - zb

    value = 0
    if in_sum <= 1:  # We're inside the tetrahedron (3-Simplex) at (0,0,0)

        # Determine which two of (0,0,1), (0,1,0), (1,0,0) are closest.
        a_point = 0x01
        a_score = xins
        b_point = 0x02
        b_score = yins
        if a_score >= b_score and zins > b_score:
            b_score = zins
            b_point = 0x04
        elif a_score < b_score and zins > a_score:
            a_score = zins
            a_point = 0x04

        # Now we determine the two lattice points not part of the tetrahedron that may contribute.
        # This depends on the closest two tetrahedral vertices, including (0,0,0)
        wins = 1 - in_sum
        if wins > a_score or wins > b_score:  # (0,0,0) is one of the closest two tetrahedral vertices.
            c = b_point if (b_score > a_score) else a_point  # Our other closest vertex is the closest out of a and b.

            if (c & 0x01) == 0:
                xsv_ext0 = xsb - 1
                xsv_ext1 = xsb
                dx_ext0 = dx0 + 1
                dx_ext1 = dx0
            else:
                xsv_ext0 = xsv_ext1 = xsb + 1
                dx_ext0 = dx_ext1 = dx0 - 1

            if (c & 0x02) == 0:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0
                if (c & 0x01) == 0:
                    ysv_ext1 -= 1
                    dy_ext1 += 1
                else:
                    ysv_ext0 -= 1
                    dy_ext0 += 1
            else:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1

            if (c & 0x04) == 0:
                zsv_ext0 = zsb
                zsv_ext1 = zsb - 1
                dz_ext0 = dz0
                dz_ext1 = dz0 + 1
            else:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz_ext1 = dz0 - 1
        else:  # (0,0,0) is not one of the closest two tetrahedral vertices.
            c = (a_point | b_point)  # Our two extra vertices are determined by the closest two.

            if (c & 0x01) == 0:
                xsv_ext0 = xsb
                xsv_ext1 = xsb - 1
                dx_ext0 = dx0 - 2 * SQUISH_CONSTANT3
                dx_ext1 = dx0 + 1 - SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb + 1
                dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT3
                dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3

            if (c & 0x02) == 0:
                ysv_ext0 = ysb
                ysv_ext1 = ysb - 1
                dy_ext0 = dy0 - 2 * SQUISH_CONSTANT3
                dy_ext1 = dy0 + 1 - SQUISH_CONSTANT3
            else:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT3
                dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3

            if (c & 0x04) == 0:
                zsv_ext0 = zsb
                zsv_ext1 = zsb - 1
                dz_ext0 = dz0 - 2 * SQUISH_CONSTANT3
                dz_ext1 = dz0 + 1 - SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT3
                dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3

        # Contribution (0,0,0)
        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * _extrapolate3(perm, perm_grad_index3, xsb + 0, ysb + 0, zsb + 0, dx0, dy0, dz0)

        # Contribution (1,0,0)
        dx1 = dx0 - 1 - SQUISH_CONSTANT3
        dy1 = dy0 - 0 - SQUISH_CONSTANT3
        dz1 = dz0 - 0 - SQUISH_CONSTANT3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * _extrapolate3(perm, perm_grad_index3, xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)

        # Contribution (0,1,0)
        dx2 = dx0 - 0 - SQUISH_CONSTANT3
        dy2 = dy0 - 1 - SQUISH_CONSTANT3
        dz2 = dz1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * _extrapolate3(perm, perm_grad_index3, xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)

        # Contribution (0,0,1)
        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * _extrapolate3(perm, perm_grad_index3, xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)
    elif in_sum >= 2:  # We're inside the tetrahedron (3-Simplex) at (1,1,1)

        # Determine which two tetrahedral vertices are the closest, out of (1,1,0), (1,0,1), (0,1,1) but not (1,1,1).
        a_point = 0x06
        a_score = xins
        b_point = 0x05
        b_score = yins
        if a_score <= b_score and zins < b_score:
            b_score = zins
            b_point = 0x03
        elif a_score > b_score and zins < a_score:
            a_score = zins
            a_point = 0x03

        # Now we determine the two lattice points not part of the tetrahedron that may contribute.
        # This depends on the closest two tetrahedral vertices, including (1,1,1)
        wins = 3 - in_sum
        if wins < a_score or wins < b_score:  # (1,1,1) is one of the closest two tetrahedral vertices.
            c = b_point if (b_score < a_score) else a_point  # Our other closest vertex is the closest out of a and b.

            if (c & 0x01) != 0:
                xsv_ext0 = xsb + 2
                xsv_ext1 = xsb + 1
                dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT3
                dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb
                dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT3

            if (c & 0x02) != 0:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT3
                if (c & 0x01) != 0:
                    ysv_ext1 += 1
                    dy_ext1 -= 1
                else:
                    ysv_ext0 += 1
                    dy_ext0 -= 1
            else:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT3

            if (c & 0x04) != 0:
                zsv_ext0 = zsb + 1
                zsv_ext1 = zsb + 2
                dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 - 3 * SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT3
        else:  # (1,1,1) is not one of the closest two tetrahedral vertices.
            c = (a_point & b_point)  # Our two extra vertices are determined by the closest two.

            if (c & 0x01) != 0:
                xsv_ext0 = xsb + 1
                xsv_ext1 = xsb + 2
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb
                dx_ext0 = dx0 - SQUISH_CONSTANT3
                dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3

            if (c & 0x02) != 0:
                ysv_ext0 = ysb + 1
                ysv_ext1 = ysb + 2
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy0 - SQUISH_CONSTANT3
                dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3

            if (c & 0x04) != 0:
                zsv_ext0 = zsb + 1
                zsv_ext1 = zsb + 2
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz0 - SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3

        # Contribution (1,1,0)
        dx3 = dx0 - 1 - 2 * SQUISH_CONSTANT3
        dy3 = dy0 - 1 - 2 * SQUISH_CONSTANT3
        dz3 = dz0 - 0 - 2 * SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * _extrapolate3(perm, perm_grad_index3, xsb + 1, ysb + 1, zsb + 0, dx3, dy3, dz3)

        # Contribution (1,0,1)
        dx2 = dx3
        dy2 = dy0 - 0 - 2 * SQUISH_CONSTANT3
        dz2 = dz0 - 1 - 2 * SQUISH_CONSTANT3
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * _extrapolate3(perm, perm_grad_index3, xsb + 1, ysb + 0, zsb + 1, dx2, dy2, dz2)

        # Contribution (0,1,1)
        dx1 = dx0 - 0 - 2 * SQUISH_CONSTANT3
        dy1 = dy3
        dz1 = dz2
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * _extrapolate3(perm, perm_grad_index3, xsb + 0, ysb + 1, zsb + 1, dx1, dy1, dz1)

        # Contribution (1,1,1)
        dx0 = dx0 - 1 - 3 * SQUISH_CONSTANT3
        dy0 = dy0 - 1 - 3 * SQUISH_CONSTANT3
        dz0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * _extrapolate3(perm, perm_grad_index3, xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0)
    else:  # We're inside the octahedron (Rectified 3-Simplex) in between.
        # Decide between point (0,0,1) and (1,1,0) as closest
        p1 = xins + yins
        if p1 > 1:
            a_score = p1 - 1
            a_point = 0x03
            a_is_further_side = True
        else:
            a_score = 1 - p1
            a_point = 0x04
            a_is_further_side = False

        # Decide between point (0,1,0) and (1,0,1) as closest
        p2 = xins + zins
        if p2 > 1:
            b_score = p2 - 1
            b_point = 0x05
            b_is_further_side = True
        else:
            b_score = 1 - p2
            b_point = 0x02
            b_is_further_side = False

        # The closest out of the two (1,0,0) and (0,1,1) will replace the furthest
        # out of the two decided above, if closer.
        p3 = yins + zins
        if p3 > 1:
            score = p3 - 1
            if a_score <= b_score and a_score < score:
                a_point = 0x06
                a_is_further_side = True
            elif a_score > b_score and b_score < score:
                b_point = 0x06
                b_is_further_side = True
        else:
            score = 1 - p3
            if a_score <= b_score and a_score < score:
                a_point = 0x01
                a_is_further_side = False
            elif a_score > b_score and b_score < score:
                b_point = 0x01
                b_is_further_side = False

        # Where each of the two closest points are determines how the extra two vertices are calculated.
        if a_is_further_side == b_is_further_side:
            if a_is_further_side:  # Both closest points on (1,1,1) side

                # One of the two extra points is (1,1,1)
                dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb + 1

                # Other extra point is based on the shared axis.
                c = (a_point & b_point)
                if (c & 0x01) != 0:
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 2
                    ysv_ext1 = ysb
                    zsv_ext1 = zsb
                elif (c & 0x02) != 0:
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb
                    ysv_ext1 = ysb + 2
                    zsv_ext1 = zsb
                else:
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb
                    ysv_ext1 = ysb
                    zsv_ext1 = zsb + 2
            else:  # Both closest points on (0,0,0) side

                # One of the two extra points is (0,0,0)
                dx_ext0 = dx0
                dy_ext0 = dy0
                dz_ext0 = dz0
                xsv_ext0 = xsb
                ysv_ext0 = ysb
                zsv_ext0 = zsb

                # Other extra point is based on the omitted axis.
                c = (a_point | b_point)
                if (c & 0x01) == 0:
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb - 1
                    ysv_ext1 = ysb + 1
                    zsv_ext1 = zsb + 1
                elif (c & 0x02) == 0:
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 + 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 1
                    ysv_ext1 = ysb - 1
                    zsv_ext1 = zsb + 1
                else:
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 + 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 1
                    ysv_ext1 = ysb + 1
                    zsv_ext1 = zsb - 1
        else:  # One point on (0,0,0) side, one point on (1,1,1) side
            if a_is_further_side:
                c1 = a_point
                c2 = b_point
            else:
                c1 = b_point
                c2 = a_point

            # One contribution is a _permutation of (1,1,-1)
            if (c1 & 0x01) == 0:
                dx_ext0 = dx0 + 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb - 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb + 1
            elif (c1 & 0x02) == 0:
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 + 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb - 1
                zsv_ext0 = zsb + 1
            else:
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 + 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb - 1

            # One contribution is a _permutation of (0,0,2)
            dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
            dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
            dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
            xsv_ext1 = xsb
            ysv_ext1 = ysb
            zsv_ext1 = zsb
            if (c2 & 0x01) != 0:
                dx_ext1 -= 2
                xsv_ext1 += 2
            elif (c2 & 0x02) != 0:
                dy_ext1 -= 2
                ysv_ext1 += 2
            else:
                dz_ext1 -= 2
                zsv_ext1 += 2

        # Contribution (1,0,0)
        dx1 = dx0 - 1 - SQUISH_CONSTANT3
        dy1 = dy0 - 0 - SQUISH_CONSTANT3
        dz1 = dz0 - 0 - SQUISH_CONSTANT3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * _extrapolate3(perm, perm_grad_index3, xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)

        # Contribution (0,1,0)
        dx2 = dx0 - 0 - SQUISH_CONSTANT3
        dy2 = dy0 - 1 - SQUISH_CONSTANT3
        dz2 = dz1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * _extrapolate3(perm, perm_grad_index3, xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)

        # Contribution (0,0,1)
        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * _extrapolate3(perm, perm_grad_index3, xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)

        # Contribution (1,1,0)
        dx4 = dx0 - 1 - 2 * SQUISH_CONSTANT3
        dy4 = dy0 - 1 - 2 * SQUISH_CONSTANT3
        dz4 = dz0 - 0 - 2 * SQUISH_CONSTANT3
        attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4
        if attn4 > 0:
            attn4 *= attn4
            value += attn4 * attn4 * _extrapolate3(perm, perm_grad_index3, xsb + 1, ysb + 1, zsb + 0, dx4, dy4, dz4)

        # Contribution (1,0,1)
        dx5 = dx4
        dy5 = dy0 - 0 - 2 * SQUISH_CONSTANT3
        dz5 = dz0 - 1 - 2 * SQUISH_CONSTANT3
        attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5
        if attn5 > 0:
            attn5 *= attn5
            value += attn5 * attn5 * _extrapolate3(perm, perm_grad_index3, xsb + 1, ysb + 0, zsb + 1, dx5, dy5, dz5)

        # Contribution (0,1,1)
        dx6 = dx0 - 0 - 2 * SQUISH_CONSTANT3
        dy6 = dy4
        dz6 = dz5
        attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6
        if attn6 > 0:
            attn6 *= attn6
            value += attn6 * attn6 * _extrapolate3(perm, perm_grad_index3, xsb + 0, ysb + 1, zsb + 1, dx6, dy6, dz6)

    # First extra vertex
    attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0
    if attn_ext0 > 0:
        attn_ext0 *= attn_ext0
        value += attn_ext0 * attn_ext0 * _extrapolate3(
                perm,
                perm_grad_index3,
                xsv_ext0,
                ysv_ext0,
                zsv_ext0,
                dx_ext0,
                dy_ext0,
                dz_ext0
                )

    # Second extra vertex
    attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1
    if attn_ext1 > 0:
        attn_ext1 *= attn_ext1
        value += attn_ext1 * attn_ext1 * _extrapolate3(
                perm,
                perm_grad_index3,
                xsv_ext1,
                ysv_ext1,
                zsv_ext1,
                dx_ext1,
                dy_ext1,
                dz_ext1
                )

    return value / NORM_CONSTANT3

#@njit(cache=True, parallel=True)
def _noise3a(X, Y, Z, perm, perm_grad_index3):
    noise = np.zeros((Z.size, Y.size, X.size), dtype=np.double)
    for z in prange(Z.size):
        for y in prange(Y.size):
            for x in prange(X.size):
                noise[z, y, x] = _noise3(X[x], Y[y], Z[z], perm, perm_grad_index3)
    return noise

#@njit(cache=True, parallel=True)
def _noise3b(X, Y, Z, perm, perm_grad_index3):
    noise = np.zeros(X.size * Y.size * Z.size, dtype=np.double)
    for z in prange(Z.size):
        for y in prange(Y.size):
            for x in prange(X.size):
                noise[(y * Y.size + x) + (z * Y.size * X.size)] = _noise3(X[x], Y[y], Z[z], perm, perm_grad_index3)
    return noise.reshape((Z.size, Y.size, X.size))

def _noise3aSlow(X, Y, T, FEATURE_SIZE, perm, perm_grad_index3):
    img = np.empty((T, X, Y), dtype=np.double)
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                img[t, x, y] = _noise3(x / FEATURE_SIZE, y / FEATURE_SIZE, t / FEATURE_SIZE, perm, perm_grad_index3)
    return img
```
## File: dataset\__init__.py
```python
# dataset package
```
## File: docs\PROJECT_SUMMARY.md
```markdown
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
```
## File: models\cms.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 out_features: int, drop: float = 0.):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SpatialGatingUnit(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(self.norm(x)))  # (B, N, 1)
        return x * gate


class CMS(nn.Module):
    """
    CV/CAD variant.

    Args:
        in_features:      token embedding dimension (e.g. 768 for ViT-B)
        hidden_features:  total hidden dim (split evenly across levels)
        out_features:     output dimension (defaults to in_features)
        drop:             dropout rate
        num_levels:       number of nested memory levels (paper default: 3)
        k:                update ratio base (paper default: 2)
        vit_layer_idx:    which ViT block this CMS lives in (set by
                          replace_mlp_with_cms); determines active levels
        use_spatial_gate: add SpatialGatingUnit at level 0 (recommended for CAD)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.,
        num_levels: int = 3,
        k: int = 2,
        vit_layer_idx: int = 0,
        use_spatial_gate: bool = True,
    ):
        super().__init__()
        out_features     = out_features or in_features
        hidden_features  = hidden_features or in_features

        self.num_levels     = num_levels
        self.in_features    = in_features
        self.out_features   = out_features
        self.k              = k
        self.vit_layer_idx  = vit_layer_idx

        level_hidden = max(hidden_features // num_levels, 1)

        self.levels = nn.ModuleList([
            MlpBlock(in_features, level_hidden, in_features, drop)
            for _ in range(num_levels)
        ])

        self.spatial_gate = SpatialGatingUnit(in_features) if use_spatial_gate else None

        self.output_proj = (
            nn.Linear(in_features, out_features)
            if out_features != in_features else None
        )

        self._active_levels = [
            i for i in range(num_levels)
            if vit_layer_idx % (k ** i) == 0
        ]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in self._active_levels:
            delta = self.levels[i](x)
            if i == 0 and self.spatial_gate is not None:
                delta = self.spatial_gate(delta)
            x = x + delta

        if self.output_proj is not None:
            x = self.output_proj(x)
        return x

    def extra_repr(self) -> str:
        active = self._active_levels
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"levels={self.num_levels}, k={self.k}, "
            f"vit_layer={self.vit_layer_idx}, active_levels={active}"
        )
```
## File: models\cnn_baseline.py
```python
"""
Simple CNN Baseline with Replay Buffer for Continual Learning
FIXED: ReplayBuffer logic to avoid Tensor comparison error
"""

import torch
import torch.nn as nn
import random
from typing import List, Tuple


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, hidden_dim=64, input_size=32):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature dimension dynamically
        feature_map_size = input_size // 8
        self.feature_dim = hidden_dim * 4 * feature_map_size * feature_map_size
        
        self.flatten = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes
        
    def forward(self, x): 
        x = self.features(x)
        x = self.flatten(x)
        x = self.projection(x)
        x = self.fc(x)
        return x
    
    def get_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size=1000, sampling_strategy='balanced'):
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        self.buffer = []
        
    def add_samples(self, images: torch.Tensor, labels: torch.Tensor, task_id: int):
        batch_size = images.shape[0]
        for i in range(batch_size):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((
                    images[i].cpu().clone(),
                    labels[i].cpu().clone(),
                    task_id
                ))
            else:
                idx = random.randint(0, self.buffer_size - 1)
                self.buffer[idx] = (
                    images[i].cpu().clone(),
                    labels[i].cpu().clone(),
                    task_id
                )
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if len(self.buffer) == 0:
            return None, None, None
        
        sample_size = min(batch_size, len(self.buffer))
        selected_indices = []

        if self.sampling_strategy == 'random':
            selected_indices = random.sample(range(len(self.buffer)), sample_size)
            
        elif self.sampling_strategy == 'balanced':
            # 1. Group indices by task_id
            indices_by_task = {}
            for i, (_, _, tid) in enumerate(self.buffer):
                if tid not in indices_by_task:
                    indices_by_task[tid] = []
                indices_by_task[tid].append(i)
            
            task_ids = list(indices_by_task.keys())
            
            # 2. Calculate how many samples per task
            per_task = sample_size // len(task_ids)
            remainder = sample_size % len(task_ids)
            
            # 3. Select indices for each task
            for tid in task_ids:
                available_indices = indices_by_task[tid]
                n_pick = per_task + (1 if remainder > 0 else 0)
                remainder -= 1
                
                # Pick random indices for this task
                if available_indices:
                    picked = random.sample(available_indices, min(n_pick, len(available_indices)))
                    selected_indices.extend(picked)
            
            # 4. Fill remaining spots if any (due to small task buffers)
            if len(selected_indices) < sample_size:
                all_indices = set(range(len(self.buffer)))
                used_indices = set(selected_indices)
                remaining_pool = list(all_indices - used_indices)
                
                needed = sample_size - len(selected_indices)
                if len(remaining_pool) >= needed:
                    extra_picks = random.sample(remaining_pool, needed)
                    selected_indices.extend(extra_picks)
        
        else:
            # Fallback to random
            selected_indices = random.sample(range(len(self.buffer)), sample_size)
        
        # Retrieve actual data using selected indices
        samples = [self.buffer[i] for i in selected_indices]
        
        # Unpack
        images = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        task_ids = [s[2] for s in samples]
        
        return images, labels, task_ids
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []


class CNN_Replay(nn.Module):
    def __init__(self, num_classes=10, buffer_size=1000, hidden_dim=64, input_size=32):
        super().__init__()
        
        self.cnn = SimpleCNN(
            num_classes=num_classes, 
            hidden_dim=hidden_dim,
            input_size=input_size
        )
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.num_classes = num_classes
        self.fc = self.cnn.fc
        
    def forward(self, x):
        return self.cnn(x)
    
    def get_features(self, x):
        return self.cnn.get_features(x)
    
    def add_to_buffer(self, images, labels, task_id):
        self.replay_buffer.add_samples(images, labels, task_id)
    
    def sample_from_buffer(self, batch_size):
        return self.replay_buffer.sample(batch_size)
        
    def get_buffer_size(self):
        return len(self.replay_buffer)
```
## File: models\dino_nsp2.py
```python
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .titans_memory import TITANSMemory


class ACCGatingHead(nn.Module):
    """Small embedding classifier used by ACC gating."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


class DinoNSP2(nn.Module):
    """
    DINO backbone wrapper with Visual Prompt Tuning and NSP2 instrumentation.

    Key features:
      - Frozen DINO/ViT backbone
      - Trainable visual prompts at early transformer blocks
      - ACC gating head for shift-vs-defect routing
      - Optional TITANS memory integration for Surprise Scalar inference
      - Extraction of ``Q_X W_k^T`` and ``S_P`` tensors for NSP2 covariance
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        img_size: int = 256,
        prompt_length: int = 8,
        prompt_layers: int = 4,
        prompt_dropout: float = 0.0,
        freeze_backbone: bool = True,
        gating_hidden_dim: int = 256,
        gating_dropout: float = 0.1,
        gating_threshold: float = 0.55,
        use_torchhub_fallback: bool = True,
        torchhub_model: str = "dinov2_vits14",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = bool(pretrained)
        self.img_size = int(img_size)
        self.prompt_length = int(prompt_length)
        self.prompt_layers = int(prompt_layers)
        self.gating_threshold = float(gating_threshold)

        self.backbone = self._build_backbone(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            use_torchhub_fallback=use_torchhub_fallback,
            torchhub_model=torchhub_model,
        )

        if not hasattr(self.backbone, "blocks"):
            raise RuntimeError("Backbone must expose transformer blocks via .blocks")

        self.embed_dim = int(getattr(self.backbone, "num_features"))
        num_blocks = len(self.backbone.blocks)
        self.num_prompt_layers = min(self.prompt_layers, num_blocks)

        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive")

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(self.num_prompt_layers, self.prompt_length, self.embed_dim)
        )
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        self.acc_gating = ACCGatingHead(
            embed_dim=self.embed_dim,
            hidden_dim=gating_hidden_dim,
            dropout=gating_dropout,
        )

        # Proxy anomaly head used for differentiable supervision.
        self.proxy_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
        )

        self._titans_memory: Optional[TITANSMemory] = None

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _build_backbone(
        self,
        model_name: str,
        pretrained: bool,
        img_size: int,
        use_torchhub_fallback: bool,
        torchhub_model: str,
    ) -> nn.Module:
        try:
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
            )
            return backbone
        except Exception as exc:
            if not use_torchhub_fallback:
                raise RuntimeError(f"Unable to load timm model '{model_name}': {exc}") from exc

            hub_model = torch.hub.load(
                "facebookresearch/dinov2",
                torchhub_model,
                pretrained=pretrained,
            )
            if not hasattr(hub_model, "blocks"):
                raise RuntimeError(
                    "TorchHub fallback model does not expose transformer blocks."
                )
            return hub_model

    def set_titans_memory(self, memory: Optional[TITANSMemory]) -> None:
        """Attach (or detach) TITANS memory used during forward routing."""
        self._titans_memory = memory

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.backbone.patch_embed(x)

        if hasattr(self.backbone, "_pos_embed"):
            tokens = self.backbone._pos_embed(patch_tokens)
        else:
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            tokens = torch.cat([cls_token, patch_tokens], dim=1)
            if hasattr(self.backbone, "pos_embed"):
                pos_embed = self.backbone.pos_embed[:, : tokens.shape[1], :]
                tokens = tokens + pos_embed
            if hasattr(self.backbone, "pos_drop"):
                tokens = self.backbone.pos_drop(tokens)

        if hasattr(self.backbone, "patch_drop"):
            tokens = self.backbone.patch_drop(tokens)
        if hasattr(self.backbone, "norm_pre"):
            tokens = self.backbone.norm_pre(tokens)

        return tokens

    def _extract_affinity_and_aggregation(
        self,
        block: nn.Module,
        tokens_with_prompts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract NSP2 tensors:
          - affinity: ``Q_X W_k^T`` approximation (B, N_image, P)
          - aggregation: prompt states ``S_P`` (B, P, C)
        """
        normed = block.norm1(tokens_with_prompts) if hasattr(block, "norm1") else tokens_with_prompts

        prompt_slice = slice(1, 1 + self.prompt_length)
        image_slice = slice(1 + self.prompt_length, normed.shape[1])

        if (
            hasattr(block, "attn")
            and hasattr(block.attn, "qkv")
            and callable(block.attn.qkv)
        ):
            qkv = block.attn.qkv(normed)
            batch_size, num_tokens, full_dim = qkv.shape
            num_heads = int(getattr(block.attn, "num_heads", 1))
            head_dim = full_dim // (3 * num_heads)

            qkv = qkv.reshape(batch_size, num_tokens, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]

            q_x = q[:, :, image_slice, :]
            k_p = k[:, :, prompt_slice, :]
            affinity = torch.einsum("bhid,bhjd->bij", q_x, k_p)
            affinity = affinity / math.sqrt(max(head_dim, 1))
        else:
            img_tokens = normed[:, image_slice, :]
            prompt_tokens = normed[:, prompt_slice, :]
            affinity = torch.matmul(img_tokens, prompt_tokens.transpose(1, 2))
            affinity = affinity / math.sqrt(max(normed.shape[-1], 1))

        aggregation = normed[:, prompt_slice, :]
        return affinity.detach(), aggregation.detach()

    def prompt_distribution_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return LayerNorm-based prompt distribution moments (mu, sigma)."""
        prompt_flat = self.prompt_embeddings.reshape(-1, self.embed_dim)
        normalized = F.layer_norm(prompt_flat, (self.embed_dim,))
        mu = normalized.mean()
        sigma = normalized.std(unbiased=False)
        return mu, sigma

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_head"):
            try:
                pooled = self.backbone.forward_head(tokens, pre_logits=True)
            except TypeError:
                pooled = self.backbone.forward_head(tokens)
            if pooled.ndim == 3:
                pooled = pooled[:, 0]
            return pooled

        if hasattr(self.backbone, "norm"):
            tokens = self.backbone.norm(tokens)

        return tokens[:, 0]

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        batch_size = x.shape[0]
        tokens = self._prepare_tokens(x)

        qxwk_mats: List[torch.Tensor] = []
        sp_mats: List[torch.Tensor] = []

        for layer_idx, block in enumerate(self.backbone.blocks):
            if layer_idx < self.num_prompt_layers:
                prompts = self.prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
                prompts = self.prompt_dropout(prompts)

                cls_token = tokens[:, :1, :]
                patch_tokens = tokens[:, 1:, :]
                tokens = torch.cat([cls_token, prompts, patch_tokens], dim=1)

                affinity, aggregation = self._extract_affinity_and_aggregation(block, tokens)
                qxwk_mats.append(affinity)
                sp_mats.append(aggregation)

            tokens = block(tokens)

            if layer_idx < self.num_prompt_layers:
                cls_token = tokens[:, :1, :]
                patch_tokens = tokens[:, 1 + self.prompt_length :, :]
                tokens = torch.cat([cls_token, patch_tokens], dim=1)

        if hasattr(self.backbone, "norm"):
            tokens = self.backbone.norm(tokens)

        embeddings = self._pool_tokens(tokens)

        acc_logits = self.acc_gating(embeddings)
        shift_probability = torch.softmax(acc_logits, dim=-1)[:, 1]

        proxy_logit = self.proxy_head(embeddings).squeeze(-1)
        proxy_score = torch.sigmoid(proxy_logit)

        if self._titans_memory is not None:
            surprise = self._titans_memory.compute_surprise(embeddings)
            route_to_titans = shift_probability <= self.gating_threshold
            image_score = torch.where(route_to_titans, surprise, torch.zeros_like(surprise))
            image_score = image_score.clamp(1e-6, 1 - 1e-6)
            image_logit = torch.logit(image_score)
        else:
            image_logit = proxy_logit
            image_score = proxy_score.clamp(1e-6, 1 - 1e-6)

        anomaly_map = image_score.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        prompt_mu, prompt_sigma = self.prompt_distribution_stats()

        return {
            "features": embeddings,
            "image_score": image_score,
            "image_logit": image_logit,
            "anomaly_map": anomaly_map,
            "patch_features": [],
            "acc_logits": acc_logits,
            "shift_probability": shift_probability,
            "proxy_logit": proxy_logit,
            "proxy_score": proxy_score,
            "qxwk_mats": qxwk_mats,
            "sp_mats": sp_mats,
            "prompt_mu": prompt_mu,
            "prompt_sigma": prompt_sigma,
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["features"]
```
## File: models\titans_memory.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TitansMatchStats:
    """Container for TITANS nearest-neighbor matching diagnostics."""

    topk_similarity: torch.Tensor
    novelty: torch.Tensor
    surprise: torch.Tensor


class TITANSMemory(nn.Module):
    """
    TITANS-style memory matcher for normal embedding prototypes.

    The module stores a rolling memory bank of normal embeddings and computes a
    ``Surprise Scalar`` per sample from nearest-neighbor mismatch.

    Surprise is defined from novelty:
      novelty = 1 - mean(top-k cosine similarity)
      surprise = clamp(0.5 * novelty, 0, 1)

    A value near 0 means highly familiar (likely normal), and near 1 means
    highly surprising (likely anomalous).
    """

    def __init__(
        self,
        embedding_dim: int,
        bank_size: int = 8192,
        k_neighbors: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if bank_size <= 0:
            raise ValueError("bank_size must be > 0")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")

        self.embedding_dim = int(embedding_dim)
        self.bank_size = int(bank_size)
        self.k_neighbors = int(k_neighbors)
        self.eps = float(eps)

        self.register_buffer("bank", torch.zeros(bank_size, embedding_dim))
        self.register_buffer("occupied", torch.zeros(bank_size, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("running_center", torch.zeros(embedding_dim))
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

    @property
    def num_items(self) -> int:
        """Number of valid items currently stored in the memory bank."""
        return int(self.occupied.sum().item())

    def clear(self) -> None:
        """Reset memory bank content."""
        with torch.no_grad():
            self.bank.zero_()
            self.occupied.zero_()
            self.write_ptr.zero_()
            self.running_center.zero_()
            self.num_updates.zero_()

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor) -> None:
        """
        Insert new normal embeddings into TITANS memory.

        Args:
            embeddings: Tensor with shape ``(B, D)``.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (B, D)")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        if embeddings.numel() == 0:
            return

        normed = F.normalize(embeddings.detach(), dim=-1, eps=self.eps)
        batch_size = normed.shape[0]

        ptr = int(self.write_ptr.item())
        for row in range(batch_size):
            self.bank[ptr].copy_(normed[row])
            self.occupied[ptr] = True
            ptr = (ptr + 1) % self.bank_size

        self.write_ptr[0] = ptr

        batch_center = normed.mean(dim=0)
        updates = int(self.num_updates.item())
        momentum = 0.995 if updates > 0 else 0.0
        self.running_center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)
        self.num_updates[0] = updates + 1

    def _valid_bank(self) -> torch.Tensor:
        valid = self.bank[self.occupied]
        if valid.numel() == 0:
            return torch.empty(0, self.embedding_dim, device=self.bank.device)
        return valid

    @torch.no_grad()
    def match(self, embeddings: torch.Tensor) -> TitansMatchStats:
        """
        Match embeddings against memory and return novelty diagnostics.

        Args:
            embeddings: Tensor with shape ``(B, D)``.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (B, D)")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        normed = F.normalize(embeddings.detach(), dim=-1, eps=self.eps)
        valid_bank = self._valid_bank()

        if valid_bank.shape[0] == 0:
            novelty = torch.ones(normed.shape[0], device=normed.device)
            surprise = novelty.clone()
            empty_topk = torch.zeros(normed.shape[0], 1, device=normed.device)
            return TitansMatchStats(
                topk_similarity=empty_topk,
                novelty=novelty,
                surprise=surprise,
            )

        similarity = normed @ valid_bank.t()
        k = min(self.k_neighbors, similarity.shape[1])
        topk_similarity, _ = torch.topk(similarity, k=k, dim=1, largest=True)

        novelty = 1.0 - topk_similarity.mean(dim=1)
        surprise = (0.5 * novelty).clamp_(0.0, 1.0)

        return TitansMatchStats(
            topk_similarity=topk_similarity,
            novelty=novelty,
            surprise=surprise,
        )

    @torch.no_grad()
    def compute_surprise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute only the scalar surprise score in ``[0, 1]``."""
        return self.match(embeddings).surprise

    @torch.no_grad()
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`compute_surprise` for module-style usage."""
        return self.compute_surprise(embeddings)

    def state_summary(self) -> Dict[str, float]:
        """Return lightweight state metadata for logging."""
        return {
            "num_items": float(self.num_items),
            "bank_size": float(self.bank_size),
            "k_neighbors": float(self.k_neighbors),
        }

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, bank_size={self.bank_size}, "
            f"k_neighbors={self.k_neighbors}, num_items={self.num_items}"
        )
```
## File: models\vit_cms.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple, Dict
from .cms import CMS
from .cnn_baseline import ReplayBuffer

def replace_mlp_with_cms(
    model: nn.Module,
    num_levels: int = 3,
    k: int = 2,
    use_spatial_gate: bool = True,
    verbose: bool = False,
) -> nn.Module:
  
    if hasattr(model, 'blocks'):
        for layer_idx, block in enumerate(model.blocks):
            for attr_name in ['mlp', 'ffn']:
                child = getattr(block, attr_name, None)
                if child is None:
                    continue
                if not hasattr(child, 'fc1'):
                    continue

                in_feat     = child.fc1.in_features
                hidden_feat = child.fc1.out_features
                out_feat    = child.fc2.out_features
                drop_rate   = 0.0
                for attr in ['drop', 'drop1']:
                    d = getattr(child, attr, None)
                    if d is not None and hasattr(d, 'p'):
                        drop_rate = d.p
                        break

                cms = CMS(
                    in_features=in_feat,
                    hidden_features=hidden_feat,
                    out_features=out_feat,
                    drop=drop_rate,
                    num_levels=num_levels,
                    k=k,
                    vit_layer_idx=layer_idx,
                    use_spatial_gate=use_spatial_gate,
                )
                setattr(block, attr_name, cms)
                if verbose:
                    print(f"  Replaced block[{layer_idx}].{attr_name} → {cms}")
        return model

    def _replace_recursive(module: nn.Module, depth: int = 0):
        for name, child in module.named_children():
            if child.__class__.__name__ in ('Mlp', 'MlpBlock', 'FeedForward'):
                if not hasattr(child, 'fc1'):
                    continue
                in_feat     = child.fc1.in_features
                hidden_feat = child.fc1.out_features
                out_feat    = child.fc2.out_features
                drop_rate   = 0.0
                for attr in ['drop', 'drop1']:
                    d = getattr(child, attr, None)
                    if d is not None and hasattr(d, 'p'):
                        drop_rate = d.p
                        break
                cms = CMS(
                    in_features=in_feat,
                    hidden_features=hidden_feat,
                    out_features=out_feat,
                    drop=drop_rate,
                    num_levels=num_levels,
                    k=k,
                    vit_layer_idx=depth,
                    use_spatial_gate=use_spatial_gate,
                )
                setattr(module, name, cms)
                if verbose:
                    print(f"  Replaced {name} at depth {depth} → {cms}")
            else:
                _replace_recursive(child, depth + 1)

    _replace_recursive(model)
    return model

class AnomalyDecoder(nn.Module):
    """
    Fuses multi-scale patch-token features into a single spatial anomaly map.

    Input:  list of (B, N_patches, C) tensors from different ViT layers
    Output: (B, 1, H_out, W_out) anomaly score map

    Architecture:
      Per-scale: 1×1 conv to reduce channels → bilinear upsample to target_size
      Fusion:    concat → 3×3 conv → sigmoid
    """
    def __init__(
        self,
        embed_dim: int,
        num_scales: int,
        patch_size: int = 16,
        img_size: int = 256,
        reduced_dim: int = 128,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.img_size    = img_size
        self.num_patches = img_size // patch_size   # spatial side length

        # Per-scale channel reduction
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, reduced_dim, kernel_size=1)
            for _ in range(num_scales)
        ])

        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_dim * num_scales, reduced_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(reduced_dim, 1, kernel_size=1),
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        scale_features: list of (B, N, C) where N = num_patches²
        returns: (B, 1, img_size, img_size)
        """
        B = scale_features[0].shape[0]
        maps = []
        for feat, proj in zip(scale_features, self.scale_projs):
            # (B, N, C) → (B, C, H_p, W_p)
            feat_2d = feat.reshape(B, self.num_patches, self.num_patches, -1)
            feat_2d = feat_2d.permute(0, 3, 1, 2).contiguous()
            feat_2d = proj(feat_2d)
            # upsample to full image resolution
            feat_2d = F.interpolate(
                feat_2d, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            )
            maps.append(feat_2d)

        fused = torch.cat(maps, dim=1)   # (B, reduced_dim*num_scales, H, W)
        return self.fusion(fused)        # (B, 1, H, W)


class ViT_CMS(nn.Module):
    """

    Args:
        model_name:       timm model identifier
        pretrained:       load ImageNet weights
        cms_levels:       number of CMS levels (default 3)
        k:                CMS update ratio base (default 2)
        extract_layers:   which ViT block indices to tap for multi-scale features
        img_size:         input image resolution (must match backbone config)
        use_spatial_gate: enable SpatialGatingUnit in CMS Level 0
        freeze_backbone:  freeze ALL backbone weights (only train decoder)
        freeze_patch_embed: freeze patch embedding only
    """

    # Standard ViT-B/16 settings as defaults; override via config
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_256',
        pretrained: bool = True,
        cms_levels: int = 3,
        k: int = 2,
        extract_layers: List[int] = (3, 6, 9),
        img_size: int = 256,
        use_spatial_gate: bool = True,
        freeze_backbone: bool = False,
        freeze_patch_embed: bool = False,
        reduced_dim: int = 128,
    ):
        super().__init__()
        self.extract_layers = list(extract_layers)
        self.img_size = img_size

        print(f"[ViT_CMS] Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, img_size=img_size
        )
        self.embed_dim  = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]

        print(f"[ViT_CMS] Replacing MLP → CMS (levels={cms_levels}, k={k}, "
              f"spatial_gate={use_spatial_gate})...")
        self.backbone = replace_mlp_with_cms(
            self.backbone,
            num_levels=cms_levels,
            k=k,
            use_spatial_gate=use_spatial_gate,
            verbose=False,
        )

        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._hooks = []
        if hasattr(self.backbone, 'blocks'):
            for idx in self.extract_layers:
                hook = self.backbone.blocks[idx].register_forward_hook(
                    self._make_hook(idx)
                )
                self._hooks.append(hook)
        else:
            print("[ViT_CMS] WARNING: backbone has no .blocks — "
                  "multi-scale extraction disabled.")

        self.decoder = AnomalyDecoder(
            embed_dim=self.embed_dim,
            num_scales=len(self.extract_layers),
            patch_size=self.patch_size,
            img_size=img_size,
            reduced_dim=reduced_dim,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("[ViT_CMS] Backbone frozen.")
        elif freeze_patch_embed:
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad_(False)
            print("[ViT_CMS] Patch embedding frozen.")

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._hook_outputs[idx] = output[:, 1:, :]
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._hook_outputs.clear()

        cls_features = self.backbone(x)

        scale_feats = [
            self._hook_outputs[idx]
            for idx in self.extract_layers
            if idx in self._hook_outputs
        ]

        if scale_feats:
            anomaly_map = self.decoder(scale_feats)          # (B, 1, H, W)
            anomaly_map = torch.sigmoid(anomaly_map)
        else:
            score = cls_features.mean(dim=-1, keepdim=True)  # (B, 1)
            anomaly_map = score[:, :, None, None].expand(
                -1, 1, self.img_size, self.img_size
            )

        image_score = anomaly_map.mean(dim=(1, 2, 3))        # (B,)
        image_logit = torch.logit(image_score.clamp(1e-6, 1 - 1e-6))

        return {
            'anomaly_map':    anomaly_map,    # (B, 1, H, W) — for Pixel-AP
            'image_score':    image_score,    # (B,)          — for AUROC
            'image_logit':    image_logit,    # (B,)          — for BCEWithLogits
            'features':       cls_features,   # (B, C)        — for KD loss
            'patch_features': scale_feats,    # list[(B,N,C)] — for memory bank
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['features']


class ViT_Replay(ViT_CMS):
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_256',
        pretrained: bool = True,
        num_classes: int = 10,
        buffer_size: int = 500,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs,
        )
        self.num_classes = num_classes
        self.replay_buffer = ReplayBuffer(buffer_size)

    def add_to_buffer(self, x, y, task_id=None):
        self.replay_buffer.add_data(x, y, task_id)

    def sample_from_buffer(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def get_buffer_size(self):
        return len(self.replay_buffer)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


class ViT_Simple(nn.Module):
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_256',
        pretrained: bool = True,
        img_size: int = 256,
        extract_layers: List[int] = (3, 6, 9),
        reduced_dim: int = 128,
    ):
        super().__init__()
        self.extract_layers = list(extract_layers)
        self.img_size = img_size

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, img_size=img_size
        )
        self.embed_dim  = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]

        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._hooks = []
        for idx in self.extract_layers:
            h = self.backbone.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(h)

        self.decoder = AnomalyDecoder(
            embed_dim=self.embed_dim,
            num_scales=len(self.extract_layers),
            patch_size=self.patch_size,
            img_size=img_size,
            reduced_dim=reduced_dim,
        )

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._hook_outputs[idx] = output[:, 1:, :]
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._hook_outputs.clear()
        cls_features = self.backbone(x)
        scale_feats = [self._hook_outputs[i] for i in self.extract_layers
                       if i in self._hook_outputs]
        anomaly_map = torch.sigmoid(self.decoder(scale_feats))
        return {
            'anomaly_map':    anomaly_map,
            'image_score':    anomaly_map.mean(dim=(1, 2, 3)),
            'features':       cls_features,
            'patch_features': scale_feats,
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['features']
```
## File: models\__init__.py
```python
"""
Models for Continual Learning Experiments
"""

from .dino_nsp2 import ACCGatingHead, DinoNSP2
from .titans_memory import TITANSMemory
from .cms import CMS, MlpBlock
from .vit_cms import ViT_CMS, ViT_Simple, ViT_Replay
from .cnn_baseline import SimpleCNN, CNN_Replay, ReplayBuffer

__all__ = [
    'ACCGatingHead',
    'DinoNSP2',
    'TITANSMemory',
    'CMS',
    'MlpBlock',
    'ViT_CMS',
    'ViT_Simple',
    'ViT_Replay',
    'SimpleCNN',
    'CNN_Replay',
    'ReplayBuffer'
]
```
## File: papers\CADIC_Continual_Anomaly_Detection_Based_on_Incremental_Coreset.pdf
```pdf
Error reading papers\CADIC_Continual_Anomaly_Detection_Based_on_Incremental_Coreset.pdf: 'utf-8' codec can't decode byte 0xbf in position 10: invalid start byte
```
## File: papers\CAD_Fundamental.pdf
```pdf
Error reading papers\CAD_Fundamental.pdf: 'utf-8' codec can't decode byte 0x8f in position 10: invalid start byte
```
## File: papers\Nested_Learning_The_Illusion_of_Deep_Learning_Architectures.pdf
```pdf
Error reading papers\Nested_Learning_The_Illusion_of_Deep_Learning_Architectures.pdf: 'utf-8' codec can't decode byte 0xbf in position 10: invalid start byte
```
## File: papers\ReplayCAD_Generative_Diffusion_Replay_for_Continual_Anomaly_Detection.pdf
```pdf
Error reading papers\ReplayCAD_Generative_Diffusion_Replay_for_Continual_Anomaly_Detection.pdf: 'utf-8' codec can't decode byte 0x8f in position 10: invalid start byte
```
## File: papers\Visual Prompt Tuning in Null Space for Continual Learning.pdf
```pdf
Error reading papers\Visual Prompt Tuning in Null Space for Continual Learning.pdf: 'utf-8' codec can't decode byte 0x8f in position 10: invalid start byte
```
## File: scripts\01_data_preparation.py
```python
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Data pipeline verification for anomaly dataset setup")
    parser.add_argument(
        "--config",
        type=str,
        default="./conf/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--run_verify",
        action="store_true",
        help="Run pipeline verification instead of printing legacy smoke-test notice",
    )
    return parser.parse_args()

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def verify_data_pipeline(config_path="./conf/config.yaml"):
    logger.info("Starting checking data pipeline")

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Loading config file is not successfully: {e}")
        return
    manager = ContinualStreamingManager(config)
    random.shuffle(manager.categories)
    
    train_loader, test_loader, task_info = manager.get_next_task()
    
    if train_loader is None:
        logger.error("DataLoader is not existing")
        return

    logger.info(f"Checking: {task_info['category'].upper()}")
    
    batch = next(iter(train_loader))
    images = batch['img']
    masks = batch['img_mask']
    labels = batch['anomaly']
    
    logger.info(f"Batch Image Shape: {images.shape}")
    logger.info(f"Batch Mask Shape:  {masks.shape}")
    logger.info(f"Batch Labels:      {labels.tolist()}")

    anomaly_idx = (labels == 1).nonzero(as_tuple=True)[0]
    normal_idx = (labels == 0).nonzero(as_tuple=True)[0]

    plot_idx = []
    if len(anomaly_idx) > 0:
        plot_idx.extend(anomaly_idx[:2].tolist())
    if len(normal_idx) > 0:
        plot_idx.extend(normal_idx[:2].tolist())

    if not plot_idx:
        logger.warning("Empty batch")
        return

    num_plots = len(plot_idx)
    fig, axes = plt.subplots(num_plots, 2, figsize=(8, 4 * num_plots))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "eda")
    os.makedirs(out_dir, exist_ok=True)

    for i, idx in enumerate(plot_idx):
        img_rgb = denormalize(images[idx])
        mask_np = masks[idx].squeeze(0).cpu().numpy()
        lbl = "Anomaly" if labels[idx].item() == 1 else "Normal"

        ax_img = axes[i, 0] if num_plots > 1 else axes[0]
        ax_mask = axes[i, 1] if num_plots > 1 else axes[1]

        ax_img.imshow(img_rgb)
        ax_img.set_title(f"Image - {lbl}")
        ax_img.axis('off')

        ax_mask.imshow(mask_np, cmap='gray')
        ax_mask.set_title(f"Mask - {lbl}")
        ax_mask.axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"pipeline_verify_{task_info['category']}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved images at: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    if args.run_verify:
        verify_data_pipeline(config_path=args.config)
    else:
        logger.info("Smoke-test entrypoint removed from this pipeline script.")
        logger.info("Run with --run_verify to execute the pipeline verification flow.")
```
## File: training\cms_optim.py
```python
import torch
from torch.optim import Optimizer

class CMSOptimizerWrapper:
    def __init__(self, optimizer: Optimizer, model: torch.nn.Module, k_factor: int = 5):
        self.optimizer = optimizer
        self.model = model
        self.k_factor = k_factor
        self.global_step = 0
        self.param_levels = self._map_params_to_levels()

    def _map_params_to_levels(self):
        level_map = {}
        for name, param in self.model.named_parameters():
            level = 0 
            parts = name.split('.')
            for part in parts:
                if part.startswith('level_'):
                    try:
                        level = int(part.split('_')[1])
                    except ValueError:
                        pass
            
            level_map[param] = level
        return level_map

    def step(self):
        self.global_step += 1
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                level = self.param_levels.get(p, 0)
                update_freq = self.k_factor ** level
                if self.global_step % update_freq != 0:
                    p.grad = None

        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
```
## File: training\evaluator.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support


class Evaluator:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            images = batch['img']
            labels = batch['anomaly']
            masks = batch.get('img_mask')
            return images, labels, masks

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
            return images, labels, None

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _is_anomaly_output(self, outputs: Any) -> bool:
        return isinstance(outputs, dict) and (
            'image_score' in outputs or 'image_logit' in outputs
        )
        
    @torch.no_grad()
    def evaluate_task(
        self,
        test_loader: DataLoader,
        task_id: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        all_scores = []
        all_pixel_preds = []
        all_pixel_labels = []
        
        active_classes = getattr(test_loader.dataset, 'task_classes', None)
        
        pbar = tqdm(test_loader, desc=f"Evaluating Task {task_id}") if verbose else test_loader
        
        for batch in pbar:
            images, labels, masks = self._unpack_batch(batch)
            images, labels = images.to(self.device), labels.to(self.device)
            if torch.is_tensor(masks):
                masks = masks.to(self.device)
            
            outputs = self.model(images)

            if self._is_anomaly_output(outputs):
                labels_float = labels.float().view(-1)
                image_logit = outputs.get('image_logit', None)
                image_score = outputs.get('image_score', None)

                if image_logit is not None:
                    image_logit = image_logit.view(-1)
                    loss = F.binary_cross_entropy_with_logits(image_logit, labels_float)
                    score_prob = torch.sigmoid(image_logit)
                elif image_score is not None:
                    score_prob = image_score.view(-1).clamp(1e-6, 1 - 1e-6)
                    loss = F.binary_cross_entropy(score_prob, labels_float)
                else:
                    raise ValueError("Anomaly output must provide image_score or image_logit")

                predicted = (score_prob >= 0.5).long()
                correct = predicted.eq(labels.long().view(-1)).sum().item()

                all_scores.extend(score_prob.detach().cpu().numpy().tolist())
                all_predictions.extend(predicted.detach().cpu().numpy().tolist())
                all_labels.extend(labels.long().view(-1).detach().cpu().numpy().tolist())

                anomaly_map = outputs.get('anomaly_map', None)
                if torch.is_tensor(masks) and torch.is_tensor(anomaly_map):
                    target_masks = masks.float().clamp(0, 1)
                    if target_masks.ndim == 3:
                        target_masks = target_masks[:, None, :, :]
                    if target_masks.shape[-2:] != anomaly_map.shape[-2:]:
                        target_masks = F.interpolate(
                            target_masks,
                            size=anomaly_map.shape[-2:],
                            mode='nearest',
                        )

                    all_pixel_preds.extend(anomaly_map.detach().cpu().reshape(-1).numpy().tolist())
                    all_pixel_labels.extend(target_masks.detach().cpu().reshape(-1).numpy().tolist())
            else:
                logits = outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs
                if active_classes is not None:
                    mask = torch.full_like(logits, float('-inf'))
                    mask[:, active_classes] = 0
                    logits = logits + mask

                loss = self.criterion(logits, labels)
                _, predicted = logits.max(1)
                correct = predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / labels.size(0):.2f}%'
                })

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
        
        avg_loss = total_loss / max(len(test_loader), 1)
        accuracy = 100. * total_correct / total_samples if total_samples > 0 else 0
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='binary' if np.unique(all_labels).size <= 2 else 'macro',
            zero_division=0,
        )

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision_scores * 100,
            'recall': recall_scores * 100,
            'f1': f1_scores * 100,
        }

        if len(all_scores) > 1 and np.unique(all_labels).size > 1:
            metrics['image_auroc'] = float(roc_auc_score(all_labels, np.array(all_scores))) * 100
            metrics['image_ap'] = float(average_precision_score(all_labels, np.array(all_scores))) * 100
        else:
            metrics['image_auroc'] = 0.0
            metrics['image_ap'] = 0.0

        if len(all_pixel_preds) > 1 and np.unique(np.array(all_pixel_labels)).size > 1:
            metrics['pixel_ap'] = float(
                average_precision_score(np.array(all_pixel_labels), np.array(all_pixel_preds))
            ) * 100
        else:
            metrics['pixel_ap'] = 0.0

        return metrics
    
    @torch.no_grad()
    def evaluate_all_tasks(
        self,
        test_loaders: List[DataLoader],
        verbose: bool = True
    ) -> Dict[int, Dict[str, float]]:
        results = {}
        
        for task_id, test_loader in enumerate(test_loaders):
            metrics = self.evaluate_task(test_loader, task_id, verbose=verbose)
            results[task_id] = metrics
            
            if verbose:
                print(
                    f"Task {task_id}: "
                    f"Acc={metrics['accuracy']:.2f}% "
                    f"F1={metrics['f1']:.2f}% "
                    f"AUROC={metrics['image_auroc']:.2f}% "
                    f"Pixel-AP={metrics['pixel_ap']:.2f}%"
                )
        
        # Calculate average metrics
        if results:
            avg_metrics = {
                'avg_accuracy': np.mean([m['accuracy'] for m in results.values()]),
                'avg_f1': np.mean([m['f1'] for m in results.values()]),
                'avg_loss': np.mean([m['loss'] for m in results.values()]),
                'avg_image_auroc': np.mean([m['image_auroc'] for m in results.values()]),
                'avg_image_ap': np.mean([m['image_ap'] for m in results.values()]),
                'avg_pixel_ap': np.mean([m['pixel_ap'] for m in results.values()]),
            }
        else:
            avg_metrics = {
                'avg_accuracy': 0,
                'avg_f1': 0,
                'avg_loss': 0,
                'avg_image_auroc': 0,
                'avg_image_ap': 0,
                'avg_pixel_ap': 0,
            }
        
        if verbose:
            print(
                f"\nAverage: "
                f"Acc={avg_metrics['avg_accuracy']:.2f}% "
                f"F1={avg_metrics['avg_f1']:.2f}% "
                f"AUROC={avg_metrics['avg_image_auroc']:.2f}% "
                f"Pixel-AP={avg_metrics['avg_pixel_ap']:.2f}%"
            )
        
        results['average'] = avg_metrics
        
        return results
    
    @torch.no_grad()
    def calculate_forgetting(
        self,
        test_loaders: List[DataLoader],
        baseline_accuracies: Dict[int, float]
    ) -> Dict[str, float]:
        current_results = self.evaluate_all_tasks(test_loaders, verbose=False)
        
        forgetting = {}
        for task_id in baseline_accuracies:
            if task_id in current_results:
                forgetting[task_id] = max(0, baseline_accuracies[task_id] - current_results[task_id]['accuracy'])
        
        avg_forgetting = np.mean(list(forgetting.values())) if forgetting else 0.0
        
        return {
            'per_task_forgetting': forgetting,
            'average_forgetting': avg_forgetting
        }
```
## File: training\memory_buffer.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SlowMemoryEntry:
    """Single memory entry for slow normal-buffer retention."""

    image: torch.Tensor
    embedding: torch.Tensor
    utility: float
    metadata: Optional[Dict[str, Any]] = None


class SlowMemory:
    """
    Diversity-aware normal image buffer using utility-based pruning.

    Utility score follows a k-center intuition: samples that are far from the
    current memory manifold receive higher utility and are preferred.
    """

    def __init__(
        self,
        capacity: int,
        embedding_dim: int,
        utility_temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        self.capacity = int(capacity)
        self.embedding_dim = int(embedding_dim)
        self.utility_temperature = float(utility_temperature)
        self.eps = float(eps)
        self._entries: List[SlowMemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[SlowMemoryEntry]:
        return self._entries

    def clear(self) -> None:
        """Remove all buffer entries."""
        self._entries.clear()

    def _stack_embeddings(self) -> torch.Tensor:
        if not self._entries:
            return torch.empty(0, self.embedding_dim)
        emb = torch.stack([entry.embedding for entry in self._entries], dim=0)
        return F.normalize(emb, dim=-1, eps=self.eps)

    def utility_score(self, embedding: torch.Tensor) -> float:
        """
        Compute utility for one candidate embedding.

        Higher score means the sample adds more geometric diversity.
        """
        if embedding.ndim != 1:
            raise ValueError("embedding must have shape (D,)")
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embedding.shape[0]}"
            )

        emb = F.normalize(embedding, dim=-1, eps=self.eps)
        existing = self._stack_embeddings()

        if existing.numel() == 0:
            return 1.0

        cosine = torch.matmul(existing, emb)
        min_cos = float(cosine.max().item())
        novelty = 1.0 - min_cos
        scaled = novelty / max(self.utility_temperature, self.eps)
        return float(scaled)

    def add_batch(
        self,
        images: torch.Tensor,
        embeddings: torch.Tensor,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Add a batch of normal samples and keep only top-K diverse entries.

        Args:
            images: Tensor ``(B, C, H, W)``.
            embeddings: Tensor ``(B, D)``.
            metadata: Optional list of length B with auxiliary fields.
        """
        if images.ndim < 2:
            raise ValueError("images must have batch dimension")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (B, D)")
        if images.shape[0] != embeddings.shape[0]:
            raise ValueError("images and embeddings must share batch size")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        batch_size = images.shape[0]
        if metadata is None:
            metadata = [None] * batch_size
        if len(metadata) != batch_size:
            raise ValueError("metadata length must match batch size")

        with torch.no_grad():
            normed_embeddings = F.normalize(embeddings.detach().cpu(), dim=-1, eps=self.eps)
            images_cpu = images.detach().cpu()

            for idx in range(batch_size):
                emb = normed_embeddings[idx]
                util = self.utility_score(emb)
                self._entries.append(
                    SlowMemoryEntry(
                        image=images_cpu[idx],
                        embedding=emb,
                        utility=util,
                        metadata=metadata[idx],
                    )
                )

            self._prune_to_capacity()

    def _kcenter_select(self, embeddings: torch.Tensor, k: int) -> List[int]:
        """
        Select representative indices via k-center greedy on cosine distance.
        """
        num_items = embeddings.shape[0]
        if k >= num_items:
            return list(range(num_items))

        selected: List[int] = []
        first_idx = int(torch.argmax(torch.norm(embeddings, dim=1)).item())
        selected.append(first_idx)

        min_dist = torch.full((num_items,), float("inf"))

        for _ in range(1, k):
            last_vec = embeddings[selected[-1]].unsqueeze(0)
            cosine = F.cosine_similarity(embeddings, last_vec.expand_as(embeddings), dim=1)
            distance = 1.0 - cosine
            min_dist = torch.minimum(min_dist, distance)
            next_idx = int(torch.argmax(min_dist).item())
            if next_idx in selected:
                break
            selected.append(next_idx)

        if len(selected) < k:
            for idx in range(num_items):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) == k:
                    break

        return selected[:k]

    def _recompute_utility(self) -> None:
        if not self._entries:
            return

        embeddings = self._stack_embeddings()
        if embeddings.shape[0] == 1:
            self._entries[0].utility = 1.0
            return

        sims = embeddings @ embeddings.t()
        sims.fill_diagonal_(-1.0)
        nearest = sims.max(dim=1).values
        utilities = (1.0 - nearest).clamp_min(0.0)

        for idx, util in enumerate(utilities.tolist()):
            self._entries[idx].utility = float(util)

    def _prune_to_capacity(self) -> None:
        if len(self._entries) <= self.capacity:
            self._recompute_utility()
            return

        embeddings = self._stack_embeddings()
        keep_indices = self._kcenter_select(embeddings, self.capacity)
        keep_set = set(keep_indices)

        self._entries = [
            entry for idx, entry in enumerate(self._entries) if idx in keep_set
        ]

        self._recompute_utility()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Randomly sample entries from slow memory.

        Returns:
            Dict with ``images``, ``embeddings``, and ``utility`` tensors.
        """
        if len(self._entries) == 0:
            return {
                "images": torch.empty(0),
                "embeddings": torch.empty(0, self.embedding_dim),
                "utility": torch.empty(0),
            }

        k = min(int(batch_size), len(self._entries))
        indices = torch.randperm(len(self._entries))[:k].tolist()

        images = torch.stack([self._entries[idx].image for idx in indices], dim=0)
        embeddings = torch.stack([self._entries[idx].embedding for idx in indices], dim=0)
        utility = torch.tensor([self._entries[idx].utility for idx in indices], dtype=torch.float32)

        return {
            "images": images,
            "embeddings": embeddings,
            "utility": utility,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serializable state for checkpointing."""
        return {
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "utility_temperature": self.utility_temperature,
            "entries": [
                {
                    "image": entry.image,
                    "embedding": entry.embedding,
                    "utility": entry.utility,
                    "metadata": entry.metadata,
                }
                for entry in self._entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state created by :meth:`state_dict`."""
        self.capacity = int(state["capacity"])
        self.embedding_dim = int(state["embedding_dim"])
        self.utility_temperature = float(state.get("utility_temperature", 1.0))

        self._entries = []
        for item in state.get("entries", []):
            self._entries.append(
                SlowMemoryEntry(
                    image=item["image"],
                    embedding=item["embedding"],
                    utility=float(item["utility"]),
                    metadata=item.get("metadata"),
                )
            )
```
## File: training\nsp2_optim.py
```python
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


class ContinualBackpropMonitor:
    """
    Continual Backpropagation (CBP) helper.

    Tracks near-zero activations and re-initializes persistently dead neurons.
    """

    def __init__(
        self,
        model: nn.Module,
        module_name_filters: Optional[Sequence[str]] = None,
        patience: int = 50,
        activation_threshold: float = 1e-5,
    ) -> None:
        self.model = model
        self.module_name_filters = tuple(module_name_filters or ())
        self.patience = int(patience)
        self.activation_threshold = float(activation_threshold)

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._tracked_modules: Dict[str, nn.Linear] = {}
        self._dead_counters: Dict[str, torch.Tensor] = {}

        self._register_hooks()

    def _should_track(self, module_name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if not self.module_name_filters:
            return True
        lowered = module_name.lower()
        return any(token.lower() in lowered for token in self.module_name_filters)

    def _register_hooks(self) -> None:
        for module_name, module in self.model.named_modules():
            if not self._should_track(module_name, module):
                continue

            self._tracked_modules[module_name] = module
            self._dead_counters[module_name] = torch.zeros(
                module.out_features,
                dtype=torch.long,
                device=module.weight.device,
            )
            handle = module.register_forward_hook(self._make_hook(module_name))
            self._handles.append(handle)

    def _make_hook(self, module_name: str):
        def hook(module: nn.Module, _inputs, output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                return
            if output.ndim < 2:
                return

            reduce_dims = tuple(range(output.ndim - 1))
            activity = output.detach().abs().mean(dim=reduce_dims)
            dead_mask = activity <= self.activation_threshold

            counters = self._dead_counters[module_name]
            counters[dead_mask] += 1
            counters[~dead_mask] = 0

        return hook

    @torch.no_grad()
    def reinitialize_dead_neurons(self) -> Dict[str, int]:
        """Re-initialize dead neuron rows and reset their counters."""
        refreshed: Dict[str, int] = {}

        for module_name, module in self._tracked_modules.items():
            counters = self._dead_counters[module_name]
            dead_indices = torch.nonzero(counters >= self.patience, as_tuple=False).flatten()
            if dead_indices.numel() == 0:
                continue

            fan_in = max(module.in_features, 1)
            bound = 1.0 / math.sqrt(fan_in)
            module.weight[dead_indices].uniform_(-bound, bound)
            if module.bias is not None:
                module.bias[dead_indices].zero_()

            counters[dead_indices] = 0
            refreshed[module_name] = int(dead_indices.numel())

        return refreshed

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: counter.detach().cpu()
            for name, counter in self._dead_counters.items()
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for name, counter in state_dict.items():
            if name not in self._dead_counters:
                continue
            self._dead_counters[name] = counter.to(self._dead_counters[name].device)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class NSP2Optimizer:
    """
    Optimizer wrapper implementing NSP2 gradient projection and CBP.

    Mathematical mapping used here for prompt gradient ``P_G``:
      - Prompt parameter is stored as ``(P, C)``
      - We project on transposed gradient ``G^T`` with shape ``(C, P)``
      - ``B1`` is derived from affinity covariance ``C1`` over prompt axis ``P``
      - ``B2`` is derived from aggregation covariance ``C2`` over channel axis ``C``
      - Update becomes ``Delta P = (B2 @ G^T @ B1)^T``
    """

    def __init__(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        prompt_param_names: Sequence[str] = ("prompt_embeddings",),
        svd_tol: float = 1e-6,
        svd_rel_tol: float = 1e-4,
        cbp_patience: int = 50,
        cbp_activation_threshold: float = 1e-5,
        cbp_module_filters: Optional[Sequence[str]] = None,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.prompt_param_names = tuple(prompt_param_names)
        self.svd_tol = float(svd_tol)
        self.svd_rel_tol = float(svd_rel_tol)

        self.prompt_params: List[nn.Parameter] = self._collect_prompt_params()
        if not self.prompt_params:
            raise RuntimeError(
                "No prompt parameters found. Check prompt_param_names and model structure."
            )

        sample_prompt = self.prompt_params[0]
        if sample_prompt.ndim < 2:
            raise RuntimeError("Prompt parameter must be at least 2D.")

        self.prompt_len = int(sample_prompt.shape[-2])
        self.embed_dim = int(sample_prompt.shape[-1])
        self.device = sample_prompt.device

        self.cov_affinity_global = torch.zeros(self.prompt_len, self.prompt_len, device=self.device)
        self.cov_aggregation_global = torch.zeros(self.embed_dim, self.embed_dim, device=self.device)
        self.cov_affinity_task = torch.zeros_like(self.cov_affinity_global)
        self.cov_aggregation_task = torch.zeros_like(self.cov_aggregation_global)

        self.B1 = torch.eye(self.prompt_len, device=self.device)
        self.B2 = torch.eye(self.embed_dim, device=self.device)
        self._projection_dirty = True

        self.cbp = ContinualBackpropMonitor(
            model=model,
            module_name_filters=cbp_module_filters,
            patience=cbp_patience,
            activation_threshold=cbp_activation_threshold,
        )

    def _collect_prompt_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in self.prompt_param_names):
                params.append(param)
        return params

    @torch.no_grad()
    def accumulate_covariances(
        self,
        qxwk_mats: Iterable[torch.Tensor],
        sp_mats: Iterable[torch.Tensor],
    ) -> None:
        """
        Accumulate uncentered covariance matrices from model forward tensors.

        Args:
            qxwk_mats: Sequence of ``Q_X W_k^T`` tensors with trailing dim ``P``.
            sp_mats: Sequence of ``S_P`` tensors with trailing dim ``C``.
        """
        for qxwk in qxwk_mats:
            if qxwk.numel() == 0:
                continue
            q_flat = qxwk.detach().to(self.device).reshape(-1, qxwk.shape[-1])
            if q_flat.shape[1] != self.prompt_len:
                continue
            self.cov_affinity_task += q_flat.t() @ q_flat

        for sp in sp_mats:
            if sp.numel() == 0:
                continue
            s_flat = sp.detach().to(self.device).reshape(-1, sp.shape[-1])
            if s_flat.shape[1] != self.embed_dim:
                continue
            self.cov_aggregation_task += s_flat.t() @ s_flat

        self._projection_dirty = True

    def _null_projection(self, covariance: torch.Tensor) -> torch.Tensor:
        if torch.count_nonzero(covariance).item() == 0:
            return torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)

        _, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
        right_vectors = vh.transpose(-2, -1)

        max_sv = float(singular_values.max().item())
        threshold = max(self.svd_tol, self.svd_rel_tol * max_sv)
        null_mask = singular_values <= threshold

        if not bool(null_mask.any()):
            min_index = int(torch.argmin(singular_values).item())
            null_mask[min_index] = True

        basis = right_vectors[:, null_mask]
        projection = basis @ basis.t()
        projection = 0.5 * (projection + projection.t())
        return projection

    @torch.no_grad()
    def refresh_projections(self) -> None:
        """Update NSP2 projection matrices from global task covariances."""
        self.B1 = self._null_projection(self.cov_affinity_global)
        self.B2 = self._null_projection(self.cov_aggregation_global)
        self._projection_dirty = False

    @torch.no_grad()
    def project_prompt_gradients(self) -> None:
        """Project prompt gradients: ``Delta P = B2 * P_G * B1``."""
        if self._projection_dirty:
            self.refresh_projections()

        for param in self.prompt_params:
            if param.grad is None:
                continue

            grad = param.grad
            if grad.ndim == 2:
                projected_t = self.B2 @ grad.t() @ self.B1
                grad.copy_(projected_t.t())
                continue

            if grad.ndim >= 3:
                grad_view = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                for idx in range(grad_view.shape[0]):
                    projected_t = self.B2 @ grad_view[idx].t() @ self.B1
                    grad_view[idx].copy_(projected_t.t())

    @torch.no_grad()
    def step(self) -> Dict[str, int]:
        """Apply NSP2 projection, optimizer update, then CBP re-initialization."""
        self.project_prompt_gradients()
        self.optimizer.step()
        return self.cbp.reinitialize_dead_neurons()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def finalize_task(self, task_id: int, save_dir: Optional[str] = None) -> None:
        """
        Merge task covariances into the stability memory and persist artifacts.
        """
        self.cov_affinity_global += self.cov_affinity_task
        self.cov_aggregation_global += self.cov_aggregation_task

        self.refresh_projections()

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "task_id": int(task_id),
                    "cov_affinity": self.cov_affinity_task.detach().cpu(),
                    "cov_aggregation": self.cov_aggregation_task.detach().cpu(),
                    "B1": self.B1.detach().cpu(),
                    "B2": self.B2.detach().cpu(),
                },
                save_path / f"cov_task_{int(task_id):02d}.pt",
            )

        self.cov_affinity_task.zero_()
        self.cov_aggregation_task.zero_()

    def state_dict(self) -> Dict[str, object]:
        return {
            "base_optimizer": self.optimizer.state_dict(),
            "cov_affinity_global": self.cov_affinity_global.detach().cpu(),
            "cov_aggregation_global": self.cov_aggregation_global.detach().cpu(),
            "cov_affinity_task": self.cov_affinity_task.detach().cpu(),
            "cov_aggregation_task": self.cov_aggregation_task.detach().cpu(),
            "B1": self.B1.detach().cpu(),
            "B2": self.B2.detach().cpu(),
            "cbp": self.cbp.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.optimizer.load_state_dict(state_dict["base_optimizer"])  # type: ignore[index]

        self.cov_affinity_global = state_dict["cov_affinity_global"].to(self.device)  # type: ignore[index]
        self.cov_aggregation_global = state_dict["cov_aggregation_global"].to(self.device)  # type: ignore[index]
        self.cov_affinity_task = state_dict["cov_affinity_task"].to(self.device)  # type: ignore[index]
        self.cov_aggregation_task = state_dict["cov_aggregation_task"].to(self.device)  # type: ignore[index]
        self.B1 = state_dict["B1"].to(self.device)  # type: ignore[index]
        self.B2 = state_dict["B2"].to(self.device)  # type: ignore[index]

        cbp_state = state_dict.get("cbp", {})
        if isinstance(cbp_state, dict):
            self.cbp.load_state_dict(cbp_state)

        self._projection_dirty = False

    def close(self) -> None:
        self.cbp.close()
```
## File: training\run_experiment.py
```python
import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager
from models import DinoNSP2
from training import Evaluator, Trainer
from utils.global_seed import set_seed


def apply_profile(config: Dict[str, Any], profile: Optional[str]) -> Dict[str, Any]:
    if not profile:
        return config

    profile = profile.lower().strip()
    if profile in ("default", "full"):
        return config

    if profile == "tiny":
        config.setdefault("dataset", {})["batch_size"] = 4
        config.setdefault("dataset", {})["num_workers"] = 0
        config.setdefault("training", {})["epochs_per_task"] = 1
        return config

    if profile == "quick":
        config.setdefault("dataset", {})["batch_size"] = 8
        config.setdefault("dataset", {})["num_workers"] = 2
        config.setdefault("training", {})["epochs_per_task"] = min(
            int(config.get("training", {}).get("epochs_per_task", 2)),
            2,
        )
        return config

    raise ValueError(f"Unsupported profile: {profile}")


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return "cpu"
    return requested_device


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config.get("model", {})

    return DinoNSP2(
        model_name=str(model_cfg.get("backbone", "vit_small_patch14_dinov2.lvd142m")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        img_size=int(config.get("dataset", {}).get("img_size", 256)),
        prompt_length=int(model_cfg.get("prompt_length", max(2, int(model_cfg.get("k", 2)) * 4))),
        prompt_layers=int(model_cfg.get("prompt_layers", model_cfg.get("cms_levels", 3))),
        prompt_dropout=float(model_cfg.get("prompt_dropout", 0.0)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        gating_hidden_dim=int(model_cfg.get("gating_hidden_dim", 256)),
        gating_dropout=float(model_cfg.get("gating_dropout", 0.1)),
        gating_threshold=float(model_cfg.get("gating_threshold", 0.55)),
        use_torchhub_fallback=bool(model_cfg.get("use_torchhub_fallback", True)),
        torchhub_model=str(model_cfg.get("torchhub_model", "dinov2_vits14")),
    )


def maybe_init_wandb(
    config: Dict[str, Any],
    run_dir: Path,
    disable_wandb: bool,
):
    logging_cfg = config.get("logging", {})
    use_wandb = bool(logging_cfg.get("use_wandb", False)) and not disable_wandb
    if not use_wandb:
        return None

    try:
        import wandb

        return wandb.init(
            project=logging_cfg.get("wandb_project", "nested-learning-for-cad"),
            entity=logging_cfg.get("wandb_entity", None),
            name=run_dir.name,
            config=config,
            dir=str(run_dir),
        )
    except Exception as exc:
        print(f"W&B initialization failed, continuing without W&B: {exc}")
        return None


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def _mean_metric(task_metrics: list, key: str) -> float:
    values = []
    for item in task_metrics:
        eval_metrics = item.get("eval", {})
        if key in eval_metrics:
            values.append(float(eval_metrics[key]))
    return float(np.mean(values)) if values else 0.0


def run_experiment(
    config: Dict[str, Any],
    *,
    config_path: str,
    profile: Optional[str] = None,
    run_name: Optional[str] = None,
    max_tasks: Optional[int] = None,
    epochs_override: Optional[int] = None,
    device_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    disable_wandb: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg = apply_profile(cfg, profile)

    if epochs_override is not None:
        cfg.setdefault("training", {})["epochs_per_task"] = int(epochs_override)
    if device_override is not None:
        cfg.setdefault("training", {})["device"] = device_override
    if seed_override is not None:
        cfg.setdefault("training", {})["seed"] = int(seed_override)

    training_cfg = cfg.setdefault("training", {})
    logging_cfg = cfg.setdefault("logging", {})

    requested_device = str(training_cfg.get("device", "cuda"))
    device = resolve_device(requested_device)
    training_cfg["device"] = device

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    results_root = Path(logging_cfg.get("results_dir", "results"))
    runs_subdir = str(logging_cfg.get("runs_subdir", "runs"))
    run_root = results_root / runs_subdir
    run_root.mkdir(parents=True, exist_ok=True)

    experiment_name = run_name or str(logging_cfg.get("experiment_name", "experiment"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    wandb_run = maybe_init_wandb(cfg, run_dir, disable_wandb)

    model = build_model(cfg)
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        use_replay=bool(training_cfg.get("use_replay", False)),
        replay_batch_size=int(training_cfg.get("replay_batch_size", 32)),
        task_type=str(training_cfg.get("task_type", "anomaly")),
        pixel_loss_weight=float(training_cfg.get("pixel_loss_weight", 0.2)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        acc_loss_weight=float(training_cfg.get("acc_loss_weight", 0.5)),
        proxy_loss_weight=float(training_cfg.get("proxy_loss_weight", 0.5)),
        ln_loss_weight=float(training_cfg.get("ln_loss_weight", 0.1)),
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
        titans_bank_size=int(training_cfg.get("titans_bank_size", 8192)),
        titans_k_neighbors=int(training_cfg.get("titans_k_neighbors", 8)),
        slow_memory_size=int(training_cfg.get("slow_memory_size", 2048)),
        nsp2_svd_tol=float(training_cfg.get("nsp2_svd_tol", 1e-6)),
        nsp2_svd_rel_tol=float(training_cfg.get("nsp2_svd_rel_tol", 1e-4)),
        cbp_patience=int(training_cfg.get("cbp_patience", 50)),
        cbp_activation_threshold=float(training_cfg.get("cbp_activation_threshold", 1e-5)),
        covariance_dir=(run_dir / "covariances").as_posix(),
    )
    evaluator = Evaluator(model=model, device=device)

    manager = ContinualStreamingManager(cfg)
    epochs_per_task = int(training_cfg.get("epochs_per_task", 1))
    save_models = bool(logging_cfg.get("save_models", True))

    checkpoints_dir = run_dir / "checkpoints"
    if save_models:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    task_metrics = []
    completed_tasks = 0

    while True:
        if max_tasks is not None and completed_tasks >= int(max_tasks):
            break

        train_loader, test_loader, task_info = manager.get_next_task()
        if train_loader is None:
            break

        task_id = int(task_info["task_id"])
        category = task_info.get("category", f"task_{task_id}")

        if verbose:
            print(f"\n=== Task {task_id}: {category} ===")

        train_metrics = trainer.train_task(
            train_loader=train_loader,
            task_id=task_id,
            epochs=epochs_per_task,
            verbose=verbose,
        )
        eval_metrics = evaluator.evaluate_task(
            test_loader=test_loader,
            task_id=task_id,
            verbose=verbose,
        )

        current_task_metrics = {
            "task_id": task_id,
            "category": category,
            "train": _to_builtin(train_metrics),
            "eval": _to_builtin(eval_metrics),
        }
        task_metrics.append(current_task_metrics)

        if wandb_run is not None:
            wandb_payload = {
                "task_id": task_id,
                "train/loss": float(train_metrics.get("loss", 0.0)),
                "train/accuracy": float(train_metrics.get("accuracy", 0.0)),
                "train/image_loss": float(train_metrics.get("image_loss", 0.0)),
                "train/pixel_loss": float(train_metrics.get("pixel_loss", 0.0)),
            }
            for metric_name, metric_value in eval_metrics.items():
                wandb_payload[f"eval/{metric_name}"] = float(metric_value)
            wandb_run.log(wandb_payload)

        if save_models:
            checkpoint_path = checkpoints_dir / f"task_{task_id:02d}.pt"
            torch.save(
                {
                    "task_id": task_id,
                    "category": category,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "config": cfg,
                },
                checkpoint_path,
            )

        completed_tasks += 1

    with open(run_dir / "task_metrics.json", "w", encoding="utf-8") as f:
        json.dump(task_metrics, f, indent=2)

    summary = {
        "experiment_name": experiment_name,
        "profile": profile,
        "config_path": config_path,
        "run_dir": run_dir.as_posix(),
        "device": device,
        "seed": seed,
        "num_tasks": completed_tasks,
        "aggregate": {
            "avg_accuracy": _mean_metric(task_metrics, "accuracy"),
            "avg_f1": _mean_metric(task_metrics, "f1"),
            "avg_image_auroc": _mean_metric(task_metrics, "image_auroc"),
            "avg_image_ap": _mean_metric(task_metrics, "image_ap"),
            "avg_pixel_ap": _mean_metric(task_metrics, "pixel_ap"),
        },
    }

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()

    print("\nRun complete")
    print(f"Run directory: {run_dir.as_posix()}")
    print(json.dumps(summary["aggregate"], indent=2))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a continual anomaly experiment")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to YAML config")
    parser.add_argument("--profile", default=None, choices=["tiny", "quick", "default", "full"], help="Quick profile override")
    parser.add_argument("--run_name", default=None, help="Optional run name prefix")
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per task")
    parser.add_argument("--device", default=None, help="Override training device")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B even if enabled in config")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars and per-step logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)

    run_experiment(
        config=config,
        config_path=config_path,
        profile=args.profile,
        run_name=args.run_name,
        max_tasks=args.max_tasks,
        epochs_override=args.epochs,
        device_override=args.device,
        seed_override=args.seed,
        disable_wandb=args.disable_wandb,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
```
## File: training\run_sweep.py
```python
import argparse
import copy
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.config import load_config
from training.run_experiment import run_experiment


def _as_int_list(values: List[int]) -> List[int]:
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for continual anomaly experiments")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to YAML config")
    parser.add_argument("--profile", default=None, choices=["tiny", "quick", "default", "full"], help="Optional profile")
    parser.add_argument("--run_prefix", default="sweep", help="Prefix for generated run names")

    parser.add_argument("--max_tasks", type=int, default=None, help="Limit tasks per run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per task")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B for all runs")
    parser.add_argument("--quiet", action="store_true", help="Disable per-task verbose output")

    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seed list")
    parser.add_argument("--generators", nargs="*", default=None, help="Anomaly generator list")
    parser.add_argument("--cms_levels", type=int, nargs="*", default=None, help="CMS level list")
    parser.add_argument("--k_values", type=int, nargs="*", default=None, help="CMS k list")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config: Dict[str, Any] = load_config(args.config)

    dataset_cfg = base_config.setdefault("dataset", {})
    model_cfg = base_config.setdefault("model", {})
    training_cfg = base_config.setdefault("training", {})
    logging_cfg = base_config.setdefault("logging", {})

    seeds = _as_int_list(args.seeds) if args.seeds else [int(training_cfg.get("seed", 42))]
    generators = args.generators if args.generators else [str(dataset_cfg.get("anomaly_generator", "superpixel"))]
    cms_levels = _as_int_list(args.cms_levels) if args.cms_levels else [int(model_cfg.get("cms_levels", 3))]
    k_values = _as_int_list(args.k_values) if args.k_values else [int(model_cfg.get("k", 2))]

    combinations = list(itertools.product(seeds, generators, cms_levels, k_values))
    total_runs = len(combinations)
    if total_runs == 0:
        raise RuntimeError("Sweep produced no run combinations.")

    results_dir = Path(logging_cfg.get("results_dir", "results"))
    sweep_root = results_dir / "sweeps"
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_name = f"{args.run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = sweep_root / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=False)

    with open(sweep_dir / "sweep_plan.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_runs": total_runs,
                "seeds": seeds,
                "generators": generators,
                "cms_levels": cms_levels,
                "k_values": k_values,
            },
            f,
            indent=2,
        )

    summaries = []

    for idx, (seed, generator, cms_level, k_value) in enumerate(combinations, start=1):
        run_config = copy.deepcopy(base_config)
        run_config.setdefault("dataset", {})["anomaly_generator"] = generator
        run_config.setdefault("model", {})["cms_levels"] = cms_level
        run_config.setdefault("model", {})["k"] = k_value
        run_config.setdefault("training", {})["seed"] = seed

        run_name = f"{args.run_prefix}_s{seed}_g{generator}_l{cms_level}_k{k_value}"
        print(f"\n[{idx}/{total_runs}] Running {run_name}")

        summary = run_experiment(
            config=run_config,
            config_path=args.config,
            profile=args.profile,
            run_name=run_name,
            max_tasks=args.max_tasks,
            epochs_override=args.epochs,
            device_override=args.device,
            seed_override=seed,
            disable_wandb=args.disable_wandb,
            verbose=not args.quiet,
        )

        summary_record = {
            "run_name": run_name,
            "seed": seed,
            "generator": generator,
            "cms_levels": cms_level,
            "k": k_value,
            "run_dir": summary.get("run_dir"),
            "num_tasks": summary.get("num_tasks", 0),
            "aggregate": summary.get("aggregate", {}),
        }
        summaries.append(summary_record)

        with open(sweep_dir / "sweep_results.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

    def _best(metric_name: str, higher_is_better: bool = True):
        metric_runs = [
            r for r in summaries
            if metric_name in r.get("aggregate", {})
        ]
        if not metric_runs:
            return None
        return sorted(
            metric_runs,
            key=lambda r: r["aggregate"][metric_name],
            reverse=higher_is_better,
        )[0]

    final_summary = {
        "sweep_name": sweep_name,
        "sweep_dir": sweep_dir.as_posix(),
        "num_runs": len(summaries),
        "best_by_image_auroc": _best("avg_image_auroc", higher_is_better=True),
        "best_by_pixel_ap": _best("avg_pixel_ap", higher_is_better=True),
    }

    with open(sweep_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    with open(sweep_dir / "sweep_config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_config, f, sort_keys=False)

    print("\nSweep complete")
    print(f"Sweep directory: {sweep_dir.as_posix()}")


if __name__ == "__main__":
    main()
```
## File: training\trainer.py
```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.titans_memory import TITANSMemory
from .memory_buffer import SlowMemory
from .nsp2_optim import NSP2Optimizer


class Trainer:
    """
    Continual anomaly trainer with ACC gating, TITANS memory and NSP2 updates.

    Backward components:
      - NSP2 null-space projected prompt gradients
      - CBP dead-neuron re-initialization via optimizer wrapper
      - Slow memory utility buffer for high-diversity normal samples
      - LayerNorm prompt drift regularization
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        optimizer: Optional[optim.Optimizer] = None,
        learning_rate: float = 1e-4,
        use_replay: bool = False,
        replay_batch_size: int = 32,
        task_type: str = "anomaly",
        pixel_loss_weight: float = 0.2,
        weight_decay: float = 1e-5,
        acc_loss_weight: float = 0.5,
        proxy_loss_weight: float = 0.5,
        ln_loss_weight: float = 0.1,
        gradient_clip_norm: float = 1.0,
        titans_bank_size: int = 8192,
        titans_k_neighbors: int = 8,
        slow_memory_size: int = 2048,
        nsp2_svd_tol: float = 1e-6,
        nsp2_svd_rel_tol: float = 1e-4,
        cbp_patience: int = 50,
        cbp_activation_threshold: float = 1e-5,
        covariance_dir: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device

        # Legacy args kept for compatibility with old entry points.
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        self.task_type = task_type

        self.pixel_loss_weight = float(pixel_loss_weight)
        self.acc_loss_weight = float(acc_loss_weight)
        self.proxy_loss_weight = float(proxy_loss_weight)
        self.ln_loss_weight = float(ln_loss_weight)
        self.gradient_clip_norm = float(gradient_clip_norm)

        self.image_criterion = nn.BCELoss()
        self.proxy_criterion = nn.BCEWithLogitsLoss()
        self.acc_criterion = nn.CrossEntropyLoss()

        embedding_dim = int(getattr(self.model, "embed_dim", 768))

        self.titans_memory = TITANSMemory(
            embedding_dim=embedding_dim,
            bank_size=titans_bank_size,
            k_neighbors=titans_k_neighbors,
        ).to(device)

        if hasattr(self.model, "set_titans_memory"):
            self.model.set_titans_memory(self.titans_memory)
        else:
            setattr(self.model, "_titans_memory", self.titans_memory)

        self.slow_memory = SlowMemory(
            capacity=slow_memory_size,
            embedding_dim=embedding_dim,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        base_optimizer = optimizer
        if base_optimizer is None:
            base_optimizer = optim.AdamW(
                trainable_params,
                lr=float(learning_rate),
                weight_decay=float(weight_decay),
            )

        self.optimizer = NSP2Optimizer(
            optimizer=base_optimizer,
            model=self.model,
            prompt_param_names=("prompt_embeddings",),
            svd_tol=nsp2_svd_tol,
            svd_rel_tol=nsp2_svd_rel_tol,
            cbp_patience=cbp_patience,
            cbp_activation_threshold=cbp_activation_threshold,
            cbp_module_filters=("acc_gating", "proxy_head"),
        )

        self.prev_prompt_mu: Optional[torch.Tensor] = None
        self.prev_prompt_sigma: Optional[torch.Tensor] = None

        self.covariance_dir = Path(covariance_dir) if covariance_dir else None

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            images = batch["img"]
            labels = batch["anomaly"]
            masks = batch.get("img_mask")
            return images, labels, masks

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
            return images, labels, None

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _acc_targets(self, labels: torch.Tensor) -> torch.Tensor:
        # ACC class 1: shift/normal accepted as new normal.
        # ACC class 0: suspected defect.
        labels = labels.long().view(-1)
        return torch.where(labels == 0, torch.ones_like(labels), torch.zeros_like(labels))

    def _ln_drift_loss(self, prompt_mu: torch.Tensor, prompt_sigma: torch.Tensor) -> torch.Tensor:
        if self.prev_prompt_mu is None or self.prev_prompt_sigma is None:
            return torch.zeros((), device=self.device)

        return (
            torch.abs(prompt_mu - self.prev_prompt_mu).mean()
            + torch.abs(prompt_sigma - self.prev_prompt_sigma).mean()
        )

    @torch.no_grad()
    def _update_prompt_reference(self, prompt_mu: torch.Tensor, prompt_sigma: torch.Tensor) -> None:
        self.prev_prompt_mu = prompt_mu.detach()
        self.prev_prompt_sigma = prompt_sigma.detach()

    @torch.no_grad()
    def _update_memories(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        outputs: Dict[str, Any],
    ) -> None:
        labels = labels.long().view(-1)
        normal_mask = labels == 0

        if not torch.any(normal_mask):
            return

        shift_probability = outputs.get("shift_probability", None)
        if not torch.is_tensor(shift_probability):
            accept_mask = normal_mask
        else:
            threshold = float(getattr(self.model, "gating_threshold", 0.55))
            accept_mask = normal_mask & (shift_probability.detach() > threshold)

        if not torch.any(accept_mask):
            return

        embeddings = outputs["features"][accept_mask]
        selected_images = images[accept_mask]

        self.titans_memory.update(embeddings)
        self.slow_memory.add_batch(
            images=selected_images,
            embeddings=embeddings.detach().cpu(),
        )

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, float]:
        self.model.train()

        history = {
            "loss": [],
            "accuracy": [],
            "image_loss": [],
            "pixel_loss": [],
            "acc_loss": [],
            "proxy_loss": [],
            "ln_loss": [],
        }

        if verbose:
            print(f"Training task {task_id} for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            epoch_image_loss = 0.0
            epoch_pixel_loss = 0.0
            epoch_acc_loss = 0.0
            epoch_proxy_loss = 0.0
            epoch_ln_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch + 1}/{epochs}") if verbose else train_loader

            for batch in pbar:
                images, labels, masks = self._unpack_batch(batch)
                images = images.to(self.device)
                labels = labels.to(self.device)
                if torch.is_tensor(masks):
                    masks = masks.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)

                self.optimizer.accumulate_covariances(
                    qxwk_mats=outputs.get("qxwk_mats", []),
                    sp_mats=outputs.get("sp_mats", []),
                )

                labels_float = labels.float().view(-1)
                image_score = outputs["image_score"].view(-1).clamp(1e-6, 1 - 1e-6)
                proxy_logit = outputs["proxy_logit"].view(-1)
                acc_logits = outputs["acc_logits"]
                prompt_mu = outputs["prompt_mu"]
                prompt_sigma = outputs["prompt_sigma"]

                image_loss = self.image_criterion(image_score, labels_float)
                proxy_loss = self.proxy_criterion(proxy_logit, labels_float)
                acc_loss = self.acc_criterion(acc_logits, self._acc_targets(labels))
                ln_loss = self._ln_drift_loss(prompt_mu, prompt_sigma)

                total_loss = (
                    image_loss
                    + self.proxy_loss_weight * proxy_loss
                    + self.acc_loss_weight * acc_loss
                    + self.ln_loss_weight * ln_loss
                )

                pixel_loss_value = torch.zeros((), device=self.device)
                anomaly_map = outputs.get("anomaly_map", None)
                if torch.is_tensor(masks) and torch.is_tensor(anomaly_map):
                    target_masks = masks.float().clamp(0.0, 1.0)
                    if target_masks.ndim == 3:
                        target_masks = target_masks[:, None, :, :]
                    if target_masks.shape[-2:] != anomaly_map.shape[-2:]:
                        target_masks = F.interpolate(
                            target_masks,
                            size=anomaly_map.shape[-2:],
                            mode="nearest",
                        )

                    pixel_loss_value = F.binary_cross_entropy(
                        anomaly_map.clamp(1e-6, 1 - 1e-6),
                        target_masks,
                    )
                    total_loss = total_loss + self.pixel_loss_weight * pixel_loss_value

                total_loss.backward()

                trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clip_norm)

                cbp_refreshed = self.optimizer.step()
                _ = cbp_refreshed

                self._update_prompt_reference(prompt_mu, prompt_sigma)
                self._update_memories(images=images, labels=labels, outputs=outputs)

                pred = (image_score >= 0.5).long()
                epoch_correct += pred.eq(labels.long().view(-1)).sum().item()
                epoch_total += labels.shape[0]

                epoch_loss += float(total_loss.item())
                epoch_image_loss += float(image_loss.item())
                epoch_pixel_loss += float(pixel_loss_value.item())
                epoch_acc_loss += float(acc_loss.item())
                epoch_proxy_loss += float(proxy_loss.item())
                epoch_ln_loss += float(ln_loss.item())

                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        {
                            "loss": f"{total_loss.item():.4f}",
                            "acc": f"{100.0 * epoch_correct / max(epoch_total, 1):.2f}%",
                        }
                    )

            batches = max(len(train_loader), 1)
            avg_loss = epoch_loss / batches
            avg_acc = 100.0 * epoch_correct / max(epoch_total, 1)

            history["loss"].append(avg_loss)
            history["accuracy"].append(avg_acc)
            history["image_loss"].append(epoch_image_loss / batches)
            history["pixel_loss"].append(epoch_pixel_loss / batches)
            history["acc_loss"].append(epoch_acc_loss / batches)
            history["proxy_loss"].append(epoch_proxy_loss / batches)
            history["ln_loss"].append(epoch_ln_loss / batches)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {avg_loss:.4f} - "
                    f"Acc: {avg_acc:.2f}%"
                )

        self.optimizer.finalize_task(
            task_id=task_id,
            save_dir=str(self.covariance_dir) if self.covariance_dir is not None else None,
        )

        return {
            "loss": sum(history["loss"]) / max(len(history["loss"]), 1),
            "accuracy": sum(history["accuracy"]) / max(len(history["accuracy"]), 1),
            "image_loss": sum(history["image_loss"]) / max(len(history["image_loss"]), 1),
            "pixel_loss": sum(history["pixel_loss"]) / max(len(history["pixel_loss"]), 1),
            "acc_loss": sum(history["acc_loss"]) / max(len(history["acc_loss"]), 1),
            "proxy_loss": sum(history["proxy_loss"]) / max(len(history["proxy_loss"]), 1),
            "ln_loss": sum(history["ln_loss"]) / max(len(history["ln_loss"]), 1),
            "history": history,
        }

    def set_learning_rate(self, lr: float) -> None:
        for param_group in self.optimizer.optimizer.param_groups:
            param_group["lr"] = float(lr)
```
## File: training\__init__.py
```python
"""
Training utilities for continual learning
"""

from .trainer import Trainer
from .evaluator import Evaluator
from .memory_buffer import SlowMemory
from .nsp2_optim import NSP2Optimizer

__all__ = ['Trainer', 'Evaluator', 'SlowMemory', 'NSP2Optimizer']
```
## File: utils\get_mvtec_meta.py
```python
import os 
import json
import glob
from PIL import Image
from tqdm import tqdm

input_path = "data/mvtec"

meta_json_path = input_path+"mvtec_meta.json"

outpath = "data/mvtec/replay_meta.json"

with open(meta_json_path, 'r') as file:
    meta_data = json.load(file)

test_data = meta_data["test"]

new_data= {"train":{},"test":test_data}
new_data["train"]["zipper"] = meta_data["train"]["zipper"]


subfolders = [f.name for f in os.scandir(input_path+"generate") if f.is_dir()]

for class_name in subfolders:
    
    class_train_data = []
    class_train_images1 = glob.glob(input_path+"generate/"+class_name+"/samples/*.jpg")
    class_train_images2 = glob.glob(input_path+"generate/"+class_name+"/samples/*.png")
    class_train_images=class_train_images1+class_train_images2
    
    for class_image_path in class_train_images:
        
        data_path="/".join(class_image_path.split("/")[-4:])
        
        single_data = {"img_path":data_path,"mask_path": "","cls_name": class_name,"specie_name": "","anomaly": 0}
        
        class_train_data.append(single_data)
    
    new_data["train"][class_name]=class_train_data
    
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
```
## File: utils\global_seed.py
```python
import os
import random
import numpy as np
import torch
from conf.config import load_config

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    print(f"Global seed set to {seed}")
    
if __name__ == "__main__":
    config = load_config()
    set_seed(config['training']['seed'])
```
## File: dataset\anomaly_generators\base.py
```python
"""
Every subclass must implement:
    generate(img_np, category) -> (result_img_np, mask_np, has_anomaly)

Parameters
----------
img_np      : np.ndarray  shape (H, W, 3), dtype float32, range [0, 255]
category    : str         e.g. "bottle", "capsule"

Returns
-------
result_img_np : np.ndarray  same shape as img_np, dtype float32, range [0, 255]
mask_np       : np.ndarray  shape (H, W), dtype float32, values in {0, 1}
has_anomaly   : bool        True if a real anomaly was placed
"""

from abc import ABC, abstractmethod
import numpy as np


class AnomalyGeneratorBase(ABC):

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def generate(
        self,
        img_np: np.ndarray,       # (H, W, 3), float32, [0-255]
        category: str,
    ):
        """
        Returns
        -------
        result_img_np : np.ndarray  (H, W, 3), float32
        mask_np       : np.ndarray  (H, W),    float32 ∈ {0.0, 1.0}
        has_anomaly   : bool
        """
        ...
```
## File: dataset\anomaly_generators\destseg.py
```python
"""
DeSTSegAnomalyGenerator

Reference: DeSTSeg — Segmentation-Based Deep Anomaly Detection with Self-Supervised
           Training (Zhang et al., CVPR 2023)

Pipeline
--------
  1. Perlin noise mask (same as PerlinAnomalyGenerator)
  2. Raw DTD texture (no augmentation)
  3. Blend:  I*(1-mask) + (1-β)*DTD*mask + β*I*mask
"""

import glob
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase
from .perlin import rand_perlin_2d 


class DeSTSegAnomalyGenerator(AnomalyGeneratorBase):
    """
    DeSTSeg-style: Perlin mask + raw (unaugmented) DTD texture.

    Config keys (all optional):
        dtd_dir             : path to DTD images directory
        perlin_scale        : max log2 of Perlin frequency  (default 6)
        min_perlin_scale    : min log2                       (default 0)
        perlin_threshold    : binarisation threshold         (default 0.5)
        destseg_beta_range  : (min, max) blend factor        (default [0.1, 0.9])
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.perlin_scale     = cfg.get("perlin_scale", 6)
        self.min_perlin_scale = cfg.get("min_perlin_scale", 0)
        self.threshold        = cfg.get("perlin_threshold", 0.5)
        beta_lo, beta_hi      = cfg.get("destseg_beta_range", [0.1, 0.9])
        self.beta_lo, self.beta_hi = beta_lo, beta_hi

        self.dtd_file_list = []
        dtd_dir = cfg.get("dtd_dir", "")
        if dtd_dir:
            self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))
        if not self.dtd_file_list:
            import logging
            logging.getLogger(__name__).warning(
                "[DeSTSeg] No DTD images found. Using random colour patch."
            )

    def _perlin_mask(self, h: int, w: int) -> np.ndarray:
        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        noise = rand_perlin_2d((h, w), (sx, sy))

        # random rotation
        angle = float(np.random.uniform(-90, 90))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        noise = cv2.warpAffine(noise, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        return (noise > self.threshold).astype(np.float32)

    def _dtd_source_raw(self, h: int, w: int) -> np.ndarray:
        """Load DTD texture WITHOUT any colour augmentation (key DeSTSeg difference)."""
        if self.dtd_file_list:
            path = np.random.choice(self.dtd_file_list)
            tex = cv2.imread(path)
            tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        else:
            tex = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return cv2.resize(tex, (w, h)).astype(np.float32)

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]
        mask = self._perlin_mask(h, w)

        if mask.sum() == 0:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        dtd = self._dtd_source_raw(h, w)
        beta = np.random.uniform(self.beta_lo, self.beta_hi)

        m = mask[:, :, None]
        # Same DRAEM blend formula
        result = img_np * (1 - m) + (1 - beta) * dtd * m + beta * img_np * m

        return result.astype(np.float32), mask, True
```
## File: dataset\anomaly_generators\mixed.py
```python
"""
MixedAnomalyGenerator
======================
Meta-generator: randomly selects one of the registered generators
each time generate() is called.

Useful for training with maximum anomaly diversity — the model sees
Superpixel, Perlin (DRAEM), DeSTSeg, and Realnet anomalies within the
same epoch.

Config keys:
    mixed_generators : list of generator names + optional weights
                       (default: all four equally weighted)
    e.g.
        mixed_generators:
          - [superpixel, 1]
          - [perlin,     2]
          - [destseg,    1]
          - [realnet,    2]

    All other generator-specific keys (dtd_dir, perlin_scale, ...) are
    passed through to the sub-generators as usual.
"""

import numpy as np

from .base import AnomalyGeneratorBase


class MixedAnomalyGenerator(AnomalyGeneratorBase):
    """Randomly picks one sub-generator per sample."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        # Import here to avoid circular import
        from .superpixel import SuperpixelAnomalyGenerator
        from .perlin     import PerlinAnomalyGenerator
        from .destseg    import DeSTSegAnomalyGenerator
        from .realnet    import RealnetAnomalyGenerator

        _cls_map = {
            "superpixel": SuperpixelAnomalyGenerator,
            "perlin":     PerlinAnomalyGenerator,
            "destseg":    DeSTSegAnomalyGenerator,
            "realnet":    RealnetAnomalyGenerator,
        }

        spec = cfg.get("mixed_generators", [
            ["superpixel", 1],
            ["perlin",     1],
            ["destseg",    1],
            ["realnet",    1],
        ])

        self._generators = []
        weights = []
        for entry in spec:
            name, w = entry[0], entry[1] if len(entry) > 1 else 1
            if name not in _cls_map:
                raise ValueError(f"[Mixed] Unknown generator '{name}'.")
            self._generators.append(_cls_map[name](cfg))
            weights.append(float(w))

        total = sum(weights)
        self._probs = [w / total for w in weights]

    def generate(self, img_np: np.ndarray, category: str):
        idx = np.random.choice(len(self._generators), p=self._probs)
        return self._generators[idx].generate(img_np, category)
```
## File: dataset\anomaly_generators\perlin.py
```python
"""
Reference: DRAEM — Discriminatively trained Reconstruction Embedding for
           Surface Anomaly Detection (Zavrtanik et al., ICCV 2021)

Pipeline
--------
  1. Generate 2-D Perlin noise (multi-scale, random resolution)
  2. Rotate the noise map randomly (−90° → +90°)
  3. Threshold at 0.5 → binary mask
  4. Pick DTD texture → augment with 3 random colour transforms
  5. Blend:  I*(1-mask) + (1-β)*DTD*mask + β*I*mask
"""

import glob
import math
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase

# ────────────────────────────────────────────────────────────────────────────
# Perlin noise (pure numpy, no extra deps)
# ────────────────────────────────────────────────────────────────────────────

def _lerp(a, b, w):
    return (b - a) * w + a


def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d(shape, res):
    """Return a 2-D Perlin noise array of `shape` with frequency `res`."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def tile(s1, s2):
        return np.repeat(np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], 0), d[1], 1)

    def dot(g, shift):
        coords = np.stack(
            [grid[:shape[0], :shape[1], 0] + shift[0],
             grid[:shape[0], :shape[1], 1] + shift[1]], axis=-1
        )
        return (coords * g[:shape[0], :shape[1]]).sum(-1)

    n00 = dot(tile([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile([0, -1], [1, None]), [0, -1])
    n11 = dot(tile([1, None], [1, None]), [-1, -1])
    t = _fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * _lerp(_lerp(n00, n10, t[..., 0]),
                                 _lerp(n01, n11, t[..., 0]), t[..., 1])


# ────────────────────────────────────────────────────────────────────────────
# Colour augmenters (same helpers as superpixel, duplicated for independence)
# ────────────────────────────────────────────────────────────────────────────

def _rand_aug(image: np.ndarray) -> np.ndarray:
    """Apply 3 random colour transforms (DRAEM augmenter set)."""
    aug_pool = [
        lambda x: np.clip(np.power(x / 255.0, np.random.uniform(0.5, 2.0)) * 255, 0, 255).astype(np.uint8),           # gamma
        lambda x: np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-30, 30), 0, 255).astype(np.uint8),     # brightness
        lambda x: cv2.addWeighted(x, 1.5, cv2.GaussianBlur(x, (0, 0), 1.0), -0.5, 0),                                # sharpness
        lambda x: _hue_sat(x),
        lambda x: np.where(x < np.random.randint(32, 129), x, 255 - x).astype(np.uint8),                              # solarize
        lambda x: ((x >> (8 - np.random.randint(3, 7))) << (8 - np.random.randint(3, 7))).astype(np.uint8),           # posterize
        lambda x: (255 - x).astype(np.uint8),                                                                          # invert
        lambda x: _autocontrast(x),
        lambda x: _equalize(x),
        lambda x: cv2.warpAffine(x,
            cv2.getRotationMatrix2D((x.shape[1]/2, x.shape[0]/2), float(np.random.uniform(-45,45)), 1.0),
            (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
    ]
    for fn in np.random.choice(aug_pool, 3, replace=False):  # type: ignore[arg-type]
        image = fn(image)
    return image


def _hue_sat(x):
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + np.random.randint(-50, 51)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-50, 51), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _autocontrast(x):
    out = x.astype(np.float32)
    for c in range(out.shape[2]):
        lo, hi = out[..., c].min(), out[..., c].max()
        if hi > lo:
            out[..., c] = (out[..., c] - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def _equalize(x):
    ycrcb = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ────────────────────────────────────────────────────────────────────────────
# Generator
# ────────────────────────────────────────────────────────────────────────────

class PerlinAnomalyGenerator(AnomalyGeneratorBase):
    """
    DRAEM-style Perlin noise mask + DTD texture blending.

    Config keys (all optional):
        dtd_dir             : path to DTD images directory
        perlin_scale        : max log2 of Perlin frequency  (default 6)
        min_perlin_scale    : min log2 of Perlin frequency  (default 0)
        perlin_threshold    : binarisation threshold        (default 0.5)
        anomaly_blend_range : (min, max) beta range         (default [0.1, 0.8])
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.perlin_scale     = cfg.get("perlin_scale", 6)
        self.min_perlin_scale = cfg.get("min_perlin_scale", 0)
        self.threshold        = cfg.get("perlin_threshold", 0.5)
        beta_lo, beta_hi      = cfg.get("anomaly_blend_range", [0.1, 0.8])
        self.beta_lo, self.beta_hi = beta_lo, beta_hi

        # DTD
        self.dtd_file_list = []
        dtd_dir = cfg.get("dtd_dir", "")
        if dtd_dir:
            self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))
        if not self.dtd_file_list:
            import logging
            logging.getLogger(__name__).warning(
                "[Perlin/DRAEM] No DTD images found. "
                "Anomaly source will be a random colour patch."
            )

    def _perlin_mask(self, h: int, w: int) -> np.ndarray:
        """Binary mask from 2-D Perlin noise, random scale + rotation."""
        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        noise = rand_perlin_2d((h, w), (sx, sy))

        # random rotation via OpenCV
        angle = float(np.random.uniform(-90, 90))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        noise = cv2.warpAffine(noise, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

        return (noise > self.threshold).astype(np.float32)

    def _dtd_source(self, h: int, w: int) -> np.ndarray:
        if self.dtd_file_list:
            path = np.random.choice(self.dtd_file_list)
            tex = cv2.imread(path)
            tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        else:
            # fallback: random colour noise
            tex = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        tex = cv2.resize(tex, (w, h))
        tex = _rand_aug(tex)           # 3 colour augmentations
        return tex.astype(np.float32)

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]
        mask = self._perlin_mask(h, w)

        if mask.sum() == 0:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        dtd = self._dtd_source(h, w)
        beta = np.random.uniform(self.beta_lo, self.beta_hi)

        m = mask[:, :, None]
        # DRAEM blend: I*(1-mask) + (1-β)*DTD*mask + β*I*mask
        result = img_np * (1 - m) + (1 - beta) * dtd * m + beta * img_np * m

        return result.astype(np.float32), mask, True
```
## File: dataset\anomaly_generators\realnet.py
```python
"""
Reference: RealNet — A Feature Selection Network with Realistic Synthetic Anomaly
           for Anomaly Detection (Zhang et al., CVPR 2024)

Pipeline
--------
  1. Perlin noise mask × foreground mask  → final mask   (both-aware)
  2. Probabilistic source choice:
       • 'dtd'  (p = dtd_weight)   → DTD texture + 3 colour augments
       • 'sdas' (p = 1-dtd_weight) → SDAS images (class-specific generated
                                      anomalies, or replay-buffer images)
  3. Blend:
       factor * (mask * src) + (1-factor) * (mask * img) + (1-mask) * img
"""

import glob
import math
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase
from .perlin import rand_perlin_2d, _rand_aug      # reuse helpers

_TEXTURE_CATEGORIES = {
    "carpet", "leather", "tile", "wood", "cable", "transistor", "grid",
}


class RealnetAnomalyGenerator(AnomalyGeneratorBase):
    """
    Realnet-style: dual-source (DTD + SDAS/replay) with foreground-aware mask.

    Config keys (all optional):
        dtd_dir                 : path to DTD images
        sdas_dir                : path to SDAS / replay-buffer images
        realnet_dtd_weight      : probability of choosing DTD source (default 0.5)
        realnet_dtd_factor_range: (min, max) blend factor for DTD   (default [0.2, 0.8])
        realnet_sdas_factor_range: (min, max) blend factor for SDAS (default [0.1, 0.6])
        perlin_scale            : max log2 Perlin frequency          (default 6)
        min_perlin_scale        : min log2 Perlin frequency          (default 0)
        perlin_threshold        : binarisation threshold             (default 0.5)
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg.get("name", "mvtec").lower()

        self.perlin_scale     = cfg.get("perlin_scale", 6)
        self.min_perlin_scale = cfg.get("min_perlin_scale", 0)
        self.threshold        = cfg.get("perlin_threshold", 0.5)

        self.dtd_weight  = cfg.get("realnet_dtd_weight", 0.5)
        dtd_lo, dtd_hi   = cfg.get("realnet_dtd_factor_range", [0.2, 0.8])
        sd_lo,  sd_hi    = cfg.get("realnet_sdas_factor_range", [0.1, 0.6])
        self.dtd_lo, self.dtd_hi = dtd_lo, dtd_hi
        self.sd_lo,  self.sd_hi  = sd_lo,  sd_hi

        # DTD source
        self.dtd_file_list = []
        dtd_dir = cfg.get("dtd_dir", "")
        if dtd_dir:
            self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))

        # SDAS / replay-buffer source
        self.sdas_file_list = []
        sdas_dir = cfg.get("sdas_dir", "")
        if sdas_dir and os.path.isdir(sdas_dir):
            self.sdas_file_list = glob.glob(os.path.join(sdas_dir, "**", "*.*"),
                                            recursive=True)
            self.sdas_file_list = [
                p for p in self.sdas_file_list
                if p.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

        import logging
        log = logging.getLogger(__name__)
        if not self.dtd_file_list:
            log.warning("[Realnet] No DTD images found.")
        if not self.sdas_file_list:
            log.info("[Realnet] No SDAS/replay images found. Will use DTD only.")
            self.dtd_weight = 1.0   # force DTD

    # ── foreground mask (same logic as Superpixel) ───────────────────────────

    def _foreground_mask(self, img_np: np.ndarray, category: str) -> np.ndarray:
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        cat  = category.lower()

        if cat in _TEXTURE_CATEGORIES or self.dataset_name != "mvtec":
            return np.ones_like(gray, dtype=np.float32)

        if cat in {"pill", "hazelnut", "metal_nut", "toothbrush"}:
            _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif cat in {"bottle", "capsule", "screw", "zipper"}:
            _, bg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            fg = cv2.bitwise_not(bg)
        else:
            _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        return (fg > 0).astype(np.float32)

    # ── Perlin mask ───────────────────────────────────────────────────────────

    def _perlin_mask(self, h: int, w: int) -> np.ndarray:
        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        noise = rand_perlin_2d((h, w), (sx, sy))
        angle = float(np.random.uniform(-90, 90))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        noise = cv2.warpAffine(noise, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        return (noise > self.threshold).astype(np.float32)

    # ── anomaly sources ───────────────────────────────────────────────────────

    def _dtd_source(self, h: int, w: int) -> np.ndarray:
        if not self.dtd_file_list:
            return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
        tex = cv2.imread(np.random.choice(self.dtd_file_list))
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        tex = cv2.resize(tex, (w, h))
        tex = _rand_aug(tex)   # 3 colour augments (DRAEM-style)
        return tex.astype(np.float32)

    def _sdas_source(self, h: int, w: int) -> np.ndarray:
        """
        Class-specific anomaly images (SDAS) or replay-buffer images.
        No augmentation — the image already looks anomalous.
        """
        img = cv2.imread(np.random.choice(self.sdas_file_list))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        return img.astype(np.float32)

    # ── public API ───────────────────────────────────────────────────────────

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]

        fg_mask    = self._foreground_mask(img_np, category)
        prl_mask   = self._perlin_mask(h, w)
        mask       = (prl_mask * fg_mask).astype(np.float32)   # intersection

        if mask.sum() == 0:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        # ── source selection ─────────────────────────────────────────────
        use_dtd = (not self.sdas_file_list) or (np.random.rand() < self.dtd_weight)

        if use_dtd:
            src    = self._dtd_source(h, w)
            factor = np.random.uniform(self.dtd_lo, self.dtd_hi)
        else:
            src    = self._sdas_source(h, w)
            factor = np.random.uniform(self.sd_lo, self.sd_hi)

        # ── Realnet blend ────────────────────────────────────────────────
        # factor*(mask*src) + (1-factor)*(mask*img) + (1-mask)*img
        m      = mask[:, :, None]
        result = factor * (m * src) + (1 - factor) * (m * img_np) + (1 - m) * img_np

        return result.astype(np.float32), mask, True
```
## File: dataset\anomaly_generators\superpixel.py
```python
"""
Nested Learning's own method

  1. Foreground mask      — Otsu thresholding, category-aware
  2. Semantic mask        — SLIC superpixel selection + area filter (0.5%–15%)
  3. Anomaly source       — DTD texture (augmented) OR self-shift + luminance jitter
  4. Alpha blend          — factor * (mask * src) + (1-factor) * (mask * img)

"""

import glob
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase


_AUGMENTER_NAMES = [
    "gamma_contrast", "brightness", "sharpness", "hue_saturation",
    "solarize", "posterize", "invert", "autocontrast", "equalize", "rotate",
]

_TEXTURE_CATEGORIES = {
    "carpet", "leather", "tile", "wood", "cable", "transistor", "grid",
}


def _apply_aug(image: np.ndarray, aug_name: str) -> np.ndarray:
    if aug_name == "gamma_contrast":
        gamma = np.random.uniform(0.5, 2.0)
        return np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)

    if aug_name == "brightness":
        return np.clip(image * np.random.uniform(0.8, 1.2) + np.random.uniform(-30, 30),
                       0, 255).astype(np.uint8)

    if aug_name == "sharpness":
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    if aug_name == "hue_saturation":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + np.random.randint(-50, 51)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-50, 51), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if aug_name == "solarize":
        threshold = np.random.randint(32, 129)
        return np.where(image < threshold, image, 255 - image).astype(np.uint8)

    if aug_name == "posterize":
        shift = 8 - np.random.randint(3, 7)
        return ((image >> shift) << shift).astype(np.uint8)

    if aug_name == "invert":
        return (255 - image).astype(np.uint8)

    if aug_name == "autocontrast":
        out = image.astype(np.float32).copy()
        for c in range(out.shape[2]):
            lo, hi = out[..., c].min(), out[..., c].max()
            if hi > lo:
                out[..., c] = (out[..., c] - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    if aug_name == "equalize":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    if aug_name == "rotate":
        h, w = image.shape[:2]
        angle = float(np.random.uniform(-45, 45))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return image


# Genetar

class SuperpixelAnomalyGenerator(AnomalyGeneratorBase):
    """Nested Learning original method."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg.get("name", "mvtec").lower()

        # DTD texture source
        self.use_dtd = cfg.get("use_dtd", False)
        self.dtd_file_list = []
        if self.use_dtd:
            dtd_dir = cfg.get("dtd_dir", "")
            self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))
            if not self.dtd_file_list:
                import logging
                logging.getLogger(__name__).warning(
                    f"[Superpixel] No DTD images at '{dtd_dir}'. "
                    "Falling back to self-shift source."
                )
                self.use_dtd = False

        # SLIC
        self.min_fg_coverage = cfg.get("superpixel_min_fg_coverage", 0.7)
        self.max_sp_fraction = cfg.get("superpixel_max_fraction", 0.15)
        self.area_min = cfg.get("anomaly_area_min", 0.005)    # fraction of image
        self.area_max = cfg.get("anomaly_area_max", 0.15)
        self.mask_retries = cfg.get("superpixel_retries", 3)

    def _foreground_mask(self, img_np: np.ndarray, category: str) -> np.ndarray:
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        cat = category.lower()

        if cat in _TEXTURE_CATEGORIES or self.dataset_name != "mvtec":
            return np.ones_like(gray, dtype=np.float32)

        if cat in {"pill", "hazelnut", "metal_nut", "toothbrush"}:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif cat in {"bottle", "capsule", "screw", "zipper"}:
            _, bg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask = cv2.bitwise_not(bg)
        else:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return (mask > 0).astype(np.float32)

    def _semantic_mask(self, img_np: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        h, w = img_np.shape[:2]
        reg_size = np.random.randint(15, 40)
        slic = cv2.ximgproc.createSuperpixelSLIC(
            img_np.astype(np.uint8),
            algorithm=cv2.ximgproc.SLIC,
            region_size=reg_size, ruler=15.0,
        )
        slic.iterate(10)
        labels = slic.getLabels()
        num_sp = slic.getNumberOfSuperpixels()

        mask = np.zeros((h, w), dtype=np.float32)
        if num_sp <= 1:
            return mask

        valid_sp = [
            sp for sp in range(num_sp)
            if np.mean(fg_mask[labels == sp]) > self.min_fg_coverage
        ]
        if not valid_sp:
            return mask

        max_select = max(3, int(len(valid_sp) * self.max_sp_fraction))
        num_select = np.random.randint(2, max_select + 1)
        for sp in np.random.choice(valid_sp, num_select, replace=False):
            mask[labels == sp] = 1.0

        k = np.random.choice([7, 11, 15])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return (mask > 0.4).astype(np.float32)

    def _dtd_source(self, h: int, w: int) -> np.ndarray:
        img = cv2.imread(np.random.choice(self.dtd_file_list))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        for aug in np.random.choice(_AUGMENTER_NAMES, 3, replace=False):
            img = _apply_aug(img, aug)
        return img.astype(np.float32)

    def _self_shift_source(self, img_np: np.ndarray) -> tuple:
        h, w = img_np.shape[:2]
        sx = np.random.randint(-w // 15, w // 15)
        sy = np.random.randint(-h // 15, h // 15)
        src = np.roll(img_np, shift=(sy, sx), axis=(0, 1))
        lum = np.random.normal(0, 15, (h, w, 1))
        src = np.clip(src * np.random.uniform(0.85, 1.15) + lum, 0, 255)
        factor = np.random.uniform(0.4, 0.8)
        return src, factor

    # API

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]
        img_area = h * w

        fg_mask = self._foreground_mask(img_np, category)

        mask_noise = None
        for _ in range(self.mask_retries):
            candidate = self._semantic_mask(img_np, fg_mask)
            area = np.sum(candidate)
            if self.area_min * img_area <= area <= self.area_max * img_area:
                mask_noise = candidate
                break

        if mask_noise is None:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        if self.use_dtd and np.random.rand() > 0.5:
            src = self._dtd_source(h, w)
            factor = np.random.uniform(0.3, 0.7)
        else:
            src, factor = self._self_shift_source(img_np)

        blurred = cv2.GaussianBlur(mask_noise, (7, 7), 0)
        m = blurred[:, :, None]
        blended = factor * (m * src) + (1 - factor) * (m * img_np)
        result = (1 - m) * img_np + blended

        return result.astype(np.float32), mask_noise, True
```
## File: dataset\anomaly_generators\__init__.py
```python
"""
Anomaly Generator Registry
==========================
Each generator implements the interface: AnomalyGeneratorBase
    .generate(img_np, category) -> (result_img_np, mask_np, has_anomaly: bool)

Usage in config.yaml:
    dataset:
      anomaly_generator: "superpixel"   # or "perlin", "destseg", "realnet", "mixed"
"""

from .base import AnomalyGeneratorBase
from .superpixel import SuperpixelAnomalyGenerator
from .perlin import PerlinAnomalyGenerator
from .destseg import DeSTSegAnomalyGenerator
from .realnet import RealnetAnomalyGenerator
from .mixed import MixedAnomalyGenerator

_REGISTRY = {
    "superpixel": SuperpixelAnomalyGenerator,
    "perlin":     PerlinAnomalyGenerator,
    "destseg":    DeSTSegAnomalyGenerator,
    "realnet":    RealnetAnomalyGenerator,
    "mixed":      MixedAnomalyGenerator,
}


def build_anomaly_generator(cfg: dict) -> AnomalyGeneratorBase:
    name = cfg.get("anomaly_generator", "superpixel").lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown anomaly_generator '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](cfg)


__all__ = [
    "AnomalyGeneratorBase",
    "SuperpixelAnomalyGenerator",
    "PerlinAnomalyGenerator",
    "DeSTSegAnomalyGenerator",
    "RealnetAnomalyGenerator",
    "MixedAnomalyGenerator",
    "build_anomaly_generator",
]
```
## File: legacy\dataset\cifar_dataloader.py
```python
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


DEFAULT_TASK_PAIRS: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


@dataclass
class CifarTaskInfo:
    task_id: int
    task_classes: List[int]
    train_size: int
    test_size: int


class CifarTaskDataset(Dataset):
    """A class-filtered view over a CIFAR dataset for a single task."""

    def __init__(self, base_dataset: Dataset, task_classes: Sequence[int]):
        self.base_dataset = base_dataset
        self.task_classes = list(task_classes)

        allowed = set(self.task_classes)
        targets = self._extract_targets(base_dataset)
        self.indices = [idx for idx, label in enumerate(targets) if int(label) in allowed]

    def _extract_targets(self, dataset_obj: Dataset) -> List[int]:
        targets = getattr(dataset_obj, "targets", None)
        if targets is None:
            raise AttributeError("The provided CIFAR dataset has no 'targets' attribute.")
        return [int(t) for t in targets]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        mapped_idx = self.indices[index]
        image, label = self.base_dataset[mapped_idx]
        return image, int(label)


class SyntheticCifar10Dataset(Dataset):
    """Lightweight CIFAR-10-like dataset for offline loader checks."""

    def __init__(self, num_samples: int, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.targets = [i % 10 for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        label = self.targets[index]
        rng = np.random.default_rng(seed=index + label * 10_000)
        array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class Cifar10TaskManager:
    """Builds pairwise CIFAR-10 continual tasks: (0,1), (2,3), ... (8,9)."""

    def __init__(
        self,
        data_root: str = "./data",
        batch_size: int = 64,
        num_workers: int = 2,
        task_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        download: bool = True,
        use_fake_data: bool = False,
        fake_train_size: int = 5000,
        fake_test_size: int = 1000,
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_pairs = list(task_pairs) if task_pairs is not None else list(DEFAULT_TASK_PAIRS)

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        if use_fake_data:
            self.train_base = SyntheticCifar10Dataset(
                num_samples=fake_train_size,
                transform=self.train_transform,
            )
            self.test_base = SyntheticCifar10Dataset(
                num_samples=fake_test_size,
                transform=self.test_transform,
            )
        else:
            self.train_base = datasets.CIFAR10(
                root=self.data_root,
                train=True,
                download=download,
                transform=self.train_transform,
            )
            self.test_base = datasets.CIFAR10(
                root=self.data_root,
                train=False,
                download=download,
                transform=self.test_transform,
            )

    def num_tasks(self) -> int:
        return len(self.task_pairs)

    def _build_loader(self, dataset_obj: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset_obj,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def get_task_loaders(self, task_id: int) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        if task_id < 0 or task_id >= self.num_tasks():
            raise IndexError(f"task_id={task_id} is out of range [0, {self.num_tasks() - 1}].")

        task_classes = list(self.task_pairs[task_id])
        train_dataset = CifarTaskDataset(self.train_base, task_classes)
        test_dataset = CifarTaskDataset(self.test_base, task_classes)

        train_loader = self._build_loader(train_dataset, shuffle=True)
        test_loader = self._build_loader(test_dataset, shuffle=False)

        task_info = {
            "task_id": task_id,
            "task_classes": task_classes,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
        }
        return train_loader, test_loader, task_info

    def iter_tasks(self, max_tasks: Optional[int] = None) -> Iterable[Tuple[DataLoader, DataLoader, Dict[str, int]]]:
        total = self.num_tasks() if max_tasks is None else min(max_tasks, self.num_tasks())
        for task_id in range(total):
            yield self.get_task_loaders(task_id)
```
## File: legacy\scripts\02_check_cifar_loader.py
```python
import argparse
from collections import Counter
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.cifar_dataloader import Cifar10TaskManager


def parse_args():
    parser = argparse.ArgumentParser(description="Quick validation for CIFAR-10 pairwise task loader")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for loader checks")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max_tasks", type=int, default=5, help="How many tasks to validate")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")
    parser.add_argument("--use_fake_data", action="store_true", help="Use synthetic CIFAR-like data")
    parser.add_argument("--fake_train_size", type=int, default=5000, help="Synthetic train samples")
    parser.add_argument("--fake_test_size", type=int, default=1000, help="Synthetic test samples")
    return parser.parse_args()


def main():
    args = parse_args()
    manager = Cifar10TaskManager(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        use_fake_data=args.use_fake_data,
        fake_train_size=args.fake_train_size,
        fake_test_size=args.fake_test_size,
    )

    print(f"Total tasks configured: {manager.num_tasks()}")

    for train_loader, test_loader, task_info in manager.iter_tasks(max_tasks=args.max_tasks):
        task_id = task_info["task_id"]
        task_classes = task_info["task_classes"]

        print("-" * 64)
        print(f"Task {task_id}: classes={task_classes}")
        print(f"Train size: {task_info['train_size']} | Test size: {task_info['test_size']}")

        train_images, train_labels = next(iter(train_loader))
        print(f"Train batch image shape: {tuple(train_images.shape)}")
        print(f"Train batch label shape: {tuple(train_labels.shape)}")

        unique = sorted(set(train_labels.tolist()))
        label_hist = Counter(train_labels.tolist())
        print(f"Unique labels in sampled train batch: {unique}")
        print(f"Label counts in sampled train batch: {dict(label_hist)}")

        disallowed = [label for label in unique if label not in task_classes]
        if disallowed:
            raise RuntimeError(
                f"Task {task_id} has labels outside task classes {task_classes}: {disallowed}"
            )

        if not hasattr(train_loader.dataset, "task_classes"):
            raise RuntimeError("train_loader.dataset is missing task_classes attribute")

        if not hasattr(test_loader.dataset, "task_classes"):
            raise RuntimeError("test_loader.dataset is missing task_classes attribute")

    print("Loader validation passed.")


if __name__ == "__main__":
    main()
```
## File: results\eda\pipeline_verify_hazelnut.png
```
Error reading results\eda\pipeline_verify_hazelnut.png: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
```
## File: results\eda\pipeline_verify_screw.png
```
Error reading results\eda\pipeline_verify_screw.png: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
```
## File: results\eda\pipeline_verify_toothbrush.png
```
Error reading results\eda\pipeline_verify_toothbrush.png: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
```
## File: results\eda\pipeline_verify_transistor.png
```
Error reading results\eda\pipeline_verify_transistor.png: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
```
## File: results\eda\pipeline_verify_wood.png
```
Error reading results\eda\pipeline_verify_wood.png: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
```
## File: results\eda\pipeline_verify_zipper.png
```
Error reading results\eda\pipeline_verify_zipper.png: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
```
