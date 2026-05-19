import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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
    Switch the generator in the active YAML config:
    dataset.anomaly_generator: "perlin"

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
        self.evaluation_cfg = config.get('evaluation', {})
        
        self.dataset_name = self.dataset_cfg['name']
        self.root_dir = self.dataset_cfg['root_dir']
        self.batch_size = self.dataset_cfg['batch_size']
        self.num_workers = self.dataset_cfg.get('num_workers', 4)
        self.drop_last_train = bool(self.dataset_cfg.get('drop_last_train', True))
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

    def _loader_kwargs(self):
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': bool(self.dataset_cfg.get('pin_memory', torch.cuda.is_available())),
        }
        if self.num_workers > 0:
            loader_kwargs['persistent_workers'] = bool(self.dataset_cfg.get('persistent_workers', True))
            loader_kwargs['prefetch_factor'] = int(self.dataset_cfg.get('prefetch_factor', 2))
        return loader_kwargs

    def get_cumulative_test_loader(self):
        if not self.test_datasets_history:
            return None
        return DataLoader(
            ConcatDataset(self.test_datasets_history),
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )

    def get_test_loader_for_task(self, task_index: int):
        if task_index < 0 or task_index >= len(self.test_datasets_history):
            return None
        return DataLoader(
            self.test_datasets_history[task_index],
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )

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
        eval_mode = str(self.evaluation_cfg.get('mode', 'current')).lower()
        if eval_mode == 'cumulative':
            eval_dataset = ConcatDataset(self.test_datasets_history)
        else:
            eval_dataset = current_test_dataset

        loader_kwargs = self._loader_kwargs()
        
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=self.drop_last_train, **loader_kwargs)
        test_loader = DataLoader(eval_dataset, shuffle=False, drop_last=False, **loader_kwargs)
        
        task_info = {
            'task_id': self.current_task_idx,
            'category': current_category,
        }
        
        self.current_task_idx += 1
        return train_loader, test_loader, task_info

if __name__ == "__main__":
    default_config = os.path.join(PROJECT_ROOT, "conf", "reference", "phase1_baseline.yaml")
    config = load_config(default_config)
    manager = ContinualStreamingManager(config)
    
    train_loader, test_loader, info = manager.get_next_task()
    if train_loader:
        batch = next(iter(train_loader))
        print(f"Task info: {info}")
        print(f"Keys in batch: {batch.keys()}")
        print(f"Image batch shape: {batch['img'].shape}")
        print(f"Mask batch shape: {batch['img_mask'].shape}")
