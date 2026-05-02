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
        return np.array(img.convert('RGB'))


def default_mask_loader(path):
    """Top-level mask loader (picklable for multiprocessing)."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def _ensure_uint8(img_np: np.ndarray) -> np.ndarray:
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np

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

        img_np = self.loader(img_path)
        img_h, img_w = img_np.shape[:2]
        mask_np = np.zeros((img_h, img_w), dtype=np.uint8)

        if self.is_train:
            if np.random.rand() > 0.5:
                result_np, mask_np, has_anomaly = self.anomaly_generator.generate(
                    img_np.astype(np.float32), self.category
                )
                if has_anomaly:
                    img_np = result_np
                    mask_np = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
                    anomaly  = 1
                else:
                    mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
                    anomaly = 0
            else:
                # Normal sample
                mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
                anomaly = 0

        else:
            if anomaly == 0 or mask_path == '':
                mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
            else:
                mask_arr = np.array(self.loader_target(mask_path)) > 0
                mask_np = (mask_arr.astype(np.uint8) * 255)

        img = Image.fromarray(_ensure_uint8(img_np), mode='RGB')
        img_mask = Image.fromarray(mask_np, mode='L')

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