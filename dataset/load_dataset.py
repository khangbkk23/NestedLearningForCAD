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
from dataset.noise import Simplex_CLASS
import cv2

from conf.config import load_config

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
    def __init__(self, cfg, category, is_train=True, transform=None, target_transform=None):
        self.root_dir = cfg['root_dir']
        self.dataset_name = cfg['name'].lower()
        self.category = category
        self.split_ratio = cfg.get('split_ratio', 0.8)
        self.is_train = is_train
        
        self.transform = transform
        self.target_transform = target_transform
        
        # use module-level loader functions so the dataset is picklable
        self.loader = default_image_loader
        self.loader_target = default_mask_loader
        
        if self.is_train:
            self.simplex = Simplex_CLASS()
            self.min_perlin_scale = 4
            self.perlin_scale = 7
            self.perlin_noise_threshold = 0.7
            
        self.use_dtd = cfg['use_dtd']
        
        if self.use_dtd:
                self.dtd_dir = cfg['dtd_dir']
                self.dtd_file_list = glob.glob(os.path.join(self.dtd_dir, '*/*.*'))
                if len(self.dtd_file_list) == 0:
                    logger.warning(f"Cannot found dtd images at: '{self.dtd_dir}'. Using random generate algorithm")
                    self.use_dtd = False
                else:
                    self.augmenters = [
                        'gamma_contrast',
                        'brightness',
                        'sharpness',
                        'hue_saturation',
                        'solarize',
                        'posterize',
                        'invert',
                        'autocontrast',
                        'equalize',
                        'rotate'
                    ]
        
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
    
    def _get_foreground_mask(self, img_np, category):
        img_gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        category = category.lower()
        
        textures = ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor', 'grid']
        if category in textures or self.dataset_name != "mvtec":
            return np.ones_like(img_gray, dtype=np.float32)

        if category in ['pill', 'hazelnut', 'metal_nut', 'toothbrush']:
            _, fg_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif category in ['bottle', 'capsule', 'screw', 'zipper']:
            _, bg_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            fg_mask = cv2.bitwise_not(bg_mask)
        else:
            _, fg_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        return (fg_mask > 0).astype(np.float32)
    
    def _get_semantic_mask(self, img_np, fg_mask):
        h, w = img_np.shape[:2]

        reg_size = np.random.randint(15, 40)
        slic = cv2.ximgproc.createSuperpixelSLIC(
            img_np.astype(np.uint8),
            algorithm=cv2.ximgproc.SLIC,
            region_size=reg_size,
            ruler=15.0
        )
        slic.iterate(10)

        labels = slic.getLabels()
        num_sp = slic.getNumberOfSuperpixels()

        mask = np.zeros((h, w), dtype=np.float32)
        if num_sp <= 1:
            return mask

        valid_sp = []
        for sp_id in range(num_sp):
            region = (labels == sp_id)
            if np.mean(fg_mask[region]) > 0.7: 
                valid_sp.append(sp_id)

        if len(valid_sp) == 0:
            return mask

        max_sp = max(3, int(len(valid_sp) * 0.15))
        num_select = np.random.randint(2, max_sp + 1)
        selected = np.random.choice(valid_sp, num_select, replace=False)

        for sp_id in selected:
            mask[labels == sp_id] = 1.0
            
        k_size = np.random.choice([7, 11, 15])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = (mask > 0.4).astype(np.float32)

        return mask

    def _apply_dtd_augmentation(self, image, aug_name):
        if aug_name == 'gamma_contrast':
            gamma = np.random.uniform(0.5, 2.0)
            adjusted = np.power(image.astype(np.float32) / 255.0, gamma) * 255.0
            return np.clip(adjusted, 0, 255).astype(np.uint8)

        if aug_name == 'brightness':
            mul = np.random.uniform(0.8, 1.2)
            add = np.random.uniform(-30, 30)
            adjusted = image.astype(np.float32) * mul + add
            return np.clip(adjusted, 0, 255).astype(np.uint8)

        if aug_name == 'sharpness':
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
            return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        if aug_name == 'hue_saturation':
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + np.random.randint(-50, 51)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-50, 51), 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if aug_name == 'solarize':
            threshold = np.random.randint(32, 129)
            return np.where(image < threshold, image, 255 - image).astype(np.uint8)

        if aug_name == 'posterize':
            bits = np.random.randint(3, 7)
            shift = 8 - bits
            return ((image >> shift) << shift).astype(np.uint8)

        if aug_name == 'invert':
            return (255 - image).astype(np.uint8)

        if aug_name == 'autocontrast':
            out = image.astype(np.float32).copy()
            for c in range(out.shape[2]):
                channel = out[..., c]
                lo = channel.min()
                hi = channel.max()
                if hi > lo:
                    out[..., c] = (channel - lo) * (255.0 / (hi - lo))
            return np.clip(out, 0, 255).astype(np.uint8)

        if aug_name == 'equalize':
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        if aug_name == 'rotate':
            h, w = image.shape[:2]
            angle = float(np.random.uniform(-45, 45))
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            return cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        return image
    
    def _get_dtd_source(self, target_h, target_w):
        idx = np.random.choice(len(self.dtd_file_list))
        dtd_img = cv2.imread(self.dtd_file_list[idx])
        dtd_img = cv2.cvtColor(dtd_img, cv2.COLOR_BGR2RGB)
        dtd_img = cv2.resize(dtd_img, (target_w, target_h))

        num_aug = min(3, len(self.augmenters))
        aug_idx = np.random.choice(np.arange(len(self.augmenters)), num_aug, replace=False)
        for i in aug_idx:
            dtd_img = self._apply_dtd_augmentation(dtd_img, self.augmenters[i])

        return dtd_img.astype(np.float32)
    
    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path = data['img_path'], data['mask_path']
        cls_name, specie_name, anomaly = data['cls_name'], data['specie_name'], data['anomaly']
        
        img = self.loader(img_path)
        img_w, img_h = img.size
        
        if self.is_train:
            # Paper (CLAD): train set is normal-only.
            # We extend with synthetic anomaly generation (50% chance) to enable
            # supervised pixel-level training — stronger than pure reconstruction.
            generate_anomaly = (np.random.rand() > 0.5)
            
            if generate_anomaly:
                img_np = np.array(img).astype(np.float32)
                fg_mask = self._get_foreground_mask(img_np, self.category)
                
                mask_noise = None
                valid_mask_found = False
                img_area = img_w * img_h
                
                for _ in range(3):
                    temp_mask = self._get_semantic_mask(img_np, fg_mask)
                    area = np.sum(temp_mask)
                    if 0.005 * img_area <= area <= 0.15 * img_area:
                        mask_noise = temp_mask
                        valid_mask_found = True
                        break
                
                if valid_mask_found:
                    mask_noise_blurred = cv2.GaussianBlur(mask_noise, (7, 7), 0)
                    mask_noise_expanded = np.expand_dims(mask_noise_blurred, axis=2)
                    
                    if self.use_dtd and np.random.rand() > 0.5:
                        anomaly_source = self._get_dtd_source(img_h, img_w)
                        factor = np.random.uniform(0.3, 0.7)
                    else:
                        shift_x = np.random.randint(-img_w//15, img_w//15)
                        shift_y = np.random.randint(-img_h//15, img_h//15)
                        anomaly_source = np.roll(img_np, shift=(shift_y, shift_x), axis=(0, 1))
                        luminance_jitter = np.random.normal(loc=0, scale=15, size=(img_h, img_w, 1))
                        anomaly_source = np.clip(anomaly_source * np.random.uniform(0.85, 1.15) + luminance_jitter, 0, 255)
                        factor = np.random.uniform(0.4, 0.8)
                    
                    blended = factor * (mask_noise_expanded * anomaly_source) + (1 - factor) * (mask_noise_expanded * img_np)
                    img_np = ((1 - mask_noise_expanded) * img_np) + blended
                    
                    img = Image.fromarray(img_np.astype(np.uint8))
                    img_mask = Image.fromarray((mask_noise * 255).astype(np.uint8), mode='L')
                    anomaly = 1
                else:
                    # Could not find a valid mask region → keep as normal
                    img_mask = Image.fromarray(np.zeros((img_h, img_w), dtype=np.uint8), mode='L')
                    anomaly = 0
            else:
                # Normal sample: no synthetic anomaly
                img_mask = Image.fromarray(np.zeros((img_h, img_w), dtype=np.uint8), mode='L')
                anomaly = 0

        else:
            # Test: load real ground-truth masks (from dataset)
            if anomaly == 0 or mask_path == '':
                img_mask = Image.fromarray(np.zeros((img_h, img_w), dtype=np.uint8), mode='L')
            else:
                mask_arr = np.array(self.loader_target(mask_path)) > 0
                img_mask = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode='L')

        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'specie_name': specie_name,
            'anomaly': anomaly,
            'img_path': img_path
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