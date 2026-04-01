import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

class ContinualAnomalyDataset(Dataset):
    """
    Unified Dataset class for Continual Anomaly Detection.
    Handles directory structures for both MVTec AD and VisA datasets.
    """
    def __init__(self, root_dir, dataset_name, category, is_train=True, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name.lower()
        self.category = category
        self.is_train = is_train
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = [] 
        
        self._load_data()

    def _load_data(self):
        category_path = os.path.join(self.root_dir, self.category)
        
        if self.dataset_name == "mvtec":
            self._load_mvtec(category_path)
        elif self.dataset_name == "visa":
            self._load_visa(category_path)
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")

    def _load_mvtec(self, category_path):
        """
        Parse MVTec AD directory structure.
        """
        if self.is_train:
            img_dir = os.path.join(category_path, 'train', 'good')
            imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
            self.image_paths.extend(imgs)
            self.labels.extend([0] * len(imgs))
            self.mask_paths.extend([None] * len(imgs))
        else:
            test_dir = os.path.join(category_path, 'test')
            defect_types = sorted(os.listdir(test_dir))
            
            for defect in defect_types:
                defect_dir = os.path.join(test_dir, defect)
                imgs = sorted(glob.glob(os.path.join(defect_dir, '*.png')))
                self.image_paths.extend(imgs)
                
                if defect == 'good':
                    self.labels.extend([0] * len(imgs))
                    self.mask_paths.extend([None] * len(imgs))
                else:
                    self.labels.extend([1] * len(imgs))
                    gt_dir = os.path.join(category_path, 'ground_truth', defect)
                    for img in imgs:
                        img_name = os.path.basename(img)
                        mask_name = img_name.replace('.png', '_mask.png')
                        self.mask_paths.append(os.path.join(gt_dir, mask_name))

    def _load_visa(self, category_path):
        """
        Parse VisA directory structure with 80/20 train/test split for normal samples.
        """
        normal_dir = os.path.join(category_path, 'Data', 'Images', 'Normal')
        anomaly_dir = os.path.join(category_path, 'Data', 'Images', 'Anomaly')
        mask_dir = os.path.join(category_path, 'Data', 'Masks', 'Anomaly')
        
        # Support both JPG and PNG extensions
        normal_imgs = sorted(glob.glob(os.path.join(normal_dir, '*.[pj][pn][gG]')))
        split_idx = int(len(normal_imgs) * 0.8)
        
        if self.is_train:
            # Training utilizes 80% of the normal samples
            self.image_paths.extend(normal_imgs[:split_idx])
            self.labels.extend([0] * split_idx)
            self.mask_paths.extend([None] * split_idx)
        else:
            # Testing utilizes the remaining 20% normal samples
            test_normals = normal_imgs[split_idx:]
            self.image_paths.extend(test_normals)
            self.labels.extend([0] * len(test_normals))
            self.mask_paths.extend([None] * len(test_normals))
            
            # Incorporate all anomaly samples
            anomaly_imgs = sorted(glob.glob(os.path.join(anomaly_dir, '*.[pj][pn][gG]')))
            self.image_paths.extend(anomaly_imgs)
            self.labels.extend([1] * len(anomaly_imgs))
            
            for img in anomaly_imgs:
                img_name = os.path.basename(img)
                mask_name = img_name.rsplit('.', 1)[0] + '.png'
                self.mask_paths.append(os.path.join(mask_dir, mask_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        mask_path = self.mask_paths[idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
        else:
            # Generate a blank mask for normal samples to maintain tensor shape consistency
            _, h, w = image.shape
            mask = torch.zeros((1, h, w))
            
        return {
            'image': image,
            'mask': mask,
            'label': self.labels[idx],
            'category': self.category,
            'path': img_path
        }


class ContinualStreamingManager:
    def __init__(self, config):
        self.dataset_name = config['dataset']['name']
        self.root_dir = config['dataset']['root_dir']
        self.batch_size = config['dataset']['batch_size']
        self.num_workers = config['dataset']['num_workers']
        img_size = config['dataset']['img_size']
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        self.categories = self._get_categories()
        self.current_task_idx = 0
        self.test_datasets_history = []

    def _get_categories(self):
        categories = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        return sorted(categories)

    def get_next_task(self):
        """
        Yields dataloaders for the next incremental step.
        Returns: (train_loader, cumulative_test_loader)
        """
        if self.current_task_idx >= len(self.categories):
            return None, None
            
        current_category = self.categories[self.current_task_idx]
        
        train_dataset = ContinualAnomalyDataset(
            root_dir=self.root_dir, 
            dataset_name=self.dataset_name,
            category=current_category, 
            is_train=True, 
            transform=self.transform, 
            mask_transform=self.mask_transform
        )
        
        current_test_dataset = ContinualAnomalyDataset(
            root_dir=self.root_dir, 
            dataset_name=self.dataset_name,
            category=current_category, 
            is_train=False,
            transform=self.transform, 
            mask_transform=self.mask_transform
        )
        
        self.test_datasets_history.append(current_test_dataset)
        concat_test_dataset = ConcatDataset(self.test_datasets_history)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            drop_last=True
        )
        
        test_loader = DataLoader(
            concat_test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        self.current_task_idx += 1
        return train_loader, test_loader