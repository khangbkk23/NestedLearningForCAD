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
