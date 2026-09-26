# dataset/benchmark_protocol_v1.py
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

CANONICAL_MVTEC_TASKS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

class _ImageDataset(Dataset):
    def __init__(self, root, rows, image_size, mean, std, training):
        self.root, self.rows = Path(root).resolve(), rows
        self.image_size, self.training = image_size, training
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["relative_path"]
        if self.training and path.resolve() != path:
            raise ValueError("Training path changed into a symlink")
        with Image.open(path) as source:
            image = source.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255
        image = (image - self.mean) / self.std
        if self.training:
            return {"images": image, "labels": 0, "relative_path": row["relative_path"]}
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if row["label"]:
            with Image.open(self.root / row["mask_path"]) as source:
                mask = np.array(source.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)) > 0
        return {"images": image, "labels": row["label"], "masks": torch.from_numpy(mask).bool(),
                "relative_path": row["relative_path"]}

class MVTecContinualProtocol:
    def __init__(self, config: dict, manifest: dict, method: dict, seed: int, num_workers=0):
        from dataset.benchmark_manifest_v1 import validate_manifest
        validate_manifest(manifest, config, seed)
        self.config, self.manifest, self.seed = config, manifest, seed
        self.root = Path(config["dataset"]["root"]).resolve()
        self.task_names = list(config["task_order"])
        runtime, pre = method["runtime"], method["preprocessing"]
        self.batch_size = int(runtime["batch_size"])
        self.num_workers = num_workers
        self.image_size, self.mean, self.std = int(pre["image_size"]), pre["mean"], pre["std"]

    def training_paths(self, task_id):
        return [e for e in self.manifest["entries"] if e["task_id"] == task_id]

    def build_train_loader(self, task_id, max_images=None):
        rows = self.training_paths(task_id)
        if max_images is not None:
            rows = rows[:max_images]
        return self._loader(rows, training=True)

    def test_paths(self, task_id):
        task = self.task_names[task_id]
        folder = self.root / task / "test"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        rows = []
        for defect in sorted(p for p in folder.iterdir() if p.is_dir()):
            for path in sorted(defect.glob("*.png")):
                label = int(defect.name != "good")
                mask = f"{task}/ground_truth/{defect.name}/{path.stem}_mask.png" if label else None
                if mask and not (self.root / mask).is_file():
                    raise FileNotFoundError(self.root / mask)
                rows.append(dict(relative_path=path.relative_to(self.root).as_posix(), label=label, mask_path=mask))
        if not rows:
            raise ValueError(f"Empty official test task: {task}")
        return rows

    def build_test_loader(self, task_id):
        return self._loader(self.test_paths(task_id), training=False)

    def _loader(self, rows, training):
        dataset = _ImageDataset(self.root, rows, self.image_size, self.mean, self.std, training)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers, generator=torch.Generator().manual_seed(self.seed))

    def metadata(self):
        return {"protocol_id": self.config["id"], "task_order": self.task_names,
                "train_manifest_digest": self.manifest["digest"], "drop_last": False,
                "evaluation_grid": [self.image_size, self.image_size],
                "image_resize": "PIL bilinear direct square", "mask_resize": "PIL nearest"}