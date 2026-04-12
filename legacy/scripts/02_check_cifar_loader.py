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
