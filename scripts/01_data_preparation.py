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