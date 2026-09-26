"""
train_mvtec.py
--------------
Script chạy thực nghiệm Continual Anomaly Detection với MVTec.
Tích hợp Weights & Biases để tracking Surprise Score và ACC real-time.
Hỗ trợ cả dữ liệu Dummy (để test) và dữ liệu thật từ DataLoader.
"""

import torch
import wandb
import os
import sys
from tqdm import tqdm

# Thêm project root vào path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.meta_nath_core import MetaNATHCore
from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager

def train_continual_learning(use_dummy=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Bắt đầu chiến dịch trên {device.upper()}")

    # 1. Khởi tạo W&B Dashboard
    wandb.init(
        project="MetaNATH-CAD",
        name="Phase1_2_MVTec_Integration" if not use_dummy else "Dry_Run_Zero_Crash",
        config={
            "backbone": "dinov2-base",
            "tau_acc": 0.25,
            "max_coreset_size": 1000,
            "d": 768,
            "is_dummy": use_dummy
        }
    )

    # 2. Khởi tạo Core System
    model = MetaNATHCore(
        d=wandb.config.d, 
        tau_acc=wandb.config.tau_acc, 
        max_coreset_size=wandb.config.max_coreset_size,
        device=device
    ).to(device)

    # 3. Chuẩn bị nguồn dữ liệu
    if use_dummy:
        print("⚠️ Đang sử dụng dữ liệu GIẢ (Dummy Data) để test pipeline...")
        # Giả lập 1 task với 10 batch
        dataloader = [
            (torch.randn(4, 3, 224, 224), torch.randint(0, 2, (4,))) 
            for _ in range(10)
        ]
    else:
        print("🔥 Đang kết nối với DataLoader THẬT từ Hồ Nam...")
        config = load_config("conf/config.yaml")
        manager = ContinualStreamingManager(config)
        # Lấy task đầu tiên để test
        train_loader, _, task_info = manager.get_next_task()
        if train_loader is None:
            print("❌ Không tìm thấy dữ liệu MVTec tại đường dẫn cấu hình!")
            return
        dataloader = train_loader
        print(f"✅ Đã kết nối Task: {task_info['category']}")

    print("🔥 Bơm dữ liệu vào hệ thống...")
    model.train() 

    for step, batch in enumerate(tqdm(dataloader)):
        # Xử lý linh hoạt cả list (dummy) và dict (real dataloader)
        if isinstance(batch, dict):
            images = batch['img'].to(device)
            labels = batch['anomaly']
        else:
            images, labels = batch[0].to(device), batch[1]
        
        # Chạy Forward Pass Phase 1 (TTT) & Phase 2 (Coreset)
        # Hệ thống tự động thích nghi và cập nhật bộ nhớ
        out = model(images, task_id=0, update_coreset=True)

        # 4. Gắn đồng hồ đo lường: Log lên W&B
        wandb.log({
            "step": step,
            "Metrics/Surprise_Score": out["surprise"],
            "Metrics/ACC_Score": out["acc_score"],
            "System/Coreset_Size": model.coreset_size,
            "System/Approval_Rate": model.gating.approval_rate(),
            "Label/Has_Anomaly": int(labels.max().item())
        })

    print(f"✅ Hoàn tất luồng học. Coreset hiện có: {model.coreset_size} mẫu.")
    wandb.finish()

if __name__ == "__main__":
    # Bạn có thể đổi use_dummy=False khi đã có dataset MVTec trong folder data/mvtec
    train_continual_learning(use_dummy=True)
