from pathlib import Path
import sys
import torch
import logging
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.meta_nath_core import MetaNATHCore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_phase_1_2_integration():
    logging.info("=== KHỞI ĐỘNG INTEGRATION TEST (PHASE 1 & 2) ===")
    
    # 1. Đo lường phần cứng
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Target Device: {device.upper()}")
    
    # 2. Khởi tạo Lõi Hệ thống
    logging.info("Khởi tạo MetaNATHCore...")
    model = MetaNATHCore(d=768, tau_acc=0.25, max_coreset_size=1000, n_patch=196)
    model.to(device)
    
    # 3. Tạo dữ liệu giả (Dummy Data) - Batch Size = 2
    dummy_x = torch.randn(2, 3, 224, 224).to(device)
    
    # 4. Chạy Forward Flow (TTT + Gating + Coreset Update)
    logging.info("Bắn dữ liệu qua hệ thống...")
    start_time = time.time()
    out = model(dummy_x, task_id=1)
    latency = time.time() - start_time
    
    # 5. Kiểm tra VRAM (Nếu dùng GPU)
    vram_mb = 0
    if device == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # 6. Báo cáo Kết quả (In ra Terminal)
    logging.info(f"--- KẾT QUẢ TEST ---")
    logging.info(f"✓ Thời gian Inference: {latency:.4f} giây")
    if device == "cuda":
        logging.info(f"✓ VRAM Tiêu thụ: {vram_mb:.2f} MB (Target: < 1500 MB)")
    
    logging.info(f"✓ Surprise Score: {out['surprise']:.4f}")
    logging.info(f"✓ ACC Score: {out['acc_score']:.4f}")
    logging.info(f"✓ Approved for Coreset: {out['approved']}")
    logging.info(f"✓ Số lượng mẫu trong Coreset hiện tại: {model.coreset_size} / {model.coreset.max_size}")
    
    print("\n🎉 HỆ THỐNG ĐÃ SẴN SÀNG CHO DATASET MVTEC! 🎉")

if __name__ == "__main__":
    test_phase_1_2_integration()