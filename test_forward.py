import torch
import os
import sys

# Đảm bảo import được các module trong models/
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.meta_nath_core import MetaNATHCore

def dry_run_test():
    print("🚀 Bắt đầu test Zero-Crash (Meta-NATH Integration)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Thiết bị sử dụng: {device.upper()}")
    
    # 1. Khởi tạo Core (Sử dụng cấu hình mặc định: DINOv2-base, 256 patches)
    try:
        core = MetaNATHCore(device=device).to(device)
        print("✅ Khởi tạo Core thành công.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return
    
    # 2. Bơm thử 1 Batch (2 ảnh fake 224x224)
    dummy_x = torch.randn(2, 3, 224, 224).to(device)
    
    print("-> Đang chạy Forward Pass qua MetaNATHCore...")
    try:
        out = core(dummy_x, task_id=0, update_coreset=True)
        print(f"✅ Pass! Surprise Score: {out['surprise']:.4f} | Coreset Size: {core.coreset_size}/{core.coreset.max_size}")
        
        # 3. Giả lập W&B logging logic
        wb_metrics = {
            "train/surprise": out['surprise'], 
            "train/acc_gating": out['acc_score'],
            "train/coreset_size": core.coreset_size
        }
        print(f"-> Dữ liệu sẵn sàng log W&B: {wb_metrics}")
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy Forward Pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dry_run_test()
