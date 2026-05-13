import os
import requests
import tarfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(filename)}")
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def setup_mvtec_bottle():
    # Tọa độ mục tiêu
    base_path = "data/mvtec"
    os.makedirs(base_path, exist_ok=True)
    
    # Link mirror cho class Bottle (khoảng 150MB)
    url = "https://www.mydrive.ch/shares/38536/703963f466487e49755107937402f1a6/download/420938113-1629951672/bottle.tar.xz"
    target_file = os.path.join(base_path, "bottle.tar.xz")
    
    if not os.path.exists(os.path.join(base_path, "bottle")):
        print(f"🚀 Bắt đầu tải class BOTTLE về {base_path}...")
        try:
            download_file(url, target_file)
            print("📦 Đang giải nén...")
            with tarfile.open(target_file) as tar:
                tar.extractall(path=base_path)
            os.remove(target_file)
            print("✅ Xong! Dữ liệu class Bottle đã sẵn sàng.")
        except Exception as e:
            print(f"❌ Lỗi tải xuống: {e}")
            print("Gợi ý: Nếu link mirror bị lỗi, hãy tải thủ công tại: https://www.mvtec.com/company/research/datasets/mvtec-ad")
    else:
        print("✅ Class Bottle đã tồn tại trong data/mvtec.")

if __name__ == "__main__":
    setup_mvtec_bottle()
