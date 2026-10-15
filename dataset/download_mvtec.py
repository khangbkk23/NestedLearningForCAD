import sys
import os
import shutil
import tempfile
import stat
import tarfile
import zipfile
import urllib.request
from pathlib import Path

# Cấu hình đường dẫn (tương đương REPO_ROOT/data)
SCRIPT_DIR = Path(__file__).parent.resolve()
# Nếu file script nằm trong thư mục con (ví dụ: dataset/), dùng .parent để ra REPO_ROOT
REPO_ROOT = SCRIPT_DIR.parent 
DATA_DIR = REPO_ROOT / "data"
TARGET_DIR = DATA_DIR / "mvtec"
DEFAULT_ARCHIVE = DATA_DIR / "mvtec_anomaly_detection.tar.xz"

DOWNLOAD_URL = "https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", 
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
]

def usage():
    print(f"""Usage:
  python download_mvtec_ad.py [ARCHIVE]

ARCHIVE may be a .tar.xz, .tar.gz, .tar, or .zip file. If omitted, the MVTec
archive is downloaded from the configured source URL into {DEFAULT_ARCHIVE}.
An existing partial download is resumed; a complete archive is reused.
Prepared output: {TARGET_DIR}""")

def validate_tree(root: Path) -> bool:
    for category in CATEGORIES:
        if not (root / category / "train" / "good").is_dir(): return False
        if not (root / category / "test").is_dir(): return False
        if not (root / category / "ground_truth").is_dir(): return False
    return True

def count_images(path: Path) -> int:
    extensions = {'.png', '.jpg', '.jpeg'}
    if not path.is_dir():
        return 0
    return sum(1 for p in path.rglob('*') if p.suffix.lower() in extensions and p.is_file())

def get_remote_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            return int(response.getheader('Content-Length', 0))
    except Exception as e:
        print(f"[ERROR] Download URL is not reachable: {url}\nDetails: {e}", file=sys.stderr)
        sys.exit(2)

def download_file_with_resume(url: str, dest: Path, remote_size: int):
    local_size = dest.stat().st_size if dest.exists() else 0
    
    if local_size == remote_size:
        print(f"[OK] Archive already downloaded: {dest} ({local_size} bytes).")
        return
    elif local_size > remote_size:
        print(f"[ERROR] Existing archive is larger than the remote file: {dest}", file=sys.stderr)
        sys.exit(2)
        
    req = urllib.request.Request(url)
    if local_size > 0:
        print(f"[INFO] Resuming archive download at byte {local_size}/{remote_size}...")
        req.add_header('Range', f'bytes={local_size}-')
        mode = 'ab'
    else:
        print(f"[INFO] Downloading MVTec AD archive ({remote_size} bytes)...")
        mode = 'wb'

    try:
        with urllib.request.urlopen(req) as response, open(dest, mode) as out_file:
            # Nếu server không hỗ trợ resume (không trả về 206), ghi đè lại từ đầu
            if local_size > 0 and response.status != 206:
                print("[INFO] Server doesn't support resume. Downloading from scratch...")
                out_file.close()
                with open(dest, 'wb') as retry_file:
                    shutil.copyfileobj(response, retry_file)
            else:
                shutil.copyfileobj(response, out_file)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        sys.exit(2)

def main():
    # 1. Kiểm tra tham số đầu vào
    archive_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    
    if archive_arg in ["-h", "--help"]:
        usage()
        sys.exit(0)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Xử lý tải xuống hoặc nhận file cục bộ
    if not archive_arg:
        archive_path = DEFAULT_ARCHIVE
        print("[INFO] Checking download URL...")
        remote_size = get_remote_size(DOWNLOAD_URL)
        if remote_size <= 0:
            print("[ERROR] Download URL did not provide a valid Content-Length.", file=sys.stderr)
            sys.exit(2)
        download_file_with_resume(DOWNLOAD_URL, archive_path, remote_size)
    else:
        archive_path = Path(archive_arg).resolve()

    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        print(f"[ERROR] Archive does not exist or is empty: {archive_path}", file=sys.stderr)
        sys.exit(2)

    # 3. Kiểm tra nếu đích đến đã sẵn sàng
    if TARGET_DIR.is_dir() and validate_tree(TARGET_DIR):
        print(f"[OK] MVTec is already prepared at {TARGET_DIR}; archive was not modified.")
        sys.exit(0)

    if TARGET_DIR.exists() and any(TARGET_DIR.iterdir()):
        print(f"[ERROR] Destination exists but is incomplete: {TARGET_DIR}", file=sys.stderr)
        print("        Move it aside before preparing a new archive.", file=sys.stderr)
        sys.exit(3)

    # 4. Giải nén vào thư mục tạm (Stage Dir)
    def safe_target(stage: Path, name: str) -> Path:
        path = (stage / name).resolve()
        if os.path.commonpath([str(stage), str(path)]) != str(stage):
            raise RuntimeError(f"unsafe archive path: {name}")
        return path

    print(f"[INFO] Extracting archive...")
    with tempfile.TemporaryDirectory(dir=DATA_DIR, prefix=".mvtec_extract.") as stage_dir_str:
        stage_dir = Path(stage_dir_str).resolve()
        
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                for info in zf.infolist():
                    safe_target(stage_dir, info.filename)
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == stat.S_IFLNK:
                        raise RuntimeError(f"symlink in archive is not allowed: {info.filename}")
                zf.extractall(stage_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                members = tf.getmembers()
                for member in members:
                    safe_target(stage_dir, member.name)
                    if member.issym() or member.islnk() or member.isdev():
                        raise RuntimeError(f"link/device in archive is not allowed: {member.name}")
                tf.extractall(stage_dir, members=members)
        else:
            print(f"[ERROR] unsupported archive format: {archive_path}", file=sys.stderr)
            sys.exit(2)

        # Tìm SOURCE_ROOT chứa dataset hợp lệ trong đống đổ nát sau giải nén
        source_root = None
        for candidate in stage_dir.iterdir():
            if candidate.is_dir() and validate_tree(candidate):
                source_root = candidate
                break
        if not source_root and validate_tree(stage_dir):
            source_root = stage_dir

        if not source_root:
            print("[ERROR] Archive extracted, but no canonical 15-category MVTec tree was found.", file=sys.stderr)
            print("        Expected <root>/<category>/{train/good,test,ground_truth}.", file=sys.stderr)
            sys.exit(4)

        # 5. Di chuyển các thư mục category vào TARGET_DIR chính thức
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        for category in CATEGORIES:
            shutil.move(str(source_root / category), str(TARGET_DIR / category))

    # 6. Kiểm tra hậu di chuyển & In bảng thống kê kết quả
    if not validate_tree(TARGET_DIR):
        print(f"[ERROR] Prepared tree failed post-move validation: {TARGET_DIR}", file=sys.stderr)
        sys.exit(5)

    failed = False
    total_train = 0
    total_test = 0
    
    print(f"{'category':<14} {'train_good':>10} {'test_imgs':>10} {'gt_masks':>12}")
    print(f"{'--------------':<14} {'----------':>10} {'----------':>10} {'------------':>12}")
    
    for category in CATEGORIES:
        root = TARGET_DIR / category
        train_count = count_images(root / "train" / "good")
        test_count = count_images(root / "test")
        gt_count = count_images(root / "ground_truth")
        
        print(f"{category:<14} {train_count:>10} {test_count:>10} {gt_count:>12}")
        total_train += train_count
        total_test += test_count
        
        if train_count == 0 or test_count == 0:
            failed = True

    print(f"\nTotal train/good images: {total_train}")
    print(f"Total test images      : {total_test}")
    
    if failed:
        print("[ERROR] One or more categories contain no train or test images.", file=sys.stderr)
        sys.exit(6)

    print(f"[OK] MVTec AD benchmark tree is ready at {TARGET_DIR}")
    print(f"[INFO] Archive retained at {archive_path}")
    print(f"[INFO] Benchmark root: {TARGET_DIR}")

if __name__ == "__main__":
    main()
