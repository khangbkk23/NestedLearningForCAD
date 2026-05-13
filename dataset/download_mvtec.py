import argparse
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from tqdm import tqdm


MVTec_URLS = [
    "https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz",
    "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection.tar.xz",
]

EXPECTED_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def _dataset_complete(root_dir):
    return all(os.path.isdir(os.path.join(root_dir, cls_name)) for cls_name in EXPECTED_CLASSES)


def _flatten_nested_root(extract_dir):
    nested_root = os.path.join(extract_dir, "mvtec_anomaly_detection")
    if not os.path.isdir(nested_root):
        return

    if _dataset_complete(extract_dir):
        return

    if _dataset_complete(nested_root):
        for entry in os.listdir(nested_root):
            src = os.path.join(nested_root, entry)
            dst = os.path.join(extract_dir, entry)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested_root, ignore_errors=True)


def _safe_extract_tar(tar_path, extract_dir):
    extract_dir_abs = os.path.abspath(extract_dir)
    with tarfile.open(tar_path, "r:xz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extract_dir, member.name))
            if not member_path.startswith(extract_dir_abs + os.sep) and member_path != extract_dir_abs:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(path=extract_dir)


def _download_with_fallbacks(urls, output_path):
    last_error = None
    for idx, url in enumerate(urls, start=1):
        try:
            print(f"Attempt {idx}/{len(urls)}: downloading from {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            
            with urllib.request.urlopen(request, timeout=60) as response:
                total_size = int(response.info().get('Content-Length', 0))
                block_size = 1024 * 8
                
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                    with open(output_path, "wb") as target:
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            target.write(buffer)
                            pbar.update(len(buffer))
            
            print(f"Download completed: {output_path}")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Download failed from {url}: {exc}")

    raise RuntimeError(
        "All download URLs failed. Please download mvtec_anomaly_detection.tar.xz manually "
        f"into {os.path.dirname(output_path)}. Last error: {last_error}"
    )


def download_and_extract_mvtec(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "mvtec_anomaly_detection.tar.xz")
    extract_dir = os.path.join(data_dir, "mvtec")
    os.makedirs(extract_dir, exist_ok=True)

    _flatten_nested_root(extract_dir)
    if _dataset_complete(extract_dir):
        print(f"MVTec dataset already prepared at: {extract_dir}. Skipping download/extract.")
        return extract_dir

    if not os.path.exists(tar_path):
        _download_with_fallbacks(MVTec_URLS, tar_path)
    else:
        print(f"Using existing archive: {tar_path}")

    print("Extracting archive...")
    try:
        _safe_extract_tar(tar_path, extract_dir)
    except (EOFError, tarfile.ReadError, lzma.LZMAError) as e:
        print(f"Archive is corrupted: {e}")
        print(f"Deleting corrupted file: {tar_path}")
        os.remove(tar_path)
        print("Please run the script again to restart the download.")
        return
    _flatten_nested_root(extract_dir)

    if not _dataset_complete(extract_dir):
        raise RuntimeError(
            "Extraction finished but dataset structure is incomplete. "
            "Expected MVTec class folders were not found."
        )

    print(f"MVTec dataset is ready at: {extract_dir}")
    return extract_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract MVTec AD dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_extract_mvtec(args.data_dir)