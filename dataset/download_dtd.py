import argparse
import os
import shutil
import tarfile
import urllib.error
import urllib.request


DTD_URLS = [
    "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
]


def _dataset_complete(data_dir: str) -> bool:
    return os.path.isdir(os.path.join(data_dir, "dtd", "images"))


def _safe_extract_tar(tar_path: str, extract_dir: str) -> None:
    extract_dir_abs = os.path.abspath(extract_dir)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extract_dir, member.name))
            if not member_path.startswith(extract_dir_abs + os.sep) and member_path != extract_dir_abs:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(path=extract_dir)


def _download_with_fallbacks(urls, output_path: str) -> None:
    last_error = None
    for idx, url in enumerate(urls, start=1):
        try:
            print(f"Attempt {idx}/{len(urls)}: downloading from {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response, open(output_path, "wb") as target:
                shutil.copyfileobj(response, target)
            print(f"Download completed: {output_path}")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Download failed from {url}: {exc}")

    raise RuntimeError(
        "All download URLs failed. Please download dtd-r1.0.1.tar.gz manually "
        f"into {os.path.dirname(output_path)}. Last error: {last_error}"
    )


def download_and_extract_dtd(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "dtd-r1.0.1.tar.gz")

    if _dataset_complete(data_dir):
        extract_dir = os.path.join(data_dir, "dtd")
        print(f"DTD dataset already prepared at: {extract_dir}. Skipping download/extract.")
        return extract_dir

    if not os.path.exists(tar_path):
        _download_with_fallbacks(DTD_URLS, tar_path)
    else:
        print(f"Using existing archive: {tar_path}")

    print("Extracting archive...")
    _safe_extract_tar(tar_path, data_dir)

    if not _dataset_complete(data_dir):
        raise RuntimeError(
            "Extraction finished but DTD structure is incomplete. "
            "Expected dtd/images to exist."
        )

    extract_dir = os.path.join(data_dir, "dtd")
    print(f"DTD dataset is ready at: {extract_dir}")
    return extract_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract DTD textures dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_extract_dtd(args.data_dir)