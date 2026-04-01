import os
import urllib.request
import tarfile

def download_and_extract_mvtec(data_dir="data"):
    mvtec_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "mvtec_anomaly_detection.tar.xz")
    extract_dir = os.path.join(data_dir, "mvtec")
    
    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(mvtec_url, tar_path)
        
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(tar_path, "r:xz") as tar:
        tar.extractall(path=extract_dir)

if __name__ == "__main__":
    download_and_extract_mvtec()