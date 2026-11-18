# Chuyển HOPE-CAD sang máy mới

Gói này giữ mã nguồn hiện tại (kể cả file chưa commit), `AGENTS.md`,
`AGENTS.local.md`, toàn bộ `agents/`, hai checkpoint pretrained và kết quả gần
đây từ 14–17/11/2026. Có đầy đủ mã và kết quả `nar_full_15`: decoder base,
checkpoint của cả hai nhánh ở cả 15 task, score, metric, source lock và protocol.

Không kèm `data/`, tensor cache đặc trưng (đặc biệt `nar_full_15/features/*.pt`), kết quả cũ,
`.git`, `.vscode`, môi trường ảo, Python cache và cấu hình tài khoản cục bộ.
`dataset/` là mã nguồn và vẫn được giữ. Không thay đổi nguồn khoa học hay kết
quả cũ; manifest trong ZIP liệt kê file và SHA-256 đã chuyển.
Metadata JSON của cache NAR được giữ để có danh sách và hash ảnh đầu vào gốc.

## Chuẩn bị môi trường

Giải nén, mở terminal ở thư mục `HOPE-CAD/`. NAR lịch sử dùng Python 3.11 và
PyTorch `2.10.0+cu126`; `pixi.toml` ở repo là cấu hình cũ Python 3.14, không phải
môi trường đã chạy NAR. Cài bản torch/torchvision phù hợp GPU/driver trước, rồi:

```bash
python -m pip install -r requirements.txt
bash scripts/download_transfer_data.sh
```

`TRANSFER_MANIFEST.json` ghi phiên bản thư viện quan sát lúc đóng gói (Python
3.11.17, torch 2.10.0+cu126, torchvision 0.25.0+cu126, timm 1.0.30, numpy 2.4.6).
Đây là snapshot môi trường hiện có; source lock lịch sử xác nhận riêng torch
2.10.0+cu126. `requirements.txt` dùng khoảng phiên bản, không phải lock tái lập.

Lệnh thứ hai tải MVTec AD, xác minh SHA-256 của archive mirror được ghim phiên
bản, rồi giải nén vào `data/mvtec`. Có thể tải tiếp khi gián đoạn. Nếu đã có
archive: `bash scripts/download_transfer_data.sh /path/to/archive.tar.xz`.
Helper này chỉ tải MVTec; không tải VisA/DTD. Không cần tải lại hai checkpoint
pretrained vì chúng có sẵn trong gói.

Mirror MVTec:
https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/blob/c67a6eab4c08d65c1990db9a273127039af7f9cc/mvtec_anomaly_detection.tar.xz
SHA-256: `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`.
Đây là kiểm tra danh tính archive tải về, không phải khẳng định dữ liệu đã tải
ở máy mới giống từng byte với dữ liệu từng dùng trong benchmark lịch sử.

## Mã NAR full-15 và cách chạy lại

Runner gốc: `exps/nar_pilot_00/full.py`; detector: `detector.py`; thuật toán:
`core.py`; đo chi phí: `cost.py`; protocol: `agents/plans/nar_full_15_protocol.md`.
Kết quả cũ: `results/hope_cad/nar_full_15/`.

Runner gốc cố ý từ chối ghi đè kết quả đã khóa. Launcher dưới đây chỉ đổi thư
mục đầu ra, giữ nguyên mã nguồn và tất cả kiểm tra của runner. Chỉ chạy các pha
GPU/đánh giá khi bạn quyết định cho phép benchmark mới theo `AGENTS.md` và task
của lần chạy mới; việc tạo gói này chưa thực thi hay cho phép benchmark mới.

```bash
python scripts/rerun_nar_full_15.py lock --out results/hope_cad/nar_full_15_rerun_01
python scripts/rerun_nar_full_15.py smoke --out results/hope_cad/nar_full_15_rerun_01 --owner-authorized
python scripts/rerun_nar_full_15.py train --out results/hope_cad/nar_full_15_rerun_01 --owner-authorized
python scripts/rerun_nar_full_15.py score --out results/hope_cad/nar_full_15_rerun_01 --owner-authorized
python scripts/rerun_nar_full_15.py metrics --out results/hope_cad/nar_full_15_rerun_01
python scripts/rerun_nar_full_15.py cost --out results/hope_cad/nar_full_15_rerun_01 --owner-authorized
```

Chạy từng pha và kiểm tra kết quả trước khi chuyển pha. `smoke` và `train` tự
tạo cache train; `score` chỉ tạo cache test sau khi cả hai nhánh hoàn tất.
Không cần copy cache 13 GB. Máy mới cần CUDA, dung lượng trống cho dataset và
cache, và phải qua các chốt tài nguyên gốc. Giới hạn 8 GB áp dụng cho file ZIP
chuyển máy, không phải dung lượng cần khi chạy.

Kiểm tra CPU của NAR:

```bash
python -m pytest -q -p no:cacheprovider exps/nar_pilot_00/test_core.py exps/nar_pilot_00/test_detector.py
```

Mã và checkpoint được giữ nguyên byte; chưa kiểm chứng benchmark trên máy mới
hay tính giống hệt kết quả giữa các GPU/phiên bản thư viện. Các kết quả lịch sử
đã được giữ để đọc lại mà không cần chạy lại train hoặc chấm lại dữ liệu test.
