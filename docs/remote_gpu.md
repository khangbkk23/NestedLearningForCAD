# Chạy v7 trên Linux/RTX5090 qua SSH

V7 là **workspace mới**, phương pháp vẫn `hope_cad_v5_1_causal_scaled_utility`. Local engineering checks không chứng minh đã chạy5090. Không thuê máy, đăng nhập hoặc chạy benchmark tự động. Không dùng old T4 checkpoint để resume bằng execution mới; existing seed0 endpoints/matrices được giữ nguyên ở archive.

## 1. Chuẩn bị và chuyển source

V7 tự đủ từ source/Git checkout; không cần `.agents`, sibling worktree hoặc old Kaggle evidence. Windows local có thể dùng `pixi install --locked`, `pixi run test`. Đóng gói:

```powershell
pixi run python -B -m hope_cad source --output dist/hope_cad_v7_source_r1.zip
```

Copy ZIP tới máy thuê bằng SSH/SCP theo host/port thật người dùng cung cấp; so SHA256 với receipt rồi giải nén vào thư mục trống. Không chứa datasets/checkpoints/.pixi/.venv/artifacts hoặc credentials. Linux script gọi bằng `bash`, không phụ thuộc executable bit của Windows.

## 2. Runtime riêng

Cần Linux x86_64, NVIDIA driver tương thích CUDA12.8, Python3.13 và `nvidia-smi`. Candidate direct pins ở requirements/remote-cu128.txt và conf/runtime/linux_cuda128.json; không là transitive lock hoặc runtime đã nghiệm thu. Setup giữ `pip freeze --all`/collect_env để ghi toàn bộ môi trường cài thật. Không đổi package để lách gate; khác candidate phải review runtime mới.

```bash
nvidia-smi
bash scripts/remote/setup.sh /absolute/path/to/python3.13
.venv/bin/python -B -m hope_cad.local_tests
```

Setup tạo venv mới, không cài vào system Python hoặc đổi driver; từ chối overwrite venv có sẵn. Nếu chưa có Python3.13, chuẩn bị bằng environment manager/image đã chọn rồi đưa path chính xác; script không tự thay host. PyTorch cung cấp [2.11.0/torchvision0.26.0 CUDA12.8 wheels](https://pytorch.org/get-started/previous-versions/); [Blackwell support bắt đầu với CUDA12.8 builds](https://pytorch.org/blog/pytorch-2-7/). Điều đó không thay actual kernel/numerical gate trên GPU thuê.

## 3. Inputs và remote preflight — bước người dùng chạy trước

Giải nén MVTec AD riêng, layout `<data-root>/bottle/train/good/...`; cung cấp ViT checkpoint đúng SHA trong revision_contract_v5_1.json. Không attach old weights/evidence. Chỉ train normals được discovery ở preflight/train; test/GT chỉ vào evaluator sau train completion.

```bash
bash scripts/remote/run.sh preflight \
  --run-dir /workspace/runs/preflight_v7_01 --device cuda:0 \
  --expected-gpu 'NVIDIA GeForce RTX 5090' \
  --data-root /workspace/data/mvtec \
  --checkpoint /workspace/weights/vit_base_patch8_224.augreg_in21k.safetensors
```

Preflight cố định: synthetic full-grid exact/FO meta-backward, reference/optimized NN, discarded24normals mỗi NESTED/CADIC +resume4 và state/score/full-budget parity. Không giữ learned pilot weights làm initialization, không anomaly test, không tune để PASS. Source/runtime/data/checkpoint binding phải khớp mọi lệnh sau; đo thời gian thật rồi mới dự trù full run. `remote_preflight_passed_pending_review` cần gửi ZIP/log cho review trước full training; `local_synthetic_smoke_passed` không mở benchmark.

Nếu đã có venv Windows, chỉ local CPU smoke:

```powershell
pixi run python -B -m hope_cad preflight --local-smoke --device cpu --run-dir results/local-smoke-01
```

## 4. Fresh matched jobs sau khi preflight được review

Chạy trong `tmux`/process manager của máy thuê để SSH disconnect không giết process. Công cụ đó cần có trên host; repo không khẳng định đã cài hay thiết lập monitoring.

```bash
bash scripts/remote/run.sh train \
  --run-dir /workspace/runs/cohort_v7_01 --job NESTED_seed0 \
  --preflight /workspace/runs/preflight_v7_01 --device cuda:0 --session-hours 8 \
  --data-root /workspace/data/mvtec --checkpoint /workspace/weights/vit_base_patch8_224.augreg_in21k.safetensors
```

`train` chỉ học stream3.599normals, lưu mỗi8ảnh/task/end; không tự eval. Pause thì cùng lệnh/run-dir/source/runtime/data để tiếp đúng cursor. Giữ log+ZIP, xem handoff; technical/scientific failure không tự retry/retune. Hard kill có thể để stale `.writer.lock`: xác minh **PID trên đúng host không còn chạy và checkpoint/evidence hợp lệ** trước khi gỡ lock thủ công; không tự xóa lock để cướp writer.

Sau `train_complete_pending_review`, dùng cùng arguments thay `train` bằng `evaluate` để final15cells; rồi `matrix` thêm105cells. Không training trong hai lệnh đó. Matched CADIC dùng `--job CADIC_seed0` cùng cohort/run-dir và recipe/runtime; giữ mỗi latest export nếu nhiều phiên. Full27jobs không tự hoàn thành chỉ từ hai seed0 jobs.

```bash
bash scripts/remote/run.sh export --run-dir /workspace/runs/cohort_v7_01 --output /workspace/backup/cohort_v7_01.zip
bash scripts/remote/run.sh aggregate --inputs /workspace/backup/cohort_v7_01.zip --output /workspace/backup/summary.json
```

`--statistics` chỉ khi đủ3seeds cho NESTED và comparator theo frozen study. Source/runtime cohorts khác bị từ chối; không ghép v6T4 với v7fresh để suy paired claims. VisA chưa có runner/protocol trong study này; không đổi dataset path rồi gọi prospective confirmation.

## 5. Lưu trữ và giới hạn

`run.sh` ghi log theo timestamp/PID, stdout unbuffered; train tiến độ mỗi8ảnh và evaluator mỗicell. Soft budget có180giây reserve, không ETA và không hard cloud-session limit. Checkpoint/evidence nằm **trên server** cho tới khi copy về máy người dùng hoặc storage bền đã cấu hình. Kiểm download/SHA trước stop/destroy; disk server không phải offsite backup.

Export có manifest và streaming SHA verification; từ chối writer đang chạy, ZIP trong payload tree hoặc symlink. Giữ failure/pause và mọi old source archives. SIGKILL/powerloss có thể ngăn export; last good atomic checkpoint vẫn cần được lấy từ disk. Không có remote agent/auto uploader/W&B mặc định.

## 6. Bước khoa học tiếp theo

[Workspace contract](main_spec/workspace_v7.md) tách G1/G2engineering, G3realGPU và G4methodrevision. Historical-query slow consolidation vẫn proposal riêng; không được gắn gain của cleanup/preprocessing/hardware vào Nested Learning. Bản v5.1 baseline giữ frozen recipe đã development-exposed; native-checkpoint normalization mismatch không được sửa ngầm. Khóa revision/protocol trước VisA.
