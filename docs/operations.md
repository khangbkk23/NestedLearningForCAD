# Vận hành workload B319

Đối chiếu source: `74d975b151f59ad0b5fbf23be392e2d891f350bf` — 11/11/2026.

[Đường đọc](../README.md#đường-đọc) · [Experiment lifecycle](../experiments/README.md) · [Artifact layout](provenance.md#artifacts-và-local)

Lệnh chạy từ root checkout. Examples dùng `b319-s0-new-001` làm placeholder;
chọn ID mới, không dùng lại execution đã đóng. Hướng dẫn không tự cấp quyền tải,
upload hoặc chạy GPU. Thay `/path/to/private-token-file` bằng path thật ngoài Git.

## 1. Environment và tests

```sh
pixi install --locked -e cad
pixi run --locked -e cad train --help
```

Versions/platforms do [pixi.toml](../pixi.toml)/[pixi.lock](../pixi.lock) định nghĩa.
CLI kiểm Python/package versions, không chỉ import. Full training/GPU smoke yêu
cầu Linux CUDA; Windows có thể package/input-check/test. Loop dùng một logical
GPU `cuda:0`, không tự dùng cả hai GPUs. Không pip-upgrade packages trong locked env.

Tests nhỏ dùng environment riêng, không tải data/train/GPU:

```sh
pixi install --locked -e cad-test
pixi run --locked -e cad-test test
```

Tests không thay scientific acceptance/runtime qualification. Chỉnh docs không
cần chạy lại benchmark.

## 2. Inputs và input-only check

Khi đã được phép download:

```sh
pixi run --locked -e cad assets
pixi run --locked -e cad train --check
```

Downloader lấy weights đúng SHA và pinned MVTec archive, kiểm extraction
members/path/cap; archive cache bị xóa sau extract thành công. Dataset directory
có sẵn không được downloader tự xác minh lại toàn tree: phải input-check tiếp.
`assets.json` không thay verification của manifest/test GT.

`train --check` kiểm config, versions, weight SHA và toàn training manifest;
**không train hoặc khởi tạo CUDA**. Nó chưa gọi `data.prepare` để kiểm full test/GT.
`test_images_expected` là số kỳ vọng, không phải test population đã verified.

Mặc định dùng `.local/inputs/`. `NRCAD_LOCAL_ROOT` chọn dedicated artifact root;
`assets --root PATH` cũng chọn root nhưng lệnh sau phải dùng cùng root.
`train --data PATH --weights PATH` hỗ trợ inputs có sẵn. Không ngầm dùng artifacts
của checkout khác hoặc sửa expected hashes khi inputs mismatch.

## 3. GPU smoke

Khác input-only check: lệnh dưới thực sự chạy encoder, một decoder fast update,
scorer, tiny conditional VJP và CTX-policy check. Chỉ chạy trên Linux GPU khi đã
được phép. Tạo parent cho output trước:

```sh
mkdir -p .local/runs/b319/b319-s0-smoke-001/receipts
pixi run --locked -e cad python scripts/preflight.py --data .local/inputs/mvtec --weights .local/inputs/weights/dinov2_vitb14_reg4_pretrain.pth --output .local/runs/b319/b319-s0-smoke-001/receipts/gpu-preflight.json
```

Smoke không giữ state cho full run. PASS không là full-model slow-event,
full-test hoặc cross-device continuation acceptance. Chọn output mới để giữ
lịch sử probes; smoke CLI không có closed-receipt overwrite guard riêng.

## 4. Fresh full run trên Linux GPU

Commit card/recipe trước execution. Exact source bytes vẫn được bundle kể cả
source dirty; nên chốt baseline rõ. Policy chi tiết ở experiment lifecycle.

```sh
pixi run --locked -e cad train --run-id b319-s0-new-001 --arm all
```

Ba arms chạy tuần tự trong child processes; arm fail thì orchestration dừng.
Có thể chọn một arm để debug, nhưng export bảng ba arms cần đủ ba study outputs.
Không có `--resume`; `final-state.safetensors` là audit checkpoint. CLI khóa seed 0,
không chỉ đổi seed config để gọi là replication seeds mới.

Logs: `B319_ARM_START`, `FOUNDATION_TRAIN`, `FULLTEST_EVAL_PROGRESS`,
`FULLTEST_CATEGORY`, `FOUNDATION_SNAPSHOT`, `FULLTEST_CHUNK_VERIFIED`.
`COMPLETE_REVIEW_PENDING` tách completion khỏi scientific acceptance. Caps thực
thi nằm trong config, trainer/storage và notebook wrappers; không suy toàn-job
ETA/cap chỉ từ một trường config. Local `--arm all` không có cùng outer watchdog
process-group/deadline như notebook.

Sau local terminal run:

```sh
pixi run --locked -e cad python scripts/record_run.py --run .local/runs/b319/b319-s0-new-001
```

Raw ở `.local`; exporter đưa receipts và matrices/results sẵn có vào experiments.
Failure/cancellation vẫn được giữ dù thiếu metrics. Hard kill có thể chưa seal
receipt: giữ live record/logs, ghi gap, không bịa terminal receipt hoặc tái dùng
run ID đang dở như fresh run.

## 5. Kaggle: package, launch, monitor và fetch

Local package không upload/launch hoặc cần GPU:

```sh
pixi run --locked -e cad package --owner qoctrn --run-id b319-s0-new-001
```

Tạo `.local/runs/b319/b319-s0-new-001/package/{dataset,kernel}/` với source ZIP,
BUNDLE hashes và notebook đã bind. **Không upload template `notebooks/b319.ipynb`
trực tiếp:** template còn source pins/run placeholder của lượt trước.

Dùng Kaggle UI hoặc CLI đã được setup để upload dataset source và kernel package,
đúng owner; kiểm dataset private, kernel private, GPU/Internet. Generated kernel
metadata đặt private/GPU/Internet, nhưng metadata local không chứng minh actual
remote visibility. `kaggle_run.py` chỉ có `pack`/`fetch`, không tự upload/launch.
Kaggle tools dùng Python/environment có Kaggle CLI đã authenticate; không thêm
Kaggle/auth packages vào locked ML environment chỉ để gọi các tools này.

Notebook dựng Pixi ở `/kaggle/tmp`, tải assets, input-check, GPU smoke rồi mới train.
Whole-job deadline/command reserves nằm trong notebook; per-arm/resource guards
trong runner. Completion vẫn cần audit.

Monitor từ root repo bằng Python của environment có Kaggle CLI:

```sh
python scripts/watch_kaggle.py --kernel qoctrn/b319-s0-new-001 --token-file /path/to/private-token-file
```

Có thể bỏ `--token-file` khi CLI đã cấu hình đúng owner. Token contents không vào
command/chat/bundle. Ctrl+C chỉ ngắt viewer, không dừng job. Viewer reconnect khi
provider còn running; trạng thái không xác định không là completed/failed. `[DONE]`
chỉ nghĩa provider execution completed, không nghĩa scientific PASS.

Sau terminal, fetch **version cụ thể**:

```sh
python scripts/kaggle_run.py fetch --kernel qoctrn/b319-s0-new-001/1 --run-id b319-s0-new-001 --token-file /path/to/private-token-file
```

Mặc định lấy logs/metadata và retained `package/source.zip`, không toàn raw ZIPs.
Thêm `--raw` khi cần raw audit và đã kiểm dung lượng. Evidence được version hóa,
terminal receipt tự export vào experiments; không auto-commit/push. Provider
COMPLETE nhưng thiếu studies có thể chỉ export receipt: kiểm `matrix_complete`,
`analysis_error` và logs trước kết luận.

Analysis lại dùng ID mới để giữ closed report:

```sh
pixi run --locked -e cad python scripts/record_run.py --run .local/runs/b319/b319-s0-new-001 --version v1 --analysis-id recheck-001
```

## Troubleshooting

| Dấu hiệu | Kiểm trước |
| --- | --- |
| Input SHA/manifest mismatch | Đúng root/bytes; không sửa expected hash để lách guard |
| Package từ chối design | Card/config tracked, committed và unchanged |
| Versions mismatch | Đúng locked env; host Kaggle Python khác ML env |
| GPU smoke fail | Giữ logs/result, kiểm device/runtime/case; chưa mở full run |
| Cap/deadline/disk fail | Đúng guard/workload; không tự nới cap hoặc xóa raw độc bản |
| Closed receipt differs | Giữ original, dùng analysis ID mới |
| Watcher disconnected | Kiểm provider; lỗi viewer không là lỗi phương pháp |
