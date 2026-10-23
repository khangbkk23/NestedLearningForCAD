# Chạy v7 với Pixi trên Linux/RTX5090 hoặc Windows local

Pixi là package/environment manager duy nhất. Method accepted vẫn v5.1; H32/full-train là candidate đã implement/local-tested, chưa remote/GPU/AD acceptance. Không tự thuê/SSH/chạy GPU. Lượt đầu chỉ chạy research preflight rồi gửi evidence để review.

## 1. Cài Pixi, clone và pin source

Trên Linux, cài Pixi theo [hướng dẫn chính thức](https://pixi.prefix.dev/latest/installation/):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

Lock được tạo/kiểm với Pixi0.66.0. Cần bản hỗ trợ lockschema6/per-package indexes; giữ phiên bản Pixi đã dùng trong records. Script installer cài Pixi, không cài NVIDIA driver.

User review/commit/push changes mới trước khi thuê; commit672abd1/720b3cd là predecessor chưa chứa Pixi/provisioning mới. Sau khi đã push:

```bash
git clone --branch hope_cad_v7 --single-branch \
  https://github.com/khangbkk23/NestedLearningForCAD.git hope_cad_v7
cd hope_cad_v7
# Pin full SHA user đã chọn/đã push, kiểm git status sạch trước cohort.
git rev-parse HEAD
git status --short
pixi install --locked
```

`pixi install --locked` cài Python, dependencyclosure từ lock và project editable. Không cần Python hệ thống, uv, pip install thủ công hoặc .venv. Không bỏ `--locked` khi chạy remote để Pixi tự resolve khác source đã chốt. Bare `pixi install` dùng cùng default environment nhưng có thể cập nhật lock nếu manifest lệch; ưu tiên locked.

`pixi.toml` là nguồn khai báo dependency; `pixi.lock` khóa cả Linux/Windows. Linux Python3.13/torch2.11+cu128/vision0.26 và scientific pins khớp runtime.json; Windows giữ Python3.14/torch2.12+cu126. Hai platform không được gọi là cùng execution cohort. Linux lock solve không là actual Linux installation/5090kernel acceptance. Không tự thay runtime.json để lách gate.

## 2. Layout tự chứa trong checkout

```text
hope_cad_v7/
  pixi.toml / pixi.lock
  .pixi/                  # Python và packages, ignore
  src/ docs/ tests/        # tracked source
  data/
    archives/             # archive tải dở/đã kiểm
    mvtec/                # bottle/train/good/... đủ15categories
    mvtec.asset.json      # archive/tree integrity receipt
  weights/                # đúng ViT safetensors + receipt
  runs/hq32_01/           # model states/events/metrics/telemetry
  runs/hq32_01.zip
  runs/hq32_01.accounting.json
  records/                # environment, allocation, billing notes
  logs/                   # stdout/stderr theo invocation
  results/                # local checks/CPU summaries
```

Data/weights/runs/records/logs/results đều bị ignore và không thuộc source inventory/hash; không commit vàiGB artifacts. Chúng là dữ liệu của checkout trên máy chạy, không tự đi theo Git clone. Thư mục repo có thể đặt ở bất cứ đâu; defaultpaths neo vào repo root, không parent/worktree khác. Có thể override assets/runtime paths để gắn persistent volume; outputs không được chứa source, đè source hoặc overlap inputs.

## 3. Provision data và backbone bằng task

```bash
pixi run --locked assets
```

Lệnh này tải cả MVTec AD và ViT bằng CPU/network, không train/GPU. Có thể chạy riêng:

```bash
pixi run --locked download-mvtec
pixi run --locked download-vit
```

MVTec mirror HuggingFace `micguida1/mvtech_anomaly_detection` revision `eb692b36147a3b3f12c561b879a062104c1f20b8`, archive5264982680bytes SHA `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`. Đây là mirror bên thứ ba có pinnedbytes, không tự chứng minh tác giả xác nhận mirror. [Dataset gốc/terms](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

ViT từ upstream timm revision4747ebd, file410434646bytes, SHA `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`; giữ nguyên safetensors, không convert .pth. Same lockedSHA với method.json.

Downloader giữ `.part`, kiểm Range response, size/SHA trước publish; lỗi không nhận partial là completed. Dataset giải nén vào staging, từ chối traversal/links/devices, kiểm15categories/3629train/209Bottle/1725test và lưu filehashmanifest. Rerun kiểm lại tree và tái dùng; existingdata không receipt/corrupt bị từ chối, không xóa/ghi đè tự động. Quy trình này chưa tải/archiveextract thật trong localtests; chỉ metadata công khai và syntheticfixtures được kiểm.

Nếu dùng archive chính bạn đã chạy trước đây, lấy SHA trên Windows bằng Get-FileHash rồi:

```bash
pixi run --locked assets --archive /absolute/path/mvtec_anomaly_detection.tar.xz \
  --archive-sha256 SHA256_BAN_DA_KIEM
```

Archive chủ động phải có operatorSHA; việc ghi hash không tự chứng minh authorprovenance. Để tiết kiệm billedGPUtime, có thể chuẩn bị/upload assets trước, nhưng không bắt buộc chép từ Windows nếu link server hoạt động. `pixi install` không tự tải5GB dataset.

## 4. Kiểm môi trường và chạy research preflight

```bash
nvidia-smi
pixi run --locked env-check --gpu
pixi run --locked test
```

`env-check` kiểm scientificpackagepins Linux, lưu installedmetadata/lockSHA vào records/environment/; `--gpu` chỉ đọc GPUinfo, không chạy kernels. Preflight riêng kiểm nativeBlackwell architecture, actualruntime/numerics. Nếu lỗi cài/test, giữ logs, không ngẫu nhiên upgrade.

Trong tmux, chạy [research handoff](research/implementation.md): tạo records/B_allocation.json từ giá/cap thật, rồi `bash scripts/remote/run.sh research --phase preflight --run-dir runs/hq32_01 --allocation records/B_allocation.json --external-billed-seconds ...`. CLI mặc định đọc data/mvtec và weights/ trong repo. Wrapper dùng `pixi run --locked cad`, thêm tee log/timestamp/unbuffered; không cài packages ngoài Pixi. `bash scripts/remote/setup.sh` chỉ là convenience wrapper cho install+envcheck, không nhận Pythonpath nữa.

Giữ ZIP+accountingsidecar+log sau mỗi invocation; ZIP ledger là snapshot trước reconciliation, sidecar gắn archiveSHA. Chưa remoteacceptance trước actualpreflightreview; không tự bật pilot/main. Baseline3.599 chạy bằng `pixi run cad preflight/train/evaluate/matrix ...`; không dùng baselinecommand thay research3.629.

## 5. Baseline, resume và CPU tools

```bash
pixi run --locked preflight-local
# Baseline GPU (sau review đúng cohort), luôn explicit job/preflight/run-dir:
bash scripts/remote/run.sh train --run-dir runs/baseline_01 --job NESTED_seed0 \
  --preflight runs/baseline_preflight --device cuda:0 --session-hours 4
pixi run --locked cad export --run-dir runs/baseline_01 --output results/baseline_01.zip
pixi run --locked cad aggregate --inputs results/baseline_01.zip --output results/summary.json
```

Evaluate/matrix chỉ sau completedtrain theo CLI, readonly; chỉ thêm missingcells. Baseline/research cần riêng source/runtime/data/review; không resume oldT4checkpoint hay trộn source. `--statistics` chỉ khi đủ registeredseeds/pairedinputs. D0 inputs riêng theo SHA; VisA chưa runner/protocolđượcaccept.

## 6. Backup và source identity

Cất artifacts trong checkout giúp dễ tìm, không bảo vệ khi Destroy instance. Trước Stop/Destroy: copy ZIP/accounting/log/records về máy khác và đối chiếu archiveSHA; live directory cần giữ để resume. Git clone không backup data/run. Hardkill có thể ngăn export; lastatomiccheckpoint phải lấy từ disk. Không tự reset ledger/folder để vượt cap.

Source identity gồm code/docs/tests/pixi manifest/lock, không outputs. Thay provisioning/source phải commit/push hoặc ghi exactdirtyinventory trước freshcohort; không pull/sửa source giữa run. Localtests/locksolve không là ADgain, speedup hay scientificacceptance. Giữ baseline89df9a2 và oldarchives bất biến; không cần tạo sourceZIP mới cho luồng Git.
