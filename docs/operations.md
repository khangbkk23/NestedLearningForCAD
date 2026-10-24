# Chạy v7 với Pixi trên Linux/RTX5090 hoặc Windows local

Pixi là package/environment manager duy nhất. Method accepted vẫn v5.1; H32/full-train là candidate đã implement/local-tested, có technical preflight cho source/runtime cũ, chưa completed/accepted pilot hoặc AD benchmark. Không tự thuê/SSH/chạy GPU. Có thể kiểm tốc độ CADIC độc lập trước research preflight, rồi mới review từng stage.

## Bootstrap qua SCP

BOOTSTRAP.sh là helper local, bị ignore và không nằm trong Git hoặc source inventory. File được cấp riêng qua SCP, không download từ branch. Ví dụ PowerShell (thay SSH_ALIAS bằng alias SSH của máy thuê):

```powershell
scp "D:\src\DATN\hope_cad_v7\BOOTSTRAP.sh" SSH_ALIAS:/workspace/BOOTSTRAP.sh
```

Trên Windows có helper tracked [Prepare-Vast.ps1](../scripts/local/Prepare-Vast.ps1). Từ checkout local, dán **thông tin máy thật** (địa chỉ dưới đây chỉ minh họa):

```powershell
# Xem trước; chưa sửa config hoặc upload.
& .\scripts\local\Prepare-Vast.ps1 `
  -SshCommand 'ssh -p 51729 root@66.115.179.150 -L 8080:localhost:8080' -WhatIf

# Khi endpoint đúng: cập nhật vast1 rồi gửi BOOTSTRAP.sh; không tạo backup config.
& .\scripts\local\Prepare-Vast.ps1 `
  -SshCommand 'ssh -p 51729 root@66.115.179.150 -L 8080:localhost:8080'

# Hoặc nhập ba trường trực tiếp:
& .\scripts\local\Prepare-Vast.ps1 -HostName '66.115.179.150' -Port 51729 -User root
```

Helper mặc định dùng `$HOME\.ssh\config`, alias `vast1`, key `$HOME\.ssh\id_ed25519_vast_datn`, bootstrap ở root checkout. Chỉ cập nhật HostName/Port/User trong block riêng `Host vast1`; giữ IdentityFile/forwarding và các hosts khác. Nếu chưa có alias thì tạo block với key này. Block dùng nhiều aliases hoặc bị Include/Match làm mơ hồ sẽ bị từ chối để tránh sửa nhầm. Không tạo backup config; lỗi upload không tự rollback config đã cập nhật. `-SkipUpload` chỉ sửa config; `-WhatIf` không sửa/upload. Không thực thi pasted SSH command hoặc `-L`, không tự SSH mở shell/chạy bootstrap. Không dùng ví dụ trên nếu đó không phải máy bạn thuê.

Trên máy thuê Debian/Ubuntu x86_64 có NVIDIA driver và root/sudo:

```bash
cd /workspace
bash BOOTSTRAP.sh
```

Script hỏi giá thuê USD/hour và capUSD Stage B (preflight+pilot) trước khi setup. Không tự chọn số tiền. Có thể truyền HOPE_USD_PER_HOUR/HOPE_STAGE_B_USD_CAP để chạy không tương tác. Base mặc định/workspace phải có sẵn; bash BOOTSTRAP.sh /absolute/base là override. Repo sinh ra tại <base>/NestedLearningForCAD.

Script kiểm máy, apt update/install git,curl,ca-certificates,tmux,unzip, rồi chạy phần dài trong session hope-cad-setup (hoặc tmux hiện có). Giữ SSH tới khi session được tạo. Tách Ctrl+B,D; vào lại bằng tmux attach -t hope-cad-setup. Session đã tồn tại thì attach kiểm thay vì tự kill hoặc chạy setup song song.

Worker tạo `~/.no_auto_tmux`, cài Codex CLI bằng installer chính thức ở chế độ không tương tác và kiểm version **trước git clone**; PATH bổ sung `$HOME/.local/bin`. Marker tắt auto-attach cho SSH mới, giữ session tmux đang chạy. Không tự đăng nhập Codex hoặc khởi động agent; user đăng nhập riêng nếu cần kết nối Codex app. Codex CLI là công cụ vận hành; Python/dependencies của dự án vẫn chỉ do Pixi quản lý.

Installer dùng `CODEX_NON_INTERACTIVE=1` theo [tài liệu installer chính thức](https://learn.chatgpt.com/docs/config-file/environment-variables); kiểm này không phải đăng nhập Codex CLI.

Sau đó worker cài Pixi nếu thiếu, clone latest hope_cad_v7, ghi exact HEAD vào log/receipt, kiểm origin/source sạch rồi install locked → check-install → env-check --gpu → tests → assets → tạo workspace/records/B_allocation.json theo số tiền đã nhập. SHA không phải điền tay; nếu muốn khóa trước, cấp EXPECTED_COMMIT=full40hex. Existing checkout không được tự pull/reset/checkout: nếu cần version khác phải preserve/review thủ công. Source không được đổi giữa setup/cohort. Helper có SHA riêng, không giả làm source file đã commit; bản sao helper/SHA và log được lưu cùng receipts.

Allocation có sẵn chỉ được tái dùng khi nội dung khớp; khác thì fail, không overwrite. Cap B còn chịu giới hạn tối đa12GPUh của runner; đây không là auto-stop/billing guard của Vast. Không tạo ledger hoặc charge run trong bootstrap. Setup/idle billed time chưa ghi phải cấp một lần qua --external-billed-seconds khi preflight.

Logs tạm /tmp/hope-cad-system.*.log và /tmp/hope-cad-setup.*.log; khi đúng checkout thì lưu complete log/exit receipt/helper copy dưới workspace/records/bootstrap/. Dispatch thành công chưa nghĩa worker hoàn tất: kiểm SETUP COMPLETE và receipt exit0. Fail trước clone/sai origin chỉ giữ log tạm.

Nếu tests thất bại, bootstrap dừng với exitcode khác0 và thông báo rõ assets/allocation chưa chạy trong invocation này; giữ log, chưa chạy diagnostic/preflight. Không bỏ tests, thêm `|| true` hoặc coi install thành công là setup hoàn tất. Khi assets được provision/kiểm thành công, có marker `[ASSETS READY]`; sau đó còn allocation/handoff gate. Asset receipts được kiểm khi tái dùng dataset/weights, không cần tải lại nếu đã đúng.

Sau setup thành công **và sau khi lưu log/receipt**, worker trong tmux mở `bash -i` tại `<base>/NestedLearningForCAD` với `$HOME/.pixi/bin` trên PATH. Không cần nhập lại cd/export trong pane đó. Non-interactive caller không mở shell để tránh treo automation; reconnect bằng SSH tạo terminal khác vẫn cần cd/export hoặc attach pane sẵn. Lệnh `bash BOOTSTRAP.sh` không đổi cwd/PATH của parent shell. Setup lỗi không mở ready shell; dispatch bên ngoài không mở shell thứ hai.

Bootstrap hoàn tất allocation/gate rồi giữ terminal để user thao tác, **không chạy preflight/training hoặc tự Stop máy**. Helper sửa fixtures/readiness được kiểm local/mocked; Linux GPU-visible suite và ready terminal thật của bản mới phải được kiểm trên host trước research. Sau setup review logs/receipts rồi dùng [CADIC diagnostic](cadic_diagnostic.md) trước research, hoặc [research handoff](research/implementation.md). Không cần commit helper, nhưng code/docs/lock muốn dùng vẫn phải được user publish lên branch trước clone. Phải gửi lại BOOTSTRAP.sh mới qua SCP; existing checkout không tự pull về commit mới.

Lỗi research `hq32_3090_01` trước đây chưa có raw `handoff.json/error/export` để chốt nguyên nhân. Trước lần chạy fresh kế tiếp, phải có SETUP COMPLETE, receipt exit0, assets đã verified và đúng `workspace/records/B_allocation.json`. Không tự ghi allocation vào một run đã mở ledger hoặc sửa identity để tiếp tục. Nếu research lỗi, giữ export và đọc ngay `run/handoff.json`; thiếu allocation của lần cũ vẫn là giả thuyết, không là nguyên nhân đã xác nhận.

## 1. Cài Pixi, clone và pin source

Trên Linux, cài Pixi theo [hướng dẫn chính thức](https://pixi.prefix.dev/latest/installation/):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

Lock được tạo/kiểm với Pixi0.66.0. Cần bản hỗ trợ lockschema6/per-package indexes; giữ phiên bản Pixi đã dùng trong records. Script installer cài Pixi, không cài NVIDIA driver.

User review/commit/push thay đổi muốn dùng trước khi thuê. Sau khi đã push:

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
  src/ tests/ scripts/ docs/ refs/    # tracked source
  pyproject.toml / pixi.toml / pixi.lock
  .pixi/                            # Pixi tự quản lý, ignore
  workspace/                        # inputs và outputs local, ignore
    data/
      archives/
      mvtec/
      mvtec.asset.json
    weights/                        # pretrained ViT + asset receipt
    records/                        # máy, environment, allocation/billing
    checks/                         # CPU checks/summaries; không là AD run
    experiments/
      hq32_01/
        run/                        # checkpoints/events/metrics/telemetry/ledger
        logs/                       # stdout/stderr theo invocation
        exports/
          session_01/
            evidence.zip
            accounting.json
          session_02/               # slot mới; không ghi đè session trước
```

workspace/ bị Git ignore và không thuộc source inventory/hash. Defaults neo repo root. .pixi/, Python caches và editable-install metadata vẫn do công cụ quản lý. Pretrained inputs dùng chung; learned checkpoints chỉ ở experiment/run. Không commit placeholder datasets/outputs.

Có thể override input/run paths ra ngoài repo cho disk/volume khác. Trong repo, assets chỉ dưới workspace/data hoặc workspace/weights; outputs mới dưới workspace/experiments/<id>/run hoặc workspace/checks/<id>. Legacy explicit runs/results/logs vẫn được chấp nhận cho baseline tools, không là default mới. Không overlap inputs/source. Không migrate checkpoint scientific cohort cũ qua bỏ source guard.

Machine/environment/allocation ở workspace/records; ledger khóa đúng allocation nội dung. Cap hiện thực thi theo study/run/stage, không là global multi-experiment account: không tạo experiment mới để reset ngân sách. Split cohort/host cần review phần đã chi và allocation còn lại.

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

`env-check` kiểm scientificpackagepins Linux, lưu installedmetadata/lockSHA vào workspace/records/environment/; `--gpu` chỉ đọc GPUinfo, không chạy kernels. Preflight riêng kiểm nativeBlackwell architecture, actualruntime/numerics. Nếu lỗi cài/test, giữ logs, không ngẫu nhiên upgrade.

Nếu mục tiêu đầu tiên là thử cấu hình/tốc độ CADIC, chạy độc lập sau setup/assets:

```bash
bash scripts/remote/run.sh cadic-diagnostic --phase train \
  --run-dir workspace/experiments/cadic_check_01/run --max-hours 1
```

Mặc định fresh K2500 rồi5000, mỗi K đủ3629ảnh; có thể pause do cửa sổ1giờ và resume đúng lệnh. Không hỏi allocation hoặc mở research gates; không tự Stop instance. Đọc `summary.json`, export `evidence.zip` + `timing.json`, logs và records theo [spec diagnostic](cadic_diagnostic.md). Optional `--phase evaluate` là bước riêng; không trộn anomaly evaluation vào train timing. Muốn dùng research sau đó vẫn cần đúng preflight/review.

Trong tmux, chạy [research handoff](research/implementation.md): tạo workspace/records/B_allocation.json từ giá/cap thật, rồi `bash scripts/remote/run.sh research --phase preflight --run-dir workspace/experiments/hq32_01/run --allocation workspace/records/B_allocation.json --external-billed-seconds ...`. CLI mặc định đọc workspace/data/mvtec và workspace/weights trong repo. Wrapper dùng `pixi run --locked cad`, thêm tee log/timestamp/unbuffered; không cài packages ngoài Pixi. `bash scripts/remote/setup.sh` chỉ là convenience wrapper cho install+envcheck, không nhận Pythonpath nữa.

Giữ ZIP+accountingsidecar+log sau mỗi invocation; ZIP ledger là snapshot trước reconciliation, sidecar gắn archiveSHA. Chưa remoteacceptance trước actualpreflightreview; không tự bật pilot/main. Baseline3.599 chạy bằng `pixi run cad preflight/train/evaluate/matrix ...`; không dùng baselinecommand thay research3.629.

## 5. Baseline, resume và CPU tools

```bash
pixi run --locked preflight-local
# Baseline GPU (sau review đúng cohort), luôn explicit job/preflight/run-dir:
bash scripts/remote/run.sh train --run-dir workspace/experiments/baseline_01/run --job NESTED_seed0 \
  --preflight workspace/checks/baseline_preflight --device cuda:0 --session-hours 4
pixi run --locked cad export --run-dir workspace/experiments/baseline_01/run --output workspace/checks/baseline_01.zip
pixi run --locked cad aggregate --inputs workspace/checks/baseline_01.zip --output workspace/checks/summary.json
```

Evaluate/matrix chỉ sau completedtrain theo CLI, readonly; chỉ thêm missingcells. Baseline/research cần riêng source/runtime/data/review; không resume oldT4checkpoint hay trộn source. `--statistics` chỉ khi đủ registeredseeds/pairedinputs. D0 inputs riêng theo SHA; VisA chưa runner/protocolđượcaccept.

## 6. Backup và source identity

Cất artifacts trong checkout giúp dễ tìm, không bảo vệ khi Destroy instance. Trước Stop/Destroy: copy ZIP/accounting/log/records về máy khác và đối chiếu archiveSHA; live directory cần giữ để resume. Git clone không backup data/run. Hardkill có thể ngăn export; lastatomiccheckpoint phải lấy từ disk. Không tự reset ledger/folder để vượt cap.

Source identity gồm code/docs/tests/Python, shell, PowerShell scripts/pixi manifest/lock, không outputs; bootstrap ignored có SHA riêng. Thay provisioning/source phải commit/push hoặc ghi exactdirtyinventory trước freshcohort; không pull/sửa source giữa run. Localtests/locksolve không là ADgain, speedup hay scientificacceptance. Giữ baseline89df9a2 và oldarchives bất biến; không cần tạo sourceZIP mới cho luồng Git.

Đổi host vật lý chưa chắc đổi runtime identity. [Handoff v1](research/stage_handoff_v1.md) có pack/check/restore cho exact source/runtime/data/backend, giữ locks và accounting. Nếu khác driver/GPU/source, gate blocked; changed-runtime/source mapping trong [proposal rộng hơn](research/stage_handoff_spec.md) chưa triển khai. Source cũ54d9049 chưa có handoff v1 nên không tự import thành run mới; triển khai/commit/push v1 trước cohort muốn portable.

## Bàn giao một ZIP sau mỗi phase/invocation

Sau khi lệnh research/export đã kết thúc và sau review acceptance cần mang đi:

```bash
bash scripts/remote/pack.sh --run-dir workspace/experiments/hq32_3090_01/run
```

Lệnh không train, nhưng cần runtime gốc để load/kiểm exact pending predictions trên GPU đó; không coi là hoàn toàn CPU-only. Nó khóa writer, kiểm source sạch đã commit, actual state/events/tasks/cells/coverage và tạo **một ZIP** dưới `workspace/experiments/hq32_3090_01/handoffs/`, in full path/SHA. Tải ZIP này lênDrive/local và giữ SHA riêng để kiểm sau download. Pack trước review thì ZIP chưa có acceptance mới; review xong phải pack bản mới, không sửa ZIP cũ. Không pack khi job đang chạy. Không dùng diagnostic experiment thay research run. Nếu runtime gốc đã mất/đổi, vẫn preserve raw experiment/exports để audit; không sửa loader cho pack PASS.

ZIP có live run, logs, immutable exports/accounting, records, asset receipts và source snapshot; không data/weights/.pixi hoặc SSH/Codex auth. Outer ZIP cũ không được lồng lại. Hash kiểm integrity, không tự chứng minh người review đáng tin. Giữ cả failure/pause sessions. Source dirty/legacy/corrupt state thì báo lỗi, không xóa dữ liệu. Nếu source hash thay giữa phase, không sửa manifest để cứu resume.

Trên **máy mới**, upload ZIP vào `/workspace/HOPE_HANDOFF.zip` trước bootstrap. Giữ đúng tên đó để tự nhận; hoặc chỉ định path tuyệt đối:

```bash
HOPE_HANDOFF_ZIP=/workspace/handoff_3090.zip bash /workspace/BOOTSTRAP.sh
```

Bootstrap đọc header để checkout đúng commit trong **fresh clone** (existing checkout không reset), cài Pixi locked/tests/assets, rồi gọi `handoff-check --restore` với giá/cap bạn nhập. Cần commit nguồn còn có trên branch GitHub; ZIP source không được tự chạy để thay code trusted checkout. Hai nhánh3090/4090 dùng IDs riêng. Marker filename trùng nhưng ZIP GPU khác sẽ blocked; không tự fallback fresh.

Gate verify ZIP/hash/source và actual runtime/data/backbone trước khi load state trên numerical device đích đã khớp; sau đó kiểm coverage/closed accounting và fixed normal-feature/read-only score probes trên CUDA. Đây là một số forwards/clones, không chạy lại pilot hoặc train. Mismatch/thiếu review/negative gate/open budget đều in lý do; không tự cấp acceptance hoặc chạy lại từ đầu. Phase1 đã accepted→Phase2; Phase2 đủ3jobs×120cells+screenPASS→Phase3; partial phase→job còn thiếu; Phase3 đủ15jobs→complete. Kernel/probe failures vẫn blocked, không gọi template giống nhau là compatible.

Chỉ khi restore đạt, stdout có `allowed:true`, `restored:true`, `run_dir`, `allocation_path`, `command`. Gate tạo allocation đúng B/C/D dưới imported origin; bạn copy lệnh `NEXT COMMAND` để tự chạy. Nhớ thêm `--external-billed-seconds` cho setup/IO/idle chưa ghi một lần; gate không tự start stage/Stop/Destroy. Existing partial stage thay giá/cap bị blocked; completedB/C vẫn giữ giá gốc và next stage mới có allocation khác. Restore vào experiment đã tồn tại bị từ chối; nếu đã restore thành công thì dùng live run/lệnh trước, không import lại ZIP cũ.

Có thể kiểm thủ công, không restore:

```bash
pixi run --locked cad handoff-check --archive /workspace/HOPE_HANDOFF.zip --rate 0.2 --cap 3
```

`compatible:true` ở chế độ này vẫn `allowed:false`, phải restore thành công trước khi chạy. Report luôn lưu dưới workspace/records/handoff_checks; statusblocked là kết quả gate, không chứng minh bootstrap cài đặt lỗi. First real cross-host run chưa được nghiệm thu, giữ bản sao thứ hai trước Destroy.

## Export sessions và logs

Research runner giữ writer lock, cấp một slot session tăng dần rồi export evidence.zip, verify/hash, sau đó ghi accounting.json và chốt live ledger. Slot lỗi/partial giữ nguyên; lần sau cấp slot mới. Export ở ngoài payload run/; budget chưa giải phóng khi export thất bại. ZIP vẫn chứa reservation mở, sidecar gắn archive SHA/attempt. Không tự import sidecar để giảm charge khi restore sang host khác.

Wrapper đặt logs canonical experiment cạnh run/ tại logs/. Tools không có canonical run dùng workspace/checks/invocations/logs. Log timestamp/PID theo invocation; telemetry attempt ID ở run/ để đối chiếu. Baseline export vẫn nhận --output explicit; auto export theo session là research runner.

Backup latest live run để resume và từng export session/log/records ra khỏi máy thuê. session_01 trong ví dụ chỉ minh họa: dùng session mới nhất của invocation vừa chạy. Không ghi đè session cũ.
