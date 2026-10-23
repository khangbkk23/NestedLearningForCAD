# Implementation H32/full-train và bàn giao SSH

**Trạng thái:** đã implement và kiểm tra local; **chưa nghiệm thu trên máy thuê RTX5090, chưa có pilot hoặc anomaly benchmark của candidate**. Baseline chính thức vẫn `89df9a21658b265731f98a7c2014afb1876f1924`; method được thực nghiệm trước đây vẫn v5.1. H32 là candidate, không tự được adopt khi tests đạt.

## Code đã có

| Đường | Implementation |
|---|---|
| D0 normal-only, không model/GPU | `benchmark/attribution.py`; SHA-locked archives, canonical events, chronology resolve→fit→predict, ridge/bias/clock/ties/metrics |
| Full-train3629 | `data/fulltrain.py`; không gọi split_prefix, Bottle209/no exclusions; source/data hashes và imagewise order |
| Research method/cohort IDs | `research.py`, `resources/research.json`; resource SHA khóa, method digest riêng từng variant; frozen baseline method/study/lineage JSON không đổi |
| Historical slow objective | `models/consolidation.py`; support4, startup query4 rồi recent2+history2; exact/FO/direct, actual phi8 giữ nguyên |
| History/RNG | `models/history.py`;32immutable completed records, AlgorithmR sau block commit, query snapshot một lần đầu block, private RNGs và strict partial resume |
| Session/controls | `models/research_session.py`; reuse bank/credit transactions/scoring, new schema; BIAS/RANDOM không model/Adam/projected-history; FAST_ONLY không slow optimizer |
| SSH orchestrator | `runtime/research.py`; preflight/pilot/main/extension, same-cohort guards, test discovery chỉ sau completed own training; train/final/matrix resume tự skip work đã hoàn tất |
| Budget | `runtime/budget.py`; cumulative B12h/C24h/D allocation riêng, session≤4h,180s reserve, interrupted reservation giữ charge |
| Aggregation | `benchmark/statistics.py`; new schema, khóa job/mode/seed/scope và per-method digest trong common cohort, completion/cell locks gắn đúng job/test/checkpoint;15registered fulljobs; paired3seed contrasts H32−CADIC/H32−R |
| Đo chi phí | `runtime/measurement.py`; receipts bất biến theo invocation trong `telemetry/`, phase/job timers và reset GPU peaks theo phase, host component inclusive/exclusive counters; không thêm model forwards hoặc đồng bộ CUDA ở mỗi hàm nhỏ |

Thay đổi session nền chỉ thêm hooks với default semantics cũ để subclass reuse, không nới loader baseline. Numeric primitives, raw feature geometry, CADIC proposer/K/Eq9/recipe và frozen contracts không đổi. Không tạo namespace theo delivery hoặc thêm notebook Kaggle.

## Kiểm tra local và phạm vi

Reduced FP64 finite differences trên5slow parameter groups; exact/FO cùng forward khác backward; DIRECT first-block theta gradient0 được giữ đúng. Recent-only scientific phi/theta parity với baseline qua nhiều blocks. Independent AlgorithmR/query-RNG oracle, mọi partial lengths1–7, malformed history/identity/future records bị từ chối, rollback khi insertion fail, score read-only và hard-crash event replay giữ bank/history/RNG/trace. Full-grid FP32 28×28/32-channel exact/FO historical backward trên CPU hữu hạn, mọi active group connected và historical-query gradient khác0.

Các tests có filesystem manifest3629 với difference chính xác30Bottle, temporal D0 future trap, sum-ridge/ties/duplicates/nonfinite, budget crash/cost cap, new/old preflight rejection và failure export. Suite baseline chạy cùng tests mới; kết quả cuối/counts/source hash trong delivery/local receipt và knowledge review, không phải AD acceptance.

D0 đã được exercise trên đúng historical archives trong quá trình local verification:3599events,3211completed labels,3179postwarm,376legacy names bị loại, censor0. Đây là kiểm analyzer/normal-event analysis, không rerun benchmark hoặc policy counterfactual; không chọn lại H32/hyperparameter từ kết quả. Dataset/test/GT/tensor checkpoints không đọc qua analyzer.

## Bước người chạy thực hiện trên Vast.ai

Đọc [runbook Pixi và assets](../operations.md) trước. Layout workspace mới cần user review/commit/push rồi clone branch và pin **full SHA chứa thay đổi layout này**. Commit32b4e5c chứa Pixi/provisioning nhưng chưa có layout workspace mới. Không tự suy currentHEAD là bản đã push. Không tạo sourceZIP mới chỉ để bàn giao Git.

```bash
# Sau khi clone/pin đúng commit mới, chạy từ repo root trong tmux.
pixi install --locked
pixi run --locked env-check --gpu
pixi run --locked test
pixi run --locked assets

# Tạo workspace/records/B_allocation.json theo mục budget trong runbook.
read -r -p "So giay billed chua ghi ledger: " HQ_EXTERNAL_SECONDS
bash scripts/remote/run.sh research --phase preflight \
  --run-dir workspace/experiments/hq32_01/run --device cuda:0 \
  --expected-gpu 'NVIDIA GeForce RTX 5090' --session-hours 4 \
  --allocation workspace/records/B_allocation.json \
  --external-billed-seconds "$HQ_EXTERNAL_SECONDS"
```

CLI mặc định dùng `<repo>/workspace/data/mvtec` và `<repo>/workspace/weights/vit_base_patch8_224.augreg_in21k.safetensors`, không phụ thuộc tên thư mục cha. Có thể override `--data-root`/`--checkpoint`; không đổi data/source/runtime identity giữa cohort. `pixi run cad` và wrapper `run.sh` dùng cùng Pixi environment; wrapper thêm log timestamp/PID, unbuffered và CUBLAS config.

**Chỉ dừng ở preflight và review trước pilot.** Giữ `workspace/experiments/hq32_01/exports/session_01/evidence.zip`, `workspace/experiments/hq32_01/exports/session_01/accounting.json`, `workspace/experiments/hq32_01/logs/` cùng `workspace/records/` và live run-directory; tải về máy khác/knowledge trước Stop/Destroy. Cất trong repo không thay thế backup ngoài máy thuê.

StageB preflight+pilot cap12GPUh, mỗi invocation≤4h mềm; StageC main cap24GPUh; D cần allocation riêng. Đây là caps, không ETA. File allocation phải cố định theo stage, ví dụ tạo một lần với giá thật/cap user chọn:

```bash
pixi run python -c 'import json; from pathlib import Path; p=Path("workspace/records/B_allocation.json"); p.parent.mkdir(parents=True,exist_ok=True); rate=float(input("USD/hour: ")); cap=float(input("USD cap Stage B: ")); assert rate>0 and cap>0; f=p.open("x"); json.dump(dict(stage="B",authorized=True,USD_per_hour=rate,USD_cap=cap),f,indent=2); f.close()'
```

`--external-billed-seconds` là setup/download/tests/idle chưa ghi, tính một lần; không là hóa đơn chính xác. Ledger giữ full reservation qua compress/verify/hash ZIP, sau export tạo accounting sidecar gắn archive SHA/attempt rồi cập nhật live budget. ZIP chứa `export_incomplete`; sidecar chốt elapsed thực +external+180s publication reserve. Crash giữ full reservation; chuyển host cần review reconciliation, không tự xóa ledger hoặc trộn checkpoint/source để cứu budget. Runner không tự Stop instance.

Sau review đạt, ghi acceptance và chạy pilot cùng inputs/run-dir/allocation; thay external time đúng invocation:

```bash
pixi run --locked cad research-review --run-dir workspace/experiments/hq32_01/run \
  --phase preflight --accept --note 'Artifacts reviewed and accepted'
bash scripts/remote/run.sh research --phase pilot --run-dir workspace/experiments/hq32_01/run \
  --session-hours 4 --allocation workspace/records/B_allocation.json \
  --external-billed-seconds "$HQ_EXTERNAL_SECONDS"
```

Pilot9fresharms×256normals209Bottle+47Cable; không import learned preflight/pilot vào main. Pause: resume cùng lệnh/run-dir/source/lock; gatefail giữ negative và review, không retune. Pilot được nghiệm thu mới reviewaccept rồi mở main từng job `CADIC_seed0`, `R_EXACT_seed0`, `H32_EXACT_seed0`, với stageCallocation riêng. Mỗi job3629→final15→matrix120; extension có screen/budget điều kiện. Chưa tự mở VisA.

CPU summary: `pixi run --locked cad research-aggregate --inputs workspace/experiments/hq32_01/exports/session_01/evidence.zip --output workspace/checks/summary.json`; paired statistics chỉ khi đủ3seeds cho H32 và comparator. D0 `pixi run --locked cad attribution ...` giữ inputs/SHA lịch sử, không cần GPU. Cài đặt không tự chạy asset download, tests, preflight hoặc benchmark.

## Các ranh giới còn nguyên

- Delivery engineering r2 sửa binding aggregator và export accounting, bổ sung telemetry, dùng chung forward factual validation/actual fast step. Thuật toán, objective, gates, data rights và số bước học không đổi. R1 giữ nguyên để truy vết; không trộn source hoặc resume checkpoint r1 qua guard. Chưa có remote run cần giữ compatibility r1.
- `telemetry/<attempt>.json` tách train/final/matrix, pilot probes nhiều actors và discarded preflight. GPU peak theo phase bao gồm backbone, allocator/cache và trạng thái thực; host peak là process lifetime, không phải peak riêng mỗi job. Summed phase wall qua **mọi** invocation gồm cả resume/restore; `train_complete.elapsed_seconds` chỉ là một phạm vi phụ. Setup/idle/export ở ledger/sidecar riêng.
- Component times là host inclusive/exclusive; CUDA dispatch có thể bất đồng bộ. Chỉ phase boundaries synchronize. Không cộng inclusive timings với nhau, không dùng chúng như CUDA kernel attribution hay bảo đảm speedup. Persistent tensor accounting khác allocator peak và tổng RAM. Phép đo tự có overhead; không thêm forwards/updates hoặc thay scientific trace để đo.

- Chưa chạy SSH/GPU thuê, chưa xác nhận kernel/runtime hay ETA5090. Local Windows/Python3.14/PyTorch2.12 khác remote pins3.13/2.11+cu128.
- Chưa có H32 AD gain, không bảo đảm bằng/hơn CADIC. Historical labels chỉ thuộc original bank pair; sameK không same total memory.
- History gradient được probe ở block sau startup và các diagnostic checkpoints (label40/mỗi32slowcommits); không chạy backward riêng cho history ở mọi block để làm instrumentation thành chi phí thuật toán. Outer exact update vẫn dùng đủ4querylosses ở **mọi** scheduled block.
- Baseline CLI train/evaluate/matrix vẫn là3599; research CLI mới fulltrain3629. Giữ loader/resource/old evidence strict; checkpoint cohorts cũ không import vào study mới.
- Spec thuật toán/gates/budgets không thay. Source/bundle identity mới, candidate chưaadopt; phải giữ negative/pause/failed evidence.
