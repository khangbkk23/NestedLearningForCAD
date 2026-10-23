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

Luồng bàn giao đã chốt là **người dùng commit/push branch `hope_cad_v7`, rồi Git clone trên máy thuê**. Candidate H32/full-train đã được người dùng commit tại **`720b3cdd9df4aae3a63c1777b1961af30d1a1543`** ngày23/10/2026. Repository: `https://github.com/khangbkk23/NestedLearningForCAD`; không dùng URL `/tree/hope_cad_v7` làm clone URL. Đã xác nhận commit local; chưa kiểm chứng push trên server. Baseline khoa học trước nâng cấp vẫn `89df9a2`.

Sau khi người dùng đã push commit trên, clone vào thư mục mới và pin đúng source commit bên dưới; `git rev-parse HEAD` phải trả về SHA đó, `git status --short` phải không có thay đổi source. Nếu chọn commit mới hơn có sửa docs/source, ghi rõ SHA mới và source digest trước preflight; không đổi checkout giữa cohort. Bản cập nhật tài liệu ghi SHA này là amendment sau commit, không tự nằm trong commit được dẫn.

```bash
git clone --branch hope_cad_v7 --single-branch \
  https://github.com/khangbkk23/NestedLearningForCAD.git /workspace/hope_cad_v7
cd /workspace/hope_cad_v7
git checkout --detach 720b3cdd9df4aae3a63c1777b1961af30d1a1543
git rev-parse HEAD
git status --short
```

Đặt data/weights/run-directory ngoài source; dùng máy/SSH credentials do bạn đã thuê. Source ZIP r1/r2 trong kho evidence chỉ giữ provenance của local verification, không là bước chuyển source bắt buộc và không cần tạo ZIP mới cho mỗi lần bàn giao. ZIP outputs cùng accounting sidecar/log của remote run vẫn là bằng chứng cần giữ. Agent không tự commit/push, thuê máy hoặc kết nối SSH.

```bash
cd /workspace/hope_cad_v7
nvidia-smi
bash scripts/remote/setup.sh /absolute/path/to/python3.13
.venv/bin/python -B -m hope_cad.local_tests

# Chỉ preflight trước. Sửa data/checkpoint paths theo máy thật.
bash scripts/remote/run.sh research --phase preflight \
  --run-dir /workspace/runs/hq32_01 --device cuda:0 \
  --expected-gpu 'NVIDIA GeForce RTX 5090' --session-hours 4 \
  --data-root /workspace/data/mvtec \
  --checkpoint /workspace/weights/vit_base_patch8_224.augreg_in21k.safetensors \
  --external-billed-seconds 0
```

`--external-billed-seconds` phải ghi **thời gian GPU đã bị tính tiền nhưng ledger chưa ghi**, gồm setup/idle trước invocation; số0 chỉ đúng nếu không có khoản chưa ghi. Ghi mỗi khoản một lần. Ledger giữ full reservation qua cả compress/verify/hash ZIP; chỉ đóng sau export thành công. Charge gồm elapsed thực đến sau export, external time và180s dự trù publication cuối; không phải hóa đơn chính xác. Export lâu hơn180s vẫn được tính đủ. Soft deadline chừa180s trước export, không bảo đảm hard stop đúng giờ khi atomic work/export dài.

ZIP chứa snapshot ledger `export_incomplete`; bản chốt sau export ở **`<run-dir>.accounting.json` cạnh ZIP**, ràng buộc bằng archive SHA và attempt ID. Live `budget_<stage>.json` được cập nhật sau sidecar. Giữ cả ZIP, accounting sidecar và log mỗi phiên. Resume bình thường dùng cùng live run-directory; ZIP đơn lẻ giữ reservation bảo thủ, không tự giảm charge từ snapshot. Nếu phục hồi sang host khác, review SHA/attempt/billing và reconciliation trước, không tự copy vài field hoặc xóa ledger. Crash/export failure giữ full reservation; không có sidecar hợp lệ nghĩa là chưa chốt export. `handoff.elapsed_seconds_before_export` không là tổng thời gian tính tiền.

Cost cap thấp hơn được cấu hình bằng `--allocation /path/B_allocation.json` với `stage:"B",authorized:true,USD_cap,USD_per_hour`; dùng cùng record cho mọi invocation stage đó. Không reset ledger/chuyển folder để lách cap; hết cap phải giữ outputs và xin phân bổ có record.

Trong SSH dùng tmux/process manager đã cài trên host. Chỉ dùng1GPU cho một job; không tự dùng GPU thứ2. Máy không phải5090 phải có runtime review riêng, không chỉ sửa expected name rồi gọi đã canonical acceptance.

Kết thúc preflight: giữ `/workspace/runs/hq32_01.zip`, `/workspace/runs/hq32_01.accounting.json`, log do run.sh tạo và run-directory. Download/copy và kiểm SHA trước Stop/Destroy máy. Gửi ZIP/accounting/log để review; trạng thái `research_preflight_passed_pending_review` chưa là chấp thuận pilot.

Sau khi preflight đã được review, ghi chấp thuận rõ và chạy pilot:

```bash
bash scripts/remote/run.sh research-review --run-dir /workspace/runs/hq32_01 \
  --phase preflight --accept --note 'Preflight artifacts da duoc review va chap thuan'

# Cùng data/checkpoint/source/runtime. Không dùng state pilot cũ.
bash scripts/remote/run.sh research --phase pilot \
  --run-dir /workspace/runs/hq32_01 --session-hours 4 --device cuda:0 \
  --expected-gpu 'NVIDIA GeForce RTX 5090' \
  --data-root /workspace/data/mvtec \
  --checkpoint /workspace/weights/vit_base_patch8_224.augreg_in21k.safetensors
```

Pilot tự đi theo thứ tự9fresh arms×256normals; pool209Bottle+47Cable, riskwindow81+47. Có pause thì chạy lại **cùng lệnh/cùng run-dir**, không reset clocks hoặc đổi source. Từng completed job được skip learning; không nhập learned pilot state vào main. `pilot_complete_scientific_gate_failed` là negative result phải giữ, không tăng thời gian/đổi thresholds để cứu gate. Budget exhaustion không là completion.

Pilot được nghiệm thu thì `research-review --phase pilot --accept --note ...`; sau đó `research --phase main --job CADIC_seed0`, rồi `R_EXACT_seed0`, rồi `H32_EXACT_seed0` với cùng arguments. Mỗi job train3629 →final15cells→matrix120cells; pause giữ task states và tự tiếp phần thiếu, không học lại để thêm matrix. Main batch24GPUh cộng dồn. Khi đủ3jobs, `development_screen.json` chỉ quyết định chi tiếp, không superiority/NI claim. Extension cần explicit stageD allocation và screen đạt; không tự mở15jobs hoặc VisA.

CPU summary sau completed outputs:

```bash
.venv/bin/python -B -m hope_cad research-aggregate \
  --inputs /workspace/runs/hq32_01.zip --output /workspace/summary.json
# --statistics chỉ khi đủ3seeds cho cảH32 và comparator.
```

D0 chạy CPU riêng bằng `hope-cad attribution --input-root /path/two-archives --output /path/new-session`; có `--nested-archive`/`--cadic-archive` khi hai ZIP ở khác thư mục. Không cần máy thuê để chạy D0. Output không overwrite session cũ, không true-resume; các operational attempts/trial limits theo spec.

## Các ranh giới còn nguyên

- Delivery engineering r2 sửa binding aggregator và export accounting, bổ sung telemetry, dùng chung forward factual validation/actual fast step. Thuật toán, objective, gates, data rights và số bước học không đổi. R1 giữ nguyên để truy vết; không trộn source hoặc resume checkpoint r1 qua guard. Chưa có remote run cần giữ compatibility r1.
- `telemetry/<attempt>.json` tách train/final/matrix, pilot probes nhiều actors và discarded preflight. GPU peak theo phase bao gồm backbone, allocator/cache và trạng thái thực; host peak là process lifetime, không phải peak riêng mỗi job. Summed phase wall qua **mọi** invocation gồm cả resume/restore; `train_complete.elapsed_seconds` chỉ là một phạm vi phụ. Setup/idle/export ở ledger/sidecar riêng.
- Component times là host inclusive/exclusive; CUDA dispatch có thể bất đồng bộ. Chỉ phase boundaries synchronize. Không cộng inclusive timings với nhau, không dùng chúng như CUDA kernel attribution hay bảo đảm speedup. Persistent tensor accounting khác allocator peak và tổng RAM. Phép đo tự có overhead; không thêm forwards/updates hoặc thay scientific trace để đo.

- Chưa chạy SSH/GPU thuê, chưa xác nhận kernel/runtime hay ETA5090. Local Windows/Python3.14/PyTorch2.12 khác remote pins3.13/2.11+cu128.
- Chưa có H32 AD gain, không bảo đảm bằng/hơn CADIC. Historical labels chỉ thuộc original bank pair; sameK không same total memory.
- History gradient được probe ở block sau startup và các diagnostic checkpoints (label40/mỗi32slowcommits); không chạy backward riêng cho history ở mọi block để làm instrumentation thành chi phí thuật toán. Outer exact update vẫn dùng đủ4querylosses ở **mọi** scheduled block.
- Baseline CLI train/evaluate/matrix vẫn là3599; research CLI mới fulltrain3629. Giữ loader/resource/old evidence strict; checkpoint cohorts cũ không import vào study mới.
- Spec thuật toán/gates/budgets không thay. Source/bundle identity mới, candidate chưaadopt; phải giữ negative/pause/failed evidence.
