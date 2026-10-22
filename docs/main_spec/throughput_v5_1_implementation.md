# Nested Learning for CAD — bàn giao profiler throughput v5.1

Ngày 20/10/2026. E1 của `hope_support_v5_1_throughput_engineering_v1`; [spec](throughput_v5_1_spec.md), [plan](throughput_v5_1_plan.md), [contract revision2](throughput_v5_1_contract.json). Contract và spec lưu trạng thái tại lúc thiết kế; tài liệu implementation và review mới ghi trạng thái thực thi. Chưa có số đo component trên T4 khi bàn giao.

## 1. Phạm vi implementation

Đã xây dựng extension riêng để đo frozen reference, không sửa166payloads của benchmark. Chỉ có E1 profiler/transport/tests; optimization candidates, paired candidate validation, migration loader/shadow bridge và AD gain vẫn chưa triển khai/nghiệm thu.

| Module | Vai trò |
|---|---|
| `training/support_throughput_v5_1/contract.py` | Hash-bound parent/source/window; safe paths; dedup ZIP/extracted evidence bằng exact manifest identity |
| `.../replay.py` | Full-manifest global179→195, không finalize/censor; checkpoint184/192; compare events và logical state, RNG guard |
| `.../instrumentation.py` | Reversible wrappers quanh frozen functions; nested host spans, coarse CUDA events, NN/block/hash logical-byte counters |
| `.../worker.py` | Một pass/process mới, canonical runtime, exact pending/state load, expected parent events/receipts, final-state parity |
| `.../runner.py` | U1/P1/U2, cumulative90min, crash-safe attempt reservation, completion seals và no automatic partial replay |
| `notebook_support/support_throughput_v5_1/bootstrap.py` |4inputs, runtime check trước extraction, chỉ16registered raw normals |
| `.../entry.py` | Failure/handoff/export/ZIP download link |
| `.../build.py` | Build từ original166-source ZIP + extension; không thay base manifest/source identity |
| `tests/support_throughput_v5_1/` | Synthetic learning trajectory, timing/cleanup/rollback, rights, input ambiguity, partial/resume/identity, delivery tests |

Profiler package có **hai manifests**: `bundle_manifest.json` gốc chỉ166basepayloads; `throughput_manifest.json` bao base manifest/base files và extension. Source identity mới từ toàn file map; frozen source/method/study digest được kiểm riêng. Không dùng source extension để resume benchmark cũ. Tênv6 là worktree, method vẫnv5.1.

## 2. Khác biệt vận hành đã công khai

- Dedicated bounded loop dùng full3599rows, không gọi `engine.train` với rows cắt. Chỉ arrivals179–194; global seen IDs/pending/replay được giữ. Parent376 không bị ghi.
- Full archive769payloads/manifest/source/assets được verify; nếu Kaggle giải nén ZIP thì dùng exact original manifest SHA cùng mọi payload SHA. Source/parent representation và paths được ghi receipts. Không accept theo Dataset name.
- Prefix0–178 được verify và nối hash chain; expected events/receipts179–194 nằm ở read-only input reference. Pass output bắt đầu mới, không prepopulate16receipts để đổi I/O branch.
- Mỗi pass cold derived cache sau load checkpoint; OS page cache ngoài quyền kiểm soát được disclosure. Không có warmup-arrival hoặc feature precompute. Thời gian replay không dùng làm ETA toàn dataset.
- U1/U2 không cài component wrappers. P1 có scope wrappers và CUDA events; counters/graphs không đi vào scientific state. NN tiles chỉ đếm/host timing, không synchronize hoặc tạo CUDA event từng tile.
- Primary time gồm provider/observe/publication/hash-chain/memory_profile và periodic save184/192. Compare/full final snapshot/collect CUDA spans/export nằm ngoài primary timer, vẫn tính session budget. Host exclusive times không cộng nested inclusive times; CUDA span có host launch gaps, không là kernel-busy time.
- Events từng ảnh so parent exact; final full scientific state và digest so U1/P1/U2. Parent không có checkpoint195 nên không giả fullstate oracle195. Global RNG phải giữ nguyên. Gradient validation của candidate thuộc G2/G3 sau này, không lấy G1 PASS thay thế.
- U1final scientific payload được giữ để đối chiếu qua session. Hai periodic checkpoints vẫn được serialize/fsync/hash đúng cadence trong primary timing; khi pass complete, fileactive.pt của disposable pass được bỏ sau đo, giữ SHA/receipt trong measurement. Partial/failure giữ checkpoint/diagnostics. Không có đường adopt state195 vào benchmark.

## 3. Budget và resume

Budget cố định5400s, gồm bootstrap/setup/load/pipeline/diagnostics. Runner giữ180s cho export. Ledger ghi thời gian thực tới trước export **cộng charge dự phòng180s**; actual export seconds nằm ở receipt in Kaggle log. Đây là charge bảo thủ, không được báo180s là chi phí export đo được. Nếu export vượt180s, notebook raise yêu cầu review; không tiếp tục automatic resume. Giới hạn này là ngân sách profiling, không giảm số bước học của method.

Trước dispatch mỗi pass, ledger atomically reserve16attempts. Worker báo actual attempted IDs, gồm event đã bắt đầu nhưng rollback. Hard kill có thể làm thiếu attempts receipt: reserve bảo thủ vẫn còn, không làm mất budget hoặc cấp lại16arrivals. Tối đa48reserved attempts, actual attempts không thể vượt trần. Một pass complete có đúng16successes, cursor195, hai slow events181/189, labels181/slow22/partial5/pending chưa censor.

Resume chỉ skip completed/sealed passes rồi bắt đầu pass kế tiếp. Không nối timing qua restart, không replay incomplete/failed pass, không tự tăng budget. `profile_incomplete_budget` nghĩa cần review ZIP/log; không mặc định attach rồi chạy lại sẽ hoàn tất. Failure do package/runtime/parent phải xử lý theo bằng chứng, không sửa method/gates.

## 4. Hướng dẫn Kaggle

1. Import [notebook](../../notebooks/hope_cad_v5_1_throughput_profile_kaggle.ipynb) trong notebook/kernel mới.
2. Upload/attach source ZIP tại `D:/src/DATN/.agents/artifacts/deliveries/hope_support_v5_1_throughput_profile_r1/hope_cad_support_v5_1_throughput_source_r1.zip`. Kaggle giải nén thành Dataset cũng được. Dùng đúng notebook/source cùng delivery.
3. Attach reviewed parent `D:/src/DATN/.agents/artifacts/runs/support_v5_1_benchmark_NESTED_seed0/session_01/hope_cad_support_v5_1_benchmark_evidence.zip` hoặc exact extracted output. Đây là paused376/3599, không W6/pilot/preflight. Parent ZIP SHA trong contract bắt đầu35f05b52.
4. Giữ MVTec raw archive/directory và checkpoint `vit_base_patch8_224.augreg_in21k.safetensors` đang dùng. Tổng4loại input. Tháo các source cũ; không cần attach riêng frozen166source ZIP vì nó nằm trong delivery mới.
5. Chọn GPU T4×2, giữ PHASE='profile' và overridesNone nếu auto discovery rõ. Dùng Save Version → Save & Run All. Không pip install/upgrade. Notebook hiển thị bootstrap/runtime, pass start, heartbeat30s và progress mỗi8arrivals.
6. Lưu ZIP `hope_cad_support_v5_1_throughput_evidence.zip` và Kaggle log vào `D:/src/DATN/.agents/artifacts/runs/support_v5_1_throughput_profile_01/session_01/`. Không ghi đè; phiên sau tăngsession. Raw evidence ở artifacts, review sẽ ở evidence/reviews.
7. Gửi kết quả để review. `profile_complete_pending_review` chỉ đủ U/P/U; phải kiểm parity/timing/overhead trước khi chọn tối ưu. Có thể chưa đo xong trong90min; giữ incomplete/failure evidence, không tự nâng budget hoặc chuyểnbenchmark.

Output includes source/parent/runtime/window locks, pass bootstrap/measurement/complete seals/events/feature receipts, one U1final state, ledger, summary, logs/handoff. Không export bản sao parent119MB, không đọc official test/GT/VisA hoặc nạp learned W6/pilot state.

## 5. Kiểm thử và nghiệm thu

Local Windows/Pixi CPU được dùng cho technical tests. Logs, delivery receipt và review tại `.agents` ghi số test/sourceSHA thật khi bàn giao; không dùng thời gian testCPU làm ETA T4. Các checks cần chạy: suites corev5.1/benchmark/profiler, frozen166hashes, original spec hashes, selected parent/window, notebook cells compile, actual bootstrap ZIP và extracted directory trong isolated processes, package import/source-map verification.

Canonical G1/T4 chưa chạy lúc viết. Sau user run, nếu profiler làm lệch events/state thì sửa instrumentation có evidence/revision, không gọi đó là model kém hoặc hạ parity. Sau profile review mới đề xuất tối đa3patches tương đương; giữ exact4+4/denominator derivative,32×8anchors, K10000, fullproposal và fullEq9. Chưa có performance/speedup/candidate-migration claim.
