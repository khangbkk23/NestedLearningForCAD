# Nested Learning for CAD — G4 chuyển execution NN-write

Ngày 21/10/2026. Addendum cho §9 của `throughput_v5_1_spec.md`, sau G3 đã được review. Đây là thay đổi execution, **không thay phương pháp/study, không nghiệm thu anomaly performance**. Các con số và hashes khóa tại `migration_nn_write_v5_1_contract.json`. Trạng thái canonical sau này nằm trong review, không sửa tài liệu này để mô tả một run chưa xảy ra.

## 1. Phạm vi và điểm dừng

Chuyển đúng `NESTED_seed0` từ original parent376 sang execution có operator `nearest_fixed_shape_writes_v1`. Không tái sử dụng clone195. Không chạy lại W6/pilot hoặc tự mở benchmark đầy đủ/VisA. Batch này giao notebook **bridge**; chỉ sau review bridge mới giao interface tiếp tục benchmark.

Giữ nguyên toàn bộ source200 đã nghiệm thu G3 và manifest cũ; thêm modules/manifest riêng. Không sửa frozen engine, loader, method hoặc candidate đã kiểm. G1/G2/G3 receipts được pin trong contract và đóng gói để không cần thêm những output này vào Kaggle.

## 2. Ba identity và ownership

Method/study giữ digest cũ. Execution source map/runtime mới có lock riêng. Suite mới lấy các scientific fields của suite gốc nhưng thay source digest, thêm parent suite digest và schema execution. Job mới tham chiếu suite mới, original job digest và migration identity; `fresh_initialization=false`. Tất cả rows/task order giữ nguyên. Không giả source mới đã tạo inherited state.

Inherited event0–375 và feature receipts giữ bytes ở run mới; bản gốc và task00 wrapper/meta giữ dưới `legacy/` cùng original job lock. Các hashes được kiểm với **raw artifact manifest gốc** đã pin, không tự khai báo allowlist từ thư mục input. Task-state resolver chỉ cho phép task00 từ owner gốc với đúng hash/job/rows; task01 trở đi phải do new job owner tạo và vẫn qua loader gốc. Cell locks và final/matrix completion dùng new job digest, trong khi checkpoint SHA của inherited task vẫn nguyên bản.

Active wrapper mới dùng new job digest, nhưng scientific session phải exact với parent khi load376. Sau bridge publish active384; đồng thời lưu immutable `bridge_checkpoint.pt` để neo receipt dù active tiếp tục di chuyển. Original parent active376 không bị ghi đè. Initialization lock giữ nguyên vì cùng initialization của cùng trajectory, không phải fresh fit mới.

## 3. Bridge nhân quả và atomic publication

Trước việc nặng, bootstrap kiểm source map, prerequisite review hashes, exact parent archive/manifest769 entries, runtime T4/backbone/method/study. Load active376 vào **hai owners độc lập trên T4**, reference và candidate. Loader gốc kiểm schema, counters, phi/theta/Adam, private RNG, replay và **recompute pending prediction/d**. Full states và pending sau load phải exact.

Chạy duy nhất8 arrivals376–383. Mỗi ảnh chỉ được đọc/hash/encode khi đến lượt; cùng raw feature đưa cho hai owners. Nếu input là tar, chỉ extract ảnh được yêu cầu hiện tại, không extract sẵn tám ảnh. Delivery r2 giữ một metadata-only index cho các paths được phép đã gặp, hỗ trợ thứ tự vật lý trong tar khác thứ tự arrival; không gọi extractfile hoặc hash/encode pixels tương lai. Sau reference và candidate cùng xử lý, so exact event/full scientific state trước khi đọc ảnh kế tiếp. Hai derived caches độc lập; activation operator có scope từng owner call. Không dùng equality để tune hoặc chọn policy. Không finalize ở384; pending/partial được giữ.

Mọi failure, NaN, mismatch hoặc hết budget: export diagnostic records, **không publish accepted checkpoint/receipt**, không lén dùng reference continuation. Không tự retry partial bridge, không tự nối hai phiên hoặc tăng budget. Default soft budget60 phút gồm bootstrap và180s export reserve; tối đa8 unique arrivals/16 actor transitions. Đây là budget bảo vệ phiên, không ETA. Hard platform kill vẫn có thể ngăn export.

Chỉ khi cả8 events/state exact mới tạo publication trong staging và rename thành `benchmark/`; receipt+completion marker gắn parent/source pair/runtime/checkpoint384. Nếu publication bị ngắt, staging không phải accepted execution. Loader chỉ đọc canonical `benchmark/` có marker hợp lệ. Bridge output còn cần người review trước khi tiếp tục; không tự train ảnh384 trong cùng notebook.

## 4. Compatibility consumers

Lớp execution chỉ mở khi identity, receipt, immutable bridge checkpoint và inherited hashes hợp lệ. Active current vẫn được engine gốc kiểm event-prefix/job/cursor/bank, bao gồm atomic PT→JSON crash recovery. Source/runtime/method/study thay thật bị từ chối; không dùng monkeypatch `lock_json` hoặc wildcard source. Owner resolver scoped chỉ thay đúng task-state dispatch; loader gốc vẫn xác minh metadata/hash cho owner tương ứng và mọi state mới.

Train sử dụng new job identity; evaluation gắn new job digest và original/new checkpoint SHA đúng owner. Completion/cell chunks vẫn qua frozen engine, không đổi metric/interpolation/reduction. API này được test synthetic cho15 final cells và120 triangular cells, score read-only, no training khi làm matrix, idempotence/resume/tamper. Interface Kaggle full benchmark chưa được mở trong batch bridge.

## 5. Receipt và compute accounting

Receipt ghi engineering ID; original archive/manifest/suite/job/active/task00 SHA và source map; new source map/digest; method/study/runtime; explicit added-file inventory (existing bytes unchanged); G1/G2/G3 receipt hashes; pending check; full-state comparison policy/exclusions; arrivals/IDs/events/digests; old/new checkpoint SHA; inherited task owner; actual reference/candidate/provider/comparison/bootstrap/export timing; parent elapsed retained. CPU copies/hash/snapshot timing không gọi là GPU utilization.

Bridge candidate elapsed, shared provider và shadow reference overhead được ghi riêng; active elapsed tiếp nối parent + provider + candidate observe, không xóa parent time hoặc cộng shadow reference vào thời gian triển khai như thể đó là workload method. Comparison, bridge serialization, bootstrap/publication/export báo riêng; active elapsed không được dùng như tổng notebook hoặc cold-start runtime. Ledger tổng wall vẫn bao mọi khoản để công bố compute thực. Gate chỉ chấp nhận exact equality, không tolerance rescue.

## 6. Handoff

Notebook `hope_cad_v5_1_nn_write_bridge_kaggle.ipynb`. Bốn inputs: source bridge r2 mới, original benchmark376 evidence, MVTec raw, locked ViT. Tháo NN-validation/profiler hoặc bridge r1 source và outputs. GPU T4×2; PHASE='bridge'; Save Version → Save & Run All. Không cần attach W6/pilot/G3 artifacts vì receipts đã đóng gói; original parent vẫn bắt buộc.

Output ZIP/log lưu tại `.agents/artifacts/runs/support_v5_1_nn_write_bridge_01/session_01/`. `bridge_complete_pending_review` có nghĩa technical bridge hoàn tất, chưa là benchmark hoàn tất hoặc quyền tự resume. Failure/incomplete cũng phải giữ evidence. Sau review đạt, tiếp tục từ384 theo new execution lineage; mọi inherited task vẫn được truy nguồn về original job.
