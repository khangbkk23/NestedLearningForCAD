# Nested Learning for CAD — kế hoạch engineering throughput v5.1

Ngày 20/10/2026. Cùng engineering ID với [spec](throughput_v5_1_spec.md) và [contract](throughput_v5_1_contract.json). Đây là **kế hoạch đã đặc tả, chưa là implementation**. Task hiện tại chỉ viết docs/spec; không sửa learner/runner/notebook, không build source mới hoặc tự chạy GPU.

## A. Phần nào đã có, phần nào phải làm

| Hạng mục | Trạng thái tại thời điểm viết | Evidence/giới hạn |
|---|---|---|
| Audit source/proposal/visual/anchor/hash paths | đã kiểm offline | [audit](../../../.agents/evidence/reviews/support-v5-1-throughput-audit-20261020/review.md); không T4 component profile |
| Checkpoint7.1 NESTED_seed0 | reviewed pause376/3599 | [partial review](../../../.agents/evidence/reviews/support-v5-1-benchmark-seed0-session1-20261020/review.md); chưa anomaly metrics |
| Method v5.1 | implemented/tested, normal pilot accepted | frozen specs và [pilot](../../../.agents/evidence/reviews/support-v5-1-pilot-session2-20261020/review.md); normal-risk không là AD superiority |
| Profiler/window/budget/parity rules | specified trong ba files này | chưa notebook/CLI/timing receipt |
| Tối ưu hot path | proposed candidates | chưa sửa code, chưa biết speedup |
| Compatible migration từ376 | specified, not implemented | source mới hiện chưa có quyền đọc locked run cũ |

Không sử dụng trạng thái design-time trong oldrevision contract để suy v5.1 chưa cócode. Ngược lại, không lấy v5.1tests/preflight làm bằng chứng profiler mới đã chạy.

## B. Lộ trình và mốc dừng

### E0 — Chốt specification và bảo toàn nguồn

Đầu ra hiện tại:

1. `throughput_v5_1_spec.md`: invariants, bounded profiling, timers/counters, optimization boundary, acceptance và migration requirements.
2. `throughput_v5_1_plan.md`: việc làm/ownership và cách bàn giao.
3. `throughput_v5_1_contract.json`: parent/window/hash/budget/status/gates máy đọc.

Kiểm JSON/speclinks/reference source map166files, ancestor/taskcheckpoint/window hashes, counters từ evidence. Cập nhật `.agents` bằng review/change/spec snapshot mới; không sửa historicalspec/evidence hoặc currentmethod ID. Không build lại source benchmark đang chạy chỉ vì thêm docs.

**Dừng của task này:** bàn giao tài liệu. Chỉ implement E1 sau khi user giao bước implement; không ngầm đưa profiler vào code cùng lượt viết spec.

### E1 — Implement profiler độc lập và tests local

Các đường dẫn dưới là **dự kiến**, chỉ tạo khi triển khai:

| Module dự kiến | Trách nhiệm |
|---|---|
| `training/support_throughput_v5_1/contract.py` | Validate engineering contract/parent/source/runtime/window, không cấp quyền học mới |
| `.../replay.py` | Temporary owner load cursor179;16 arrivals; giữ partial/pending; reset deterministic |
| `.../instrumentation.py` | Coarse scopes/counters; host exclusive tree; deferred CUDA events; không đổi RNG/state |
| `.../runner.py` | U1→P1→U2; cumulative90min/48 transitions; cleanup/export/failure |
| `notebook_support/support_throughput_v5_1/` | Verified bootstrap; kiểm runtime trước việc nặng; assets; chọn parent rõ ràng |
| `notebooks/hope_cad_v5_1_throughput_profile_kaggle.ipynb` | Chỉ interface profiling; hướng dẫn input/heartbeat/export; không launch benchmark |
| `tests/support_throughput_v5_1/` | Window/rights/clock/parity/timers/stop/receipt/duplicate-input tests |

Profiler extension bọc frozen base trong subprocess/module roots xác định rõ. Không sửa166 reference payloads để đo baseline. Hook bắt buộc phải nằm trong candidate/extension có source identity riêng; không tráo code rồi ghi “unprofiled reference” cho modified base.

Tests local tối thiểu: synthetic complete U/P/U trajectory, không đổi RNG, timer cleanup khi exception, reject malformed parent/runtime, xử lý duplicate/ZIP/extracted-parent discovery, partial pass pause/export, cumulative budget resume, checkpoint prefix/taskstate hash chain và không có đường gọi GT/test loader. Compile notebook cells và bootstrap cả ZIP lẫn Kaggle-extracted form trong isolated processes. Local Windows không thay canonical T4.

Bàn giao mới phải có source ZIP/hash/file map, contract/window manifest, notebook thật và checklist. **Không chạy lại24-normal pilot preflight rồi256-normal pilot**: chỉ technical bootstrap tests và bounded16-image replay đã đặc tả. Old normal acceptance vẫn có scope cũ trên đúng source/runtime, không được ghi thành acceptance mới của optimization.

**Mốc cần người dùng:** upload source mới và chạy notebook profiling T4. Agent không tự dùng tài khoản/quota hoặc GPU. Không cần thêm W&B/Drive để bắt đầu; evidence ZIP/log là đầu vào review bắt buộc.

### E2 — Đo trên T4 và review thành phần

Inputs dự kiến bốn loại:

1. Source profiler riêng đã verify, chứa pinned base/extension và source receipts.
2. **Reviewed benchmark session01 evidence376**, không phải pilot/W6/v4 recovery evidence.
3. MVTec AD raw dataset đang dùng.
4. ViT-B/8 checkpoint SHA đang dùng.

Giữ parent source delivery cũ ở kho; bootstrap verify original base hashes trong profiler delivery. Notebook hướng dẫn inputs chính xác theo artifact đã build lúc E1, không dựa tên Dataset để suy bytes. Nhiều parent inputs thì dùng explicit override hoặc dedup nội dung exact kèm receipt, không tự chọn phiên có cursor lớn nhất.

Mỗi pass load task00 cursor179, không load active376 rồi replay backward. Run U1/P1/U2 từ179→195;16 normals cable000–015 đều đã ở evidence. Soft1,5h là cumulative budget cả phiên và resume, không ETA. Resume chỉ tiếp metadata/pass work của profiler; không trở thành benchmark resume. Kết quả thiếu pass giữ incomplete, không chọn nửa nhanh để báo.

Giữ parent ZIP/log nguyên vẹn; output mới dưới:

`.agents/artifacts/runs/support_v5_1_throughput_profile_01/session_01/`

Review đủ timings, overhead, region coverage, ops/cache/copies/memory, exact trace/state parity, source/runtime và actual compute ledger. Report nằm trong `.agents/evidence/reviews/` riêng, không bỏ whole ZIP vào review. Nếu P1 làm lệch state hoặc overhead che signal, sửa instrumentation, không sửa model. Retry kỹ thuật phải có attempt ledger, chỉ sau khi báo nguyên nhân; không lặp90min vô hạn mà coi là một lượt nhỏ.

**Mốc dừng:** xuất bảng bottlenecks đã đo và chọn tối đa3 patches. Chưa tự sửa phương pháp; không gọi +53%pilot là expected speedup. Nếu không có overhead đủ đáng kể để tối ưu, báo kết quả và giữ lựa chọn tiếp tục bằng source cũ.

### E3 — Implement các tối ưu tương đương được chọn

Mỗi patch nhỏ có bản mô tả old/new operation graph, assumptions về immutability/autograd, cache keys/invalidation, failure/resume và ảnh hưởng memory. Quyết định dựa chi phí đã đo, không theo AD metrics/loss/risk đẹp hơn.

Thứ tự ứng viên để đánh giá, không lệnh áp dụng cả loạt:

1. Reuse duplicate actual prediction trong learner, giữ denominator/higher-order ở meta branch.
2. Reuse exact digest/finite-check/result trong immutable owner scope.
3. Giảm CPU roundtrips hoặc device-local rollback nếu peak memory đã đo còn đủ; giữ đầy đủ snapshot/rollback/evidence.

Không đổi policy/geometry/proposal kernel/precision trong batch này. Không “cứ detach cho nhanh”, đổi sang FIRST_ORDER hoặc giảm32 anchors thành8. Proposal bị reject vẫn phải được dựng đủ vì đó là input/target của method hiện hành; early-veto là phương pháp khác.

Đầu ra E3: diffs nhỏ, logs kiểm thử local, candidate file map riêng, bằng chứng source tham chiếu không đổi, danh sách những gì chưa được xác minh. Nếu chỉ gần bằng về số học, chưa đủ điều kiện tiếp tục cùng thuật toán/trajectory; không hạ tiêu chí từ exact xuống close để giữ công việc.

### E4 — T4 paired validation và quyết định adoption

Candidate được chốt trước run; không search learning recipe. Hai unprofiled pairs trên cùng cửa sổ16 normals đã quan sát: R→O rồi O→R. Mỗi pass dùng temporary owner mới từ179. Bốn passes/64 transitions, soft budget riêng90min, reserve180s; không coi đây là full benchmark hoặc thay learning budget của method.

Kiểm toàn bộ trajectory/state và gradient semantics theo G2/G3. Gradient recomputation của hai meta-episodes được tách khỏi primary timing, không thêm image arrivals. Báo từng time, aggregate, memory và phạm vi đo. Hai pairs không cho statistical CI đáng tin cậy. Không adopt nếu lệch scientific state, OOM hoặc không tiết kiệm rõ sau biến động. Không bắt phải đạt mục tiêu tốc độ tùy ý bằng cách cắt phương pháp.

**Mốc cần người dùng:** chạy validation notebook riêng sau review E2/E3. Profiling và candidate validation là hai lượt khác mục đích; tổng chi phí phải được báo đầy đủ.

### E5 — Bảo toàn/tiếp tục7.1

Nếu G3 đạt, mới implement/chốt compatibility loader, lineage/receipt và technical tests theo §9 spec. Một số tối ưu giữ method/study ID nhưng execution/source identity phải đổi. Không đơn giản thay source ZIP rồi dùng `--resume` cũ; runtime guard hiện hành phải tiếp tục reject unknown source.

Một canonical bridge8 arrivals376–383 so reference/candidate trước mỗi commit; không đọc trước và không dùng AD labels. Bridge phải được bàn giao/cho phép khi đến bước đó. Fail thì giữ parent376, giữ evidence, không tự publish candidate. Đạt mới xuất checkpoint384 với execution identity mới và liên kết parent376; tiếp7.1 tới3599 và15 final cells. Compute của từng execution segment được ghi riêng, thời gian parent không bị xóa.

Nếu không đủ compatibility evidence, user vẫn có thể resume source cũ từ376 mà không làm lại; không tự rerun W6/full pilot/toàn bộ7.1. Chỉ trình lineage/rerun mới khi có thay đổi không thể giữ tương đương, nêu rõ chi phí và lý do. Không đánh giá VisA để “xem tối ưu có ổn không”.

## C. Bảng acceptance dùng khi bàn giao

| Gate | Điều kiện | Không được suy thành |
|---|---|---|
| E0/G0 | Spec/contract/hash/window nhất quán;166 files nền không đổi | Profiler đã implement hoặc checkpoint376 đã migrate |
| E1 | Modules/notebook/tests và packaged bootstrap thật | Throughput T4 hoặc AD gain |
| E2/G1 | U/P/U scientific parity; timings/counters/overhead và coverage đủ | Speedup của candidate hoặc full dataset ETA |
| E3/G2 | Candidate code và gradient/cache/rollback tests đạt | Canonical compatibility |
| E4/G3 | Paired T4 state/trajectory exact; cost/memory báo rõ | Universal bit-exact hoặc model accuracy tốt hơn |
| E5/G4 | Receipt gắn đúng parent/sources/runtime; shadow bridge đạt | W7 complete hoặc VisA authorized |
| G5 |3599 train+15 final cells reviewed; matched CADIC job riêng | Full27-job study hoặc kết luận ba seeds |

Task này dừng ở E0. Các E1–E5 là **planned**, không ghi implemented vào `.agents`. Phải giữ cumulative quota cost, failed attempts và negative findings. Khi có số liệu profiling mới, kết luận chi phí được bổ sung qua review mới; không rewrite audit cũ.

## D. Câu trả lời thực dụng cho người chạy

- Giữ ZIP/log376 đã lưu; không xóa hoặc upload đè.
- Chưa có notebook profiling mới để chạy từ tài liệu này; không tùy ý biến notebook7.1 thành profiler.
- Bước tiếp theo khi được giao là E1: bàn giao notebook/source riêng và hướng dẫn inputs thực tế. Sau E2 mới biết nên sửa đâu.
- Đề tài vẫn là **Nested Learning for CAD**, method v5.1; bộ docs throughput không biến NESTED thành CADIC thuần hoặc đổi sang FIRST_ORDER.

## E. Điều kiện sẵn sàng implementation sau double-check

Đọc §11 spec và contract revision2 trước E1. Các điểm đã làm rõ từ code:

1. Implement bounded loop riêng giữ full manifest/global cursor; không gọi train với rows bị cắt và vô tình finalize ở195. Periodic saves184/192, diagnostic final snapshot195 ngoài timing.
2. Pass output riêng; expected parent events/receipts ở read-only reference store, không prepopulate16 output receipts/events. Cold derived cache sau checkpoint load được disclosure.
3. Primary timing không chứa per-event full-state audit. So events từng ảnh và full final scientific state ngoài timing; parent không có state oracle195. Peaks và diagnostic capture có accounting rõ.
4. Chỉ complete uninterrupted passes vào speed comparison; resume giữa passes. Partial pass không tự replay/nối timing. Attempts bị rollback vẫn tính48/64cap và cumulative90min.
5. G4 là bước sau G3; phải cover train/task/eval/completion identity consumers bằng synthetic tests. Không tự coi profile PASS là được đổi source cho checkpoint376.

E0 được đánh giá đủ cụ thể để bắt đầu **E1 profiler**, chưa phải acceptance của optimization/migration. Nếu implementation gặp nhu cầu thêm replay/arrivals hoặc đổi frozen arithmetic, báo bằng amendment trước; không ứng biến trong notebook.
