# Nested Learning for CAD — matched CADIC seed 0, backend NN tối ưu

Ngày 21/10/2026. Đặc tả thực thi đối chứng thuộc study v5.1 đã khóa. Không sửa phương pháp, partition, bank budget, scoring, metrics hoặc study gốc. NESTED seed0 đã được review endpoint; việc chuẩn bị launcher CADIC sau khi xem endpoint phải ghi là MVTec development. Không dùng kết quả NESTED để chọn cấu hình CADIC.

## 1. Phép so sánh

Chỉ expose `JOB='CADIC_seed0'`, `PHASE='7.1'`. Dùng nguyên `SupportSession('CADIC',seed=0)` của core đã khóa, một bank mới K10000/raw784×768FP32; không learner, pending hoặc meta-update. Cùng ViT-B/8 augreg ImageNet21k checkpoint, block8 zero-based, RGB direct224bilinear/ImageNet normalization, Eq8/full Eq9 b9. Cùng strict nearest/tile256/tie policy. Scoped backend `nearest_fixed_shape_writes_v1` là backend NESTED continuation đã dùng và CADIC cost đã xác minh exact/speed/memory. Không thêm tăng K, normalize, smoothing, AMP/TF32 hoặc tuning từ anomaly test.

3.599 online normals theo task order15 và seed0 partition gốc, loại cùng30normal validation/probe taskđầu. Trước khi học, stage3629registered raw train bytes, hash và build lại prefix manifest từ config gốc; phải khớp nguyên manifest đã khóa của NESTED. Staging raw bytes/manifest hashing không cấp future features cho learner. Features trích tuần tự từng online image. Không dùng30excludednormals để score hoặc update. Không input bank/learned weights của NESTED, pilot hoặc CADIC cost clones.

Sau train_complete3599 mới stage official test/GT, build manifest1725ảnh và so cả digest/rawcanonicalmanifestSHA với NESTED trước khi score. Labels/masks chỉ evaluator. Giữ pixel map28→224CPUFP32bilinear, mask224nearest/rawnonzero, score direction và macro15 như study. Undefined/nonfinite metric fail; không bỏcategory. Chỉ15finalcells; không matrix/VisA/khácseed trong notebook này.15taskstates vẫn lưu cho7.2 sau này.

## 2. Source và ownership

Source mới chứa nguyên235entries của continuation r1 (234payloads +continuation_manifest.json) byte-exact; thêm namespace `notebook_support/matched_cadic_v5_1`, docs/tests/notebook và metadata receipts. Manifest riêng khóa toàn gói; verify original continuation/r2 guards trước model/data. Core/model/engine/evaluator giữ bytes. New suite/job/execution IDs phản ánh operational source mới; scientific method/study/normalrows/backbone/runtime giữ nguyên. Không relabel state của job khác thành CADIC.

Metadata đối chiếu chỉ gồm manifest train gốc, testmanifestSHA/digest, runtime, method/study/source hashes và reviewed receipts. Không đóng kèm NESTED metrics, tensors, checkpoint hoặc feature cache; không đọc NESTED metrics trong runner. Source dựng từ raw evidence đã review và source archives được pin SHA. Ba inputs lần đầu: source mới, MVTec, ViT. Không cần NESTED output hoặc benchmarkpreflight riêng.

Khóa bankconfig vào matched reference và method. Fresh initialization bằng `session_for('CADIC_seed0')`; `fresh_initialization=true`, `learned_state_imported=false`. CADIC_seed0 trong frozen study là matched baseline của HOPE/CMS: có prequential normal-risk, event/hash/checkpoint trong harness. Thời gian này không là đo code tác giả paper thuần. Chỉ descriptive single-seed comparison, chưa CI/NI/superiority.

## 3. Kiểm tra tích hợp và lifecycle

Notebook có preflight tích hợp riêng trên **hai online normals đầu**, hai fresh discarded CADIC owners reference/optimized. Cùng features, so exact events và full state, strict state roundtrip, readonly Eq9score finite và shape. Không đọc official test. Không phải rerun pilot/W6 hoặc thay accepted full-bank G3/CADIC cost gates; không tạo AD claim. Receipt nêu rõ bank chưa đầy trong probe này. Sau preflight, benchmark CADIC vẫn fresh cursor0. Resume bỏ qua preflight chỉ khi receipt/hash/operation/source/runtime/row bindings khớp. Failure không tự retry; pause trong preflight xuất evidence nhưng chưa có accepted preflight thì probe sẽ restart trên discarded owners, không có learned benchmark state để salvage.

Engine frozen giữ checkpoint mỗi8ảnh/task boundary/pause, atomic PT→receipt, event trace và exact hard-crash replay. Luồng train không reset tại boundary. CADIC không có delayed pending/learning, nhưng finalize đúng trueend3599 và giữ15taskstates. Nếu đã train_complete thì engine chỉ kiểm lại state/owner/trace rồi trả về, không observe hoặc học thêm ảnh; nếu final_complete15 thì không compute scores lại. Có thể resume evaluation theo chunks8ảnh. Tất cả task states thuộc new CADIC job owner.

Mỗi kernel/workdir sạch. Resume chỉ nhận schema matchedCADIC với status `paused_at_atomic_boundary` hoặc `7.1_complete_pending_review` và operation lock chính xác. Từ chối failed run, source/runtime/config/data khác, job/mode khác hoặc nhiều parent phiên khác nhau. Directory và ZIP cùng manifest có thể coi cùng identity; không tự chọn cursor cao nhất. Đọc fullpayload hashes trước restore, không overwrite local files khác. Source/canonical runtime mismatch fail trước data staging/extractor. Cần GPU T4 và runtime đúng phiên NESTED; không tự pip upgrade.

SESSION_HOURS mặc định9h, hợp lệ lớn hơn600sreserve và≤11h. Deadline là started+budget−600s; boundary chỉ giữa image hoặc evaluationchunk. Cấu hình này là soft wall budget, không ETA/learningbudget; atomic event có thể vượt softdeadline và hardplatformkill vẫn có thể ngăn export. Heartbeat30s và progress8ảnh. Logs/config/resources/timing/import receipts append trong session_history; root handoff latest. Luôn export verifiedZIP kể cảfailure, rồi raise nếufailed. Completion chỉ khi workerreturn0, train_complete đúngjob3599 và final_complete15cell tồn tại. Không chỉnh rawstatus để tạo PASS.

## 4. Runtime, báo cáo và nghiệm thu

Đo riêng bootstrap/staging/verification, preflight, train session wall, evaluation, worker total, export. `engine.train_complete.elapsed_seconds` tích lũy actual benchmarktrain gồm init/restore/validation/IO/checkpoints theo engine, không preflight/export/eval. Ghi pause/resume sốphiên và initial/finalcursor để không double-count. NESTED đã dùng376ảnh chưa tốiưu trướcbridge: không lấy cumulative NESTEDtime coi fresh toànstreamoptimized. So final inference timings vẫn phải kiểm workload/backend/hardware scope. Đây không là paired fullstreamtiming giữamethods.

Local verification cần freshCADIC/noimport, sourcebyteparity, manifest/runtime guards, forbiddenparent/tamper, preflightdiscard/read-only/probeparity, interrupted→resume events/taskstates/finalmetrics, final15/chunks và no-retrain complete, atomic crash recovery/owner rejection, source ZIP/Kaggle-extracted bootstrap thật, failure export và softdeadline. Dùng CPU fixtures với geometry nhỏ để testdriver; canonical execution còn cần user chạy T4 và reviewZIP/log thật. Không tạo data, metrics hoặc timing giả.

Output `hope_cad_matched_cadic_seed0_evidence.zip` +Kagglelog lưu trong `D:/src/DATN/.agents/artifacts/runs/support_v5_1_matched_CADIC_seed0/session_01/`; pause tiếp session_02… Source và notebook theo implementation doc. Giữ nguyên NESTED endpoint; đối chiếu matched metrics sau review, rồi mới quyết định compute7.2/7.3. Không mở VisA tự động.
