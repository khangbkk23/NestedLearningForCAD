# Nested Learning for CAD — Hướng dẫn dự án và implementation v5

Ngày 17/10/2026. Đọc cùng `architecture_decision_v5.md` và `revision_contract_v5.json`. Đây là bộ đặc tả mới của Stage 2; không sửa lịch sử v4. Method ID: `hope_cad_v5_causal_support_controller`. Trạng thái: **implemented và locally tested; chưa experimentally validated**.

## 1. Bối cảnh và câu hỏi nghiên cứu

Đồ án học các category tuần tự từ ảnh normal, rồi đánh giá detection và localization trên các category đã đến. Frozen ViT cung cấp hình học chung với CADIC. V4 dùng masked teacher prediction và output CMS làm anomaly score; kết quả matched seed 0 kém CADIC đáng kể. Recovery và chẩn đoán normal có giới hạn đã tìm thấy dư địa tối ưu readout, nhưng tăng plasticity cũng gây interference. Các kết quả đó không được đổi tên thành một software bug đã chữa, hoặc suy rộng thành kết luận rằng paper Nested Learning thiếu giá trị.

V5 chọn câu hỏi mới: **Nested Learning có thể học chính sách quản lý normal-support bank tốt hơn một luật cập nhật cố định không?** Learner dự đoán utility của proposal CADIC; quyết định của nó thay đổi bank thật; bank thật quyết định anomaly score. Raw feature geometry được giữ nguyên, thay vì bắt reconstruction decoder học lại toàn bộ không gian 768 chiều.

Nested Learning ở đây có ba quá trình với context và clock riêng: self-modification trong từng ảnh; fast CMS học sau mỗi delayed utility label; slow encoder/CMS/head nhận exact meta-gradient sau mỗi tám label. Bank là bộ nhớ explicit gồm raw patches; fast CMS là neural state. Hai loại memory này có vai trò khác nhau. Đây là HOPE-derived CAD adaptation, không phải bản sao toàn bộ HOPE hoặc VisionHOPE.

Nguồn bằng chứng trước revision: `docs/recovery_stage1_kaggle/main/review.md`, `docs/normal_causal_kaggle/review_v1.md` và `results/normal_causal/main_review.json`. Các benchmark cũ giữ nguyên; không dùng chúng như evidence về performance v5.

## 2. Dữ liệu, input, output và quyền truy cập

RGB resize trực tiếp về 224 × 224 bằng PIL bilinear, ImageNet normalization, frozen ViT-B/8 ImageNet21k, block 8 zero-based/layer 9 one-based, bỏ CLS: output là 784 raw patches × 768 chiều. Checkpoint SHA, preprocessing, tie handling và Eq9 dùng cùng matched CADIC. Không thêm LayerNorm hoặc L2 normalization lên raw patches. Fixed projection 32 chiều chỉ phục vụ context và meta replay; không đổi geometry của bank hay score.

`observe` nhận đúng một normal image đã đến và stable identity để kiểm cursor. Task ID không dùng để routing. Test labels/masks chỉ thuộc evaluator; learner không đọc chúng. Manifest giữ canonical MVTec order và 30 held-out exclusions của matched protocol, còn 3.599 online train images. Inventory và hash không cấp quyền nhìn trước future features. Mỗi job, seed hoặc dataset có learned state mới.

V5 không dùng offline calibration hoặc hyperparameter search trên prefix. Tất cả fast/slow learning đều causal trong stream; VisA cũng dùng cùng luật với fresh state. Đây là protocol revision được công khai so với v4. Cache frozen features chỉ tái sử dụng khi image identity, checkpoint, preprocessing và tensor hashes khớp; driver vẫn chỉ cấp từng image theo cursor.

## 3. Toán học và nhân quả

Với finite observed normal set S và coreset C ⊆ S, đặt epsilon = max distance từ S đến C. Triangle inequality cho `d(q,S) ≤ d(q,C) ≤ d(q,S) + epsilon`. Muốn suy ra separation cần giả thiết normal gần S và anomaly đủ xa S. Không có theorem bảo đảm AUROC, Eq9 ranking hoặc thắng CADIC.

Normal risk là tổng bằng nhau của mean và max squared nearest-neighbor distance. Tại proposal t, C− là bank cũ và C+ là clone đã chạy đúng CADIC image update. Cả hai cố định đến image kế tiếp. Anchor risk được tính từ reservoir trước image t và cache ngay lúc proposal. Khi normal image t+1 thực sự đến, trước write của nó, tính risk trên cả hai banks rồi tạo relative advantage trong [−1,1]. Admission ở t không được dùng reward này trước khi nó đến. Cả proposal được nhận hoặc bị từ chối đều có delayed label. Terminal pending proposal bị censor; task boundary hoặc pause không tạo label giả.

Fast CMS cập nhật sau từng completed label. Slow objective bắt đầu từ fast state ở đầu block tám label: bốn chronological support labels tạo bốn differentiable fast steps; bốn query labels còn lại tạo meta loss. Gradient đi qua inner gradients tới slow parameters. FIRST_ORDER giữ cùng forward nhưng detach inner-gradient dependence. Replay lưu fixed projected tokens, known statistics và labels đã đến; encoder phải được tính lại, không lưu detached learned embeddings rồi gọi đó là exact encoder meta-gradient.

Actual fast state đã học cả tám records được giữ sau slow commit; không reset sang meta clone. Partial block và pending reward giữ qua task boundary và pause. Không flush partial block ở cuối stream. Phương trình, epsilon, margins, initialization, parameter ownership và clocks đầy đủ nằm ở architecture §§3–6 và JSON contract.

## 4. Codespace và đường chạy

| File hoặc namespace | Trách nhiệm |
|---|---|
| `models/hope_cad_v5/contract.py` | Đọc contract, canonical identity và design hash; không tự search |
| `models/hope_cad_v5/geometry.py` | Fork chính xác proposer CADIC; risk, context statistics, projection, anchors |
| `models/hope_cad_v5/learner.py` | Visual encoder, fast/slow CMS, utility head, exact meta-gradient và clocks |
| `models/hope_cad_v5/session.py` | Causal event, transaction, pending bank, score read-only, checkpoint và memory accounting |
| `models/hope_cad_v5/benchmark_adapter.py` | RGB adapter riêng, frozen teacher và Eq9; không sửa registry cũ |
| `training/hope_support_v5.py` | Manifest-scoped pilot, frozen-feature cache, resume, profiling và gates |
| `notebook_support/support_v5/` | Bootstrap, runtime lock, normal-only assets, worker, heartbeat, export và delivery verification |
| `tests/hope_cad_v5/` | Numerical, causal, geometry, gradient, atomicity, resume và lifecycle tests |
| `notebooks/hope_cad_support_v5_kaggle.ipynb` | Giao diện Kaggle duy nhất cho preflight và pilot |
| `ultils/support_v5/` | Source ZIP, hash receipts, verification và hướng dẫn bàn giao |

Dependency cũ vẫn cần: visual block/recurrence và CMS trong `models/hope_cad_v2/`, cached coreset và score trong CADIC, protocol/hash helpers đã khóa. Tên namespace cũ không làm chúng thành legacy có thể xóa. Không sửa recurrence hoặc baseline để giảm gap. Source bundle mới bảo toàn 242 file nền r4 và bổ sung namespace v5.

`v5_implementation.md` ghi trạng thái thực. Sự hiện diện của adapter hoặc notebook không có nghĩa full benchmark đã hoàn tất. Shared `.agents` ở ngoài worktrees chứa ADR, change records, snapshot và evidence registry; snapshot là context có hash, không thay source bundle tái lập code dirty.

## 5. Chạy Kaggle và resume

Chỉ cần ba inputs: source v5, archive MVTec AD và checkpoint ViT đã khóa. Không attach W6 learned weights, cache cũ hoặc recovery output làm initialization. Notebook default `PHASE='pilot'`: preflight trên 24 real normal images trong clone bị discard, rồi bốn jobs mới CADIC/GEOMETRIC/FIRST_ORDER/NESTED, mỗi job 256 normal images, seed 0, một trial. Có thể dùng `PHASE='preflight'` để chỉ kiểm môi trường.

Chọn GPU T4, dùng Save Version → Save & Run All. Runtime packages phải khớp lock; không nâng packages để vượt lỗi. V5 khai báo thêm deterministic settings và CUBLAS workspace trước CUDA; mọi control mới dùng cùng settings. Local reduced fixtures hoặc RTX profiling không thay thế canonical T4 real-normal evidence.

`SESSION_HOURS` là soft budget, không phải dự báo thời gian hoàn thành. Runner dừng ở image boundary và dành thời gian xuất ZIP. Nếu pause, attach latest v5 output và resume cùng source/runtime/manifest/config. Không gọi finalize ở pause. Sau hard crash, có thể replay tối đa bảy images kể từ checkpoint gần nhất; events đã ghi phải khớp chính xác. Tuple/list được canonical hóa bằng JSON nhưng thay đổi giá trị thực phải bị từ chối.

ZIP có manifest, handoff, checkpoints, utility/coverage traces, profiling, gates và logs. Luôn cố xuất failure ZIP; hard platform kill vẫn có thể ngăn export. Giữ saved version, tải ZIP và log về. Không dùng failed bootstrap làm completed parent hoặc bỏ kiểm hash. `complete_pending_review` cần review, không tự cho phép anomaly benchmark.

## 6. Nghiệm thu và giới hạn

Pilot PASS kỹ thuật không chứng minh gain. Mechanism gate đòi hỏi learner thực sự đổi bank và normal score, đủ fast/slow commits và gate không luôn PASS/REJECT. Normal risk gate so last 128 prequential images với CADIC; toàn bộ ngưỡng khóa ở architecture §7. Failure hoặc no-effect phải được giữ, không đổi threshold rồi gọi cùng method.

Đo overhead gồm alternative bank, anchors, replay, fast/slow parameters, optimizer, pair cache và IO. Pilot timing chịu ảnh hưởng cache warmup nên không là fair latency benchmark. Một ViT forward không có nghĩa nhanh hơn CADIC: proposer, delayed credit và meta-learning đều có chi phí.

Sau review mới triển khai full matched benchmark runner, ba seeds, controls, task matrix và uncertainty. MVTec đã development-exposed. VisA là confirmation sau freeze source/spec/transfer rules, với fresh causal state và không đổi hyperparameters theo test. Thesis cần báo cả negative v4 findings, assumptions và limitations. ReplayCAD là baseline do Khang phụ trách, cần matched artifacts trước so sánh chính thức. Không có phần nào bảo đảm ngang hoặc hơn CADIC trước thực nghiệm.
