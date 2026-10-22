# Nested Learning for CAD — Chốt nghiệm thu trước implementation v5.1

Ngày 19/10/2026. Đây là **clarification bổ sung** cho bốn files architecture/instruction/planning/contract v5.1 đã seal, không sửa phương trình, hyperparameters, gate hoặc evidence trước đó. Method vẫn `hope_cad_v5_1_causal_scaled_utility`; chưa implementation, production-tested hoặc AD-validated. Đọc file này cùng [architecture](architecture_decision_v5_1.md), [instruction](instruction_CAD_v5_1.md), [planning](planning_CAD_v5_1.md) và [contract](revision_contract_v5_1.json).

## 1. Kết luận kiểm tra lại

Không phát hiện blocker toán học hoặc dependency khiến phải đổi thiết kế trước khi bắt đầu I1. Local fast partial, derivative của quotient trong exact meta, whole-increment FO detach, scalar casts, causal pending và sự phân biệt objective J / safety gate R nhất quán với code primitives được tái dùng.

Kết luận này chỉ đủ để triển khai theo spec và kiểm thử. Nó không chứng minh không còn mọi bug, đã tận dụng tối đa Nested Learning, model ổn định trên mọi stream, pilot sẽ PASS hoặc benchmark sẽ ngang CADIC. Không tự chạy Kaggle từ tài liệu này.

## 2. Khóa định nghĩa reused-normal score probe

Gate `score_differs > 1e-6` đã có trong contract nhưng chưa nêu rõ probe identities/count. Chốt kế thừa phép đo ở `training/hope_support_v5.py:74–77,90–94`:

1. Probe là **đúng 8 online train-normal identities đầu tiên** của ordered/hash-bound `pilot_rows.json` cho trial này, tức `pilot_rows[:8]`. Không sampling, đổi category, chọn theo score hoặc thay vì một probe không cho kết quả khác biệt.
2. Tất cả controls dùng đúng cùng 8 identities, raw feature producer và feature hashes. Ghi ordered probe manifest với relative path/image SHA256/raw feature digest; đối chiếu equality trước khi so score. Nếu thiếu hoặc thứ tự không khớp thì technical failure, không zip/truncate arrays cho qua.
3. Đo sau khi job đã đi đủ 256 images và `finalize_stream` chỉ censor pending như spec. Score là read-only actual **final bank**. Probe không tạo label, học, chọn checkpoint hoặc thay đổi scientific state/RNG.
4. Đọc **784 raw patch NN distances của mỗi ảnh**, trước map resize và trước Eq9 image weighting. Reduction duy nhất: `max_{i=0..7,j=0..783} abs(score_NESTED[i,j]-score_CADIC[i,j])`. Mọi tensor đúng shape và finite; maximum phải **lớn hơn** 1e-6.
5. Đây là reused training-normal **mechanism probe**, không independent validation, retention estimate hoặc anomaly performance. Final-bank hash gate và risk last128 vẫn giữ riêng, không thay thế cho nhau.

Driver/contract identity executable mới phải đưa quy tắc này và probe-manifest hash vào payload trước pilot. Không rewrite `revision_contract_v5_1.json` cũ để làm receipt thiết kế đã seal trông như đã có field; builder có thể đọc acceptance này hoặc encode quy tắc tương đương, nhưng source identity phải bao gồm nó và tests phải kiểm.

## 3. Quy ước clock/optimizer theo mode

`labels` đếm completed delayed labels hợp lệ. `completed_label_blocks=labels//8` chỉ là số cửa sổ hoàn tất, không tự là số optimizer updates. `fast_updates` đếm actual fast steps thành công; `slow_updates` đếm **actual successful slow Adam commits**; fail trong image transaction rollback cả các counters. Không gọi một no-op schedule là parameter learning.

| Mode | fast_updates | slow_updates / Adam calls | Partial replay window |
|---|---:|---:|---|
| NESTED, FIRST_ORDER | labels | labels//8 | labels%8 records; clear sau successful commit |
| FAST_ONLY (planned) | labels | 0; không slow Adam update | giữ cùng 8-label boundaries, clear khi đủ8; phi_start iff partial window |
| NO_FAST_ADAPTATION (planned) | 0 | 0; không tạo Adam steps với zero/disconnected gradients | giữ 8-label bookkeeping, clear khi đủ8; không pretend đã meta-learn |
| CADIC, GEOMETRIC | không learner labels/updates | không learner optimizer | không utility replay window |

NO_FAST_ADAPTATION vẫn có completed-label clock để thực hiện learned-policy warmup32 và record chronology. Zero readout làm prediction=gamma trên **trajectory riêng**; forced-PASS có thể khiến bank khác GEOMETRIC. Adam state của mode này là null/absent theo schema executable mới phải khai rõ, không serialize fabricated trained optimizer. Các planned controls chưa được mở pilot hoặc claim contribution bởi bảng này. NO_SELF_MODIFICATION/NO_ANCHOR_OBJECTIVE khi triển khai về sau giữ learning counters của NESTED với active ownership/objective của ablation được khóa.

## 4. Những assertion phải có trong implementation

- FO/NESTED phải có initial theta, phi, projection và anchor-RNG hashes giống nhau khi cùng seed trước image đầu. Init theo private RNG và deterministic construction; không chỉ ghi cùng số seed rồi mặc định state giống nhau.
- Shared `training/runtime.py` tắt TF32 nhưng tự nó chưa bật tất cả deterministic settings. Wrapper v5.1 phải áp dụng đầy đủ canonical deterministic/CUBLAS setup từ contract, không sửa hoặc dựa nhầm vào v5 identity.
- `dataset/benchmark_protocol_v2.py:203–206` còn metadata `prefix_calibrated_not_strict_single_pass` của quyền v4. Successor giữ đúng partition/hash nhưng phải xuất quyền riêng: không offline calibration/search; one-image causal delivery; replay của observed projected contexts được cho phép. Không gọi pipeline strict single-pass theo nghĩa không replay, cũng không báo prefix calibration đã chạy khi nó không chạy.
- Tái dùng `CMSLevel` và visual primitives, không tái dùng optimizer của `ContinuumMemory.event` như thể đó là update rule v5.1. Với current fast tensors là các functional state outputs đồng thời, `autograd.grad(loss,current_phi,create_graph=True)` là local fast partial; outer derivative vẫn đi qua dependency lịch sử trong unroll. Không request total gradient của slow theta rồi gọi là local fast gradient.
- Canonical FP32/full28×28 numerical/state tests, malformed checkpoint rejection và crash/resume tests vẫn bắt buộc trong I1/I2. Synthetic FP64/reduced CPU audit không thay thế chúng. Benchmark evaluator, uncertainty, full resource fairness và VisA còn planned như bộ spec đã nêu.

Không đổi learning rate, head ownership, target, c/d, warmup32, task/data rights hoặc risk gate1.10 trong clarification này. Nếu implementation phát hiện cần đổi các semantics ấy, báo thiết kế trước và tạo revision/lineage rõ; không cứu gate trong lúc chạy.
