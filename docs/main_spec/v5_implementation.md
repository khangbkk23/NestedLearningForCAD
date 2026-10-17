# V5 implementation — trạng thái kỹ thuật Stage 2

**Scope lịch sử:** các counts/receipts dưới đây ghi Stage2 trước cleanup. V6 integration giữ method, có tests/delivery riêng và đường dẫn hiện hành trong `workspace_v6.md`; không cần khôi phục ultils hoặc ZIP r4 để build v6.


Ngày 17/10/2026. Worktree `hope_cad_v5`, method `hope_cad_v5_causal_support_controller`. Namespace v5 mới; frozen v4/r4 giữ nguyên. **Implemented và locally tested; chưa experimentally validated.** Không có benchmark gain v5 đã đo.

## Đã triển khai

- Contract và ba specs v5; exact fork của cached CADIC proposer, raw geometry và full Eq9.
- Utility model: visual self-modification, persistent fast CMS, slow CMS/encoder/head; exact bốn-step meta-gradient và FIRST_ORDER control.
- Delayed reward từ next normal và cached historical anchors; admission ảnh hưởng actual bank và score. Terminal censor, không task-boundary flush.
- Atomic session/checkpoint, read-only scoring, RGB adapter riêng không sửa registry frozen.
- Normal-only pilot driver; canonical preflight, source/runtime/data locks, heartbeat, pause/failure export và Kaggle notebook.
- Core ablation modes FAST_ONLY/NO_FAST_ADAPTATION/NO_SELF_MODIFICATION/NO_ANCHOR_OBJECTIVE có executable paths và resume checks. Pilot chỉ đăng ký bốn jobs; không lấy chúng làm full ablation.

## Kiểm chứng thực tế

`ultils/support_v5/local_verification.json`: **26 tests v5 +58 frozen dependency regressions PASS**, không errors/failures/skips. Local Windows/Python3.14.4/PyTorch2.12.0+cu126/RTX3050Ti; khác canonical T4 environment. Test v5 có full 28×28/32-channel FP32 exact four-step meta-backward; reduced FP64 finite differences trên năm slow parameter groups; forced-PASS CADIC bank/counters/Eq9 parity; causal credit/censor; score read-only; atomic rollback; pending/partial episode resume; hard-crash event replay; driver data rights và adapter tests.

`ultils/support_v5/delivery_verification.json`: all payload SHA256 khớp live; 242 file nền r4 byte-exact; notebook cells compile; chạy bootstrap thực trong isolated processes cho ZIP và Kaggle-extracted directory; suite v5 cũng chạy từ packaged source. Source/hash cuối cùng ở `delivery_receipt.json`; không dùng HEAD một mình mô tả dirty checkout.

Chưa chạy canonical T4 real-normal preflight hoặc bốn-job pilot v5. Full-grid local learner test không thay evidence đó. Không anomaly evaluation/VisA mới trong lần triển khai này. `reportable=false` giữ nguyên.

## Việc chưa triển khai hoặc chưa nghiệm thu

Full anomaly benchmark orchestration, toàn bộ ba seeds/task matrices/uncertainty, equal-total-memory CADIC control và VisA confirmation vẫn là công việc sau pilot. Memory counter hiện đếm unique persistent tensor storages và pair-cache layout; chưa bao phủ mọi Python object/shared backbone/allocator/IO peak. Pilot elapsed chịu cache/IO warmup; không là fair compute benchmark. Chưa có bằng chứng gate hoặc exact meta-gradient cải thiện CADIC.

## Bàn giao tiếp theo

Xem `ultils/support_v5/README.md`. Chạy notebook với ba inputs source v5/MVTec/ViT. Default pilot có preflight 24 normals trên discarded clone rồi bốn fresh jobs ×256 normals. Soft budget mặc định ba giờ, có atomic resume; không là ETA. Giữ ZIP/log và review gates trước benchmark. Scientific gate fail hoặc always-PASS phải được báo; không tự đổi threshold hoặc tăng trial budget.
