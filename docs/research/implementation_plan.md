# Kế hoạch implementation và nghiệm thu

**Trạng thái: chưa thực hiện các bước dưới đây.** Đây là kế hoạch có acceptance, không phải danh sách lệnh candidate chạy được. Dùng [contract](contract.json), [candidate](candidate.md), [study](study.md) và [pipeline](../pipeline_alignment_spec.md) cùng nhau.

| Bước | Phần cần làm | Điều kiện hoàn thành |
|---|---|---|
| 1. D0 offline | Analyzer allowlist/streaming SHA, causal chronology, các predictors/metrics đã chốt; output root do người chạy cấp | Đúng input SHA/3.599 events/3.211 credits/3.179 postwarm; tests temporal/future trap/QA; negative result vẫn hợp lệ |
| 2. Successor full-train | New protocol/resources/factory/manifest; giữ loader/JSON baseline strict | 3.629 normals, Bottle209, zero exclusions; chênh đúng Bottle070–099 với manifest cũ; all seeds cùng IDs/order; current runtime vẫn có thể đọc contract cũ đúng guard |
| 3. Cohort identity | Bind producer/checkpoint/recipe/backend/precision/evaluator; per-job method digest khác common cohort digest | Same-cohort CADIC/recent so được, khác data/source/runtime không được gộp; old checkpoints bị từ chối; không đổi transform hoặc numerical policy một phía |
| 4. H32 và controls | Immutable completed history, hai RNG, sampled query snapshot, exact/FO/direct; actual phi8 khác functional phi4 | FP64 oracle + full-grid FP32 meta-backward; R_EXACT scientific parity với base equations; FIFO/future/overlap/ownership traps; DIRECT zero-first-gradient hợp lệ |
| 5. State/resume | Session/learner/history schemas, partial blocks1–7, transaction, pending/censor, debug traces | Interrupted/uninterrupted parity, RNG rollback, hard-crash không double insert, score read-only, malformed state từ chối; true-end partial không flush |
| 6. Delivery local | Clean install, inventory, CLI/factories, resource accounting, outputs/receipts | New source digest/bundle; không ghi đè archive cũ; public docs/status khớp code thật; tests không bị gọi là AD gain |
| 7. Remote preflight/pilot | Đúng máy/runtime đã cấp quyền; discarded clones rồi9fresh×256 | Toàn gates trong study; full-train pilot209Bottle+47Cable, last12881+47; cap12GPUh tính cả setup/IO; không chọn lại checkpoint hoặc kéo dài để cứu gate |
| 8. Matched development | Fresh CADIC/R_EXACT/H32_EXACT seed0×3.629 +120cells/job | Cap24GPUh; raw scores/recompute/lineage/resources đủ; report cả3; screen chỉ quyết định chi tiếp |
| 9. Attribution/replication | Sáu seed0controls +sáu seed1/2jobs của3main methods khi đạt screen và có allocation | Tối đa15fulljobs tính cả3đã có; không chọn bestseed/arm; finite-seed exploratory statistics theo study |
| 10. Quyết định và VisA | Chọn/giữ method theo evidence, freeze code/recipe/protocol; VisA suite riêng | Exposure ledger, official split, fresh bank/learner, runner và budget riêng; không chỉ thay data path |

## Ranh giới thay đổi

Không tăng width/unroll/K/anchors, đổi utility target/LR, sửa preprocessing, thay proposer ContCore hoặc thêm component để cứu H32 trong vòng này. Lỗi đã chứng minh phải được sửa có version/provenance và nghiệm thu lại đúng phạm vi; kết quả âm không được tự gán là bug. Chi phí/caps không phải ETA hay quyền tự thuê máy.

Trong lần tổ chức docs này chỉ công bố/đồng bộ đặc tả: model/runtime code và frozen resources chưa đổi. Cần task implementation riêng để thực hiện kế hoạch; các lệnh hiện có chỉ được mô tả trong [operations](../operations.md).
