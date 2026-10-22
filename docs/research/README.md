# Đặc tả nghiên cứu tiếp theo

**Đã đặc tả; chưa triển khai analyzer D0, successor full-train hoặc H32; chưa có kết quả mới.** Phương pháp đang chạy trong repo vẫn là v5.1, mô tả ở [method](../method.md). Tài liệu này không là chỉ dẫn chạy CLI của candidate.

Đây là bộ spec công khai, tự đủ cho implementation và phản biện. Người clone repo không cần chat, worktree khác hoặc kho context riêng. Các archives thực nghiệm là inputs bên ngoài phải xin riêng nếu muốn tái tính số liệu.

| Câu hỏi | Nguồn chuẩn trong repo |
|---|---|
| Vì sao chọn thay đổi này, evidence hỗ trợ tới đâu? | [Rationale và giới hạn](rationale.md) |
| Phân tích offline các quyết định đã lưu thế nào? | [D0 attribution](attribution.md) |
| H32 học gì, reservoir/RNG/gradient/state ra sao? | [Candidate](candidate.md) |
| Dữ liệu, preprocessing, CADIC route cần chốt gì? | [Pipeline spec](../pipeline_alignment_spec.md), [JSON](../pipeline_alignment_contract.json) |
| Controls, pilot, metrics, ngân sách, stop rules? | [Study](study.md) |
| Làm theo thứ tự nào và nghiệm thu từng phần thế nào? | [Implementation plan](implementation_plan.md) |
| Hằng số và IDs tổng hợp máy đọc? | [Research contract](contract.json) |

Contract tổng hợp đã áp dụng quyết định full-train 3.629 vào bản thiết kế H32: không cần người đọc tự ghép một bản study 3.599 với amendment ngoài repo. D0 vẫn phân tích trace lịch sử 3.599. H32 history32/support4/recent2+history2, optimizer, RNG, causal order và gates không đổi so với thiết kế đã chốt.

## Ranh giới nguồn chuẩn

- **Code hiện tại:** [method.json](../../src/hope_cad/resources/method.json), [study.json](../../src/hope_cad/resources/study.json), CLI và source hiện hành; cohort 3.599. Chúng chưa nạp research contract này.
- **Implementation sắp làm:** toàn bộ bộ spec tại đây và hai file pipeline. Candidate `hope_cad_hq32_fulltrain_v1`, recent contract `hope_cad_v5_1_fulltrain_v1`, protocol `mvtec_1x15_fulltrain_v7_v1`, study `hq32_mvtec_fulltrain_development_v1` là IDs dành trước, chưa có runtime factories.
- **Kết quả đã có:** [results](../results.md), thuộc matched seed0 v5.1 cũ, không phải full-train/H32/v7 GPU benchmark.
- **Báo cáo trao đổi:** [QUICK_UPDATE](../../QUICK_UPDATE.md) là tổng quan và giải thích cho nhóm; khi triển khai dùng spec/contract, không tự suy thuật toán từ đoạn tóm tắt.

Nếu phát hiện mâu thuẫn giữa các spec prospective, dừng và ghi correction trước implementation. Không coi JSON nghiên cứu đã tồn tại là chương trình đã hỗ trợ nó; không sửa frozen resources hoặc archived evidence để làm khớp lời kể. Bản chính thức trước nâng cấp là commit `89df9a21658b265731f98a7c2014afb1876f1924`.
