# Tài liệu dự án

Repo tự chứa code, cấu hình, phương pháp, kết quả tổng hợp và spec/plan nghiên cứu. Không cần truy cập context/chat/worktree riêng để hiểu hoặc implement. Dataset, checkpoint và raw result archives là tài sản bên ngoài được định danh bằng receipts/SHA, không commit trong Git.

## Code hiện tại — đọc trước

| Tài liệu | Nội dung |
|---|---|
| [Architecture](architecture.md) | Package/ownership/luồng code và điểm vào thật |
| [Method](method.md) | Toán học và hợp đồng của v5.1 đang implement |
| [Benchmark](benchmark.md) | Protocol3.599 hiện hành, evaluator/controls/statistics và coverage còn thiếu |
| [Operations](operations.md) | Các lệnh đang tồn tại: setup/preflight/train/evaluate/matrix/resume/export |
| [Results](results.md) | Benchmark đã có, provenance/hashes, chi phí và giới hạn |

Nguồn thực thi máy đọc nằm tại [resources](../src/hope_cad/resources). Runtime vẫn dùng3599normals; workspace tênv7 không đổi algorithm thànhH32.

## Kế hoạch nâng cấp — chưa implement

| Tài liệu | Trách nhiệm |
|---|---|
| [Pipeline alignment spec](pipeline_alignment_spec.md) + [JSON](pipeline_alignment_contract.json) | Chốt full-train3629, recipe giữ nguyên, identity/fairness và yêu cầu successor pipeline |
| [Research specs](research/README.md) | Đầy đủD0/H32/controls/study/acceptance và contract tổng hợp đã áp dụng full-train |
| [Implementation plan](research/implementation_plan.md) | Thứ tự làm, nghiệm thu từng bước, stop rules và phần cần remote GPU |

Hai file pipeline không mô tả một tính năng đã chạy: chúng là design decision cho cohort tiếp theo. Spec nghiên cứu không được runtime hiện tại tự động nạp. Khi triển khai xong phải đồng bộ docs/resources/status theo code và evidence thật.

[QUICK_UPDATE](../QUICK_UPDATE.md) là báo cáo trao đổi với nhóm, có Overview và milestone/per-category details. Nó dẫn tới specs trong repo; không thay contract thuật toán. Papers ở thư mục `papers/` và provenance upstream ở `refs/`.
