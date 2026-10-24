# Tài liệu dự án

Repo tự chứa code, cấu hình, phương pháp, kết quả tổng hợp và spec/plan nghiên cứu. Không cần truy cập context/chat/worktree riêng để hiểu hoặc implement. Dataset/checkpoint/raw outputs mặc định nằm trong workspace/ được ignore ngay trong checkout; định danh bằng receipts/SHA và không phát hành trong Git.

## Code hiện tại — đọc trước

| Tài liệu | Nội dung |
|---|---|
| [Architecture](architecture.md) | Package/ownership/luồng code và điểm vào thật |
| [Method](method.md) | Toán học và hợp đồng của v5.1 đang implement |
| [Benchmark](benchmark.md) | Protocol3.599 hiện hành, evaluator/controls/statistics và coverage còn thiếu |
| [Operations](operations.md) | Các lệnh đang tồn tại: setup/preflight/train/evaluate/matrix/resume/export |
| [CADIC diagnostic](cadic_diagnostic.md) | Kiểm riêng K2500/K5000 trên full-train trước research preflight; timers/QA/resume/export độc lập |
| [Stage handoff v1](research/stage_handoff_v1.md) | One-ZIP pack, exact-identity qualification/restore và bootstrap readiness; chưa real transport acceptance |
| [Results](results.md) | Benchmark đã có, provenance/hashes, chi phí và giới hạn |

Nguồn thực thi máy đọc nằm tại [resources](../src/hope_cad/resources). Baseline CLI dùng3599normals; CLI research mới dùng3629 và new method identities. Workspace tênv7 không tự gán benchmark cũ choH32.

## Nâng cấp — đã implement local, chưa remote acceptance

| Tài liệu | Trách nhiệm |
|---|---|
| [Pipeline alignment spec](pipeline_alignment_spec.md) + [JSON](pipeline_alignment_contract.json) | Chốt full-train3629, recipe giữ nguyên, identity/fairness và yêu cầu successor pipeline |
| [Research specs](research/README.md) | Đầy đủD0/H32/controls/study/acceptance và contract tổng hợp đã áp dụng full-train |
| [Implementation plan](research/implementation_plan.md) | Thứ tự làm, nghiệm thu từng bước, stop rules và phần cần remote GPU |

Hai file pipeline là design decision đã được runtime research thực thi qua resource riêng `resources/research.json`; baseline không tự nạp candidate. Xem [implementation/SSH handoff](research/implementation.md) và [phạm vi technical preflight đã có](results.md#technical-preflight-tren-may-thue); chưa có completed/accepted H32 pilot hoặc AD result. [Handoff v1](research/stage_handoff_v1.md) đã có exact-identity implementation; changed-runtime/source mapping trong [proposal rộng hơn](research/stage_handoff_spec.md) vẫn prospective.

[QUICK_UPDATE](../QUICK_UPDATE.md) là báo cáo trao đổi với nhóm, có Overview và milestone/per-category details. Nó dẫn tới specs trong repo; không thay contract thuật toán. Papers ở thư mục `papers/` và provenance upstream ở `refs/`.
