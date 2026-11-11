# Nested Learning for CAD

Đối chiếu source: `74d975b151f59ad0b5fbf23be392e2d891f350bf` — 11/11/2026.

Gói nghiên cứu của Tài về **learned replay cho continual anomaly detection**:
encoder DINOv2 frozen, decoder Dinomaly trainable, memory features và controller
học trọng số replay. Workload được hỗ trợ hiện tại là **B319 seed 0**, ba arms
`er_uniform`, `a0_tail`, `a0_tail_ctx`.

Detector có thành phần kế thừa qua Khang và upstream; xem
[provenance](docs/provenance.md). Worktree này không chứa workflow NL-CAD của
Khang và không kế thừa kết quả benchmark của bạn ấy.

## Đường đọc

| Muốn làm gì | Đọc |
| --- | --- |
| Hiểu cấu trúc và từng bộ phận code | [Architecture](docs/architecture.md) |
| Hiểu memory, learned replay, fast/slow update và CTX | [Method](docs/method.md) |
| Hiểu dữ liệu, task order, metrics và giới hạn kết luận | [Protocol](docs/protocol.md) |
| Setup, chạy, monitor, lấy kết quả | [Operations](docs/operations.md) |
| Truy nguồn, hiểu `.local/` và backup gaps | [Provenance](docs/provenance.md) |
| Tra lịch sử v6/v7 → hiện tại | [Experiment history](experiments/HISTORY.md) |
| Tạo experiment/run, giữ failure và đóng receipts | [Experiment lifecycle](experiments/README.md) |

Docs mô tả implementation ở baseline trên, không tạo bộ locked specs mới.
Source/scripts + config/manifest/lock định nghĩa hành vi chạy. Khi có mâu thuẫn
với protocol đã đăng ký, ghi defect và review thay vì âm thầm đổi tiêu chí.
Cards/receipts ghi thiết kế/evidence của từng lượt; không áp mọi spec lịch sử
cho phương pháp hiện tại.

## Bắt đầu

Từ root checkout, cài environment đã khóa (cần mạng và dung lượng phù hợp):

```sh
pixi install --locked -e cad
pixi run --locked -e cad train --help
```

Training hỗ trợ **Linux + NVIDIA CUDA GPU**. Windows dùng để đọc code,
package, kiểm inputs và tests phù hợp; chưa có full training CPU/Windows.
Xem [các bước vận hành](docs/operations.md) trước khi tải inputs hoặc chạy GPU.
Pixi activation đưa `src/` vào `PYTHONPATH`; không cần `pip install -e .`.
Cài distribution riêng không provision ML dependencies.

## Trạng thái bằng chứng

[B319 card](experiments/B319/card.md) giữ kết quả/disposition: cả learned arms
chưa đạt development screen; MVTec test đã dùng trong phát triển. Standalone
khớp recorded matrices của original trong phạm vi audit. Các đổi namespace,
environment/tooling về sau không tự kế thừa qualification của bundle cũ.
Không suy Phase4B closure, paper parity hoặc independent confirmation từ docs.

Git chứa source và reports nhẹ; datasets, weights, checkpoints, raw scores và
credentials nằm ngoài Git. Khả năng raw audit phụ thuộc payloads/locators trong
[provenance](docs/provenance.md). [LICENSE](LICENSE) giữ MIT notice gốc và notices
Apache-2.0 của phần kế thừa.
