# B320 — kiểm chứng phi-gradient có điều kiện trên episode thật

**Đăng ký trước execution ngày 13/11/2026, sau khi đã thấy screen B319 không đạt.**
Diagnostic số học exploratory, không đo lợi ích AD. Kết quả: lần 1 `FAILED` ở checker,
lần 2 `PASS_SCOPED` ([runs](runs/)).

## Câu hỏi

Gradient của 97 tham số phi (controller CTX) mà code production tính ở các slow event
thật có khớp một reference độc lập không? B317 mới kiểm trên fixture; actual episode
khi đó là `NOT_RUN`.

## Thiết kế

- Seed 0 mới, arm primary `a0_tail_ctx`; giữ nguyên model, optimizer, p-ramp, sampling và
  admission của B319. Lấy hai slow event đầu của prefix tối đa 64 ảnh train/good; không đọc
  test, GT hay VisA.
- So phi-gradient production với một reference FP64 độc lập (native autograd), dùng đúng
  primal FP32, nhánh và quy tắc tie của source. G_i và query lấy từ chính episode production:
  phần sampling và đạo hàm query-loss không được kiểm độc lập. Không dùng sai phân hữu hạn
  FP32 làm oracle.
- Tám hướng Rademacher cố định (seed 31901), tách query và KL; kiểm parity toàn tọa độ của
  fast optimizer, control null/nonzero và kiểm restore/RNG. Hằng số và cap ở
  [configs/b320.json](../../configs/b320.json); không nới tolerance sau khi thấy kết quả.
- Tolerance `2e-8 + 2e-4·max(|prod|, |ref|)` là gate chốt trước, không phải cận sai số tổng
  quát. Các control nhỏ (zero, nonzero, tie) phải đạt trước khi chạy episode; lệch thì điều
  tra reference trước khi gọi là bug production.
- `PASS_SCOPED` chỉ khi cả hai episode và mọi gate đạt, và tín hiệu query có thông tin; gần 0
  ở mức floor tuyệt đối, chạm cap hoặc thiếu tính độc lập là `INCONCLUSIVE`. Non-finite hoặc
  lỗi integrity thì dừng. Một attempt, không sweep.
- Cap: cả job ≤ 3.600 s, worker ≤ 2.400 s, RAM nhóm 8 GiB, VRAM cấp phát 12 GiB, output 1 GiB,
  CPU tổng 7.200 s. Đây là cap, không phải ETA.

[B317](../B317/card.md) (`NOT_RUN` cho actual episode) và [B319](../B319/card.md)
(`SCREEN_NOT_MET`) giữ nguyên.

## Các lần chạy

| Run | Kết quả | Ghi chú |
| --- | --- | --- |
| [b320-s0-20261113-01-v1](runs/b320-s0-20261113-01-v1/results.md) | `FAILED` (checker) | Episode 1 đạt; episode 2 dừng ở phép so bằng tuyệt đối khi tính lại |
| [b320-s0-20261114-02-v1](runs/b320-s0-20261114-02-v1/results.md) | `PASS_SCOPED` | Cả hai episode qua tám hướng; tolerance không đổi |

**Lần 2 sửa gì.** Lần 1 dựng lại graph query/KL để so bitwise; chỉ đổi thứ tự dựng graph đã
có thể làm gradient FP32 khác vài bit (fixture CPU: 10/16 trường hợp, lệch tối đa 2,91e-11),
nên checker có cơ chế báo sai. Lần 2 chụp gradient production trực tiếp trước bước slow Adam,
kiểm gradient đã ghi khớp bản chụp, rồi áp đúng gate theo hướng đã đăng ký. Không đổi source
model, config, seed, hai slow event đầu, tám hướng hay tolerance. Lần 1 vẫn giữ là `FAILED`.

## Kết luận và giới hạn

- Trong phạm vi hai episode thật, phi-gradient production khớp reference FP64 theo gate đã
  đăng ký. Đây là audit có phạm vi: reference độc lập cho optimizer và policy nhưng dùng
  G_i/query của production.
- Không qualification toàn bộ 97 chiều trên mọi episode, không chứng minh lợi ích AD, không
  đóng phase.
