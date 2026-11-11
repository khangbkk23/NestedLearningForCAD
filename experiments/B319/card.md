# B319 — Full-test corrected replay và standalone reproduction

**Card hồi cứu. Workload seed0 hiện hành; chưa có scientific adoption.**

## Câu hỏi và thiết kế

Protocol input `c239…` ghi ER / corrected-tail / corrected-tail+CTX, primary
`a0_tail_ctx`, gain Pixel AP ≥0,005 so ER, floors mỗi class Pixel AP ≥−0,01 và
Image AUROC ≥−0,005. Intake đối chiếu source/receipt, không coi card này là bằng
chứng preregistration. Numerical/CTX scope xem [B317](../B317/card.md).

Một lượt 3.629 train/good, 15 tasks bottle → zipper, một fast update/arrival.
Full1725 test sau mỗi task trên tasks đã học;120 observed cells/arm. Reservoir 24
+ FIFO 8; learned arms ghi 225 slow updates. Cùng fast/memory nhưng chưa matched
compute. MVTec đã development-exposed.

## Kết quả được giữ

| Arm | Final Pixel AP (%) | Image AUROC (%) | Δ Pixel AP vs ER (pp) |
| --- | ---: | ---: | ---: |
| er_uniform | 59.83418 | 96.75749 | +0.00000 |
| a0_tail | 60.14776 | 96.82860 | +0.31359 |
| a0_tail_ctx | 59.62896 | 96.90572 | -0.20522 |

**Corrected tail và primary CTX đều `SCREEN_NOT_MET`.** Không đổi primary/margins.
JSON giữ full precision và signed prior-max/acquisition forgetting trên 14 tasks
có lịch sử; không lấy max gồm final rồi tự coi cùng định nghĩa.

Standalone v5 khớp 360 recorded metric cells và 45 reported model hashes với B319
gốc. Intake tính lại aggregates từ recorded matrices, không re-rank raw scores
hay inference. Audit gốc ghi kiểm raw final 45 cells; 315 historical raw cells và
standalone raw tensors không được nâng thành đã tái kiểm ở đây.

## Lineage và giới hạn

B317 là qualification dependency; B319 gốc là replication reference, không phải
checkpoint initialization. Năm versions standalone: 3 cancel, 1 bootstrap error,
1 COMPLETE; cancel cause/progress thiếu evidence vẫn `unknown` trong [runs](runs/).

Namespace/env/reporting hiện tại đổi sau payload đã tái lập; bundle mới không tự
kế thừa qualification. VisA chỉ là ứng viên holdout chưa chạy. Không suy independent
confirmation hoặc Phase4B closure. Config hiện hành có một bản tại
[configs/b319.json](../../configs/b319.json).

## Nguồn và verification

- [R-e54916184dbc23cfe1](../history/catalog.json#/records/R-e54916184dbc23cfe1) — `S-c239b64fb510cd616b6b`.
- [R-2cc61f989a3afc5fcb](../history/catalog.json#/records/R-2cc61f989a3afc5fcb) — `S-a02811f4d1828808737b`.
- [R-754e9f33a678b3e5cf](../history/catalog.json#/records/R-754e9f33a678b3e5cf) — `S-5b20282db2c6eb0709bf`.

[V-B319-recorded-matrices](../history/audit.json#/verifications/V-B319-recorded-matrices)
ghi chính xác phạm vi đối chiếu.
