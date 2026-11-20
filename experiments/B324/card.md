# B324 — replay lưu đầu vào và cấu trúc cập nhật hai tần số

**Đăng ký trước execution ngày 16/11/2026.** Hai thay đổi protocol được chốt trong card này: lưu đầu vào
encoder dạng uint8 làm định dạng replay của baseline mới, và mở khóa seed để chạy seed 0 và seed 1. Thiết kế
được chọn sau khi đã xem kết quả B319–B323 trên MVTec test, nên mọi kết quả là exploratory. Không VisA.
Config: [configs/b324.json](../../configs/b324.json).

| Job | Câu hỏi | Nhãn | Kết quả |
| --- | --- | --- | --- |
| Q0 | `q0` lưu uint8 có cho đúng quỹ đạo B319 ER không? | `IDENTITY_REPRODUCED` | [q0_report.json](runs/b324-q0-20261116-01/q0_report.json) |
| S1 | Lưu đầu vào ở cùng byte budget có xóa được quên macro không? | `COVERAGE_SUPPORTED` | [b324-s1-20261116-01](runs/b324-s1-20261116-01/results.md) |
| S2 (a, b) | Hai tần số có hơn cập nhật phẳng không? | **`STRUCTURE_ADDS_NOTHING`** | [phân tích S2](analyses/S2-structure/README.md) |
| R1 | S1 có lặp lại ở seed 1, cùng job không? | `REPLICATED` | [b324-r1-20261116-01](runs/b324-r1-20261116-01/results.md) |

## 1. Hai câu hỏi, tách riêng

- **S1 (baseline, không phải NL):** ở cùng byte budget của B319, lưu đầu vào encoder (uint8) thay cho
  feature có xóa được quên macro như `er_bytes` của B323 (82 feature, 398 MB) đã làm không?
- **S2 (NL):** với cùng bộ nhớ đó và cùng tổng số cập nhật decoder, tiêu ngân sách theo hai tần số (nhanh
  mỗi arrival + đợt consolidation định kỳ chỉ trên bộ nhớ dài hạn) có hơn tiêu phẳng (k cập nhật mỗi arrival)
  không?

Prior viết trước: S1 nhiều khả năng đạt; S2 nhiều khả năng ra "cấu trúc không thêm gì".

## 2. Ràng buộc chốt trước

- **Byte budget replay:** 155.123.712 byte, bằng B319 (32 × 4.847.616). Mỗi mẫu lưu đúng tensor uint8
  3×392×392 sau `resize((392, 392), BILINEAR)` của `encoder.preprocess`: 460.992 byte. Vậy 336 mẫu
  (154.893.312 byte): **reservoir 328 + FIFO 8**.
- **Chỉ lưu không mất mát.** Không JPEG, không giảm độ phân giải, không lượng tử hóa hay nén feature.
- **Không cache feature của bộ nhớ** giữa các lần cập nhật: mỗi lần dùng, mẫu bộ nhớ đi lại qua encoder đóng
  băng (no-grad). Feature của mẫu hiện tại được dùng lại trong một arrival.
- **Category-blind:** không arm nào dùng ranh giới task lúc train.
- Phần còn lại giữ recipe B319 ([configs/b319.json](../../configs/b319.json)): thứ tự stream, optimizer, LR,
  hard-mining, clip gradient, scorer, full test 1.725 ảnh, 15 boundary. k, C, K mỗi thứ một giá trị, không sweep.

## 3. Arms

| Arm | Bộ nhớ | Cập nhật decoder | Tổng cập nhật |
| --- | --- | --- | ---: |
| `q0` | 24 + 8, lưu uint8 | 1 mỗi arrival | 3.629 |
| `raw1` | 328 + 8, lưu uint8 | 1 mỗi arrival | 3.629 |
| `flat` | 328 + 8, lưu uint8 | 3 mỗi arrival | 10.887 |
| `nested` | 328 + 8, lưu uint8 | 1 mỗi arrival + đợt consolidation | 10.887 |
| `iid` | 328 + 8, lưu uint8 | 3 mỗi arrival, stream đã xáo | 10.887 |

- **Bước nhanh** (mọi arm): như B319, mẫu hiện tại (hệ số 0,5) và tối đa 4 support (mỗi mẫu 0,5 / số support).
- **`flat`:** sau bước nhanh là 2 cập nhật thêm cùng cấu trúc, 4 mẫu rút đều không hoàn lại từ reservoir bằng
  luồng RNG `consolidation`.
- **`nested`:** bước nhanh mỗi arrival; khi `arrival % 64 == 0` chạy một đợt **K = 128** cập nhật, mỗi cập nhật 5
  mẫu rút đều từ reservoir, hệ số 1/5, không có mẫu hiện tại. Sau arrival cuối chạy thêm một đợt 90 cập nhật để
  tổng bằng `flat`: 3.629 + 56 × 128 + 90 = 10.887.
- **Thứ tự trong một arrival:** bước nhanh → cập nhật thêm hoặc đợt consolidation → admission → observer nếu là
  boundary.
- **Luồng RNG `consolidation`:** luồng PCG thứ năm, độc lập; bốn luồng sẵn có không bị đụng tới, nên `raw1`,
  `flat`, `nested` cùng seed có cùng quỹ đạo nội dung bộ nhớ.
- **`iid`** (tham chiếu, không vào gate): như `flat` nhưng 3.629 arrival được xáo một lần; một observer ở cuối.
  Không phải Dinomaly joint training.
- **Seed** đổi khởi tạo decoder, các luồng sampling, dropout và hoán vị `iid`; thứ tự stream không đổi theo seed.

## 4. Thay đổi guard của trainer (chỉ cho các arm B324)

Store là uint8 `(3, 392, 392)` thay cho float32 `(2, 789, 768)`, giới hạn capacity + 8 phần tử và tổng byte ≤
155.123.712; bộ đếm `replay_encoder` riêng có trần theo arm; trần `support_forwards` theo arm; runner dừng nếu số
optimizer step cuối khác bảng ở mục 3; observer kiểm store không đổi (hash). Recipe B319 không có các khóa mới và
giữ nguyên mọi guard.

## 5. Kiểm tra tính đúng

1. **Q0, identity:** `q0` (seed 0) phải cho đúng model hash B319 ER tại boundary 209 và 433 (`84c3e401…bfd8`,
   `2d4401de…62fc`). Job Q0 dừng ở arrival 433; GPU thứ hai chạy `nested` tới 433 để kiểm đường code và đo thời
   gian. Kết quả: `IDENTITY_REPRODUCED`, hai hash trùng, số bước đúng, replay trong budget.
2. **Tests CPU trong job, trước GPU:** round-trip uint8 → tensor chuẩn hóa trùng `encoder.preprocess`; byte bộ nhớ
   không vượt budget; lịch `flat`/`nested` đúng 10.887 cập nhật; luồng `consolidation` không đổi bốn luồng kia;
   hoán vị `iid` tất định.
3. **Cùng quỹ đạo bộ nhớ:** mỗi event ghi `memory_sha256`; chuỗi hash của `flat` và `nested` cùng seed phải trùng
   ở mọi arrival.
4. Gate pytest: ở Q0 gate chạy sau các bước GPU và không đạt vì lỗi notebook (thư mục cha của `--basetemp` không
   có trong bundle; 12 test đạt, 6 test dùng `tmp_path` lỗi ở setup). Từ S1, gate chạy ở đầu mỗi job, trước dữ
   liệu và GPU; lỗi là dừng.

Implementation: [coverage.py](../../src/nested_replay_cad/coverage.py), hook trong
[trainer.py](../../src/nested_replay_cad/trainer.py) (chỉ bật khi config có `replay_format`), launcher
[coverage_pilot.py](../../scripts/coverage_pilot.py), báo cáo [coverage_report.py](../../scripts/coverage_report.py),
notebook [b324.ipynb](../../notebooks/b324.ipynb), tests [test_coverage_pilot.py](../../tests/test_coverage_pilot.py).

## 6. Quy tắc đọc kết quả (chốt trước)

- **S1** (`raw1` seed 0 so với ER 32 của B319, khác job, mô tả): macro Facq ≤ 0,010 **và** macro Final Pixel AP ≥
  0,620 → `COVERAGE_SUPPORTED`; ngược lại `COVERAGE_NOT_SUPPORTED` và S2 không chạy.
- **S2** (`nested` − `flat`, cùng job, seed 0 và 1): `STRUCTURE_SIGNAL` khi ở cả hai seed macro Final Pixel AP ≥
  +0,005, không category nào < −0,010, macro Image AUROC ≥ −0,005; `STRUCTURE_ADDS_NOTHING` khi |trung bình hai seed|
  < 0,005; còn lại `MIXED`.
- `flat`, `nested`, `iid` tốn khoảng ba lần số cập nhật của ER 32; mọi so sánh với ER 32 hay `raw1` phải nêu điều đó.

## 7. Các job

| Job | GPU 0 | GPU 1 | Seed |
| --- | --- | --- | --- |
| Q0 | `q0` tới 433 | `nested` tới 433 | 0 |
| S1 | `raw1` | `iid` | 0 |
| S2a | `flat` | `nested` | 0 |
| S2b | `flat` | `nested` | 1 |
| R1 | `q0` đủ 3.629 arrival | `raw1` | 1 |

Mỗi job một attempt; không retry, sweep hay early stop theo chất lượng AD. Cap `per_arm_seconds` = 18.000 s, đặt sau
Q0 (tới arrival 433: `q0` 555 s, `nested` 1.254 s với 768 bước consolidation, ≈ 0,91 s mỗi bước thêm).

### 7b. Job R1 (thêm 16/11/2026, sau S1 và S2)

Kết quả dương chính (`raw1` hơn ER 32) mới có một seed và so khác job. R1 chạy seed 1 với `q0` đủ 3.629 arrival
(tương đương ER 32 nhưng lưu uint8) và `raw1` trong cùng job. Không đổi thiết kế, gate S1/S2 hay cap. Quy tắc
(chốt trước khi chạy): `REPLICATED` khi `raw1 − q0` macro Final Pixel AP ≥ +0,005 **và** macro Facq của `raw1` ≤
0,010; còn lại `NOT_REPLICATED`. Screen lịch sử theo category báo kèm.

## 8. Kết luận

- **S1 và R1:** lưu 336 ảnh uint8 thay cho 32 feature, cùng ngân sách 155 MB và cùng số cập nhật, nâng Pixel AP
  cuối +0,0379 (seed 0, khác job) và +0,0254 (seed 1, cùng job), quên macro về khoảng 0. Screen lịch sử theo
  category không đạt nguyên văn ở cả hai seed (seed 0: Image AUROC transistor; seed 1: Pixel AP carpet, tile,
  toothbrush và Image AUROC toothbrush).
- **S2:** cấu trúc hai tần số không hơn cập nhật phẳng (`nested − flat` = −0,0001 và −0,0034).
- Cải thiện có thật trong chuỗi B319–B324 đến từ định dạng lưu bộ nhớ, không từ cấu trúc lồng nhau. Kết luận
  không nói gì về setting không replay.
- **Lưu ý cho việc dùng lại `coverage_report.py`:** nên yêu cầu hai seed khác nhau, đủ 3.629 memory hash mỗi cặp
  và trajectory khớp trước khi phát nhãn, kèm test âm cho seed lặp, ledger rỗng hoặc thiếu, và hash không khớp.

## 9. Ngoài phạm vi

Không VisA: holdout chỉ dùng khi MVTec đã ổn, với cấu hình đóng băng. Không nén có mất mát. Không sweep k, C, K
hay kích thước bộ nhớ. CMS bên trong decoder để dành cho card sau nếu S2 có tín hiệu.
