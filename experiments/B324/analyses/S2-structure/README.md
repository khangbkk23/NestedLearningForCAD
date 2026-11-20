# B324 S2 — cấu trúc hai tần số so với phẳng (hai seed)

Exploratory, MVTec test đã lộ. Nhãn theo card mục 6: **`STRUCTURE_ADDS_NOTHING`**. Trung bình hai seed của
`nested − flat` về macro Final Pixel AP là −0,0017 (ngưỡng |·| < 0,005). Đây cũng là kết quả đã dự đoán trước khi
chạy ([S1](../../runs/b324-s1-20261116-01/results.md), khoảng 60%). Số đầy đủ: [s2_report.json](s2_report.json); bản
ghi chạy: [S2a seed 0](../../runs/b324-s2a-20261116-01/publication.json), [S2b seed 1](../../runs/b324-s2b-20261116-01/publication.json).

## Kiểm tính đúng

- Cả hai job hoàn tất, gate pytest đầu job đạt, mỗi arm exit 0, 3.629 arrival, 15 boundary.
- `flat` và `nested` đều đúng 10.887 optimizer step (7.258 bước thêm) ở cả hai seed.
- Chuỗi `memory_sha256` của `flat` và `nested` trùng ở mọi arrival trong từng seed: cùng nội dung bộ nhớ, chỉ khác
  lịch cập nhật.
- Bộ nhớ replay 154,4 MB, dưới budget 155,1 MB.

## Kết quả

| Seed | Arm | Final Pixel AP | Image AUROC | Acq | Fmax | Facq | Thời gian arm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `flat` | 0,6460 | 0,9884 | 0,6378 | 0,0117 | −0,0088 | 14.624 s |
| 0 | `nested` | 0,6459 | 0,9859 | 0,6266 | 0,0143 | −0,0207 | 14.477 s |
| 1 | `flat` | 0,6475 | 0,9893 | 0,6411 | 0,0165 | −0,0068 | 13.772 s |
| 1 | `nested` | 0,6441 | 0,9878 | 0,6370 | 0,0121 | −0,0077 | 15.012 s |

`nested − flat`: seed 0 là −0,0001, seed 1 là −0,0034. Không seed nào đạt quy tắc tín hiệu; mỗi seed có 4–5
category dưới sàn −0,010, và Image AUROC của `nested` thấp hơn 0,002–0,003.

## Đọc kết quả

1. **Cấu trúc hai tần số không thêm gì đo được.** Với cùng bộ nhớ, cùng nội dung bộ nhớ và cùng số cập nhật, gom cập
   nhật thêm thành đợt consolidation định kỳ cho Final ngang hoặc hơi thấp hơn rải đều mỗi arrival. Hai arm khác
   nhau cả ở lịch cập nhật lẫn thành phần batch: cập nhật chậm của `nested` chỉ lấy replay, còn `flat` trộn mẫu hiện
   tại với replay. `nested` có Acquisition thấp hơn (−0,011 và −0,004), đúng như card đã lường: boundary rơi giữa hai
   đợt.
2. **Chênh lệch theo category là nhiễu nhiều hơn hiệu ứng.** Dấu đổi giữa hai seed ở nhiều category (tile +0,012 rồi
   −0,049; cable −0,009 rồi −0,051; carpet −0,011 rồi +0,008).
3. **Ngân sách cập nhật gấp ba mua khoảng 1 điểm.** `flat` seed 0 (0,6460) hơn `raw1` seed 0 (0,6362) +0,0098, với chi
   phí hơn gấp đôi thời gian arm.
4. **`flat` so với `iid` (mô tả):** `flat` trên stream thật (0,6460) hơn `iid` trên stream đã xáo (0,6426) +0,0034, cùng
   seed và ngân sách; hai seed không đủ để ước lượng nhiễu, nên không rút ra kết luận về tính không dừng của stream.

## Kết luận trong phạm vi

Trên recipe này (encoder DINOv2 đóng băng, decoder Dinomaly, MVTec 15 task, bộ nhớ replay 155 MB lưu đầu vào uint8),
sau khi bộ nhớ đủ phủ:

- Facq macro quan sát được ≤ 0 ở các arm dùng bộ nhớ 336 mẫu (`raw1`, `flat`, `nested`, `iid`); không áp dụng cho
  `q0` ở R1 (+0,0280);
- cấu trúc cập nhật hai tần số không hơn cập nhật phẳng.

Cùng với B319/B321 (trọng số replay học được), B322 (EMA thụ động) và B323 (neo hàm số), đây là kết quả null thứ tư
cho một cơ chế kiểu Nested Learning trên bài toán replay này, và là lần đầu có hai seed với đối chứng khớp cả byte lẫn
số cập nhật. Phần cải thiện có thật trong chuỗi thí nghiệm đến từ định dạng lưu bộ nhớ, không từ cấu trúc lồng nhau.

## Giới hạn

Hai seed, cùng thứ tự task; một giá trị cho mỗi k, C, K. So sánh `raw1` với ER 32 vẫn một seed và khác job; `flat` so
với `raw1` và `iid` cũng khác job. MVTec test đã định hướng thiết kế từ B298 tới B324; chưa có xác nhận trên VisA.
Kết luận không nói gì về setting không replay.
