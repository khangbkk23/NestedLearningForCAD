# B321 — grouped replay khớp số lượt cập nhật

**Đăng ký trước execution ngày 13/11/2026, sau khi đã thấy kết quả B319 trên MVTec.**
Exploratory, không holdout, không confirmatory. Hai arm: ER uniform và primary tail+CTX,
seed 0, biến thể grouped. Config: [configs/b321.json](../../configs/b321.json).
Kết quả cuối (lần 5): **`SCREEN_NOT_MET`** ([kết quả](runs/b321-s0-20261114-05/results.md)).

## Câu hỏi

Cùng số update và cùng số lượt dùng mẫu, tail+CTX có trọng số học được và chia nhóm có hơn
ER uniform không? Không claim tái lập B319 hay parity chính xác với code của Khang hoặc paper.

## Thiết kế

- 667 update/task × 15 = 10.005 update/arm; batch 16 = 160.080 lượt dùng mẫu. Stage 1: 16
  current; từ stage 2: 8 current + 8 replay phân biệt. Loss = 0,5·mean(8 current) + 0,25·nhóm
  bốn A có trọng số + 0,25·nhóm bốn B có trọng số. 97 phi dùng chung; CTX riêng cho bốn support.
- Admission: lần dùng current thành công đầu tiên đưa ảnh vào reservoir 24 / FIFO 8; dùng lại
  không làm mới. Tuổi theo serial admission; query gồm 2 FIFO gần nhất + 2 lịch sử không trùng;
  thiếu query thì uniform; thiếu 8 support thì dừng. Mỗi epoch current là một hoán vị
  `drop_last`; RNG dữ liệu độc lập; cách chia nhóm bốn ghép cặp giữa hai arm.
- FP32, micro-batch 4 nhưng loss và hard-mining theo từng ảnh; LR hằng 0,002, mang moment fast
  qua các task; phi và slow state mang qua task; hard-mining cập nhật toàn cục mỗi 26; slow step
  mỗi 16 batch đủ điều kiện, KL 0,001·mean 2 nhóm, hệ số alignment 0,25. Cách ghép batch mới
  phải qua control riêng; không thừa hưởng PASS của B319. Khác biệt với B319 ở
  [method](../../docs/method.md#b321-khác-b319-ở-đâu).
- **So với Khang:** RH01/RH01b/NLCAD (source `9b2fc7c`) cũng dùng 667 × 15, batch 16 nhưng BF16
  micro 8, optimizer/scheduler/hard-mining ramp theo task và replay task đã xong khác. Chỉ khớp
  số đếm, không khớp compute, bộ nhớ, split hay recipe.
- Đánh giá sau 667·t update trên full test các task 1..t; cuối là 1.725 ảnh; 120 ô quan sát/arm,
  bốn ma trận 15 × 15 (ô chưa học là null).
- **Screen exploratory chính:** macro Pixel AP gain ≥ +0,005 so ER, mỗi category Pixel AP delta
  ≥ −0,01 và Image AUROC delta ≥ −0,005. Báo cáo Fmax/Facq có dấu trên 14 task có lịch sử.
  Không early stop, chọn, tune hay sweep theo chất lượng AD. Mọi failure, cancel và partial
  đều được giữ.
- **Tài nguyên:** một job, một attempt, cả job ≤ 43.200 s gồm setup, preflight, cache, checkpoint
  và export; RSS nhóm 8 GiB, VRAM cấp phát 12 GiB, payload 18 GiB. Preflight đo thời gian thật
  và dự trù cộng 20% margin + 1.800 s export phải vừa cap còn lại; không vừa thì
  `PREFLIGHT_INSUFFICIENT_CAP` và dừng trước training. Không bao giờ giảm count, cohort hay
  precision để lọt gate. Không VisA.

Thay đổi context, grouping và admission là thay đổi phương pháp, không chỉ là thêm budget;
một seed ghép cặp không tách được nguyên nhân của khác biệt với điểm lịch sử.

## Các lần chạy

| Run | Kết quả | Ghi chú |
| --- | --- | --- |
| [b321-s0-20261113-01-v1](runs/b321-s0-20261113-01-v1/results.md) | `PREFLIGHT_INSUFFICIENT_CAP` | Chạy tuần tự hai arm cần ≈ 17,1 h, cap còn 11,5 h; không train |
| [b321-s0-20261114-02-v1](runs/b321-s0-20261114-02-v1/results.md) | `PREFLIGHT_INSUFFICIENT_MEMORY` | Hai arm song song trên hai T4; dự trù RSS 10,2 GiB > cap 8 GiB; không train |
| `b321-s0-20261114-03` | Lỗi kỹ thuật, không có kết quả | Cap RAM nhóm 16 GiB / 8 GiB mỗi worker. `FileNotFoundError` trong census đĩa khi dọn score archive (race giữa `is_file` và `stat`); chưa có checkpoint nên không resume được |
| `b321-resume-qual-20261114-01` | Lỗi đóng gói | Lỗi dependency của notebook, trước mọi phép kiểm |
| [b321-resume-qual-20261114-02-v4](runs/b321-resume-qual-20261114-02-v4/results.md) | `PASS_SCOPED` | Qualification checkpoint/resume CUDA trên dữ liệu tổng hợp |
| `b321-s0-20261114-04` | `BLOCKED_RESOURCE_PREFLIGHT` | Chạy lại từ đầu sau qualification; preflight chặn, không train |
| [b321-s0-20261114-05](runs/b321-s0-20261114-05/results.md) | **`SCREEN_NOT_MET`** | Hai arm hoàn tất; chạy theo phân đoạn có checkpoint |

Các thay đổi giữa các lần chỉ thuộc về thực thi (topology hai GPU, kế toán tài nguyên, census
chịu race, checkpoint nguyên tử mỗi 500 update và ở update 10.005, chạy theo phân đoạn với
điểm dừng ở START + 11 h). Recipe khoa học, seed, count, đánh giá, metric và sàn giữ nguyên
qua mọi lần. Lần 3 và 4 không có thư mục run vì không tạo ra kết quả khoa học; nguyên nhân
được ghi ở bảng trên.

## Kết quả và phân tích

- Lần 5: ER 55,5500 % so với tail+CTX 55,4776 % Pixel AP cuối (−0,0725 pp; cần +0,5 pp); năm
  category vi phạm sàn. **`SCREEN_NOT_MET`.** Không hạ tiêu chí nào.
- [Phân tích A5](analyses/a5-mechanism-audit/results.md): controller có học, có đổi trọng số
  support và đổi quỹ đạo, nhưng trọng số có độ tương phản thấp (0,224–0,269 quanh 0,25). Không
  phải controller chết. Mắt xích chưa chứng minh là liệu trọng số và objective này có tạo ra cải
  thiện AD hay không.

Thêm budget update không mang lại lợi ích learned replay đã khai báo ở seed này. Đây là bằng
chứng chống adoption tail+CTX với recipe này, không phải bằng chứng cả phương pháp không thể
hoạt động hay implementation sai. Không chạy lại hoặc tune trên cùng test đã lộ.
