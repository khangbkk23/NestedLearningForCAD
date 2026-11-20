# B324 R1 seed 1 — kết quả và cách đọc (b324-r1-20261116-01)

Exploratory, MVTec test đã lộ. Job được thêm sau khi đã thấy S1 và S2 (card mục 7b); quy tắc đọc chốt trước khi
chạy. Nhãn: **`REPLICATED`** (`raw1 − q0` macro Final Pixel AP = +0,0254, cần ≥ +0,005; macro Facq của `raw1` =
−0,0008, cần ≤ 0,010). Số đầy đủ: [metrics.json](metrics.json); bản ghi chạy: [publication.json](publication.json).

Kiểm tra thực thi: gate pytest đầu job đạt (18 test); hai arm cùng seed 1, cùng hash khởi tạo, mỗi arm 3.629 arrival,
3.629 bước optimizer, 15 boundary, không bước thêm; bộ nhớ replay `q0` 14,8 MB (32 mẫu uint8), `raw1` 154,4 MB (sức
chứa 336; cuối chuỗi lưu 335 tensor vì một mẫu nằm ở cả reservoir và FIFO nên chỉ lưu một lần); log không có
traceback. Cả job 7.566 s (`q0` 7.295 s, `raw1` 7.074 s).

| Seed 1, cùng job | Mẫu lưu | Final Pixel AP | Image AUROC | Acq | Fmax | Facq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q0` (ER 32, lưu uint8) | 32 | 0,6039 | 0,9759 | 0,6300 | 0,0524 | 0,0280 |
| `raw1` | 336 | **0,6293** | 0,9861 | 0,6285 | 0,0193 | −0,0008 |
| `raw1 − q0` | | +0,0254 | +0,0102 | −0,0015 | −0,0331 | −0,0288 |

| | Baseline ER 32 | `raw1` | Gain Pixel AP | Facq `raw1` | Kiểu so sánh |
| --- | ---: | ---: | ---: | ---: | --- |
| Seed 0 (S1) | 0,5983 | 0,6362 | +0,0379 | −0,0085 | khác job (ER feature của B319) |
| Seed 1 (R1) | 0,6039 | 0,6293 | +0,0254 | −0,0008 | cùng job |

## Đọc kết quả

1. **Kết quả dương chính lặp lại ở seed thứ hai, lần này so cùng job.** Trung bình hai seed khoảng +0,032; hai seed
   là ít, không gắn khoảng tin cậy.
2. **Gain đến từ giảm quên, không từ học tốt hơn.** Acquisition macro gần như bằng nhau (−0,0015). Category bị quên
   nặng ở `q0` được giữ lại: screw +0,139, cable +0,100, pill +0,050, metal_nut +0,048, grid +0,043.
3. **Screen lịch sử nguyên văn không đạt ở seed này.** Pixel AP giảm quá 0,01: toothbrush −0,045, tile −0,036, carpet
   −0,033; toothbrush còn giảm Image AUROC 0,019 (0,983 → 0,964). 11/15 category tăng.
4. **Bản chất ba vi phạm:** đều là category mà ER 32 **tăng** sau acquisition (carpet 0,683 → 0,700; tile 0,757 →
   0,837; toothbrush 0,404 → 0,505). `raw1` giữ mô hình gần trạng thái lúc học hơn nên nhận ít phần tăng đó hơn; với
   carpet, tile còn thấp hơn ngay lúc acquisition khoảng 0,023. Mô tả từ một seed, chưa phải cơ chế đã kiểm.
5. **Giới hạn so sánh:** `q0` dùng 14,8 MB còn `raw1` 154,4 MB: R1 đo "nhiều mẫu hơn nhờ lưu uint8" ở cùng ngân sách
   byte với ER 32 lưu feature (155 MB), không phải cùng số byte thực dùng trong job. Tương đương giữa `q0` và ER
   feature đã kiểm ở seed 0 tới arrival 433; ở seed 1 và chuỗi đầy đủ là suy ra, không đo.

## Hệ quả

Cấu hình `raw1` (336 mẫu uint8, 1 cập nhật mỗi arrival) là ứng viên baseline replay cho MVTec, với ghi chú trung
thực: tăng macro ở hai seed, không đạt sàn theo category của screen lịch sử. Chưa có gì ở đây là thành phần Nested
Learning; S2 đã cho `STRUCTURE_ADDS_NOTHING` trên chính nền này.
