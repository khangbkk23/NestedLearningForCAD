# B324 S1 seed 0 — kết quả và cách đọc (b324-s1-20261116-01)

Exploratory, một seed, MVTec test đã lộ. Nhãn theo card mục 6: **`COVERAGE_SUPPORTED`** (dự đoán viết trước:
macro Facq ≤ 0,010 và macro Final Pixel AP ≥ 0,620; quan sát −0,0085 và 0,6362). Gate pytest đầu job đạt; số bước
khớp lịch (`raw1` 3.629, `iid` 10.887). Số đầy đủ: [metrics.json](metrics.json); bản ghi chạy:
[publication.json](publication.json).

| | Bộ nhớ replay | Cập nhật | Final Pixel AP | Image AUROC | Acq | Fmax | Facq |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ER 32 feature (B319) | 155 MB | 3.629 | 0,5983 | 0,9676 | 0,6256 | 0,0457 | 0,0292 |
| `er_bytes` 82 feature (B323) | 398 MB | 3.629 | 0,6273 | 0,9837 | 0,6264 | 0,0232 | −0,0010 |
| `raw1` 336 uint8 | 154 MB | 3.629 | **0,6362** | 0,9829 | 0,6282 | 0,0108 | −0,0085 |
| `iid` (stream xáo) | 154 MB | 10.887 | 0,6426 | 0,9904 | — | — | — |

## Đọc kết quả

1. **Ở gần cùng byte budget, đổi định dạng lưu đi cùng Facq macro ≈ 0.** `raw1` hơn ER 32 +0,0379 Pixel AP và
   +0,0153 Image AUROC, cùng ngân sách 155 MB và cùng số cập nhật. Pixel AP không giảm quá 0,0095 ở category nào
   (12/15 tăng). So khác job (với B319) nên chỉ là mô tả. Facq macro gần 0 vẫn có thể đi cùng suy giảm ở từng
   category.
2. **Chưa đạt nguyên văn screen lịch sử:** Image AUROC transistor giảm 0,0108 (0,976 → 0,965), vượt sàn 0,005.
3. **`raw1` ngang `er_bytes` với ít hơn 2,6 lần byte** (+0,0089 macro, theo category từ −0,034 đến +0,052); hai chuỗi
   mẫu khác nhau, một seed.
4. **Quên còn lại tập trung và nhỏ:** cable +0,065, screw +0,058, transistor +0,044. Toothbrush (−0,125), tile, carpet
   tăng sau acquisition.
5. **`iid` so với `raw1` (mô tả):** `iid` dùng gấp ba số cập nhật trên stream đã xáo và hơn `raw1` 0,0064 (9/15
   category tăng). So sánh đổi cùng lúc thứ tự stream và số cập nhật nên không tách được hai yếu tố, không đọc
   thành "trần".

## Hệ quả cho S2 (viết trước khi chạy S2)

S2 so `nested` với `flat`, cả hai 10.887 cập nhật. Tham chiếu `iid` cho thấy ngân sách đó mua được khoảng +0,006
trên stream lý tưởng, nên `STRUCTURE_SIGNAL` (≥ +0,005 ở cả hai seed) khó xảy ra. Ước lượng trước khi chạy:
`STRUCTURE_ADDS_NOTHING` khoảng 60%, `MIXED` khoảng 30%, `STRUCTURE_SIGNAL` khoảng 10%. S2 vẫn chạy đúng như đăng ký
vì đó là phép thử duy nhất của thành phần NL trong B324. Điểm yếu lớn hơn là S1 mới có một seed và so khác job;
job R1 được thêm để có so sánh cùng job ở seed thứ hai.
