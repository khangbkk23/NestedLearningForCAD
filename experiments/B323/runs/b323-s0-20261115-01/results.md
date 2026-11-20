# B323 seed 0 — kết quả (b323-s0-20261115-01)

Exploratory, một seed, MVTec test đã lộ. Verdict theo card mục 8: **`NO_SIGNAL`** (`anchor` − `er_bytes`).
Prefix: `PREFIX_REPRODUCED` (hash `anchor` tại arrival 209 bằng hash B319 ER). 14 lần consolidation,
12.673 lượt forward neo.

| Arm | Final Pixel AP | Final Image AUROC | Acq Pixel AP | Fmax | Facq |
| --- | ---: | ---: | ---: | ---: | ---: |
| er_bytes (82 mẫu) | 0,6273 | 0,9837 | 0,6264 | 0,0232 | −0,0010 |
| anchor (32 mẫu + mức chậm) | 0,5952 | 0,9591 | 0,6205 | 0,0497 | 0,0271 |
| ER 32 mẫu (B319, tham chiếu) | 0,5983 | 0,9676 | 0,6256 | 0,0457 | 0,0292 |

Gate, `anchor` − `er_bytes`: Final Pixel AP −0,0321 (cần ≥ +0,005: không đạt); category tệ nhất −0,2309 (cần
≥ −0,01: không đạt, 9 category vi phạm); Acq macro −0,0059 (đạt); Image AUROC −0,0246 (cần ≥ −0,005: không đạt).

Tham chiếu, không vào gate: `anchor` − ER 32 = −0,0032 Pixel AP; `er_bytes` − ER 32 = +0,0289 Pixel AP, +0,0162
Image AUROC, Fmax giảm 0,0225 (category tệ nhất −0,0241). So sánh với ER 32 không phải so sánh đã đăng ký; chuỗi
mẫu của `er_bytes` khác ER 32.

| Category | er_bytes | anchor | anchor − er_bytes |
| --- | ---: | ---: | ---: |
| bottle | 0,853 | 0,879 | +0,026 |
| cable | 0,546 | 0,467 | −0,079 |
| capsule | 0,526 | 0,550 | +0,024 |
| carpet | 0,700 | 0,686 | −0,015 |
| grid | 0,406 | 0,392 | −0,014 |
| hazelnut | 0,756 | 0,718 | −0,038 |
| leather | 0,511 | 0,549 | +0,038 |
| metal_nut | 0,750 | 0,765 | +0,015 |
| pill | 0,733 | 0,690 | −0,044 |
| screw | 0,433 | 0,202 | −0,231 |
| tile | 0,826 | 0,822 | −0,004 |
| toothbrush | 0,501 | 0,443 | −0,059 |
| transistor | 0,475 | 0,497 | +0,023 |
| wood | 0,711 | 0,633 | −0,078 |
| zipper | 0,681 | 0,636 | −0,045 |

Compute: byte-matched, không compute-matched. Arm `anchor` mất 7.937 s, `er_bytes` 6.038 s. Khoảng cách neo trung
bình 0,008–0,012; loss replay của `anchor` cao hơn `er_bytes` ở cuối chuỗi (0,083 so với 0,068).

## Đọc kết quả

- Neo hàm số về bản sao theo boundary **không** giúp ở recipe này: thấp hơn đối chứng cùng byte 0,032 Pixel AP và
  ngang ER 32 (−0,003). Dùng chính số byte đó để lưu thêm 50 mẫu replay tốt hơn hẳn.
- Kết quả dương phụ (`er_bytes` hơn ER 32 +0,029) là gợi ý rằng độ phủ của bộ nhớ là đòn bẩy chính; B324 kiểm
  điều đó.

Số đầy đủ theo boundary, category và tuổi task: [metrics.json](metrics.json). Bản ghi chạy:
[publication.json](publication.json).
