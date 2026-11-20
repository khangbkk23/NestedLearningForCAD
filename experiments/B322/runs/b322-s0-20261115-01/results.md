# B322 seed 0 — kết quả (b322-s0-20261115-01)

Exploratory, một seed, MVTec test đã lộ. Verdict theo card mục 6: **`NO_SIGNAL`** (primary τ = 512).
Secondary τ = 128: `NO_SIGNAL`. Screen lịch sử: `SCREEN_NOT_MET` (vi phạm sàn: tile, transistor, zipper).

Fast readout tái lập B319 ER đã ghi: 15/15 model hash ở boundary, 480 ô ma trận lệch tối đa 0.

| Readout | Final Pixel AP | Final Image AUROC | Acq Pixel AP | Fmax | Facq |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast | 0,5983 | 0,9676 | 0,6256 | 0,0457 | 0,0292 |
| ema512 | 0,6021 | 0,9764 | 0,4384 | 0,0019 | −0,1754 |
| ema128 | 0,6016 | 0,9700 | 0,6130 | 0,0372 | 0,0122 |

Primary, shadow − fast: Final Pixel AP +0,0038 (cần ≥ +0,005: không đạt); Fmax giảm 0,0438 (đạt);
Acq macro −0,1872, category tệ nhất −0,4816 (không đạt); Image AUROC +0,0088 (đạt).

Lưu ý đọc: Fmax của τ = 512 thấp một phần vì shadow trễ lúc acquisition (Facq âm: AP của shadow còn tăng
sau khi task kết thúc), nên mức giảm Fmax không thuần là giữ kiến thức tốt hơn.

Số đầy đủ theo category và boundary: [metrics.json](metrics.json). Bản ghi chạy:
[publication.json](publication.json).
