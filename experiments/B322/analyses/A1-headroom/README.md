# B322-A1 — headroom và chất lượng teacher (hậu kiểm)

**Exploratory, post-hoc, một seed, MVTec test đã lộ.** Viết sau khi đã thấy verdict `NO_SIGNAL` của
[run seed 0](../../runs/b322-s0-20261115-01/results.md); không đổi verdict hay ngưỡng nào. Không train,
không GPU, không đọc score tensor: chỉ đọc `metrics.json`. Script:
[shadow_headroom.py](../../../../scripts/shadow_headroom.py); số đầy đủ: [result.json](result.json).

## Câu hỏi

1. Một cơ chế chống quên trên recipe này tối đa cải thiện được bao nhiêu Pixel AP cuối?
2. Trọng số chậm (EMA) có giữ kiến thức task cũ tốt hơn trọng số nhanh không, tức có đáng làm teacher
   cho vòng kín không?

## Kết quả

**Headroom của fast (ER uniform):**

| Đại lượng | Macro Pixel AP |
| --- | ---: |
| Final | 0,5983 |
| Acquisition (15 category) | 0,6256 |
| Acquisition − Final | +0,0273 |
| Đỉnh theo từng category qua các boundary | 0,6417 (+0,0433) |

- +0,0273 là mức một teacher "đông cứng lúc vừa học xong" có thể mang lại nếu giữ được hoàn toàn;
  +0,0433 là trần lạc quan (max theo từng category trên một seed).
- Quên tập trung: screw (0,133), cable (0,132), grid (0,110) chiếm 62% tổng phần mất dương.
- **Backward transfer có thật:** tile +0,069, toothbrush +0,045, carpet +0,034, metal_nut +0,021, leather
  +0,016 tăng *sau* acquisition. Một cơ chế neo cứng về lúc acquisition sẽ làm mất phần này và vi phạm sàn
  −0,01 mỗi category. Sàn đó là ràng buộc chặt nhất cho thiết kế tiếp theo.

**EMA có phải teacher tốt hơn trên task cũ không** (shadow − fast, gộp mọi boundary, chỉ task đã học trước):

| Readout | tuổi ≥ 1 boundary | tuổi ≥ 2 | tuổi ≥ 3 |
| --- | ---: | ---: | ---: |
| EMA τ = 512 | −0,0141 | −0,0054 | −0,0007 |
| EMA τ = 128 | +0,0026 | +0,0024 | +0,0031 |

- EMA τ = 512 không tốt hơn fast trên task cũ ở bất kỳ tuổi nào; τ = 128 hơn khoảng 0,003, cỡ nhiễu của
  một seed. EMA của một quỹ đạo đang quên cũng quên theo, chỉ trễ hơn.
- Chọn readout tốt nhất theo từng category ở cuối (cần biết task) cũng chỉ đạt 0,6128 (+0,0144).

## Kết luận trong phạm vi

- Nhánh shadow thụ động đóng ở `NO_SIGNAL`. Không replication thêm seed (card chỉ mở việc đó khi `SIGNAL`).
- Tiền đề "mức chậm EMA là teacher tốt hơn" **không được hỗ trợ** ở recipe này; vòng kín dùng teacher EMA
  bị giảm ưu tiên. Thụ động khác vòng kín, nên đây không phải bác bỏ.
- Teacher có chất lượng đọc được từ ma trận đã lưu là snapshot của fast tại boundary; đó là căn cứ cho
  [B323](../../../B323/card.md).
- Trần cải thiện của mọi cơ chế chỉ chống quên trên recipe này nhỏ (+0,03 đến +0,04).

## Giới hạn

Một seed, không có khoảng tin cậy; chênh lệch giữa category không phải phương sai giữa seed. Thiết kế tiếp
theo được chọn sau khi đã nhìn kết quả theo category trên test, nên nó là exploratory.
