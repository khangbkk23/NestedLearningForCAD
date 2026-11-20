# B321 lần 5 — so sánh development hoàn tất (b321-s0-20261114-05)

**Audit thực thi: `PASS_SCOPED` (metadata/ma trận). Screen khoa học: `SCREEN_NOT_MET`.** Cả hai arm
xong trước mốc dừng mềm 11 h; không cần resume. Một seed trên MVTec test đã lộ; không confirmatory,
không tái lập chính xác điều kiện của Khang hay paper.

| Metric | ER uniform | tail+CTX | Chênh lệch (CTX − ER) |
| --- | ---: | ---: | ---: |
| Pixel AP macro cuối (%) ↑ | 55,5500 | 55,4776 | −0,0725 pp |
| Image AUROC macro cuối (%) ↑ | 94,5523 | 94,6010 | +0,0487 pp |
| Fmax có dấu (pp) ↓ | 11,2812 | 11,6977 | +0,4165 |
| Facq (pp) ↓ | 9,6783 | 9,8716 | +0,1933 |

Yêu cầu gain chính +0,5 pp Pixel AP; quan sát −0,0725 pp nên không đạt. Vi phạm sàn Pixel AP (−1 pp):
cable −3,1949, capsule −2,3997, grid −2,1572, wood −4,9782. Vi phạm sàn Image AUROC (−0,5 pp): cable
−0,6184, screw −0,6559, wood −0,6140. Năm category vi phạm. Không hạ tiêu chí nào.

## Đã kiểm

- Code notebook thực chạy khớp code đã đăng ký và đóng gói (chỉ chuẩn hóa biểu diễn chuỗi/list
  JSON). Source manifest khớp manifest gốc. Cùng hash model khởi tạo, cùng transcript chọn dữ liệu và
  cùng danh tính cache feature theo stage giữa hai arm.
- Mỗi arm 10.005 update, 160.080 lượt mẫu = 85.376 current + 74.704 replay; 15 stage, 120 ô mỗi ma
  trận, bốn ma trận, 1.725 ảnh cuối và 13.950 lượt đánh giá. Cùng ID ảnh và số GT giữa hai arm.
- Tái dựng độc lập cả 960 ô quan sát và tính lại metric theo category, macro cuối, Fmax/Facq có dấu
  (14 category) khớp summary producer ở 1e−12. Controller tail+CTX hoạt động: 9.338 batch đủ điều
  kiện, 583 slow step, 2.332 forward query / 4.664 forward alignment, khớp lịch. ER có 0 controller,
  query, alignment. Đây là kiểm receipt và bộ đếm, không phải audit hypergradient từng update.
- Mỗi arm có 21 mục trong ledger checkpoint (500…10.000, 10.005); hai commit marker cuối khớp ledger.
- Thời gian: cả notebook 34.051,879 s; ER 26.234 s; CTX 33.772 s. Peak RSS nhóm 11,56 GiB < 16 GiB;
  mỗi worker < 8 GiB; VRAM < 12 GiB. Decoder 245.701.632 byte; replay 155.123.712 byte / 32 bản ghi;
  encoder đóng băng 346.334.208 byte; optimizer 737.104.896 byte. CTX thêm 97 tham số / 388 byte
  cộng 776 byte slow optimizer.

## Giới hạn audit

ZIP score thô và `state.pt` chưa được tải về để tính lại AP/AUROC từ tensor pixel ở audit này (phân
tích A5 sau đó kiểm một phần). Log event lớn hơn ngưỡng đọc nên chưa audit từng update. Sáu ô image
AP trung gian của metal_nut bằng 1,0000000000000002 (một ULP FP64 trên 1); giữ nguyên, không vào
screen.

## Kết luận

Thêm budget update không mang lại lợi ích learned replay ở seed này: Pixel AP gần bằng, quên hơi
nhiều hơn, có category giảm. Đây là bằng chứng chống adoption tail+CTX với recipe này, không phải
bằng chứng cả phương pháp không thể hoạt động. Giữ ER làm tham chiếu. Seed mới có thể đo độ dao
động nhưng không xóa screen đã trượt.

Số đầy đủ: [audit.json](audit.json); bản ghi chạy: [publication.json](publication.json).
