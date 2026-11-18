# Gói chuyển máy mở rộng — ngân sách 10 GB

Đây là bản mở rộng của gói 2,96 GB trước đó. Phạm vi kết quả 14–17/11 trong
`TRANSFER_README.md` mô tả gói trước; gói mới bổ sung lịch sử các phiên bản cũ
từ tháng 10 tới tháng 11. Hướng dẫn môi trường, tải MVTec và chạy lại NAR trong
file đó vẫn áp dụng.

Toàn bộ file đã có trong `agents/` được giữ nguyên tên, đường dẫn và nội dung,
kể cả các file bị Git bỏ qua. Gói mới chỉ thêm task/report đóng gói riêng;
không sửa hoặc xóa bất kỳ task/report nào đã có.

Gói giữ tất cả mã nguồn hiện có, hai checkpoint pretrained, đầy đủ kết quả NAR
full-15 đã có trong gói trước và metadata/log của các phiên bản lịch sử. Trong
ngân sách còn lại, gói bổ sung trọn các nhóm artifact nhỏ, lịch sử NL-SGM gần
đây và checkpoint đại diện của các nhóm cũ lớn hơn. Không phải mọi checkpoint
trung gian và mọi tensor prediction cũ đều có mặt.

`TRANSFER_MANIFEST.json` trong ZIP là danh mục chính xác: file được giữ, SHA-256,
nhóm artifact được chọn, file bỏ qua do dung lượng/cache và snapshot `agents/`.
Checkpoint "latest available" là checkpoint có số stage lớn nhất còn trên đĩa
trong run đó; tên này không khẳng định run đã hoàn thành hay benchmark đạt.
Các file có tên gợi ý sealed/confirmation được bỏ qua mà không đọc nội dung.

Vẫn loại toàn bộ `data/`, tensor cache đặc trưng, `.git`, `.vscode`, môi trường
ảo và cấu hình tài khoản cục bộ. Không sửa nguồn khoa học, checkpoint, protocol
hoặc kết quả cũ. Budget 10 GB áp dụng cho file ZIP; dung lượng khi giải nén và
chạy có thể lớn hơn.

Sau khi giải nén vào `HOPE-CAD/`, tải MVTec bằng:

```bash
bash scripts/download_transfer_data.sh
```

Mã NAR chính là `exps/nar_pilot_00/full.py`. Hướng dẫn từng pha và launcher tạo
thư mục kết quả mới ở phần chạy lại NAR trong `TRANSFER_README.md`.
