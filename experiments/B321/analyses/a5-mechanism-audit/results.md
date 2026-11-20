# B321 A5 — audit cơ chế và một phần dữ liệu thô

**Audit có phạm vi, không phải thí nghiệm mới.** Controller có học, có đổi trọng số support và đổi
quỹ đạo training và dự đoán. Screen AD âm vẫn giữ. Không có bằng chứng đây là controller chết hay chỉ
là dư số học gradient bằng 0. Không claim tính đúng toàn cục của implementation hay một nguyên nhân
khoa học duy nhất.

## Bằng chứng

| Kiểm | Quan sát |
| --- | --- |
| Event và sampling | Đọc đủ 10.005 event/arm. Vai trò current/replay/recent/history, tính đủ điều kiện và serial admission được tái sinh chính xác từ metadata, seed 0 |
| Trọng số có hoạt động | Trọng số khác uniform lần đầu ở update 684, ngay sau slow update đầu tiên ở 683. 9.322/9.338 replay update khác uniform > 1e−6. ER giữ đúng 0,25 |
| Prefix ghép cặp | Loss và norm gradient fast trùng chính xác tới 683; khác lần đầu ở 684. Hash model ở boundary 667 và cả 83 thành phần score thô trùng |
| Độ lớn trọng số | Trên 74.704 trọng số CTX: 0,2236–0,2694; trung vị độ lệch tuyệt đối so 0,25 là 0,00168, trung bình 0,00266. ESS nhóm trung vị ≈ 3,9996/4. Độ tương phản thấp, không bão hòa |
| Slow state | 583 norm phi-gradient khác 0: trung vị 4,85e−5, nhỏ nhất 1,10e−6, lớn nhất 8,81e−3. Cả 97 tọa độ phi và moment đều có state học được |
| Query cục bộ | Đóng góp tuyến tính hóa đã log của trọng số học được so với uniform là âm (có lợi) ở 46,655% trong 583 slow event; trung vị +2,70e−8, trung bình −1,34e−7. Đây là đóng góp bậc nhất đã log, không phải cải thiện đo được sau update |
| Checkpoint | Hai `state.pt` cuối được tải độc lập; SHA và CRC ZIP khớp commit marker. Fast step 10.005 / slow 583; mọi phần tử moment hữu hạn, variance ≥ 0. Feature bộ nhớ và RNG CPU/CUDA trùng giữa hai arm |
| Model cuối | So 61,425 triệu tham số: chênh lệch L2 tương đối ≈ 1,03; 99,996% tọa độ khác > 1e−6. Đây là phân kỳ tích lũy, không phải phản thực một bước hay cải thiện chất lượng |
| Tính lại metric thô | Hai ZIP score ở boundary 667 có SHA/CRC đúng; 166 thành phần được kiểm SHA. AP có xử lý tie và AUROC Mann–Whitney tính độc lập trên 83 ảnh bottle / 5.439.488 pixel mỗi arm trùng metric đã lưu |
| Dự đoán cuối | Tám thành phần / bốn ảnh mỗi arm (cable, wood), chọn hậu kiểm cho audit cơ chế. RMS chênh lệch pixel 0,0067–0,0221, lớn nhất 0,058–0,082; GT trùng. Không phải dự đoán giống hệt chỉ khác bộ đếm |

## Xem lại công thức

Sau khi replay bắt đầu, tổng hệ số current là 0,5; mỗi trọng số trong hai nhóm bốn nhận 0,25 × w.
Uniform w = 0,25 vì vậy cho 0,0625/mẫu trên 8 replay, khớp 8 current. Slow objective là
Σ(w × alignment) + 0,001 × trung bình KL(w‖uniform) của hai nhóm; query trung bình bốn ảnh normal,
loss patch trung bình cho recent và top-79 patch cho lịch sử. Không thấy sai lệch hệ số, dấu hay nối
dây ở các đường này. Context và gradient support của từng step không được lưu trong ledger, nên
không tái dựng độc lập đầy đủ phi-gradient cho cả 583 step từ log.

## Phân loại nguyên nhân

- **Bị loại trong run này:** controller không bao giờ cập nhật; trọng số không được áp; sampler, vai
  trò hay seed chung bị hỏng; payload được chọn bị hỏng; dự đoán giữ nguyên.
- **Quan sát được:** đổi trọng số với độ tương phản yếu trong các nhóm đã chọn, trong khi lựa chọn bộ
  nhớ và truy xuất giữ nguyên; vẫn có thể gây phân kỳ lớn về lâu dài.
- **Có thể nhưng chưa chứng minh:** objective query trên ảnh normal có thể không chọn được thay đổi
  replay giúp xếp hạng AD.
- **Chưa xác lập:** KL áp đảo, learning rate chậm sai, CTX bão hòa, hay trọng số mạnh hơn sẽ giúp.
- **Chưa giải quyết:** dao động theo seed so với yếu kém hệ thống của phương pháp.

**Kết luận làm việc:** implementation có kích hoạt learned replay; mắt xích chưa chứng minh là trọng
số độ tương phản thấp và objective này có mang lại cải thiện AD hay không. Screen vẫn là Pixel AP
−0,0725 pp với năm category vi phạm sàn. Một câu hỏi tiếp theo nên tách cường độ can thiệp khỏi
giá trị của objective (uniform, học được, và control cùng độ tương phản nhưng gán xáo trộn); chưa
có run nào như vậy.

Số liệu có cấu trúc, hash và giới hạn: [audit.json](audit.json).
