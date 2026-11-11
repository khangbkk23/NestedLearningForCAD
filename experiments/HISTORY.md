# Lịch sử thí nghiệm: v6/v7 → B319

Chắt lọc hồi cứu từ `23b3a17` (22/10/2026), snapshot v6 trước migration v7, tới
inventory ngày 11/11/2026. Dependencies cần cho B319 được truy ngược dù trước mốc.
Không nhập implementation cũ vào cây chạy; không coi các records dưới đây là
preregistration chỉ vì chúng được lưu trong Git hôm nay.

## Đường đi nghiên cứu

**[H01 — v7/CADIC-NL](H01-v7-cadic-nl/card.md).** Hướng cũ đặt learned utility /
admission quanh normal-support memory CADIC. Migration v7 chủ yếu hợp nhất
package/vận hành, không tự đổi scientific identity. Baseline source và từng
pilot có protocol riêng; Git tip mới không làm artifacts cũ thành đã chạy trên
tip ấy. Matched CADIC nội bộ, số Khang và paper phải giữ provenance khác nhau.
Population, calibration exclusions, preprocessing, scorer và budgets cần được
đối chiếu trước khi gọi một so sánh là matched hoặc paper reproduction.

**[H02 — D0/D1](H02-d0-d1/card.md).** RDR/specs mới chuyển sang learned replay với
D0. Technical checks giúp kiểm gradients/state/resume, không tự xác nhận detector
hữu ích. Diagnostics ghi D0 có học nhưng chưa giải quyết F2/LOO; B148 cho fitting
tốt hơn với Adam trong normal-only 256 mà original F2 vẫn INCONCLUSIVE. Các sửa
D1/readout/update được giữ đúng scope, không kể thành AD gains. Design lock,
implementation conformance, usefulness và adoption là các kết luận khác nhau.

**[H03 — D2](H03-d2/card.md).** D2 thay predictor/scorer bằng associative prediction
và metric học được trên dictionary. Có evidence policy tác động update/prediction
và mechanism screens, nhưng không bảo đảm anomaly ranking. Review sau development
nhiều seeds khuyến nghị không benchmark dài recipe hiện tại nếu không có defect
mới. Đó là practical no-go, không chứng minh mọi NL thất bại. Current-only của D2
vẫn có retrieval memory, nên không phải no-memory baseline.

**[H04 — Dinomaly](H04-dinomaly/card.md).** Nhóm kế thừa nền detector có source
Dinomaly/DINOv2 qua Khang, kiểm nền trước khi thêm learned replay. Không kế thừa
điểm benchmark của Khang. Pilot256, GPU qualification, full 15 cohort 240 và full 1.725
là các tầng kiểm khác nhau. B298/B310 không được nối endpoints thành matched
gain. Full-test B310 chưa support primary benefit và dẫn tới numerical audit.

**[B317](B317/card.md).** Correction bind derivative convention với forward-source-
dtype primals/branches; CTX có scope riêng. Qualification report không phải AD
acceptance. Intake đối chiếu 17 overlay bytes. Parent probe là checkpoint ER B298,
không phải B310; payload cha còn gap local được ghi riêng.

**[B319](B319/card.md).** Corrected tail tăng nhẹ trong seed0 nhưng chưa đạt screen;
primary CTX cũng không đạt. Standalone khớp recorded matrices/model identifiers
của original, không tạo một kết quả thắng mới. Chưa có independent confirmation
hoặc Phase4B closure từ việc dọn code hay khớp checksum.

## Cách đọc và phần còn thiếu

[Catalog](history/catalog.json) công bố **4,371 records nghiên cứu và tham chiếu**,
giữ đủ 44 provider/version identities và 151 local attempts đã được nhận diện.
Đây không phải số scientific trials. Records chưa rõ thời điểm hoặc measurement
scope không được coi là executions khoa học đã xác nhận. Selection không dựa vào
PASS/FAIL, score hoặc gain; operational-only/unclassified context giữ ở hồ sơ riêng.

Mỗi record có `verification_refs` tới [audit](history/audit.json). Purpose chỉ gán
khi có nguồn; ordering chưa chứng minh thì preregistration `not_proven`. Bản đổi
links/semantic reconstruction không giả làm original. Hash raw không xác nhận
science. Intake đọc/hash metadata và chọn lọc source/checkpoint/archive bytes,
không train, inference, GPU hoặc đọc outcomes VisA; không sửa raw hay tự backup.

Audit liệt kê nguy cơ độc bản của archive lịch sử, B148 checkpoints, original B319
raw archives và standalone source ZIP. B298 ER payload chưa có bản đã xác
minh trong scan. Có pointer không đồng nghĩa backup độc lập. Người dùng quyết
định khôi phục/sao lưu; các raw ranks chưa kiểm lại không được nâng thành PASS.

Kiểm tra cấu trúc bản công bố: [kết quả](history/audit.json#/validation), chỉ áp dụng
cho IDs, pointers, links, source projections và phạm vi thay đổi; không thêm
scientific acceptance.

## Phạm vi bản công bố

Catalog/audit là `semantic_derivative`, không phải original producer bytes. IDs và
hashes tham chiếu nguồn gốc được bảo toàn, nhưng raw source payloads, exact source
selectors và đường dẫn riêng không nằm trong public tree. Có hash không đồng
nghĩa người đọc tải được payload hoặc tái lập toàn bộ trials lịch sử.
Xem [publication manifest](history/publication.json) và [provenance](../docs/provenance.md).
