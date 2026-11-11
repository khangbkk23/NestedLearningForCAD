# H04 — Dinomaly foundation và mở rộng evaluation

**Tách hiệu quả detector kế thừa khỏi đóng góp learned replay.**

## Câu hỏi và thiết kế

FND-01 hỏi detector kế thừa có học normal representation và tạo ranking phù hợp
trong exposure budget của nhóm trước khi đầu tư controller hay không. Nguồn
thiết kế cùng giai đoạn được dẫn dưới; không coi mọi lựa chọn đã preregister khi
chưa xác minh temporal ordering độc lập.

Components qua snapshot Khang có provenance Dinomaly/DINOv2; không kế thừa điểm
benchmark của Khang. Kiểm current-only/uniform làm nền trước, rồi mới so uniform
với learned replay trên cùng nền. Source, encoder, decoder/scorer có identity riêng.

## Bằng chứng và kết luận

Pilot capsule/carpet 256, CPU/GPU qualification, learned trajectories, replication
và full 15 là các records riêng. Native GPU qualification không chứng minh mọi
checkpoint portable CPU/GPU hoặc giữa mọi runtime.

B298 học đủ 15 tasks nhưng dùng cohort development240. B310 dùng full 1.725 seen-task
test; không nối hai endpoints thành matched gain. Audit B310 ghi tail/mean chưa
đạt screen so ER, dẫn tới numerical/design review; chưa chứng minh tìm đủ mọi
nguyên nhân. Phần correction/qualification tách ở [B317](../B317/card.md).

## Giới hạn

Cùng fast commits/memory không phải matched compute khi learned arms có slow
updates. MVTec đã development-exposed; seed mới trên cùng test không là
confirmation độc lập. Memory presence không bảo đảm accuracy cao hơn.

## Nguồn và verification

- [R-b55a7ab3bfb102ac3b](../history/catalog.json#/records/R-b55a7ab3bfb102ac3b) — `S-996d4b7b4cd5b543ecb0`.
- [R-63ed0ea6d136c3d90f](../history/catalog.json#/records/R-63ed0ea6d136c3d90f) — `S-51e649e3ca85147e6a66`.
- [R-b062fe37d288a646c9](../history/catalog.json#/records/R-b062fe37d288a646c9) — `S-b9a014352cef3d770408`.
- [R-273d2572033999bc7b](../history/catalog.json#/records/R-273d2572033999bc7b) — `S-6656593dbf15548f3bbb`.
