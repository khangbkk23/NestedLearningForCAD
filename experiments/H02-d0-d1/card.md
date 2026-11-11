# H02 — D0/D1, F2 và fast Adam

**Hồi cứu; giữ riêng normal-only, technical qualification và AD.**

## Câu hỏi và thiết kế

Protocol D0-OPT-01 đặt phép so SGD–Adam hữu hạn để kiểm fitting và tác động replay
trong stream. Đó là nguồn thiết kế cùng giai đoạn; intake chưa chứng minh temporal
precedence cho mọi trial. Record thiếu nguồn giả thuyết riêng giữ `unknown`.

D0 là restoration detector thiết kế cho A0. Diagnostics xem readout/output span,
geometry, optimizer và state; D1 thử các sửa predictor/update giới hạn. Đổi fast
optimizer/state là scientific revision riêng, không chỉ numerical backend.
Design lock không tương đương usefulness acceptance.

## Bằng chứng và kết luận

D0 có học so initialization trong probes đã báo cáo nhưng chưa giải quyết F2/LOO.
B148 ghi Adam cải thiện fitting trong paired normal-only 256; original F2 vẫn
`INCONCLUSIVE`, không lấy endpoint hoặc snapshots thuận lợi để gọi toàn gate PASS.

D1 có các phép fitting/support-loss tốt hơn nhưng screen tổng chưa được giải
quyết. Reconstruction loss giảm không tự chứng minh retention, anomaly ranking
hoặc ích lợi controller. Qualification gradients/state/resume là các lớp riêng.

## Giới hạn

Không biến D0/D1 thành AD benchmark, không coi readout/optimizer là nguyên nhân
duy nhất đã chứng minh. Negative/null/failure trước revision vẫn giữ. Intake hash
lại hai checkpoint B148, không load tensor, train hoặc tái xác nhận behavior.

## Nguồn và verification

- [R-47f9a36ec4730c2d2e](../history/catalog.json#/records/R-47f9a36ec4730c2d2e) — `S-f256ae15e6d682883ec1`.
- [R-7550e610d6ffca6a0a](../history/catalog.json#/records/R-7550e610d6ffca6a0a) — `S-9f7a69e16ae178c7b55f`.
- [R-80eb033f862ba3a89f](../history/catalog.json#/records/R-80eb033f862ba3a89f) — `S-bfdaf6f81771777f7cca`.
