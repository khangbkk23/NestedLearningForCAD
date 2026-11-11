# B317 — Qualification numerical correction và CTX

**Technical qualification; không phải AD benefit hoặc adoption.**

## Câu hỏi và correction

Input `revision_protocol.json` ghi mode `FND-VJP-FWD32-01`, CTX-RN-01 và scope
exploratory. Intake xác minh source hashes, chưa chứng minh độc lập timestamp
preregistration; không bổ sung giả thuyết AD gain từ kết quả probe.

Audit trước chỉ ra recompute FP64 có thể chọn nhánh khác forward FP32 tại rounding
ties. Correction bind forward-source-dtype primals/branches rồi tính first
conditional VJP theo convention. Nó không là literal derivative của quantized
floating-point function và không hứa higher-order. CTX transform hai residual
columns trên active replay set, giữ detached state.

## Bằng chứng

Producer ghi 128 math checks, 74 context checks và actual-optimizer probe. Intake
đọc/hash receipt, base bundle và đủ 17 overlays, không thực thi lại model/probes.

Parent qualification là checkpoint ER **B298/version1**: model identifier khớp
`parent_evidence.json` và report probe. B310 dẫn tới diagnostic, không phải
checkpoint cha. Xem [V-B317-parent](../history/audit.json#/verifications/V-B317-parent).

## Kết luận và giới hạn

Giữ producer PASS đúng scope cùng failures B313/B315. Actual-episode phi-gradient
là `NOT_RUN`; toy/context không là finite-difference qualification cho toàn model.
Payload checkpoint B298 chưa được tìm như file local đã xác minh trong scan.
Metadata binding không thay hash tensor bytes. Không suy AD/higher-order/Phase4B
acceptance. B319 chạy fresh, không resume checkpoint B298.

## Nguồn và verification

- [R-e54916184dbc23cfe1](../history/catalog.json#/records/R-e54916184dbc23cfe1) — `S-c239b64fb510cd616b6b`.
- [R-30277429906929740e](../history/catalog.json#/records/R-30277429906929740e) — `S-d3bb2e98d2a7ede22421`.
- [R-649ffcf8d2d74a6519](../history/catalog.json#/records/R-649ffcf8d2d74a6519) — `S-10a53e9950cbcbc4d577`.

[V-B317-source](../history/audit.json#/verifications/V-B317-source) kiểm source
identity, không tái xác nhận behavior qualification.
