# H03 — D2 associative prediction và quyết định đổi nền

**Scientific candidate lịch sử; không phải reproduced ReplayCAD.**

## Câu hỏi và thiết kế

Design review sau B220 đề xuất predictor thay thế trước trial dài: context-keyed
associative prediction, metric học được để truy hồi normal values. Đây là mục
tiêu trong nguồn thiết kế, không phải giả thuyết thắng benchmark bổ sung sau
outcome. Temporal preregistration chưa được intake chứng minh độc lập.

Đường tác động: policy → replay weights → fast metric update → retrieval →
prediction. Memory admission và phần stream/sampler phù hợp được giữ. Thay
predictor/scorer/state cần review riêng, không kế thừa acceptance/checkpoint D0.

## Bằng chứng và kết luận

Prototype, scaling, direction/controller và online trials nằm trong catalog.
B239 hỗ trợ mechanism screen trong scope đã ghi, không chứng minh AD benefit.
Normal-only và AD development là hai lớp evidence khác nhau.

Review B262 ghi primary AD development đã có ba seeds và không đạt screen;
khuyến nghị không benchmark dài D2 hiện tại nếu không có defect mới có evidence.
Đây là practical disposition cho recipe, không chứng minh mọi D2/NL đều vô ích.

## Giới hạn

Current-only của D2 vẫn có retrieval dictionary, không phải no-memory control.
Policy tác động replay weights, không tự học admission/toàn bộ dictionary.
Không so pilot ít classes/ảnh với full 15 của Khang/paper khi backbone, scorer,
population và budgets khác nhau. Raw runs chưa kiểm sâu vẫn có gaps riêng.

## Nguồn và verification

- [R-0296fe5f4f9a8de896](../history/catalog.json#/records/R-0296fe5f4f9a8de896) — `S-5c86bb169abbd43106f9`.
- [R-5a1d28b203fb49c6dd](../history/catalog.json#/records/R-5a1d28b203fb49c6dd) — `S-c4205af1bbf6a11cfd8b`.
- [R-0c5bba02b99b23f5d5](../history/catalog.json#/records/R-0c5bba02b99b23f5d5) — `S-4222459534b134492730`.
