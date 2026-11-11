# H01 — v6 → v7 và NL quanh CADIC

**Hồ sơ hồi cứu; không phải method đang chạy.** Mốc nhập `23b3a17` (22/10/2026)
bảo toàn source v6 trước migration. ADR012 chốt baseline v7 `89df9a2`; tag tip
v7 về sau là mốc khác. Không relabel artifacts cũ thành đã chạy trên source mới.

## Câu hỏi và thiết kế

ADR010 mô tả learned utility/admission cho normal-support memory quanh CADIC.
Đây là mục tiêu ghi trong nguồn thiết kế lịch sử; giả thuyết gain và việc đăng
ký trước của từng run chưa được chứng minh đồng loạt. Không suy từ outcome.

v7 chủ yếu đổi tổ chức/package/vận hành; scientific identity cũ vẫn là hướng
v5.1 causal-scaled utility. Nó khác replay learner và detector Dinomaly hiện tại.
Baseline source, overlays, protocol và population phải đi cùng nhau: một Git tip
không đại diện mọi working changes hoặc artifacts đã chạy.

## Bằng chứng, kết luận và giới hạn

Legacy register giữ run/review tìm được; engineering reports không được đếm như
scientific successes. Digest ghi các pilot/retention observations có scope hạn
chế, không đủ khẳng định AD gain hay VisA confirmation.

Matched CADIC nội bộ, số paper và số Khang có provenance khác nhau. Stream,
calibration exclusions, preprocessing và readout phải được đối chiếu trước so
sánh. Reproduction gap không chứng minh paper sai hoặc NL thất bại tổng quát.
Run không có thời điểm rõ vẫn là candidate/context; semantic records không thay
raw bytes. Archive lịch sử chưa có backup độc lập được xác minh.

## Nguồn và verification

- [R-a4281c61f8a6210e80](../history/catalog.json#/records/R-a4281c61f8a6210e80) — `S-dbdc0bed45aa1caa6117`.
- [R-bae0756cd1ac5c7dfa](../history/catalog.json#/records/R-bae0756cd1ac5c7dfa) — `S-3d12f2f11049bf233ad1`.
- [R-615b45de94a256f3d0](../history/catalog.json#/records/R-615b45de94a256f3d0) — `S-0795b6dd6b0a9ba39797`.
- [R-157b42da203092e5a0](../history/catalog.json#/records/R-157b42da203092e5a0) — `S-2246728e743d05307687`.

Mỗi record dẫn verification riêng; byte identity không xác nhận science.
