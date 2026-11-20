# B321 lần 1 — terminal audit (b321-s0-20261113-01-v1)

**`PREFLIGHT_INSUFFICIENT_CAP`; không bắt đầu full training hai arm.**

- Composition check đạt; gradient decoder micro-batch 4 so với chạy tuần tự lệch L2 tương đối
  9,01e-7 (< 2e-4). Preflight có 6 optimizer commit tổng hợp; không có update khoa học, không có
  AP/AUROC.
- Đo được: batch ER 1,768 s; CTX thường 2,050 s; CTX có slow step 6,538 s. Dự trù training hai arm
  11,34 h, cache 0,25 h, đánh giá đầy đủ 2,22 h; cộng margin và export cần 17,08 h trong khi worker
  còn 11,48 h. Dừng trước training đúng theo cap; không giảm update, cohort hay precision.
- Notebook wrapper cũ coi mọi exit 2 là exception nên provider báo `ERROR` thay vì giải thích việc
  chặn tài nguyên; đã sửa để chỉ nhận đúng schema dừng đã đăng ký. Không sửa kết quả producer.

Hướng tiếp theo khi đó: chạy hai arm độc lập trên hai T4 của cùng một job, giữ nguyên khoa học.
Bản ghi chạy: [publication.json](publication.json).
