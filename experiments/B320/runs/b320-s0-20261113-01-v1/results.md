# B320 lần 1 — terminal audit (b320-s0-20261113-01-v1)

**`FAILED` ở checker; chưa hoàn tất kiểm hai episode.**

- Episode 1 (arrival 30) đạt `PASS_SCOPED`: query/reference lệch tối đa 9,29e-13; forward toàn
  bộ optimizer lệch tuyệt đối 0; tín hiệu 4,10e-6.
- Production đã xử lý 46 arrival / 46 fast update. Episode 2 (arrival 46) dừng ở phép kiểm
  bằng tuyệt đối khi tính lại `hidden.bias`, chưa ra verdict theo hướng.
- Một regression CPU tổng hợp cho thấy chỉ đổi thứ tự dựng graph query/KL đã có thể làm
  gradient FP32 khác vài bit (10/16 fixture, lệch tối đa 2,91e-11). Điều này cho thấy checker
  có cơ chế báo sai, nhưng **không đo được sai lệch cụ thể của episode 2** và không chứng minh
  episode 2 đạt. State và gradient của episode 2 không được lưu đủ để hậu kiểm.
- Sửa instrumentation cho lần sau: chụp gradient production trực tiếp trước slow Adam; dựng
  lại graph chỉ còn là diagnostic; không nới tolerance.

Không có claim lợi ích AD. Bản ghi chạy: [publication.json](publication.json).
