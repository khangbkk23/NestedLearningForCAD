# B321 lần 2 — admission tài nguyên bị chặn (b321-s0-20261114-02-v1)

Provider `COMPLETE`; notebook báo đúng `BLOCKED_RESOURCE_PREFLIGHT`. Worker
`PREFLIGHT_INSUFFICIENT_MEMORY`; **full training chưa bắt đầu.**

- Hai T4 khác UUID; hai profile chạy chồng thời gian 49,92 s đã được kiểm. Composition và gradient
  toàn decoder đạt trên cả hai GPU; 12 commit profile tổng hợp; 0 update khoa học.
- RSS profile 7.115.829.248 byte (≈ 6,63 GiB) cộng dự phòng hai cache current-task 3.790.835.712 byte
  thành dự trù 10.906.664.960 byte (≈ 10,16 GiB), vượt cap 8 GiB. Không có OOM thật: đây là từ chối
  theo dự báo. Dự trù thời gian 41.930,93 s > thời gian worker còn lại 41.318,40 s, cũng sát cap.
- Nhãn `remaining_whole_cap` trong bộ dự báo thực chất dùng min(deadline worker, deadline cả job);
  cách tính reserve cần xem lại trước lần sau, không tự nới cap.

Không kết luận khoa học từ việc bị chặn tài nguyên. Lần 3 dùng cap RAM nhóm 16 GiB / 8 GiB mỗi
worker, giữ cả job 12 h. Bản ghi chạy: [publication.json](publication.json).
