# B320 lần 2 — diagnostic có phạm vi đạt (b320-s0-20261114-02-v1)

**`PASS_SCOPED`.** Kernel version 1 hoàn tất; producer báo `PASS_SCOPED`.

- Hai episode thật (arrival 30 và 46) đều qua gate query/KL/tổng trên tám hướng, tolerance
  `2e-8 + 2e-4` không đổi; giá trị được kiểm lại từ JSON.
- Forward toàn bộ optimizer trên mọi tọa độ và kiểm bản ghi gradient live đều đạt.
- Hash source Python khớp package đã chạy.

Phạm vi: audit metadata, danh tính và gate; reference độc lập cho optimizer/policy dùng G_i và
query của production. Không qualification toàn 97 chiều trên mọi episode, không chứng minh
lợi ích AD. B317 `NOT_RUN` và B320 lần 1 `FAILED` không bị sửa. Bản ghi chạy:
[publication.json](publication.json).
