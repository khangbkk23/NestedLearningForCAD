# B321 qualification checkpoint/resume CUDA — version 4 (b321-resume-qual-20261114-02-v4)

**`PASS_SCOPED`**, provider `COMPLETE`, cả job 253,705 s. Không dùng weights thật, MVTec, AD hay VisA.

- Notebook và source manifest khớp package đã commit (source ZIP `669ef9e4…`). Hai T4 khác UUID;
  mỗi arm chạy bản gốc và bản resume trên cùng GPU. Mỗi arm khớp 11 trường hash; slow step 16 của
  CTX chạy trên cả quỹ đạo liên tục lẫn quỹ đạo khôi phục. Năm trường hợp filesystem Linux đạt.
- Fixture tổng hợp 40 feature [2, 789, 768], decoder thật, StableAdamW và 97 phi. Liên tục 17 update
  so với checkpoint ở 15 → process mới → 16/17. Qualification CPU riêng phủ mọi mốc 500, mốc cuối
  10.005, cursor và retention.
- Checkpoint native ≈ 1,05 GiB/arm; ghi mất 10,12 s (CTX) / 13,99 s (ER). Peak RSS nhóm 6,29 GiB,
  mỗi worker ≤ 3,25 GiB, VRAM cấp phát ≤ 2,58 GiB, CPU 272,67 s, payload 2,10 GiB.

Phạm vi: so receipt nhẹ, hash và danh tính source; không hash lại checkpoint hay tensor thô. Không
chứng minh mọi chế độ lỗi GPU; test lỗi filesystem không thay cho test mất điện thật. Run đầy đủ vẫn
phải qua admission tài nguyên với 21 checkpoint/arm. Version 1 bị cancel, không output; version 4
là bản hoàn tất. Bản ghi chạy: [publication.json](publication.json).
