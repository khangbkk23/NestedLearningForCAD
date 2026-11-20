# B323 — mức chậm cập nhật theo boundary làm neo hàm số trên replay

**Đăng ký trước execution ngày 15/11/2026.** Thiết kế được chọn sau khi đã xem kết quả theo category của
B322 trên MVTec test, nên mọi kết quả là exploratory, một seed. Không phải adoption, không phải bằng chứng
"NL thắng ER". Config: [configs/b323.json](../../configs/b323.json). Kết quả: **`NO_SIGNAL`**
([seed 0](runs/b323-s0-20261115-01/results.md)).

## 1. Vì sao là thiết kế này

- B322: readout EMA thụ động ra `NO_SIGNAL`; theo [A1](../B322/analyses/A1-headroom/README.md), EMA không
  giữ kiến thức task cũ tốt hơn fast, nên teacher EMA cho vòng kín không có căn cứ.
- Teacher có chất lượng đo được từ ma trận đã lưu là **bản sao của fast tại boundary**.
- Khung NL: hai mức tham số có tần số cập nhật khác nhau. Mức nhanh θ_F cập nhật mỗi arrival; mức chậm θ_S
  cập nhật một lần mỗi task và tác động ngược lại mức nhanh qua loss.
- **Tiền lệ cơ chế:** neo hàm số trên replay với snapshot đầu stage đã có ở NLCAD-06C-ANCHOR của Khang
  (λ = 1,0, cùng dạng cosine, đầu ra decoder eval-mode FP32 tại đầu mỗi stage trên các mẫu replay). 06C chỉ
  chạy stage 7–9, đo loss trên ảnh normal, không có đối chứng ER cùng byte, và kết quả của nó không có trong
  repository. Ý tưởng này cũng có trong LwF và DER. B323 không có tính mới ở cơ chế; phần mới là Pixel AP /
  Image AUROC trên 15 task và đối chứng ER cùng byte.

## 2. Câu hỏi

Trên recipe B319, neo đầu ra của decoder nhanh về đầu ra của mức chậm trên các mẫu replay của task cũ có
nâng Pixel AP cuối so với ER **cùng byte budget** không, mà không làm mất backward transfer và khả năng học
task mới?

## 3. Arms

| Arm | GPU | Memory | Thêm |
| --- | --- | --- | --- |
| `er_bytes` (đối chứng) | slot 0 | reservoir 74 + FIFO 8 = 82 mẫu | Không. 50 mẫu thêm = 242,4 MB, lệch dưới 2% so với một decoder (245,7 MB) |
| `anchor` (ứng viên) | slot 1 | reservoir 24 + FIFO 8 = 32 mẫu | θ_S: một bản sao decoder |

- Hai arm chạy song song trong một job, mỗi arm một GPU, output riêng. Tham chiếu thứ ba, không chạy lại và
  không vào gate: ER 32 mẫu của B319 (B322 đã tái lập bit-exact).
- Mọi thứ khác giữ recipe B319: seed 0, 3.629 arrival, một fast update mỗi arrival, 4 support mỗi update,
  full test 1.725 ảnh, 15 boundary.
- `er_bytes` chỉ đổi `capacity` của sampler (mặc định 24 giữ hành vi cũ). Chuỗi mẫu của nó khác ER 32 từ khi
  reservoir vượt 24, nên không có tham chiếu bit-exact cho arm này.

## 4. Định nghĩa arm `anchor`

Implementation: [anchor.py](../../src/nested_replay_cad/anchor.py), hook trong
[trainer.py](../../src/nested_replay_cad/trainer.py), launcher [anchor_pilot.py](../../scripts/anchor_pilot.py).

- **Mức chậm:** θ_S là bản sao cứng của θ_F, lấy ngay sau observer của mỗi boundary (14 lần). Giữa hai
  boundary θ_S đứng yên. Trước boundary đầu tiên arm này trùng ER 32.
- **Mẫu được neo:** support có arrival lúc vào memory ≤ boundary gần nhất. Mẫu hiện tại và mẫu của task đang
  học không bị neo.
- **Khoảng cách:** với `de_F`, `de_S` là hai nhóm đầu ra đã fuse của decoder,

```text
D(x) = (1/2) · Σ_g  mean_vị_trí [ 1 − cos( de_F,g(x), de_S,g(x) ) ]
```

  `de_S` tính bằng `functional_call` ở eval, no-grad: tất định, không tiêu RNG.
- **Loss mỗi arrival:** loss ER như cũ cộng `c · λ · D(x)` cho mỗi mẫu được neo, `c` là hệ số của support đó
  trong ER (0,5 / số support), **λ = 1,0** (một giá trị, không sweep).
- **Thứ tự backward:** hook hard-mining của loss ER nhân gradient với 0,1 ở vị trí "dễ"; để hook không bóp
  gradient của neo, số hạng neo được backward trước, riêng, rồi mới dựng loss ER.
- **Inference:** chỉ θ_F; bộ nhớ detector không đổi (592,0 MB). Bộ nhớ để tiếp tục training tăng 245,7 MB,
  xấp xỉ `er_bytes` (+242,4 MB). Byte-matched nhưng **không compute-matched**: mỗi arrival `anchor` thêm tối đa
  4 forward no-grad của θ_S và 4 backward riêng.

## 5. Giả định mới của protocol

Training của `er_uniform` không dùng ranh giới task; arm `anchor` cần tín hiệu boundary lúc train. Đây là giả
định thêm so với baseline và phải được nêu trong mọi báo cáo.

## 6. Rủi ro biết trước

Neo có thể chặn backward transfer (tile, toothbrush, carpet, metal_nut, leather); trần nhỏ (Acquisition −
Final của fast là +0,0273, trần lạc quan +0,0433); neo dây chuyền chỉ làm chậm trôi, không khôi phục; thêm 50
mẫu replay có thể tự nó thu hẹp gần hết khoảng đó. Giảm D(x) không phải bằng chứng retention.

## 7. Kiểm tra tính đúng và qualification

1. Tests [test_anchor_pilot.py](../../tests/test_anchor_pilot.py): config, nhãn, báo cáo; θ_S không đổi
   RNG/mode/tham số fast, D = 0 ngay sau consolidation, gradient neo không bị hook bóp; `capacity=24` trùng
   mặc định.
2. **Prefix:** hash `anchor` tại arrival 209 phải bằng hash B319 ER (`84c3e401…bfd8`); lệch là `PREFIX_MISMATCH`.
3. **Qualification:** hai arm trên dữ liệu thật, dừng ở arrival 433 để đo thời gian, bộ nhớ và kiểm đường
   code. Số AD của đoạn này không dùng để đổi thiết kế, λ hay gate.

## 8. Quy tắc đọc kết quả (chốt trước)

So sánh chính `anchor` − `er_bytes`, cùng job. `SIGNAL` khi đồng thời: (a) macro Final Pixel AP ≥ +0,005;
(b) mọi category Final Pixel AP ≥ −0,01; (c) macro Acquisition ≥ −0,010; (d) macro Image AUROC cuối ≥ −0,005.
`NO_SIGNAL` khi (a) không đạt; còn lại `MIXED`. Báo cáo thêm, không vào gate: so với ER 32; Fmax/Facq; chênh
lệch theo tuổi task. Tính bằng [anchor_report.py](../../scripts/anchor_report.py). `SIGNAL` chỉ mở đề xuất thêm
seed.

## 9. Chi phí và các lần chạy

Cap `per_arm_seconds` = 14.400 s, đặt sau qualification 02 (mỗi arrival sau boundary đầu: `er_bytes` 0,73 s,
`anchor` 1,12 s; peak RAM và VRAM 2,2 GB mỗi arm ở arrival 433; khoảng cách neo trung bình sau boundary 0,0075,
bằng khoảng 7% loss replay ER). RAM 8 GiB mỗi tiến trình, 16 GiB cả nhóm.

| Run | Kết quả |
| --- | --- |
| `b323-qual-20261115-01` | Lỗi ở bước test CPU (notebook gọi test của B322 đọc config không có trong bundle), trước dữ liệu và GPU ([bản ghi](runs/b323-qual-20261115-01/publication.json)) |
| `b323-qual-20261115-02` | Hoàn tất; số đo ở trên ([bản ghi](runs/b323-qual-20261115-02/publication.json)) |
| [b323-s0-20261115-01](runs/b323-s0-20261115-01/results.md) | **`NO_SIGNAL`**; `PREFIX_REPRODUCED` |

## 10. Ngoài phạm vi

Không VisA. Không đổi baseline đồ án. Không sweep λ hay kích thước memory. Biến thể không cần boundary (sao
chép theo chu kỳ cố định) để dành cho card sau.
