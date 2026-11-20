# B322 — pilot shadow EMA thụ động trên recipe B319

**Đăng ký trước execution ngày 15/11/2026.** Exploratory, một seed, trên MVTec test đã lộ trong
development. Không phải adoption, không phải tái lập B321, không phải bằng chứng "NL thắng ER".
Config: [configs/b322.json](../../configs/b322.json). Kết quả: **`NO_SIGNAL`**
([seed 0](runs/b322-s0-20261115-01/results.md); [phân tích A1](analyses/A1-headroom/README.md)).

## 1. Câu hỏi và phạm vi kết luận

Trên **cùng một quỹ đạo ER uniform**, readout bằng trung bình trọng số theo thời gian (shadow EMA)
có giữ kiến thức cũ tốt hơn readout bằng trọng số hiện tại không, mà không trả giá bằng khả năng
học task mới?

| Bước | Thay đổi | Kết luận được |
| --- | --- | --- |
| 1. Shadow thụ động (B322) | Chỉ thêm readout; training không đổi | Trung bình trọng số có giúp readout giữ kiến thức không |
| 2. Vòng kín | Mức chậm tác động lại mức nhanh | Consolidation vòng kín có ích không |
| 3. Mức chậm có học | Cổng/objective học được, kèm ablation | Thành phần NL đề xuất đóng góp gì |

B322 chỉ là bước 1. EMA có tiền lệ trong CLS-ER (Arani et al., ICLR 2022); B322 không tuyên bố đóng
góp NL. Bước 2 và 3 cần card riêng. Kết quả chỉ nói về nền DINOv2-reg ViT-B/14 + decoder Dinomaly
của hướng replay.

## 2. Recipe và nguồn

- Nền: recipe B319 nguyên bytes ([configs/b319.json](../../configs/b319.json), SHA-256 ghi trong config
  B322): seed 0, 15 task bottle → zipper, 3.629 arrival `train/good`, một fast update mỗi arrival,
  reservoir 24 + FIFO 8, full test 1.725 ảnh, 120 ô boundary × task.
- Một arm: `er_uniform`. Không controller, query hay slow update. Execution mới, ID mới. Recipe khác
  B321 (667 update/task, batch 16); không suy nguyên nhân chênh lệch B319/B321 từ pilot này.

## 3. Định nghĩa shadow EMA

Gọi θ_t là toàn bộ tham số decoder ngay sau `optimizer.step()` của arrival t (t = 1…3629); θ_0 là
tham số khởi tạo.

```text
s_0 = θ_0
s_t = a_t · s_{t−1} + (1 − a_t) · θ_t
a_t = min(1 − 1/τ, 1 − 1/(t + 1))
```

- τ là số nguyên dương, đơn vị optimizer step. Khi t + 1 ≤ τ, s_t là trung bình cộng θ_0…θ_t; sau đó
  là EMA hệ số 1 − 1/τ.
- Hai shadow cố định, không sweep: **τ = 512 (primary)** và τ = 128 (secondary). Độ dài task 60–391
  arrival (trung bình 242).
- **Thứ tự trong mỗi arrival:** forward/backward → clip gradient → `optimizer.step()` → cập nhật mọi
  shadow đúng một lần → admission → observer (nếu là boundary). Runner dừng nếu số lần cập nhật shadow
  khác số arrival. Không reset ở ranh giới task.
- Cập nhật in-place FP32 trong `no_grad` trên bản sao riêng. **Observer không đổi state:** kiểm hash fast
  decoder, hash từng shadow, RNG CPU và CUDA trước và sau observer; lệch thì dừng.

Implementation: [shadow.py](../../src/nested_replay_cad/shadow.py), hook trong
[trainer.py](../../src/nested_replay_cad/trainer.py), launcher [shadow_pilot.py](../../scripts/shadow_pilot.py).

## 4. Readout và metrics

- Tại mỗi boundary, mỗi ảnh test của các task đã học được trích feature một lần và chấm bằng ba bộ
  tham số (fast, τ = 512, τ = 128), cùng decoder ở `eval`, cùng scorer, cùng GT. Một readout dùng
  xuyên suốt; không chọn shadow theo task.
- Final(c) = AP[15][c]; Acq(c) = AP tại boundary đầu tiên có c (macro trên 15 category);
  Fmax(c) = max_{b<15} AP[b][c] − Final(c); Facq(c) = Acq(c) − Final(c); có dấu; macro trên **14 category**
  (định nghĩa B319). Image AUROC cuối: macro 15 category. Tính bằng
  [shadow_report.py](../../scripts/shadow_report.py).

## 5. Kiểm tra tính đúng trước khi đọc kết quả

1. Fast readout phải khớp B319 ER đã ghi (15 model hash ở boundary và 4 ma trận của `er_uniform`, run
   `b319-s0-20261110-kaggle-v5`). Lệch thì ghi `REPRO_MISMATCH`.
2. Tests: [test_shadow_pilot.py](../../tests/test_shadow_pilot.py) (stdlib: hệ số, công thức, nhãn, config;
   torch: tính thụ động và forward bằng tham số shadow).
3. Đây là kiểm tái lập cùng seed, không đo độ nhạy với seed.

## 6. Quy tắc đọc kết quả (chốt trước)

Mọi chênh lệch là **shadow − fast**, cùng run. Verdict lấy từ τ = 512. `SIGNAL` khi đồng thời:

- (a) macro Final(shadow) − macro Final(fast) ≥ +0,005;
- (b) macro Fmax(fast) − macro Fmax(shadow) ≥ +0,010;
- (c) macro Acq(shadow) − macro Acq(fast) ≥ −0,010 **và** min_c [Acq_shadow(c) − Acq_fast(c)] ≥ −0,030;
- (d) macro Image AUROC cuối: shadow − fast ≥ −0,005.

`NO_SIGNAL` khi (a) hoặc (b) không đạt; còn lại `MIXED`. Shadow secondary chỉ báo cáo. Screen lịch sử
giữ nguyên và được tính nguyên văn cho shadow so với fast (gain ≥ +0,005, mỗi category Pixel AP ≥ −0,01,
Image AUROC ≥ −0,005). `SIGNAL` chỉ mở đề xuất replication; `NO_SIGNAL` nghĩa là giả thuyết readout thụ
động không được hỗ trợ ở recipe này, không bác bỏ vòng kín.

## 7. Hạch toán bộ nhớ

| Hạng mục | B319 ER | B322 pilot | Ghi chú |
| --- | ---: | ---: | --- |
| Decoder fast (FP32) | 245,7 MB | 245,7 MB | 61.425.408 tham số |
| Shadow decoders | 0 | 491,4 MB | 2 × 245,7 MB; chi phí của pilot |
| Moment optimizer | 737,1 MB | 737,1 MB | cần để tiếp tục training |
| Replay features | ≤ 155,1 MB | ≤ 155,1 MB | 32 mẫu × 4,85 MB |
| **Cần giữ để tiếp tục training** | **1.137,9 MB** | **1.629,3 MB** | |
| Encoder đóng băng | 346,3 MB | 346,3 MB | |
| **Detector lúc inference** | **592,0 MB** | **592,0 MB** | encoder + một bộ tham số decoder |

Một shadow tương đương khoảng 50 mẫu replay về byte; bước 2 vì vậy phải có đối chứng ER cùng byte.

## 8. Chi phí và các lần chạy

- Một attempt cho mỗi job, không retry, không sweep, không early stop theo chất lượng AD.
- Cap `per_arm_seconds` = 14.400 s, đặt sau profile. Căn cứ: profile `b322-profile-20261115-02` (Tesla T4,
  tensor tổng hợp) cho cập nhật hai shadow 10,9 ms/arrival, thêm hai lượt chấm 64 ms/ảnh so với 31 ms của
  fast, peak VRAM 2,37 GB, peak RAM 2,82 GB; 6/6 kiểm thụ động đạt.

| Run | Kết quả |
| --- | --- |
| `b322-profile-20261115-01` | Lỗi script profile (`torch.allclose` nhận list); chưa có số đo. Đã sửa, không đổi phạm vi ([bản ghi](runs/b322-profile-20261115-01/publication.json)) |
| `b322-profile-20261115-02` | Hoàn tất; số đo ở trên ([bản ghi](runs/b322-profile-20261115-02/publication.json)) |
| [b322-s0-20261115-01](runs/b322-s0-20261115-01/results.md) | **`NO_SIGNAL`**; fast tái lập B319 ER 15/15 hash |

## 9. Ngoài phạm vi

Đối chứng lịch LR và budget cho ER thuộc card sau. Không VisA. Không đổi baseline đồ án hay kết luận NL.
