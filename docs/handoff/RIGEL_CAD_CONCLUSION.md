# RIGEL-CAD: Kết luận cuối cùng (handoff, 2026-11-28)

Tài liệu này tự đứng một mình, đọc không cần các report nội bộ. Mọi số liệu dưới đây đã được đo trong repo; file nguồn ghi ở cuối mỗi mục.

- Code: `rigel_cad/`, phiên bản **1.1.1**, commit trên nhánh `dikhang_space`.
- Hướng dẫn dùng chi tiết: `rigel_cad/README.md`.

---

## 0. Tóm tắt

1. **RIGEL-CAD** là bộ phát hiện bất thường liên tục (continual anomaly detection):
   - học tuần tự từng category công nghiệp, chỉ dùng ảnh bình thường;
   - không huấn luyện bằng gradient;
   - không cần biết task id khi test;
   - bộ nhớ lưu giữ cố định, mặc định 8 MiB.
2. **Kết quả chính ở 8 MiB, định dạng fp16 mặc định** (pixel AP / image AUROC):

   | Dataset | Pixel AP | Image AUROC |
   |---|---|---|
   | MVTec AD | 0.593 | 0.983 |
   | VisA | 0.460 | 0.935 |
   | MPDD | 0.508 | 0.940 |

   Định dạng nén tuỳ chọn **int4_triangular**, cùng 8 MiB:

   | Dataset | Pixel AP | Image AUROC |
   |---|---|---|
   | MVTec AD | 0.630 | 0.993 |
   | VisA | 0.475 | 0.971 |
   | MPDD | 0.517 | 0.960 |

   Lặp lại trên 3 thứ tự task × 3 seed, int4 hơn fp16 trong 9/9 cell trên MVTec và VisA, 7/9 cell trên MPDD.
3. **Trục bộ nhớ gần như đã hết dư địa:**
   - int4 ở 8 MiB chỉ kém bản không giới hạn bộ nhớ (46–107 MB) khoảng 0.004–0.012 AP;
   - đường cong ngân sách 2–24 MiB xác nhận điều này (§3).
4. **Dư địa lớn duy nhất còn lại là biểu diễn đặc trưng:**
   - huấn luyện joint (không continual) Dinomaly-B đạt 0.693 / 0.532 pixel AP (MVTec / VisA);
   - khoảng cách là chừng 0.06 AP;
   - đây là biến thể khoa học mới, cần owner duyệt.
5. **Bản 1.1.1 chỉ tối ưu kỹ thuật** và cho kết quả giống hệt từng bit trên cả 3 dataset:
   - train nhanh gấp 2.2 lần;
   - chấm điểm nhanh gấp 2 lần.

---

## 1. Phương pháp (ngắn)

**Encoder.** DINOv2 ViT-B/14 có 4 register, đóng băng, ảnh 392 px. Mỗi ảnh thành 28×28 patch token 768 chiều, đã chuẩn hoá L2.

**Hai bộ nhớ, hai tần suất cập nhật:**

| Bộ nhớ | Cập nhật | Nội dung |
|---|---|---|
| Exemplar (nhanh) | Mỗi task: lấy coreset 1 % bằng greedy k-centre, gộp với key cũ, rồi k-centre lại về đúng dung lượng | ngân hàng key |
| Residual geometry (chậm) | Cộng dồn suốt luồng | mô-men bậc hai S của phần dư (token − key gần nhất cùng task), cùng số đếm n |

**Chấm điểm:**
- Metric W = shrink(S/n, 0.5)^(−1/2).
- Năng lượng của mỗi token q là e(q) = ½·min_m ‖qW − mW‖².
- Map 28×28 được nội suy song tuyến lên 392², rồi làm mượt Gaussian với σ = 4.
- Điểm ảnh = trung bình top 1 % của map.

**Ngân sách bộ nhớ.** Chỉ tính key và S. Encoder dùng chung, đóng băng (144.8 MB) và không tính vào bộ nhớ continual.

**Hai định dạng lưu trữ.** Thuật toán giống hệt nhau, chỉ khác cách biểu diễn:
- **fp16:** 1,536 B mỗi key, S là ma trận fp32 đầy đủ. 8 MiB chứa được 3,925 key.
- **int4_triangular:** 386 B mỗi key (int4 với scale absmax theo từng key), S chỉ lưu tam giác trên. 8 MiB chứa được 18,672 key.
- Với int4, việc lượng tử hoá diễn ra ngay trong luồng học. Consolidation và chấm điểm đều chạy trên key đã giải lượng tử.

---

## 2. Kết quả

**Giao thức.** Theo UCAD: MVTec 15 task, VisA 12, MPDD 6, đánh giá model cuối, không dùng task id.
- Pixel AP là AP nhị phân hoá ở độ phân giải gốc của ảnh.
- FM là độ quên.

### 2.1 Cell chính (order a, seed 2711, 8 MiB)

| Dataset | Định dạng | Pixel AP | Image AUROC | FM |
|---|---|---|---|---|
| MVTec AD | fp16 | 0.5927 | 0.9826 | 0.0333 |
| | int4_triangular | 0.6304 | 0.9926 | 0.0074 |
| VisA | fp16 | 0.4603 | 0.9349 | 0.0119 |
| | int4_triangular | 0.4749 | 0.9707 | 0.0053 |
| MPDD | fp16 | 0.5082 | 0.9399 | 0.0128 |
| | int4_triangular | 0.5170 | 0.9598 | 0.0012 |
| *Tham chiếu không giới hạn bộ nhớ (fp16)* | | 0.6348 / 0.4868 / 0.5175 | | dùng 46.1 / 106.6 / 13.1 MB |

CI95 bootstrap cho pixel AP của fp16:
- MVTec AD [0.575, 0.612];
- VisA [0.430, 0.485];
- MPDD [0.490, 0.530].

### 2.2 Độ bền: 3 thứ tự task × 3 seed (mean ± std)

| Dataset | fp16 | int4_triangular | Δ AP (số cell dương) |
|---|---|---|---|
| MVTec AD | 0.5944 ± 0.0063 | 0.6310 ± 0.0017 | +0.037 (9/9) |
| VisA | 0.4505 ± 0.0052 | 0.4738 ± 0.0031 | +0.023 (9/9) |
| MPDD | 0.5102 ± 0.0078 | 0.5171 ± 0.0046 | +0.007 (7/9) |

int4 không thắng đều ở mọi category. Mức giảm lớn nhất của từng category so với fp16:
- zipper: −0.073;
- macaroni1: −0.031;
- MPDD connector: −0.020.

int4 tốt hơn là nhờ chứa được nhiều key hơn khoảng 4.8 lần, không phải nhờ key chính xác hơn. Thực tế lượng tử hoá int4 làm đổi 19–25 % phép gán key gần nhất.

### 2.3 MVTec AD 2 (chỉ fp16, chấm trên server chính thức)

| Split | AU-PRO 0.05 | Chỉ số khác |
|---|---|---|
| test_private | 38.2 % | SegF1 20.5 |
| test_private_mixed | 34.0 % | SegF1 15.3 |

- Kết quả này cao hơn các baseline 256 px của tác giả dataset, nhưng thấp xa các phương pháp joint độ phân giải cao.
- int4 chưa được đánh giá trên AD 2, và owner đã quyết định không chạy lại AD 2.

### 2.4 So với các phương pháp đã công bố (số do paper tự báo cáo)

Định dạng ô: image AUROC / pixel AP.

| Phương pháp | Backbone | Bộ nhớ | MVTec | VisA |
|---|---|---|---|---|
| **RIGEL-CAD fp16 @8 MiB** | DINOv2 @392 | 8.4 MB | 0.983 / 0.593 | 0.935 / 0.460 |
| **RIGEL-CAD int4 @8 MiB** | DINOv2 @392 | 8.4 MB | 0.993 / 0.630 | 0.971 / 0.475 |
| RIGEL-CAD fp16, backbone khớp | ViT-B/8 @224 | 8.4 MB | 0.887 / 0.502 | 0.764 / 0.236 |
| CADIC 2.5k / 10k | ViT-B/8 @224 | 7.7 / 30.7 MB | 0.955 / 0.559 · 0.972 / 0.584 | 0.828 / 0.415 · 0.891 / 0.438 |
| CMPMB | CLIP | ≈ 23 MiB | 0.974 / 0.604 | 0.901 / 0.365 |
| DeCoFlow | WRN50 / ViT-B/16 | ≈ 136 MB | 0.984 / 0.582 | 0.930 / 0.370 |
| GCR | OpenCLIP ViT-B/16 | 9 MB | 0.978 / 0.513 | 0.893 / 0.302 |
| ReplayCAD | SD 1.5 | 3 MB + generator | 0.948 / 0.537 | 0.903 / 0.415 |

**Cách đọc bảng.** Lợi thế của RIGEL-CAD **phụ thuộc vào backbone**. Khi dùng cùng backbone ViT-B/8 @224 với CADIC thì RIGEL-CAD thấp hơn CADIC. Vì vậy **không claim state of the art**; chỉ claim một phương pháp không cần gradient, ngân sách cố định, đã đo đầy đủ chi phí.

---

## 3. Đường cong ngân sách 2–24 MiB

Pixel AP cuối, order a, seed 2711.
- Chỉ mang tính mô tả: test set đã từng được dùng khi phát triển, và không chọn gì từ đường cong này.
- Các điểm fp16 4 / 8 / 16 MiB và int4 8 MiB khớp đúng số đã khoá ở mọi stage.

| MiB | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | Không giới hạn |
|---|---|---|---|---|---|---|---|---|---|---|
| MVTec fp16 | — | — | .561 | .584 | .593 | .608 | .622 | .628 | .632 | .635 |
| MVTec int4 | .583 | .609 | .621 | .626 | .630 | .633 | .633 | .633 | | |
| VisA fp16 | — | — | .367 | .442 | .460 | .462 | .461 | .474 | .481 | .487 |
| VisA int4 | .421 | .453 | .466 | .471 | .475 | .476 | .482 | .487 | | |
| MPDD fp16 | — | — | .469 | .489 | .508 | .517 | .518 | .518 | .518 | .518 |
| MPDD int4 | .488 | .510 | .517 | .517 | .517 | .517 | .517 | .517 | | |

fp16 cần tối thiểu 2.25 MiB chỉ riêng cho S, nên không có điểm 2 và 3 MiB.

**Đọc đường cong:**
- **int4 chỉ cần khoảng ¼ ngân sách của fp16 để đạt cùng AP.** Ví dụ trên MVTec, int4 4 MiB ≈ fp16 16 MiB; trên VisA, int4 4 MiB > fp16 16 MiB.
- **Ngoại lệ:** MPDD từ 12 MiB trở lên. Ở đó cả hai định dạng đều giữ toàn bộ key, nên int4 thấp hơn khoảng 0.0005; đây là cái giá thuần của lượng tử hoá.
- **Điểm bão hoà:**
  - MVTec: int4 bão hoà ở 12 MiB (giữ toàn bộ 28,458 key, 12.2 MB), đạt 0.6331 so với 0.6348 của bản không giới hạn (46 MB);
  - VisA: int4 24 MiB đạt 0.4869, bằng bản không giới hạn vốn cần 107 MB.
- **Ở ngân sách rất nhỏ** (phù hợp thiết bị edge): int4 3 MiB đã vượt fp16 8 MiB trên MVTec và MPDD.

---

## 4. Bản 1.1.1: tối ưu kỹ thuật, kết quả không đổi

Các thay đổi:
- **k-centre chạy bằng CUDA graph:** chia khối 8,192 hàng, không đồng bộ CPU mỗi bước.
- **Tách `fit_task`** thành `compute_fit` (không phụ thuộc ngân sách) và `write_task`.
- **Cache chấm điểm theo state:** key fp32, W và key·W chỉ tính một lần.
- **Decode ảnh đa luồng** khi train.

Kiểm tra:
- Mọi thứ giống hệt từng bit với 1.1.0 trên MVTec, VisA và MPDD: fit từng task, k-centre, state cuối, grid điểm ở mọi stage.
- 43 test đạt.

| | 1.1.0 | 1.1.1 |
|---|---|---|
| Train MVTec từ ảnh gốc (3,629 ảnh, 3 arm) | 682 s | 305 s |
| Fit coreset MVTec / VisA / MPDD | 92 / 640 / 16 s | 37 / 243 / 7 s |
| Chấm điểm mỗi ảnh p50, fp16 / int4 | 3.7 / 17.9 ms | 1.8 / 8.2 ms |

Chưa làm: giảm tiếp chi phí chấm điểm int4 (cần TF32 hoặc fp16 GEMM, sẽ làm đổi số). Encoder (khoảng 115 s trên MVTec) giữ nguyên.

---

## 5. Các hướng đã thử và đã đóng

| Hướng | Kết quả |
|---|---|
| Đóng góp Nested Learning (hai tần suất cập nhật: key nhanh, metric chậm) | **Đã kết luận:** ở cùng ngân sách byte, E (hai mức) > F (chỉ key) trên cả 3/3 dataset |
| Writer hoặc adapter học bằng gradient | Không hỗ trợ (không cải thiện) |
| GG-01: consolidation có dùng W để dẫn hướng | **Bị loại ở DEV:** −0.049 AP |
| D1: thêm tầng CMS giữ các key bị đẩy ra ở dạng int4 | **Bị loại ở DEV:** thua Q4T |
| Q8T: key int8 | Đạt DEV, nhưng kém hơn int4_triangular |
| Q4T: int4 + S tam giác | **Được hỗ trợ trên benchmark** (27 cell), đã tích hợp vào 1.1 |
| Xoay trực giao trước khi lượng tử int4 | Méo giảm (25 % → 11 % key gần nhất bị đổi), nhưng trần AP chỉ khoảng +0.006 nên không làm |
| Dùng task id (oracle) | Chỉ +0.001 macro AP. Cable +0.19 nhưng các texture bị âm, nên không phải đòn bẩy |

---

## 6. Giới hạn cần nói rõ

- **Test set đã lộ.** Test set MVTec, VisA và MPDD đã được dùng trong quá trình phát triển, nên:
  - kết quả của int4 và đường cong ngân sách là bằng chứng benchmark, không phải kiểm định held-out;
  - đường cong ngân sách mới chạy 1 cell, chưa có CI.
- **So sánh với backbone khớp.** Khi dùng cùng ViT-B/8 @224, RIGEL-CAD thấp hơn CADIC. Còn một nghi vấn chưa kiểm: CADIC có thể chấm pixel AP trên mask 224 px, trong khi ta chấm ở độ phân giải gốc. Muốn kiểm thì chấm lại các grid ViT-B/8 đã lưu với mask 224; việc này chạy trên CPU và rẻ.
- **Chi phí của int4:**
  - head chậm khoảng 4.5 lần so với fp16 (8.2 so với 1.8 ms);
  - file state vượt 8 MiB vài kB do overhead của `torch.save` (phần tensor vẫn nằm trong ngân sách).
- **Không phải streaming thật sự.** Chọn k-centre hồi cứu trên toàn bộ coreset quá khứ cho kết quả tốt hơn, nhưng cách đó cần dữ liệu cũ nên không dùng được trong luồng học.

---

## 7. Việc còn chờ quyết định

1. Paper có nên khuyến nghị int4_triangular cho ngân sách ≤ 8 MiB không, dưới dạng đường cong mô tả? Claim chính vẫn là fp16 @ 8 MiB.
2. Có kiểm nghi vấn mask 224 px khi so với CADIC không (§6)? Rẻ, chạy CPU.
3. Có duyệt hướng biểu diễn đặc trưng không (khoảng +0.06 AP)? Hướng này cần:
   - trích xuất token mới trên GPU;
   - một giao thức DEV mới, vì mọi test set công khai đã lộ.

---

## 8. Tái lập

```bash
python -m pytest -q rigel_cad/tests                                    # 43 test (equivalence cần CUDA + cache)
python -m rigel_cad stream --dataset mvtec --data-root data/mvtec --budget-mib 8 [--storage int4_triangular]
python -m rigel_cad bench  --dataset mvtec --data-root data/mvtec [--storage int4_triangular]
python -B -m exps.rigel_eng_01.check {mvtec|visa|mpdd}                 # kiểm bitwise 1.1.1
python -B -m exps.rigel_budget_00.sweep {mvtec|visa|mpdd}              # đường cong ngân sách
```

**Nguồn số liệu:**

| Mục | File |
|---|---|
| Kết quả chính và độ bền | `results/hope_cad/g3e_paper_01/RESULTS.json` |
| int4 | `results/rigel_cad/q4t_integration/VERIFY_SUMMARY.json` |
| Đường cong ngân sách | `results/rigel_cad/budget_sweep/SWEEP_TABLES.md` |
| Kiểm 1.1.1 | `results/rigel_cad/eng_01/CHECK_SUMMARY.json` |
| Benchmark tốc độ | `results/rigel_cad/bench_v111/` |
| Bảng cho paper | `results/rigel_cad/paper_v111/RIGEL_CAD_TABLES.md` |
