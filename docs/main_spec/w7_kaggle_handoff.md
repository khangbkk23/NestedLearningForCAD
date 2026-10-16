# Hướng dẫn W7: preflight → 7.1 → 7.2 → 7.3

Đọc `w7_implementation.md` để biết contract và acceptance. Không dùng runner v3
hoặc source ZIP W6 cho W7. Model/recipe vẫn là HOPE-CAD v4.

## Lượt đầu — short preflight

1. Upload **`ultils/hope_cad_w7_source_bundle.zip`** thành Kaggle Dataset mới;
   source ZIP còn nguyên hoặc được Kaggle giải nén đều được notebook kiểm hashes.
2. Import `notebooks/hope_cad_w7_preflight_kaggle.ipynb`.
3. Add Input: source W7 mới; **completed W6** `hope_cad_v4_normal_evidence.zip`
   trong `docs/w6_kaggle_v4.1`; MVTec AD archive/root; ViT-B/8 ImageNet21k cũ.
   Không attach nhầm paused W6 hoặc preflight W6 có cùng filename.
4. Chọn cùng T4/runtime như W6. Internet chỉ cần nếu Kaggle yêu cầu để khởi tạo
   environment; extractor chỉ dùng local checkpoint, không tự download model khác.
5. Save Version → Save & Run All. Đây là tests + 9 normal images, không W6 refit.
6. Output phải có `handoff.status=w7_preflight_passed`, `preflight.passed=true` và
   ZIP `hope_cad_w7_evidence.zip`. Nếu fail, giữ ZIP/log để review, không mở test.

## 7.1 — lần có final metrics để so sơ bộ

1. Import `hope_cad_w7_7_1_kaggle.ipynb`.
2. Giữ bốn inputs ở trên. Add Input → Notebook Output của W7 preflight vừa PASS,
   hoặc đúng ZIP W7 của nó. Chỉ attach một previous W7 output nếu auto-detect.
3. Giữ `PHASE='7.1'`, `JOB='M_seed0'`. Budget mặc định 10.5 giờ tính từ first
   code cell, có export reserve. Không đổi thành 24 giờ để mong vượt platform limit.
4. Save & Run All. Nó train đầy đủ normal stream, lưu mọi task checkpoint, rồi
   evaluate final state trên 15 categories. Không search lại hoặc train W6 nữa.
5. Nếu `paused_at_atomic_boundary`: attach output mới nhất và chạy lại cùng
   notebook/source/JOB. Resume từ event hoặc score shard đã commit.
6. Nếu `7.1_complete_pending_review`: lưu ZIP/log; gửi review. Final metrics tại
   `runs/M_seed0/metrics/final_macro.json` và `final_per_task.json`.
   Đây là preliminary quick comparison, chưa matched claim với historical CADIC.

## 7.2 — thêm matrix từ checkpoint sẵn có

1. Import `hope_cad_w7_7_2_kaggle.ipynb`.
2. Attach completed 7.1 Output/ZIP, cùng source W7, completed W6, MVTec và ViT.
3. Giữ `JOB='M_seed0'`. Chạy; chỉ thêm 105 task-evaluations còn thiếu, không train.
4. Pause thì resume latest 7.2 output; complete thì giữ ZIP/log. Matrix và F/BWT
   ở `runs/M_seed0/metrics/`; final row cũ vẫn giữ nguyên.
5. Không đổi model sau khi thấy metrics rồi coi đây là cùng frozen benchmark.

## 7.3 — chạy từng registered job còn lại

1. Import `hope_cad_w7_7_3_kaggle.ipynb`. Đọc registry trong `suite_lock.json` /
   `training/hope_w7_contract.py::jobs` để chọn JOB.
2. Attach completed 7.2 Output (hoặc một W7 output khác có cùng preflight/suite),
   completed W6, source, MVTec và ViT. Một job mới chỉ nhập shared locks/preflight;
   không nhập terminal state của M vào C0 hoặc seed1.
3. Ví dụ `JOB='C0_seed0'`, `C1_seed0`, `M_seed1`, `C1_seed1`, `A1_seed1`,
   `CADIC10000_seed0`, `CADIC2500_seed0`, `M_seed0_order100`.
4. Seeds1/2 cần prefix cache từ completed W6. Refit dùng đúng recipe, có thể
   pause trong calibration/gate/train/eval; không phải search W6 thêm lần nữa.
5. **A1 seed1/2:** chạy C1 cùng seed trước; attach C1 Output ZIP và đặt
   `C1_EVIDENCE_OVERRIDE` tới ZIP đó. Đặt `RESUME_OVERRIDE` rõ ràng nếu nhiều W7
   outputs được attach. A1 không được tự fit C1 lần thứ hai hoặc lấy C1 seed khác.
6. Lưu từng output theo JOB ở local/server; không ghi đè ZIP cũ chỉ vì tên file
   tải về giống nhau. Ví dụ folder `M_seed1/`, `CADIC10000_seed0/`.
7. `7.3_job_complete_pending_suite` nghĩa là **một job xong**, chưa full W7 xong.

## CPU report và kết thúc W7

Chạy `report_w7.py --evidence <các ZIP W7> --output-dir <report dir>` trên CPU;
`--inventory-only` kiểm jobs/matrices còn thiếu mà chưa bootstrap. Tool đọc ZIP
trực tiếp, không cần giải nén toàn bộ raw maps. Cần giữ tất cả outputs để chạy
paired statistics; không chỉ copy bảng macro.

Report phải đủ 35 registered streams (M seed0 reused), complete matrices, exact
point metrics, seed mean/std, order studies và paired CIs. Pixel bootstrap có
32-replicate accuracy gate; fail thì giữ kết quả/đổi sang exact CPU reporting,
không đổi model hoặc bỏ endpoint. Không cần GPU để tính CI.

Kết thúc W7 là đã chạy và review **thật** các jobs/statistics, gồm matched CADIC;
không còn một bước “chạy baseline chính sau W7” riêng. ReplayCAD theo tiến độ
Khang là baseline bổ sung có native/protocol disclosure. Sau đó mới W8 VisA
fresh-prefix với recipe khóa, rồi W9 báo cáo/repro.

## Inputs/lỗi thường gặp

- Có nhiều W7 ZIP: dùng `RESUME_OVERRIDE` và `C1_EVIDENCE_OVERRIDE`, không chọn
  tùy tiện output có metrics đẹp hơn.
- Runtime mismatch: notebook fail sớm. Giữ log; cần parity review, không xóa lock.
- Hash mismatch: dùng đúng source/data/checkpoint. Không sửa hashes bằng tay.
- Session pause không phải failed scientific gate. Failed refit/gate là failure
  thật, giữ nguyên và review trước mọi thay đổi recipe.
- Save Version lưu server outputs khi phiên kết thúc; tải/giữ version Outputs
  và ZIP/log. Không dựa vào RAM hoặc một interactive session còn mở.
- Thời gian W7 phải dựa trên preflight/actual progress, chưa có benchmark timing
  đo được ở lượt triển khai code này. Một phần có thể cần nhiều saved sessions.
