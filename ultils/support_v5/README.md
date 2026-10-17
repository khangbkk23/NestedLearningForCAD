# Chạy pilot HOPE-CAD v5 trên Kaggle

Đây là revision mới của **Nested Learning for CAD**: learned utility điều khiển normal-support bank, raw ViT +CADIC Eq9 quyết định anomaly score. Không chạy lại W6 hoặc full W7 cũ.

## 1. Tạo notebook và inputs

Import `notebooks/hope_cad_support_v5_kaggle.ipynb`. Chọn accelerator **GPU T4 x2**. Attach đúng ba inputs:

1. Source mới `ultils/support_v5/hope_cad_support_v5_source_v1.zip`, upload dưới dạng Kaggle Dataset; giữ nguyên ZIP hoặc để Kaggle giải nén đều được.
2. MVTec AD archive đang dùng: `mvtec_anomaly_detection.tar.xz`.
3. ViT-B/8 ImageNet21k: `vit_base_patch8_224.augreg_in21k.safetensors` có SHA256 `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`.

Tháo source r4, normal-causal source và W6/W7/recovery evidence cũ khỏi notebook này. V5 không khởi tạo từ learned weights v4. Không cần Internet để tải thêm packages.

## 2. Chạy

Giữ `PHASE='pilot'`, `SESSION_HOURS=3.0`, các overrides=None. Dùng **Save Version → Save & Run All**. Bootstrap phải nhận schema `hope_support_v5_source_v1`, runtime T4 đúng lock và checkpoint đúng SHA. Nếu lỗi, không pip upgrade hoặc bỏ guard.

Pilot tự chạy preflight trên 24 real normals và discard clone; sau đó chạy CADIC, GEOMETRIC, FIRST_ORDER, NESTED, mỗi job 256 normals. Chỉ train/good được extract; không anomaly test/GT. Worker dùng một T4; không giả định hai GPU cộng VRAM. Chưa có ETA đo thật; ba giờ là soft budget có reserve export, không hứa hoàn thành trong ba giờ.

Nếu muốn tách kiểm trước, chạy `PHASE='preflight'`. Sau PASS, attach output preflight v5 và chuyển `PHASE='pilot'`. Không bắt buộc tách hai phiên.

## 3. Pause, failure và kết quả

- `paused_at_image_boundary`: giữ output/ZIP mới nhất; attach nó vào lần sau cùng ba inputs, tháo output v5 cũ hơn và chạy cùng notebook/source. Không làm lại từ đầu, không finalize pending ở pause.
- `preflight_passed`: chỉ preflight đã xong; tiếp pilot nếu đã chọn preflight riêng.
- `pilot_complete_pending_review`: bốn jobs xong và gates đạt ở phạm vi normal pilot; gửi ZIP/log để review trước benchmark.
- `pilot_complete_scientific_gate_failed`: finite pilot xong nhưng mechanism/risk gate fail; vẫn giữ kết quả, không tự đổi threshold hay chạy anomaly benchmark.
- `failed_preserve_evidence`: lỗi kỹ thuật; giữ ZIP/log để xác định nguyên nhân. Failed bootstrap chưa có pipeline lock không là resume parent hợp lệ.

Saved output chứa `hope_cad_support_v5_pilot_evidence.zip` và thư mục `hope_support_v5_pilot`. Tải ZIP và log về `docs/support_v5_kaggle/` trong worktree này, giữ saved version trên Kaggle. Nếu hard platform kill trước export, cần kiểm saved checkpoint/output thực tế; không có bảo đảm tự lưu mọi RAM.

## 4. Acceptance và ranh giới

Một trial, không grid/search. Chỉ normal data và causal delayed labels. Gates khóa ở architecture §7/JSON contract; không đổi sau khi xem kết quả. Pilot PASS chưa chứng minh ngang hoặc hơn CADIC, chưa là VisA confirmation. Full benchmark runner/ba seeds/task matrix/equal-total-memory control là bước sau review.

Local evidence: 26 v5 tests và 58 dependency regressions PASS. `delivery_receipt.json` ghi source SHA/digest cuối; `delivery_verification.json` kiểm từng packaged payload, 242 file r4 giữ nguyên và cả hai bootstrap formats. Đây là engineering evidence, không là benchmark.
