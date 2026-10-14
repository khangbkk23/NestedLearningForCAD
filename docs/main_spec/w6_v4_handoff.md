# Chạy preflight và W6 v4 — Nested Learning for CAD

Code/spec v4 đã chuẩn bị. Chưa có nghiệm thu W6 v4; không chạy W7 bằng notebook này.

## 1. Lượt preflight mới

1. Upload `ultils/hope_cad_v4_source_bundle.zip` thành Kaggle Dataset mới. Kaggle
   giải nén ZIP cũng được: notebook kiểm `bundle_manifest.json` và từng file hash.
   Không dùng source bundle v3 cũ; chỉ attach một source v4.
2. Import `notebooks/hope_cad_v4_preflight_kaggle.ipynb`.
3. Add Input: source v4, MVTec AD archive/root, checkpoint
   `vit_base_patch8_224.augreg_in21k.safetensors` SHA256
   `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`.
4. Tùy chọn attach đúng **một** output/ZIP W6 v3 cũ để tái dùng `prefix_features.pt`.
   V4 chỉ sao chép file frozen này, kiểm parent manifest/hash rồi kiểm contract/data
   và RGB parity. Không nhập selection, initialization, probe hay resume v3.
   Không có parent vẫn chạy được: cache được trích xuất mới bằng frozen ViT.
5. Chọn T4, Internet ON. Dùng **Save Version → Save & Run All**, bật lưu outputs.
   Dự trù ban đầu 20–45 phút để bố trí; chưa phải timing đã đo của v4.
6. Kiểm saved Output có `hope_cad_v4_normal_evidence.zip` và `handoff.json` với
   `status=preflight_passed`. Failure/pause phải lưu evidence; không bắt đầu fit
   vì chỉ thấy notebook chạy hết cells.

## 2. Lượt W6 chính

1. Import `notebooks/hope_cad_v4_normal_kaggle.ipynb`.
2. Giữ đúng source v4, MVTec và ViT. Add Input → Notebook Output của đúng
   **preflight v4 PASS** vừa chạy, hoặc ZIP evidence của nó. Chỉ attach một v4
   resume source. Output preflight v3 không thay preflight v4.
3. Giữ `METHODS=['M','C0','C1','C2','C3','C4','A1']`, `SESSION_HOURS=10.5`.
   Ngân sách tính từ first code cell, chỉ là thời gian một session, không thay
   400-step fits/gate hoặc chọn lại recipe. Notebook dùng một GPU; T4x2 không tự
   chia đôi thời gian. Không đổi thành 24h để vượt session limit.
4. Save Version → Save & Run All. Recipe mới có cap 400, warm validation và
   numerical derivatives khác: không hứa hoàn tất trong 8h11m như v3. Xem timings
   preflight/commands thực tế; có thể cần nhiều saved sessions.
5. Nếu `status=paused_for_session_limit`, tải/giữ ZIP output mới nhất rồi attach
   output đó cho lượt tiếp, cùng source/runtime/methods. Fit resume từ atomic
   completed-step checkpoint; gate từ atomic 16-event boundary. Đây là cùng run,
   không reset, không chọn lại candidate. Partial-period state không flush.
6. Khi `ready_for_artifact_review`, tải `hope_cad_v4_normal_evidence.zip` và log
   về `docs/w6_v4_kaggle/`. Nếu `stopped/selection_failed`, tải failure evidence
   để audit; không bật extra rates, đổi guards hoặc dùng probe cứu selection.

## 3. Kết quả cần kiểm tra

ZIP gồm immutable protocol/baseline lock, nguồn/runtime, frozen-cache parent
receipt, tests/resource profile, real RGB + outer optimizer/RNG resume, toàn
400-step learning curves, loss validation tại 0/16, shortlisted/rejected fits,
1024-event gates và failures, selected recipes/zero-clock initialization, các
normal-probe controls và interventions, center/border spatial readout, logs và
resource/read accounting. Manifest ghi SHA256 từng entry.

Probe vẫn chỉ đọc sau selection. Vì đã xem W6 v3/probe để phát triển v4, không
gọi nó confirmation độc lập. Numerical PASS không bảo đảm AD accuracy. Sau review
W6 v4 mới quyết định chuyển W7; W7 sẽ chạy matched CADIC, locked methods/controls,
seeds và task matrix. Không tự sửa kiến trúc theo anomaly-test feedback.

Nếu file inputs không được auto-detect duy nhất, dùng biến override tương ứng
ở first code cell; không sửa business logic trong notebook. Không trộn nguồn v3
và v4 hay restore v3 learned state vào v4.
