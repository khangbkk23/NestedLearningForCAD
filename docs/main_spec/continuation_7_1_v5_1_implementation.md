# Continuation7.1 v5.1 — implementation và hướng dẫn

Lớp vận hành trong `continuation_runtime_v1/{common,entry,worker}.py`, builder riêng `build.py`. Source/model/engine/Execution của bridge r2 giữ nguyên222entries. Continuation manifest khóa thêm lớp điều phối; source identity G4 không bị đổi tên hoặc bypass. Đặc tả đầy đủ: `continuation_7_1_v5_1.md`. Chỉ triển khai7.1 NESTED_seed0; matched CADIC độc lập và7.2/7.3 không tự launch.

## Chạy trên Kaggle

1. Import `notebooks/hope_cad_v5_1_continuation_7_1_kaggle.ipynb` mới.
2. Attach **source continuation r1**, **reviewed completed bridge384 output**, MVTec AD, ViT-B/8 checkpoint cũ. Source continuation chứa core r2 byte-exact nên tháo source bridge riêng. Không attach paused376/preflight/pilot/W6 hoặc nhiều output khác nhau.
3. Chọn T4×2, kernel/session mới. Giữ PHASE7.1/JOB NESTED_seed0. SESSION_HOURS9 là soft limit,600s reserve, có thể đổi hợp lệ≤11h; không phải ETA. OverridesNone trừ khi mount ambiguous.
4. Save Version → Save & Run All. Cell đầu verify source, cell sau verify runtime/evidence/lineage, copy registered raw normals sang temporary assets rồi worker train tiếp. Không pip upgrade. Heartbeat30s, train progress mỗi8ảnh. Test/GT chỉ stage/evaluate sau train_complete.
5. `paused_at_atomic_boundary`: lưu ZIP + Kaggle log, phiên sau input **latest continuation output** thay bridge/output cũ và dùng chính notebook/source này. Giữ model/source/runtime. Mỗi session có log/config/timing/import receipt riêng, không overwrite history.
6. `7.1_complete_pending_review`: dừng, gửi artifacts review; kết quả15categories ở `benchmark/runs/NESTED_seed0/metrics/final_macro.json` và `final_per_task.json`. Checkpoints15tasks vẫn giữ để7.2 sau này không train lại. Đây chưa phải matched CADIC/fullstudy acceptance.
7. `failed_preserve_evidence`: giữ ZIP/log để review, không tự resume failed state. Kernel bị platform kill chưa có ZIP thì không coi SavedVersion đã bảo toàn checkpoint hiện tại.

Source delivery: `D:/src/DATN/.agents/artifacts/deliveries/hope_support_v5_1_continuation_7_1_r1/hope_cad_support_v5_1_continuation_source_r1.zip`.
Output: `hope_cad_support_v5_1_continuation_evidence.zip` + exported Kaggle log, đặt tại `D:/src/DATN/.agents/artifacts/runs/support_v5_1_benchmark_NESTED_seed0_continuation/session_01/`, tiếp theo session_02… Không đặt đè session cũ.

## Phạm vi kiểm chứng

Implementation và CPU/package verification được ghi trong delivery review/receipt riêng sau khi thực chạy tests. Không coi nội dung hướng dẫn là bằng chứng testsPASS. G4 T4 checkpoint384 đã nghiệm thu độc lập; continuation trên T4 và final AD metrics **chưa chạy**. Không thay đổi science, không hứa gain; timing windows cũ không là fullstream ETA. Bootstrap/restore, train/eval và export đo riêng; active elapsed G4 không là toàn cold-start wall time.
