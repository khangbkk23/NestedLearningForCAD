# CADIC — đối chiếu chi phí NN-write trên T4

## Câu hỏi và phạm vi

Đo CADIC K10k reference/optimized trên cùng16 normal arrivals179–194 như G3 NESTED. Không đổi protocol3599, không test/GT/VisA, không chỉnh phương pháp. Đây là engineering measurement, không phải reproduction paper hoặc benchmark accuracy. Source cũ và checkpoint376 không bị sửa; không adopt clone195.

CADIC cần predecessor riêng: chạy reference CADIC seed0 trên179 normals0–178 một lần, theo đúng manifest đã khóa. Không dùng bank NESTED, không force-pass một checkpoint NESTED, không giả clocks. Chi phí PREP179 được báo riêng. Sau đó bốn fresh worker processes R1/O1/O2/R2 load cùng predecessor179 với derived caches cold. Mỗi pass16 ảnh, feature receipts phải exact với reviewed parent NESTED vì cùng raw bytes/backbone/preprocessing.

## Timing và parity

Dùng bounded_replay của profiler gốc: bao feature extraction/normal-risk/proposal/update/events/hash/memory accounting và checkpoint ở184/192, synchronized đầu/cuối. Bootstrap/parent verification/predecessor load/final audits/export báo riêng. CADIC vẫn là matched harness (có normal-risk), không là pure paper kernel. Không learner capture/meta recomputation vì CADIC không có learner.

R/O giữ distance math, tile256, K10000, strict comparison, seeds và chronology. Event-by-event và full scientific endpoint195 phải exact giữa bốn lượt; kiểm RNG và no-finalize. Báo replacements để thấy trajectory CADIC khác NESTED. Peak PyTorch allocated/reserved riêng, không đồng nhất tổng VRAM. Không tự bỏ anchors/learning/proposals của NESTED để tăng tốc.

Speed gate: cả hai paired O/R<1, mean saving vượt relative-noise envelope như G3; memory gate candidate peak≤reference peak+64MiB. Inconclusive vẫn xuất toàn bộ kết quả, không retry để kiếm speedup. Hai pairs không là confidence interval.

So với historical NESTED G3 bằng mean seconds, ratio và delta; pin source ZIP/review/measurement SHA và runtime. Đây là cross-session descriptive comparison, không paired NESTED–CADIC: GPU tải/clock có thể khác, bank trajectories khác, NESTED timing loại bốn diagnostic capture intervals đồng bộ trong khi CADIC không capture. Không suy ETA3599 hoặc overhead% toàn stream từ16 ảnh chuyển category.

## Budget và outputs

Cumulative soft180min gồm180s export reserve, không ETA. Tối đa243 arrivals =179PREP+4×16. Durable reservation trước mỗi worker; không retry/resume tự động hay nối partial timing. Partial/failure giữ ZIP/log để review. Deadline kiểm giữa arrivals; một atomic event có thể vượt soft boundary; hard platform kill có thể ngăn export.

Notebook `notebooks/hope_cad_v5_1_cadic_cost_kaggle.ipynb`. Bốn inputs: source CADIC-cost mới; original reviewed NESTED_seed0 paused376 evidence; MVTec raw; locked ViT-B/8 safetensors. Tháo profiler/NN-validation/bridge source và outputs khác. GPU T4×2, canonical runtime cũ, no pip upgrades; cuda:0 thực hiện công việc, không pool GPU.

Save Version → Save & Run All. Output `hope_cad_cadic_cost_evidence.zip` và log đặt ở `D:/src/DATN/.agents/artifacts/runs/support_v5_1_cadic_cost_01/session_01/`. Complete cần review trước kết luận. Không phải bridge G4, không mở full benchmark. Tình trạng bridge/continuation đọc riêng từ status hiện hành; phép đo này không ghi đè publication của benchmark.
