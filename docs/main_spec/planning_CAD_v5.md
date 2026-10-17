# Nested Learning for CAD — Kế hoạch Stage 2 / v5

Ngày 17/10/2026. Chỉ triển khai tại worktree `hope_cad_v5`; không merge hoặc sửa sibling. Architecture v5 và JSON contract khóa quyết định trước pilot. Negative evidence v4 giữ nguyên.

## A. Tiến độ và bằng chứng

| Phần | Trạng thái |
|---|---|
| Matched CADIC10k, C1/A1/M seed 0 | Recovery completed và reviewed; gap M lớn, không cần rerun gói để chẩn đoán |
| W6 v4, W7.1 và W7.2 | Giữ artifacts/source/weights; full W7.3 chưa hoàn tất |
| Bounded normal causal diagnostic | Reviewed 119 payloads, 12 capacity fits, 48 oracle fits, continuation đến 500; có headroom và interference, chưa AD gain |
| Architecture/spec v5 | Frozen for implementation theo yêu cầu Stage 2 ngày 17/10 |
| Code/tests/pilot v5 | Xem `v5_implementation.md` và receipts; code đã có không đồng nghĩa pilot được nghiệm thu |
| Anomaly benchmark v5, VisA, thesis | Chưa hoàn tất; chưa có performance gain được xác nhận |

## B. Công việc và acceptance

1. **Raw geometry và proposer.** Module riêng fork chính xác cached CADIC, giữ raw score và Eq9. Kiểm ties, duplicates, fill/full transitions, cache parity và parent immutability. Không sửa baseline registry cũ.
2. **Utility learner.** Fixed projection → self-modifying visual context → fast/slow CMS → utility head. Kiểm exact bốn support/bốn query gradient, FIRST_ORDER cùng forward, ownership, finite differences và đường ảnh hưởng tới bank/score thật. Không tái áp decoder 768 chiều.
3. **Causal atomic session.** Reward từ next normal và cached historical anchors; hai banks cố định trong thời gian chờ. Kiểm terminal censor, không flush ở task boundary/pause, score read-only, rollback toàn optimizer/RNG/episode và uninterrupted/resume parity.
4. **Protocol và runner.** Cùng 3.599 online train identities với matched protocol; một image mỗi event. Bốn pilot controls × 256 images, cache/checkpoint hashes, canonical runtime, soft deadline và failure export. Không discover hoặc evaluate official test trong normal pilot.
5. **Canonical preflight.** Kaggle T4, 24 real normal images, K10k shapes, full-grid nested gradient, proposer/Eq9 parity, resume và resource profile. Clone bị discard trước pilot. Local fixtures không thay nghiệm thu GPU này.
6. **Pilot một trial.** CADIC/GEOMETRIC/FIRST_ORDER/NESTED. Technical, mechanism và normal-risk gates theo architecture §7. Nếu fail, lưu evidence và review trước mọi method revision; không tự sửa threshold hoặc tăng budget.
7. **Sau pilot acceptance.** Chuẩn bị full anomaly benchmark orchestration và toàn bộ controls; lock source/spec/protocol/endpoints trước test. Pilot không tự chạy benchmark.

## C. Giả thuyết và ablation

| Giả thuyết | Đối chiếu cần có | Điều không được suy ra |
|---|---|---|
| H1: raw support và coverage utility tránh reconstruction bottleneck v4 | Geometry/score locks và matched CADIC | Không tự bảo đảm AD accuracy |
| H2: exact nested utility có ích | GEOMETRIC và FIRST_ORDER; bank/score effect, normal risk, AD endpoints và CI | Always-PASS không là learned contribution |
| H3: anchors giảm harmful replacement | NO_ANCHOR_OBJECTIVE, past coverage và task matrix | Finite anchors không chứng minh global retention |
| H4: self-modifying context có ích | NO_SELF_MODIFICATION, công khai inactive capacity | Không quy mọi gain cho VisionHOPE |
| H5: fast/slow coupling có ích | FAST_ONLY, NO_FAST_ADAPTATION, FIRST_ORDER | Một contrast không chứng minh toàn bộ Nested Learning |

Core modes ngoài bốn pilot jobs phải có tests và full runner trước claim liên quan. Noninferiority margin hai điểm phần trăm trên cả I-AUROC và P-AP theo CI đã khóa; không đổi margin sau khi thấy kết quả. Superiority cần cả hai lower bounds lớn hơn 0. Negative result hoặc tradeoff vẫn phải báo trung thực.

## D. Tài nguyên, dữ liệu và chia phiên

Pilot không có grid 16 fits × 400 steps như W6; mỗi job bắt đầu mới và chỉ đi 256 normal images. Chưa có ETA đáng tin trước real profile. Jobs chạy tuần tự trên một T4 để đơn giản hóa audit/resume; T4 ×2 không cộng VRAM cho worker này.

Main raw bank K10k có 30,72 MB; alternative pending bank thêm 30,72 MB. Anchors 32 ×8 ×768 ×4 khoảng 0,786 MB; tám fixed projected contexts khoảng 0,803 MB. Còn parameters, fast state, Adam, episode start, pair cache và bookkeeping. Full memory claim phải đo mọi thành phần; không chỉ nhìn K. CADIC K24.576 là conservative total-payload control đã đề xuất cho full benchmark, chưa có pilot performance của nó.

Proposer vẫn chạy trước admission nên gate không tiết kiệm chi phí proposer. Một frozen ViT forward mỗi normal image không loại chi phí context, delayed reward, NN và meta-learning. Pilot elapsed time bao gồm cache/IO và thứ tự warmup; không dùng nó như fair compute benchmark.

Người dùng chạy notebook với source v5/MVTec/ViT bằng saved version và giữ ZIP/log. Agent chuẩn bị code, tests, specs, runner và review. Dừng bàn giao ở GPU evidence pending; nếu scientific gate fail thì review, không tự mở anomaly test.

## E. Đường tới hoàn thành đồ án

1. Review preflight và pilot v5, ghi pass/fail đúng phạm vi.
2. Khóa một executable matched benchmark plan; hoàn thiện adapter orchestration, controls, resource accounting và uncertainty.
3. Chạy MVTec seed 0 progress view và task matrix; giữ checkpoint để tiếp tục các seeds/controls đã đăng ký.
4. Hoàn tất full development evidence, limitations và reproducibility package. Không đổi method theo test rồi gộp vào cùng frozen result.
5. Freeze transfer/initialization/data/metrics trước VisA; fresh causal state, cùng model và rates, không tune bằng VisA test.
6. Chạy confirmation VisA và viết thesis: nền tảng toán, paper comparison, hypotheses, methods, kết quả, negative findings, ablations, chi phí và giới hạn.

Full W7 v4 không là benchmark v5. Source IDs, artifacts và lineage phải tách rõ. Nếu cần revision tiếp, dùng version mới và exposure ledger; không ghi đè v5 rồi gọi là preregistered trước MVTec.
