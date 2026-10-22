# Nested Learning for CAD — Kế hoạch thực hiện v5.1

Ngày 19/10/2026. Active checkout/branch `hope_cad_v6`; successor method `hope_cad_v5_1_causal_scaled_utility`. Bộ [architecture](architecture_decision_v5_1.md), [instruction](instruction_CAD_v5_1.md), [contract](revision_contract_v5_1.json) đã đặc tả thiết kế; **code/model/notebook/pilot v5.1 chưa triển khai hoặc chạy**. Lượt thiết kế này không tự cấp quyền GPU.

## A. Trạng thái hiện tại và mục tiêu

| Hạng mục | Evidence/status |
|---|---|
| Historical v4 | Matched seed 0: M 66.80/17.02 so với CADIC 88.36/46.01% I-AUROC/P-AP; development evidence, không phải v5 benchmark |
| Implemented v5 | Source v6-r4 và 115 packaged files; technical tests/T4 preflight có receipts riêng |
| V5 normal pilot | Bốn controls đều complete 256/256; risk ratio 1.128397 > 1.10, FAIL; chưa có AD gain |
| Offline diagnosis/audit | Đã review; tách observed lag, mathematical mismatch, reliability hazards và claims bị bác bỏ; không quy causal share cho toàn gap |
| Part B proposal cũ | Weighted ratio/atanh/current-gradient bounded rule không được implement nguyên xi; giữ report lịch sử |
| V5.1 ở lượt này | Chốt một direct scaled target/linear fast readout/context-energy step, full specs và mathematical reference check; chưa production-tested |
| Successor AD benchmark/VisA/thesis | Chưa hoàn tất; chỉ planned theo gates dưới đây |

Câu hỏi nghiên cứu: liệu self-modifying context + fast/slow nested utility có cải thiện raw normal-support management so với fixed proposer mà không đánh đổi acquisition/retention quá mức? Target toán học khớp expected combined advantage hơn v5; điều đó không dẫn tới promised benchmark gain.

## B. Một revision, không mở grid

Giữ ViT-B/8/checkpoint/preprocessing/raw 768D/Eq9/K10k/image proposal/numeric route/data exclusions/anchors 32×8/task-agnostic clock/warmup 32. Chỉ đổi:

1. Readout33+bias sang persistent fast state, zero-init, p=gamma+linear residual, bỏ tanh/atanh.
2. Causal scale c=1+current±+anchor±; y=(futureJminus−futureJplus)/storedc; plain half-squared loss, không weighted ratio.
3. Fast update `.05*grad(loss)/||[h32,log1pc,1]||²`; không lấy gradient của loss đã chia hoặc dùng current gradient norm/EMA.
4. Exact meta giữ derivative của denominator; FO detach whole increment; dùng state/load/telemetry contracts mới.

Không thêm drift warmup, task reset, batching/backbone/L2/blur/patch retention hoặc gate 1.05/transition. Nếu cần thay sau run thì dùng revision mới/trial ledger; không sửa method đã freeze để cứu gate.

## C. Các chặng và điểm dừng

### Chặng D — Thiết kế (lượt hiện tại)

Deliverables: bốn full specs, scoped design verification trong shared review, ADR/change và context/snapshot mới. Acceptance: equations/ownership/shapes/clocks/causal rights không mâu thuẫn; direct scaled objective proof, denominator derivatives và first zero-head step được kiểm trên reference fixtures; old source/spec/evidence giữ nguyên. Design PASS không phải production PASS.

### Chặng I1 — Learner và cơ sở toán học executable (chưa thực hiện)

Tạo namespace v5_1 riêng; contract/schema/digest, model readout mới, loss/step/meta. Giữ visual/CMS primitives. Tests cần kiểm:

- Unbounded target/known scale/conditional counterexample; d positive trước label và bias coordinate bằng 1.
- `grad(loss)/d` khác `grad(loss/d)`; zero-head CMS gradient bằng 0 và same-context error factor .95.
- Same-context encoder gradient cancellation và varied-context dependency; full 4+4 FP64 finite differences cho slow groups, exact/FO forward parity, canonical FP32 gradient finiteness.
- Không detach d trong exact hoặc chỉ detach g trong FO. Ownership không double-update; giữ actual phi8, không thay bằng meta φ4.

Stop: sửa lỗi numerical/contract trước integration; không dùng small-toy tests làm bằng chứng về độ chính xác của model hoặc AD.

### Chặng I2 — Session, reliability và pilot driver (chưa thực hiện)

Giữ matched fork/proposer/scoring; implement causal pending và các field c/gamma/y/d, temporary-owner load validation, atomic rollback, resume/finalize. Kiểm bad shapes/keys/dtypes/nonfinite/RNG/clocks/state hashes, partial blocks 1–7 records, warmup boundary 32, no-op/bootstrap remainder, uninterrupted/resume parity, hard-crash replay tối đa 7 events và read-only scientific score.

Driver chỉ nhận train/good normals; feature cache khóa producer/checkpoint/image/tensor hashes. Dùng source/runtime/method locks, output schema và inventory mới. Reject old learned weights. Build notebook/source bundle sau khi code/tests sẵn sàng; không dùng old v5 delivery để chạy successor.

Runtime T4 FP32, không AMP/TF32; scalar FP64 theo contract. Finite-check failure → rollback/export/stop; không clip/drop/retry/fallback. Thêm telemetry đã khóa trong cùng run, không dựng thêm GPU diagnostic stage.

Stop tại bàn giao: source ZIP/notebook/receipts mới được review, chưa tự run GPU. Report local PASS kèm shape/device/dtype/pass/fail/skip; ghi rõ canonical evidence còn pending.

### Chặng P — Một canonical normal pilot sau bàn giao (chưa chạy)

User chạy Saved Version trên Kaggle, T4; ba inputs mới: successor source, MVTec archive và locked ViT. Default pilot tích hợp 24 discarded normal preflight, rồi 4 fresh jobs×256 images, seed 0. Khởi đầu không dùng learned parent. Khi soft pause, attach latest output cùng source/runtime/protocol và resume; không reset hoặc gọi đó là trial mới.

Preflight kiểm real ViT geometry, K10k, CUDA full 4+4/readout denominator path, forced-PASS raw bank/Eq9, checkpoint/resume và resource profile. Clone 24 ảnh bị discard. Preflight fail thì stop trước pilot; technical PASS không phải scientific PASS.

Pilot có bốn controls CADIC/GEOMETRIC/FIRST_ORDER/NESTED; cùng 179 Bottle+77 Cable identities, matched learned initialization/seed, một trial, không grid. Giữ source/spec file hashes, full commands/runtime/cursor/resume chain và failure ZIP/log. Session hours là soft-pause budget, không phải ETA; resume phải giữ đúng source/runtime/protocol/method identity. Chuyển tài khoản không nới khóa runtime; mismatch phải dừng, không tự migrate parent bằng một ghi chú parity.

### Chặng R — Nghiệm thu pilot (chưa thực hiện)

| Gate | Quy tắc khóa trước run |
|---|---|
| Technical | Source/data/runtime/parent/topology/finite/causality/parity/resume hợp lệ |
| Learning activity | NESTED≥64 completed labels, ≥8 slow updates; mode-aware finite gradient paths |
| Nontrivial policy | Post-warmup eligible reject fraction [.01,.99], final bank khác CADIC, reused-normal patch-score delta>1e−6 |
| Normal-risk safety | Mean last 128 finite **pre-write actual-bank R** <=1.10×CADIC; không anchors/accepted-only/window changes |
| Claim boundary | PASS chỉ đủ để review đề xuất AD benchmark; cùng FO actions không chứng minh higher-order benefit |

Gắn nhãn riêng cho ratio FAIL/no-effect/technical fail; lưu artifacts, không tăng time/rate/threshold hoặc bỏ transition để cứu kết quả. Pause chưa phải complete; thiếu artifacts thì chưa acceptance. V5 counts 242/30 không phải target bắt successor phải giống, vì no-op/proposal trajectories có thể khác.

Trong cùng pilot: response probes ở labels 1,33,65,97,129,161,193,225; saved actual meta episode 33–40. Review current/next/anchor, prior/residual, c/y/d, actual step norms và action/bank changes. Không đồng nhất gradient energy với Adam-step contribution; không claim causal share từ correlation.

### Chặng B — Full MVTec development benchmark (planned sau R)

Chuẩn bị explicit v5_1 evaluator, test image/mask hashes, raw/resized mask QA, undefined metric policy, correct finalize và read-only trace; tránh trộn legacy CLI v1/v2. Canonical metrics: `macro_final_i_auroc`, `macro_final_p_ap`; legacy p_aupr alias=average precision phải map rõ. ReplayCAD unit parser phải explicit trước so sánh.

Chạy full methods/controls/seeds 0/1/2 và task matrix; FM=prior peak−final, BWT=final−diagonal, không trộn định nghĩa. Các ablations FAST_ONLY/NO_SELF_MODIFICATION/NO_ANCHOR_OBJECTIVE phải có executable/tests. NO_FAST_ADAPTATION với zero head là geometry admission trên own trajectory ngoài learned warmup, không phải independent slow baseline hoặc toàn policy GEOMETRIC tương đương. Không chọn control sampler sau khi nó thua.

Giữ cấu trúc bàn giao W7-type nếu được duyệt: seed 0 final view dùng saved checkpoints; thêm full matrix bằng evaluation, không train lại; rồi các registered jobs/seeds/controls còn lại. Successor khác method v4/v5, không resume learned checkpoints của chúng. Executable orchestration/source phải freeze trước anomaly test.

Primary I-AUROC/P-AP; NI 2 pp cho cả hai, superiority đòi cả hai lower CI>0. Dùng paired whole-image bootstrap theo category/label strata, 2000 replicates, percentile 1.25/98.75, cùng draws cho methods/paired seeds. Aggregate của ba finite seeds không phải population-seed uncertainty. Chốt chi tiết deterministic sampler seed/hash cùng evaluator trước benchmark; không chọn CI method theo kết quả.

Matched CADIC K10k/main protocol là đối chứng chính. K24576 payload control còn planned; phải đo actual total bytes trước fair-memory claim. ReplayCAD của Khang chỉ được so sánh chính thức khi có matched protocol/source/assets/artifacts; không đồng nhất một log với toàn bộ reproduction.

### Chặng V/T — VisA và thesis (planned)

Khóa method/source/transfer/data/mask/scoring/metrics/seeds/budget trước VisA test; dùng same constants và fresh causal state, không retune. Công khai prior exposure và normal-development rights. Chạy matched controls/CI, sau đó viết thesis/reproducibility package.

Thesis phải gồm background CAD/NL/HOPE/VisionHOPE, related work, hai cấp utility learning, conditional proofs/limits, negative v4/v5 results, trial history, matched benchmarks/ablations/uncertainty, resource cost và future work. Không cần thêm NSP2/CBP/TITANS để đúng tên đề tài; contribution phải được controls/evidence hỗ trợ.

## D. Giả thuyết và điều bác bỏ

| Giả thuyết | Phép đối chiếu | Giới hạn kết luận |
|---|---|---|
| Causal scaled target tránh ratio sign mismatch | Conditional toy + finite differences + arrival rights | Chứng minh objective/step, không convergence hoặc AD |
| Fast readout phản ứng rõ hơn | Zero-head .95 proof, actual functional-response probes | Không suy rằng toàn bộ acquisition lag đã được sửa |
| Learned utility tốt hơn geometry | NESTED so với GEOMETRIC/CADIC, risk/bank/AD | Nếu giống geometry thì không claim NL gain |
| Exact nested credit có ích | NESTED so với FO, valid ablations và paired results | Gradient khác 0 chưa đủ; cùng policy là negative information |
| Retention regularizer giúp | NO_ANCHOR_OBJECTIVE + past coverage/task matrix | Common training anchors không phải independent holdout |
| Visual self-modification có ích | NO_SELF_MODIFICATION với cùng data/ownership | Không quy mọi gain cho VisionHOPE |

## E. Tài nguyên và trách nhiệm

Không EDA/Kaggle/GPU trong design step. Reference CPU fixtures không train model; chúng dùng để bắt math/graph mistakes. Kế hoạch không đưa ETA khi chưa có số đo.

Raw bank 10k×768×4=30,720,000 B; pending alternative cùng kích thước; anchors 32×8×768×4=786,432 B; tám projected contexts 8×784×32×4=802,816 B. Ngoài ra phải đo θ/φ/readout33+bias, phi_start/Adam/labels/coefs/private RNG/cache và transactional peaks. Không có fast EMA state mới; readout/chuyển ownership không làm overhead biến mất.

Agent phụ trách spec/code/tests/config/notebook/bundle/receipts và review theo từng chặng được giao. User chạy canonical notebook khi bàn giao, giữ Saved Version và tải ZIP/log; quyết định revision nếu gate FAIL. Không giấu kết quả âm hoặc lặp grid tới PASS. Mục tiêu là hoàn thành đồ án bằng evidence đủ mạnh, không bằng lời hứa vượt CADIC.
