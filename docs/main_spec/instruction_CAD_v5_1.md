# Nested Learning for CAD — Hướng dẫn đầy đủ v5.1

Ngày 19/10/2026. Method ID `hope_cad_v5_1_causal_scaled_utility`. **Specified for implementation; chưa có production implementation/tests hoặc nghiệm thu thực nghiệm.** Đọc cùng [architecture](architecture_decision_v5_1.md), [planning](planning_CAD_v5_1.md), [JSON contract](revision_contract_v5_1.json). Tài liệu này không biến code v5 hiện có thành v5.1.

## 1. Bài toán, mục tiêu và lịch sử cần biết

Tên đồ án là **Nested Learning for Continual Anomaly Detection (CAD)**. Mỗi category đến tuần tự chỉ có ảnh normal để học; inference phát hiện ảnh/patch bất thường trên các category đã quan sát, không có task routing. Cần cân bằng khả năng tiếp nhận normal patterns mới, giữ patterns cũ và ngân sách bộ nhớ cố định.

CADIC cung cấp baseline: frozen feature extractor, patch coreset tăng dần và nearest-neighbor anomaly scoring. Phương pháp nghiên cứu dùng HOPE-derived visual self-modification, persistent fast CMS và slow meta-learning để học **utility của bank proposal**. Quyết định của learner thay đổi actual bank, nên có đường ảnh hưởng tới prediction thật. Controller không cần chạy trên ảnh test.

V4 reconstruction đã có kết quả matched seed 0 kém CADIC; giữ làm negative evidence, không dùng làm số của v5/v5.1. V5 raw-support controller đã được implement và qua technical preflight, nhưng normal pilot thất bại: NESTED/CADIC risk ratio 1.128397 > 1.10. Fast response yếu và acquisition transient là những quan sát đã có; normalized ratio objective và gradient normalization có các giới hạn toán học. V5 chưa có AD benchmark. Audit mới không chứng minh mọi veto đều vô ích hoặc bắt buộc phải đổi backbone.

V5.1 giữ bài toán và raw geometry, thay readout/objective/update rule để loại các mismatch cụ thể đó. Chưa có bằng chứng revision này sửa được toàn bộ lag, giữ memory tốt hơn hoặc đạt CADIC. Các proof dưới đây là cơ sở khoa học để thử một revision với budget hữu hạn, không phải dự đoán kết quả benchmark.

## 2. Nguồn chuẩn và trạng thái

| Câu hỏi | Nguồn quyết định |
|---|---|
| V5.1 phải được implement thế nào? | Bốn file v5_1 trong bộ này; equations trong architecture, constants/state keys trong JSON |
| Code hiện tại thực sự chạy gì? | Active checkout `D:/src/DATN/hope_cad_v6`, namespace `models/hope_cad_v5`, HEAD và source hashes tại thời điểm đọc |
| Đã chạy thành công/thất bại gì? | Original ZIP/log và scoped review tại `D:/src/DATN/.agents/artifacts` / `evidence/reviews`; không lấy status lúc build làm current runtime truth |
| Vì sao đổi thiết kế? | Offline diagnosis ngày 18/10 và external audit ngày 19/10, ADR/change tương ứng trong shared `.agents` |
| Làm tiếp ở đâu? | Worktree/branch `hope_cad_v6`; primary giữ shared Git/Khang reference, không sửa sibling |

Full historical specs/snapshots là bất biến. Current source có thể dirty vì docs mới: HEAD một mình không định danh dirty payload. Old source ZIP v6-r4 vẫn là **v5 delivery**; không upload lại nó như một gói chạy v5.1. Source bundle mới chỉ được build sau implementation và verification.

## 3. Input, output và data rights

Raw RGB dùng PIL direct bilinear 224×224, ImageNet mean (.485,.456,.406)/std (.229,.224,.225), frozen ViT-B/8 ImageNet21k `vit_base_patch8_224.augreg_in21k`, block 8 zero-based, bỏ CLS. Checkpoint SHA256 `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`. Features FP32 [1,784,768], không thêm LN/L2.

`observe(raw_features, stable_identity)` chỉ nhận đúng một normal image đã đến. Stable relative path/hash/cursor kiểm order và resume; category/task ID không vào model. Frozen feature cache chỉ hợp lệ với cùng image bytes, backbone/preprocessing/producer và tensor hashes. Caching toàn file không cho learner nhìn future features hoặc tổng hợp future statistics.

MVTec order: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper. Cùng matched protocol với 3,599 online identities; giữ 30 excluded Bottle normals cho mọi control. V5.1 không học offline trên prefix/validation/probe, không search. Normal pilot mới là development repeat sau exposure, không phải independent holdout.

VisA planned order: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum; dùng split_csv/1cls.csv, giữ rule exclusions của matched protocol. Chưa được evaluate VisA test để chọn revision. Mỗi method/seed/dataset dùng fresh state, không dùng v4/v5/pilot learned initialization.

Output inference gồm patch NN distances, map 28×28→224×224 bilinear và complete Eq9 image score. Test labels/masks thuộc evaluator. Score không resolve pending hoặc cập nhật learner/bank/RNG. Preprocessing/metric QA không được tùy ý drop tiny defects, đổi label, flip score hoặc normalize theo category test.

## 4. Mô hình và cơ sở toán học

Raw bank C chứa tối đa 10,000 observed patch vectors. CADIC proposer chạy trên fork, có thể thay nhiều row của một image. Gate nhận hoặc từ chối **candidate bank update**; nó không coi ảnh là memory unit. Exact full-PASS parity cần bank/counters/score giống CADIC, không chỉ metric gần nhau.

Với observed support S và epsilon=max_s d(s,C), `d(q,S)<=d(q,C)<=d(q,S)+epsilon`. Muốn suy anomaly separation phải giả thiết normal/anomaly margin chưa được chứng minh. Inherited FP32 kernel có hazard với near pairs; giữ numeric route cho matched comparison và công khai giới hạn, không gọi đây là bit-for-bit full paper replication.

Risk `R=.5 mean squaredNN+.5 max squaredNN`. Historical anchor reservoir gồm 32 images×8 patches, dùng uniform private RNG, không class quotas. Tại proposal image t, cache current risks r± và anchors a± **trước khi thêm x_t**. C−/C+ fixed đến next normal.

Định nghĩa:

`c=1+r-+r++a-+a+`, `gamma=((r-+a-)-(r++a+))/c`.

Khi image t+1 đến: `J±=R(next_image,C±)+cached_a±`, `Delta=J−−J+`, `y=Delta/stored_c`. Constant 1 có cùng đơn vị risk của frozen raw geometry; không fit từ future samples. Cả accepted/rejected proposals đều nhận credit. Target y có thể ngoài [-1,1]; không còn ratio theo next-image sum hoặc weighted loss.

Context neural dùng fixed Rademacher 768→32/private seed 20261017, image RMS normalization eps 1e-6, visual four-scan five-map block 28×28/32 channels/head 8/row retention .9, pooled 32 + affine 16 geometry stats, CMS_fast rồi CMS_slow. CMS là residual pre-LN MLP 32→64→32 (GELU); visual wrapper có depthwise 3×3 và FFN 32→128→32. Full recurrence, guards/rounding/initialization nằm trong architecture §5; không thay primitive để tối ưu pilot.

Utility readout thuộc fast state: `u=[h32,log1p(c)]` có 33 neural features, w33/b1 zero-init. `p=gamma+w^T u+b`; không tanh/atanh, không phải probability. Augmented `a=[u,1]` có 34 coordinates; `d=a^Ta>=1`.

Loss `.5(p-y)^2`. Fast step **`phi_next=phi-.05*grad_phi(loss)/d`**, không phải `grad(loss/d)`. d dùng context/pre-update weights trước credited outcome, không dùng current label/gradient norm. Tại zero head, với cùng context và theta cố định, first-step error được nhân .95. Đây là algebraic response property, không phải 5% benchmark gain hoặc guarantee cho các bước sau.

Conditional Bayes optimum `E[Delta|full causal history]/c` có cùng dấu expected combined advantage. Positive known d không đảo conditional expected gradient của loss. Kết quả này không bảo đảm cho shared finite model, normalization across contexts, anchor surrogate, slow Adam, nonstationary banks hoặc AD ranking.

## 5. Ownership và học lồng nhau

| State | Chủ sở hữu và update |
|---|---|
| Frozen ViT/P/raw banks/risks | Không gradient; bank chỉ nhận raw proposer transaction |
| Visual working maps | Self-modify theo patch scan nội ảnh; reset mỗi image/context |
| Fast phi | Toàn bộ LN/MLP của CMS_fast và readout33/bias; cập nhật sau mỗi completed label, persistent qua tasks |
| Slow theta | Visual wrapper/learned initial maps, geometry affine, CMS_slow; exact meta update mỗi 8 completed labels |
| Replay | Fixed projected tokens, stats/c/gamma và labels đã đến; không lưu detached learned embeddings |

Ba clocks là spatial scan nội ảnh, completed delayed labels và eight-label slow commit. Chúng có objectives/contexts riêng và dependency thật; không chỉ là hai timer đặt cạnh bank. Muốn claim benefit vẫn cần controls/AD evidence.

Trong mỗi block 8: lưu detached phi_start trước label 1; theta không đổi; actual phi học cả 8 records. Slow replay clone phi_start, recompute encoder/h/d ở từng support 1–4 và unroll exact steps. Query 5–8 đều dùng cùng phi4; lấy mean plain losses. Exact meta differentiate toàn `grad(loss)/d`, bao gồm derivative của d; FIRST_ORDER detach toàn quotient và giữ forward giống nhau khi cùng start/records. Không backprop qua raw risk/hard bank hoặc toàn actual stream.

Sau slow Adam 1e-4, betas .9/.999, eps 1e-8, weight decay 0, global clip 1, giữ actual phi8. Không reset/rebase sang meta clone. Representation drift sau slow commit là limitation phải đo. Partial blocks qua task boundary/pause/end không flush.

Learned modes force PASS trong 32 completed labels đầu nhưng học bình thường; sau đó p>=0 thì accept. CADIC luôn nhận proposal; GEOMETRIC dùng sign gamma, không có warmup override. Geometry prior parity chỉ đúng ngoài override hoặc khi prior>=0 trong warmup. No-op không tạo utility label; bootstrap chạy nguyên image, kể cả remainder. Không thêm drift reset, task routing hoặc oracle category warmup.

`NO_FAST_ADAPTATION` với zero head không học readout, nên dùng geometry admission trên trajectory riêng ngoài learned warmup; zero slow gradient ở đây là đúng, không phải bug. Không gọi nó globally GEOMETRIC-equivalent vì forced-PASS có thể làm bank trajectories khác nhau. Không dùng nó như trainable slow-only baseline hoặc đòi mọi group nonzero ở mọi sample. Nondegenerate full 4+4 fixtures mới kiểm encoder meta-dependency.

## 6. Đường code cần triển khai

Các path v5.1 ở cột sau đây **là đích dự kiến, chưa tồn tại**. Không tạo placeholder rồi ghi implemented.

| Hiện có / tái dùng | Phần successor cần làm |
|---|---|
| `models/hope_cad_v2/visual_memory.py`, `visual_block.py`, `cms.py` | Giữ primitive đã khóa; wrapper successor phải khai đúng dims 32/head 8; không dùng old primitive metadata làm full-method identity |
| `models/cadic_paperfaithful/` | Giữ bank/proposer/cache/Eq9 cùng numeric route cho controls |
| `models/hope_cad_v5/` | Evidence/source tham chiếu; không sửa equations hoặc checkpoints cũ thành v5.1 |
| Proposed `models/hope_cad_v5_1/contract.py` | Đọc contract mới, canonicalization và identity; reject old learned states |
| Proposed `models/hope_cad_v5_1/geometry.py` | Inherited fork/risk/anchors/context; FP64 current c/gamma và shared geometry decision |
| Proposed `models/hope_cad_v5_1/learner.py` | Fast readout + direct target loss + context-energy step, exact denominator derivative/FO rule và clocks |
| Proposed `models/hope_cad_v5_1/session.py` | Causal pending/state/atomicity/topology validation/read-only score |
| Proposed `models/hope_cad_v5_1/benchmark_adapter.py` | Fresh RGB adapter, explicit lifecycle/finalize and metadata; full evaluator còn planned |
| Proposed `training/hope_support_v5_1.py` | Normal-only pilot, frozen-cache provenance, new schema, telemetry/gates/resume |
| Proposed `notebook_support/support_v5_1/`, `tests/hope_cad_v5_1/`, `notebooks/hope_cad_support_v5_1_kaggle.ipynb` | Bootstrap/runtime/source locks, tests, interface/export sau implementation; chưa có notebook để user chạy |

Root `conf/delivery/support_source.json` và old bundle v6-r4 vẫn khóa v5. Builder successor phải có inventory/schema/delivery riêng, không replace source ZIP tại cùng hash/ID. Không đặt training dataset hoặc raw evidence ZIP trong docs. Shared artifacts ở `D:/src/DATN/.agents/artifacts/{deliveries,runs}`.

## 7. Numerical/state/resume requirements

Features/model/phi dùng FP32; scalar c/gamma/y/log/d accumulation/final p/loss/step dùng FP64. Log feature được cast FP32 để vào neural readout và dùng đúng cast đó trong d. Promote cả readout weights và input u lên FP64 trước dot/reduction; không dot FP32 rồi mới cộng prior FP64. Next parameters được tính bằng step FP64 rồi cast FP32 differentiably. Exact replay recompute d và giữ graph; saved decision_d chỉ dành audit. Exactness theo declared autograd/derivative-bridge semantics, không phải derivative của machine rounding rời rạc. Không AMP/TF32; dùng deterministic canonical T4 lock và ghi rõ source/file hashes.

Không assert y/p nằm trong [-1,1]. Kiểm finite của input/context/scalars/loss/gradient/norm/parameter/Adam; c/d>=1. Vi phạm phải rollback **toàn image transaction**, export failure và stop; không clip/drop/retry/fallback âm thầm. d chỉ chuẩn hóa readout sensitivity, không chuẩn hóa full CMS Jacobian hoặc mọi Hessian.

Observe order: validate → resolve pending/learn → pre-write risk → proposal/context/prior → commit chosen/alternative → reservoir → trace/cursor. Snapshot toàn learner/Adam/replay/banks/pending/RNG/counters trước event. Score không learning; terminal finalize chỉ censor pending, không slow flush. Pause không finalize.

State schemas của successor khác v5. Load vào temporary owner và validate keys/shapes/dtypes/finite values, source/protocol/runtime/config, phi/readout/optimizer, actual clock/records 0–7/phi_start, pending banks/context/c/gamma/d, anchors/RNG và unique order trước commit. Bank SHA không đủ nếu replay shape sai. Pair cache là derived state cần rebuild; không serialize `_version` làm portable identity.

Save atomic mỗi 8 images và khi pause/end; hard crash replay<=7 images, existing event trace phải match. Context c/gamma từ decision không được recompute theo later banks/anchors. Không migrate old weights để “resume” revision mới.

## 8. Validation và pilot sau implementation

Design algebra/reference checks của docs chỉ chứng minh phương trình/fixtures, không kiểm successor code. Production acceptance phải kiểm:

1. Direct target optimum và phản ví dụ cho ratio/current-gradient rule cũ; c/d independent of credited outcome.
2. Gradient của loss trước chia d, zero-head .95 contraction; same-context encoder derivative cancellation; varied-context full 4+4 finite differences và denominator dependency.
3. Ownership, whole-increment FO detach, identical forward at matched state, finite active paths sau support adaptation.
4. Real backbone/preprocessing/canonical shapes, original CADIC proposal/Eq9 forced-PASS parity, partial-block/no-op/terminal chronology.
5. Malformed state rejection, atomic rollback, uninterrupted/resume/hard-crash trace parity và read-only scientific score.

Sau đó mới bàn giao source/notebook mới. Planned run duy nhất: 24 discarded real-normal preflight, rồi bốn fresh controls CADIC/GEOMETRIC/FIRST_ORDER/NESTED×256 images, seed 0. Không thêm search hoặc old NESTED rerun làm control thứ 5. Preflight fail thì stop; same-identity soft-pause resume là cùng trial, không phải fresh fit.

Mechanism gates: NESTED≥64 labels, ≥8 slow updates, reject fraction post-warmup [.01,.99], actual bank khác CADIC và reused-normal patch scores khác >1e-6. Safety gate: last 128 pre-write actual-bank R<=1.10×CADIC. Không dùng gate 1.05/transition 1.15, không đổi window hoặc chọn accepted-only cases. PASS chỉ đủ để review benchmark; no-effect/FAIL không được cứu bằng tăng time/rate.

Ghi current/next/anchor, c/gamma/y/pred/residual/d, counts/actions/censor, gradient và actual step norms, bank hash/operations/changed rows, resources. Response probes theo labels 1,33,65,97,129,161,193,225, trên cùng stored context/prior và fixed theta trước/sau một actual fast update: `(abs(p_before-y)-abs(p_after-y))/abs(p_before-y)`; mẫu số<=1e-12 thì null/small_error, không NaN hoặc extra learning. Saved meta episode 33–40, trong cùng pilot, không thêm GPU sweep.

## 9. Benchmark, fairness và hoàn thành đồ án

Full benchmark v5.1 còn planned sau pilot review: full orchestration, hashed test manifest/metric QA, matched baselines/valid ablations, seeds 0/1/2, final metrics/task matrix/FM/BWT và uncertainty. Canonical primary keys: `macro_final_i_auroc`, `macro_final_p_ap`; AP=average_precision_score, legacy AUPR alias phải được map rõ. Whole-image paired bootstrap theo category/image-label strata, 2000 draws giống nhau cho các method, percentile 1.25/98.75; NI margin 2 pp cho cả hai endpoint, superiority đòi cả hai lower bounds>0. Không coi duplicate deterministic baseline seeds là independent replicates.

Equal K không đồng nghĩa equal memory: phải đo main/alternative banks, anchors, replay, phi_start, models/Adam, pair cache, shared ViT, allocator/transactional CPU+GPU/IO. K24576 là planned payload control, chưa phải measured memory match. Latency phải sync đúng scope; cache warmup khác nhau trong pilot không tạo fair runtime comparison.

VisA chỉ là confirmation sau freeze architecture/source/transfer/data/mask/metric/seeds/budget, dùng fresh state và same rates, không retune bằng test. Public prior exposure phải được disclose. Raw/resized mask QA và empty/undefined metric policy áp dụng cho mọi method; không lọc những ảnh khó để tô số.

Đồ án hoàn tất khi có reproducible implementation, đạt technical/scientific gates phù hợp, matched AD results cùng controls/CI, phân biệt rõ dữ liệu development/confirmation và thesis giải thích background/method/negative findings/limits/chi phí. Đúng tên Nested Learning for CAD không thay thế yêu cầu chứng minh contribution thật. Không hứa CADIC parity hoặc VisA performance trước evidence.
