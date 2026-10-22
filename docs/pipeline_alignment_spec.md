# Chuẩn hóa pipeline v7 trước candidate H32 — amendment 1

Ngày: 22/10/2026. ID: `v7_pipeline_alignment_v1`. Trạng thái: **đã đặc tả, chưa implementation runtime hoặc benchmark mới**. [Contract máy đọc](pipeline_alignment_contract.json), [báo cáo nhóm](../QUICK_UPDATE.md).

## Vai trò hai file pipeline

File Markdown này giải thích quyết định về dữ liệu/recipe/fairness và acceptance. JSON cạnh nó mã hóa các IDs/counts/flags của cùng quyết định; đây là **design contract, chưa là runtime config**. [Bộ research spec](research/README.md) cung cấp thuật toán và study đầy đủ ngay trong repo. Baseline thực thi đọc resources riêng; không cần nguồn ngoài repo để phân biệt hai lớp.

## 1. Nguồn chuẩn và ranh giới phiên bản

Baseline source chính thức được người dùng chốt lại là commit `89df9a21658b265731f98a7c2014afb1876f1924`, method `hope_cad_v5_1_causal_scaled_utility`, design digest `6f6b2c847a815619b6ad5ed899a0859aab1d7b532d62b27ae3217d926695db99`. Các benchmark accepted cũ dùng 3.599 online normals; số liệu/provenance giữ nguyên.

Amendment này có ưu tiên cho **cohort nghiên cứu tiếp theo** đối với partition, pilot composition, identity và các yêu cầu so sánh. Định nghĩa equations/precision/ownership hiện hành trong [method](method.md) tiếp tục kế thừa. Candidate historical-query giữ thuật toán history32/4support/2recent+2historical đầy đủ tại [candidate](research/candidate.md); dữ liệu của candidate mới theo full-train amendment này.

`resources/method.json`, `study.json`, `protocols/mvtec_matched.yaml` và runtime hiện tại **vẫn là baseline cũ**, không được sửa đè để làm biến mất lịch sử. Lệnh CLI hiện chưa cung cấp full-train/H32 cohort mới; không dùng lệnh cũ rồi báo đã chạy amendment. Khi implementation, thêm successor resources/factory với guards riêng, giữ predecessor strict. Source và delivery identity mới phải công bố cả khi chỉ đổi docs.

## 2. Quyết định partition và data rights

**Cohort mới dùng full MVTec AD train/good: 3.629 normals, Bottle 209; không loại Bottle 070–099.** Các category theo thứ tự cố định:

`bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper`.

Trong từng category sort canonical relative path; không shuffle/drop_last, task reset hoặc grouping bank updates theo loader batch. Hash từng ảnh và toàn manifest; path/bytes/role rights phải bind vào source/runtime/cohort. Chỉ normals đã đến được đưa vào learner. Discovery/hash inventory không cho phép dùng future feature statistics. Test/GT chỉ evaluator, không optimizer/sampler/selection. Full train không là offline pretraining hoặc calibration thêm.

Protocol ID mới dự kiến `mvtec_1x15_fulltrain_v7_v1`; cùng full-training semantics với route `mvtec_1x15_v1` của Khang nhưng không giả hai manifests/digests khác nguồn là cùng identity. Future recent-only contract ID `hope_cad_v5_1_fulltrain_v1`; H32 contract ID `hope_cad_hq32_fulltrain_v1`; study `hq32_mvtec_fulltrain_development_v1`. Đây là IDs đã dành trước, chưa implemented. Recent-only giữ learning equations v5.1, nhưng data contract khác; candidate phải kế thừa đúng cohort đó.

CADIC, recent-only, H32 và các controls dùng **cùng 3.629 IDs/order**, fresh learner/bank/optimizer/RNG. Không resume checkpoint 3.599 rồi chèn 30 ảnh, không carry learned pilot state, không so pipeline cũ và candidate mới rồi gán toàn chênh lệch cho H32. Dataset test giữ cùng official 15 category/1725 images nếu bytes xác minh khớp; source masks/hash QA vẫn bắt buộc.

Phân tích D0 offline **không đổi**: nó vẫn đọc logged 3.599 events của completed benchmark cũ. Không sửa allowlist/expected counts hoặc thêm targets giả để D0 trông như full-train. D0 và prospective cohort là hai dữ liệu/phạm vi khác nhau.

## 3. Recipe ảnh và checkpoint — quyết định giữ, không giả đã sửa

Giữ recipe current/matched CADIC của Khang: PIL direct square 224 bilinear, mean(.485,.456,.406), std(.229,.224,.225); không center crop, không raw-LN/L 2, frozen ViT-B/8 AugRegIN21K block 8 zero-based, bỏ CLS, raw [1,784,768] FP32.

**Lý do:** cô lập history-query effect trên một recipe chung; native classification metadata không chứng minh author anomaly recipe, crop có thể bỏ defect, và chưa đo causal gain từ đổi preprocessing. Không mở grid chọn transform bằng MVTec anomaly metrics. Native mean/std 0.5 và bicubic khác hiện hành là rủi ro còn công bố, không gọi là defect đã được sửa hoặc không còn quan trọng.

Checkpoint của cohort khóa SHA `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`. Shared source upstream không tự chứng minh `.pth` từng run Khang và safetensors của ta cùng tensors. Nếu so trực tiếp với run Khang, nhận actual receipt hoặc checkpoint, kiểm exact keyset/dtype/shape/values của non-head tensors và ghi mapping chỉ cho phép head.weight/head.bias bị loại theo extractor contract; không wildcard-ignore missing/unexpected keys. Khác SHA container không là proof weights khác. Thiếu artifacts phía Khang chặn claim cross-run parity, **không tự chặn fresh internal matched cohort** đã bind cùng checkpoint.

## 4. CADIC execution route và numerical policy

Route chuẩn Khang được kiểm tại `dikhang_space@794d7e4`: shell `run_cadic_paperfaithful.sh` override sang paper-faithful adapter và per-image update. Generic adapter khác có thể truyền cả loader batch vào bank dù YAML có image label. Vì coreset selection phụ thuộc candidate grouping, cần pin actual adapter class/source/backend/update-unit trong receipt, không suy semantics từ filename.

Cohort mới giữ mỗi image 784 patches là một proposal, repeated CADIC replacements và deterministic fill/ties theo baseline. Feature extraction batch chỉ được đổi nếu chứng minh producer parity theo policy; không để đổi batch đồng thời đổi bank update unit. Chọn K 10000 cho main comparison; không dùng config default 2500 làm actual K. Full Eq9 b 9, raw NN patch maps→CPU FP32 bilinear 224, align_corners=False, nonzero GT→nearest 224, average_precision_score và task macro giữ nguyên.

Backend cached pair/NN/write và FP32 numerical route hiện hành được giữ cho mọi arm cùng cohort. Runtime pins/TF 32 off / AMP off / actual device phải bind, preflight trên máy thật. Synthetic/reference/optimized fixtures và selected actual-feature parity phải kiểm bank/counters/ties/score/resume. FP32 cancellation đã biết không được silently sửa một phía; nếu chứng minh cần numerical repair thì version backend/cohort cho mọi comparator trước run tiếp. Scoped parity không là global bit-exact guarantee hoặc author-paper parity.

## 5. Pilot mới — giữ tiêu chí, đổi đúng tập và cửa sổ

Preflight discarded 24 allowednormals đầu +4 events kế tiếp; historical-gradient checks có synthetic fixtures riêng, không bịa labels để đủ history. Pilot fresh 9 variants× 256 normals: **209 Bottle+47 Cable**, thay 179 Bottle + 77 Cable của partition cũ.

Last 128 image window mới là **81 Bottle+47 Cable**; window cũ là 51 Bottle+77 Cable. First 24 score probeIDs vẫn Bottle 000–023. Category annotation chỉ reporting, không chia reservoir/sampler hoặc reset learner. Phải ghi manifest và windowIDs trong receipts để tránh gọi thaypool là improvement.

Giữ một trial, seed 0, số ảnh 256, completed-label minimum 64, scheduled slowminimum 8 cho các R/H có Adam; other arms slow 0, warmup 32 completed labels. H32 history-gradient path finite/nonzero sau startup; reject fraction[.01,.99], bank khác CADIC, after 256 read-only maps của first 24 trainIDs có max absolute delta>1e-6. Last 128 mean prequential risk(H32)/CADIC<=1.10 là safety heuristic đã chọn; pool mới giảm số Cable observations nên không claim mạnh về acquisition/generalization. Không kéo dài/cắtwindow/đổi margin để cứu output. Cả 9 jobs thiếu completion vì cap là budget_unresolved, không PASS.

Budgets giữ 12 GPU wall-hours cho preflight+pilot và 24 GPU wall-hours cho 3 fresh main jobs, gồm setup/features/IO; sessionmax 4 h/reserve 180 s. Đây là caps, không ETA hoặc auto-spend. Nếu measuredcost vượt cap, user quyết định phân bổ tiếp bằng record riêng; không tự đổi algorithm hoặc trial count.

## 6. Fairness và bằng chứng: việc phải đo, không phải sửa bừa model

| Điểm audit | Quyết định/acceptance | Không được suy |
|---|---|---|
| Full-train mismatch | New manifest 3.629/excluded 0, fresh all arms, successor IDs/guards | Benchmark 3.599 cũ sai hoặc trở thành full-train sau đổi tên |
| Native preprocessing mismatch | Giữ recipe hiện hành có lý do/disclosure; không thêm preprocessing grid | Đã paper parity hoặc chắc không ảnh hưởng |
| Actual CADIC route | Source/class/backend/update-unit receipts, imagewise fixtures | Cùng YAML là cùng execution |
| Checkpoint provenance | Locked single checkpoint trong new cohort; tensor check nếu so run ngoài | Cùng tag/khác SHA là proof tương đương/khác weights |
| Same K vs memory | Báo bank/alternative/anchors/learner/Adam/replay/cache, persistent/peak CPU/GPU và disk riêng | K=10k đồng nghĩa equal total memory |
| Numeric/runtime | Cùng fixed flags/runtime/backend +preflight parity; version repair nếu có | T4 vs 5090 hoặc batched extraction luôn bit-exact |
| Seeds/order/attribution | Main seed 0→conditional seeds 1/2; controls exact/FO/direct/bias/random; report single-order scope | 120 cells là 120 seeds, hoặc exact/CMS frequency benefit đã cô lập |
| Timing scope | Fresh same-cohort train/extract/update/checkpoint/eval/export tách; cumulative across sessions | Continuation/fresh ratio là exact overhead hoặc paper timing |

Không tự tăng K của CADIC để gọi equal memory. Existing `CADIC_TOTAL_BYTES` K 24576 là payload control, chưa equal total memory hoặc có completed result. New minimum H32 study chưa có measured-total-memory-matched arm; không mở claim memory efficiency. Nếu cần comparator tương ứng phải có separate registered budget/contract trước chạy, không chọn K bằng AD metric.

## 7. Candidate/study và attribution

H32 giữ history 32, Algorithm R sau successful block tám labels, riêng reservoir/query RNG offset 96136/96137, support r0…r3, startup queries r4…r7 rồi queries [r6,r7,h0,h1], mean đồng trọng số. Historical labels original fixed pair, actual phi8 vẫn deploy, replay phi4 chỉ meta; FO detach toàn increment g/d; DIRECT zero functional support, first-block zero-head gradient hợp lệ. Không làm candidate nhận old learned state. New schemas/transactions giữ pending/partial/history/Adam/RNG/score read-only strict. Chi tiết thuật toán ở [candidate](research/candidate.md), controls/gates ở [study](research/study.md), D0 ở [attribution](research/attribution.md) và thứ tự làm ở [plan](research/implementation_plan.md). [Research contract](research/contract.json) đã tích hợp data amendment; equations nền ở [method](method.md). QUICK_UPDATE là báo cáo, không thay full spec.

Study mới giữ 9 pilot variants: CADIC,R_EXACT,R_FO,H32_EXACT,H32_FO,H32_DIRECT,FAST_ONLY,BIAS_VETO,PILOT_RATE_RANDOM_VETO. Recent arms giữ history bookkeeping để so context intervention. Không sửa study cũ 9 methods × 3 seeds để giả đó là cùng một study.

Main batch 3 fresh seed 0 CADIC/R_EXACT/H32_EXACT đều học 3.629 và 120 cells. Spend-screen vẫn H32−R P-AP>0, H32−R I-AUROC>=−.02, H32−CADIC cả primary>=−.02. Screen là quyết định chi tiếp, không statistical acceptance. Conditional 6 remaining seed-0 controls +6 replicate jobs cho 3 main methods, seeds 1/2: tối đa 15 full jobs; D allocation do user cấp sau measured B/C, không automatic paid jobs.

MVTec development-exposed; paired whole-image bootstrap/finite 3-seed/CI/multiple-contrast caveats giữ nguyên. Same task order được cố ý khóa để đối chiếu Khang; không claim order-robust nếu chưa registered extra orders. VisA cần riêng official split/exposure/method freeze/protocol/runner fresh, chưa biến thành runnable chỉ bằng đổi data path. Native ReplayCAD/ContCore có protocol/budget/evaluator riêng; adapter tồn tại không là accepted baseline result.

## 8. Implementation plan và Definition of Done

**Task tài liệu này không implement runtime, model, tools đo checkpoint, GPU hoặc source delivery mới.** Source metadata/inventory cập nhật để docs đi cùng package; hash cũ không được dùng cho working tree mới.

Khi được giao implementation:

1. Implement D0 theo old archived trace, không đổi scientific target vì phân tích bất lợi.
2. Thêm successor full-train protocol/manifest/study/factories; giữ old loading path và frozen JSON. Kiểm old/new manifest set difference chính xác 30 Bottle IDs và counts 3.629/209; per-image order, hashes, no future features/labels, task IDs reporting only.
3. Bind checkpoint/extractor/backend/precision/evaluator/identity; không đổi numerical policy hoặc preprocessing trong cùng amendment.
4. Registered fresh CADIC/R_EXACT phải chạy được và reject cross-protocol/source/runtime/old checkpoint imports; test resume/atomicity/end counts theo new manifest length, không hardcode 3.599.
5. Implement H32/controls với full-train contract đã định danh; old tests giữ và thêm independent gradient/RNG/state tests. Fixtures và new pilot acceptance phải được review trước GPU.
6. New source bundle +local installation/inventory/CLI checks; real GPU preflight/pilot theo budget, raw artifacts/logs đầy đủ. Không launch paid GPU từ spec.
7. Full-train pilot/main benchmark review chỉ là new cohort results; historical numbers giữ provenance. Adoption/VisA cần empirical gates riêng, không vì docs hoặc tests PASS.

**Bản chính thức để rollback vẫn là commit 89df9a2.** Commit89df9a2 đã chứa bộ docs/spec/plan công khai; bản chỉnh tham chiếu baseline này chưa thêm implementation prospective pipeline; không coi current HEAD hoặc dirty source bằng shipped r3. Không automatic Git reset/merge/push.
