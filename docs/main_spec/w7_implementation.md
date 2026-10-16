# Nested Learning for CAD — W7 v4, ba phần 7.1 / 7.2 / 7.3

## 1. Phạm vi và trạng thái

W7 đo continual anomaly detection trên MVTec AD cho **HOPE-CAD v4**. Đây là
orchestration/evaluation của method đã khóa, không phải method revision v5.
Architecture/objective/retention/backbone/masks/scoring/periods/learning rates và
data permissions tiếp tục theo `architecture_decision_v4.md`,
`instruction_CAD_v4.md`, `planning_CAD_v4.md`, `revision_contract_v4.json`.
Tài liệu này bổ sung cách chạy ba phần mà người dùng đã yêu cầu.

Implemented: transactional stream runner, resumable raw-score evaluator, fixed
seed refits, matched CADIC jobs, suite/order registry, paired bootstrap/report,
normal-only W7 preflight, notebooks và source packaging. Local tests được lưu
riêng dưới `results/hope_cad_v2/w7_local/`; không dùng fixtures làm benchmark.
**Chưa có W7 GPU/AD benchmark được chạy/nghiệm thu trong lượt triển khai này.**
W8 VisA và W9 báo cáo/reproduction không được coi là đã hoàn thành từ W7 code.

Worktree được phép: `D:/src/DATN/hope_cad_v2_2`, branch `hope_cad_v2_2`, bắt đầu
tại `b49a103`. Không merge sang `hope_cad_v2_1` và không commit/push thay người dùng.
Predictor/visual/CMS/outer learning và các spec frozen giữ nguyên. Thay đổi nhỏ tại
`benchmark_adapter_v4.py` chỉ cho phép kiểm tra **receipt refit có seed thật**;
đường forward, score và update của adapter giữ nguyên.

## 2. W6 input và quyết định trước khi mở official test

Input bắt buộc là completed W6 tại `docs/w6_kaggle_v4.1/`:

- ZIP SHA256: `ae9d257ecf1ced738df651fa07887b838a96da5fe4cdda84e46c51533c9bcd71`.
- `artifact_manifest.json` SHA256: `5892af759a096a6f2ba658eff5a3095aadc94c05bdb7c5849bfb6b122f37d4cd`.
- Đủ 16 completed fits × 400 steps, 11 long gates, 7 selected initializations.
- M/C2/C3/C4/A1 dùng beta 0.01; C4 slow rate 0.00125/event.
- M/C2/C3/C4 outer LR 0.0001; C0/C1/A1 inherited outer LR 0.0003.
- Tất cả selected steps = 400. A1 không có optimizer fit riêng.

Audit W6: engineering PASS, cache roles 50/20/15/15, resume và receipts hợp lệ.
Normal mechanism evidence mixed: M thích nghi nhẹ và visual contribution rõ hơn,
nhưng normal fitting chưa vượt static/one-level/equal-period controls. Không gọi
đây là AD validation hoặc bằng chứng thắng CADIC. Việc triển khai W7 do yêu cầu
trực tiếp của người dùng sau audit này, không do pipeline tự cấp quyền.

W6/probe đã development-exposed; các nghiên cứu cũ cũng có official-test exposure
đã ghi ở spec v4. Không gọi toàn bộ MVTec là untouched confirmation. Không dùng
W7 metrics để chọn beta, step, seed, order hay thay model rồi giữ cùng identity.
Một kết quả thấp ở 7.1 không cho phép bỏ controls/seed bất lợi và báo full suite.

W6 ZIP gốc được kiểm hash; nếu Kaggle đã giải nén thì kiểm manifest hash đã ghim
và từng file được nhập. Nhập initialization/selection/locks, không refit W6.
Seeds 1/2 cần frozen `prefix_features.pt`; cache này chỉ gồm 85 ảnh được phép.
Normal validation/probe vẫn bị loại khỏi online stream cho HOPE và CADIC matched.

## 3. 7.1 — M seed 0, stream đầy đủ, final metrics

1. Chạy W7 normal-only preflight trên đúng source/runtime/backbone; xem §8.
2. Dùng lại selected M initialization của W6, không train calibration lần nữa.
3. Stream một lượt toàn bộ `manifest.online`, đủ 15 categories theo alphabetic.
4. Mỗi ảnh là một CMS event; view = global cursor modulo 4. Task boundary không
   reset/flush CMS, pending gradients hoặc mask clock. Static controls cũng
   consume identities nhưng giữ CMS clock 0 như contract hiện hành.
5. Lưu full method checkpoint sau mỗi task. Atomic progress theo mỗi 16 ảnh và
   cuối task, bao gồm model/current weights/accumulators/counters/cursor/RNG.
6. Từ checkpoint cuối, score toàn bộ official test của đủ 15 categories. Mọi
   image dùng clean teacher + 4 masked views, final CMS discrepancy và max-patch
   image score; mask GT chỉ ở evaluator. Test không cập nhật persistent state.
7. Lưu per-class và macro I-AUROC/P-AUPR/I-AP/P-AUROC, raw FP32 scores/maps,
   labels/masks/relative paths và hashes. Exact point AP dùng sklearn AP.
8. **Dừng và export**. Đây là quick comparison với Khang, chưa phải matched
   confirmatory claim hoặc forgetting result. Kết quả CADIC cũ vẫn là real
   historical numbers; khác protocol phải được ghi rõ.

Data integrity kiểm per-category train/test-normal/test-anomaly counts theo
[MVTec AD Table 1](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf):
raw train 3,629, official test 467 normal + 1,258 anomaly. Test counts chỉ được
đọc/kiểm ở evaluator; không đưa test images/features vào calibration. Sau khi
loại 15 validation + 15 probe, online stream có 3,599 images.

CLI: `run_w7.py --phase 7.1 --job M_seed0 ...`.
Notebook: `notebooks/hope_cad_w7_7_1_kaggle.ipynb`.

## 4. 7.2 — bổ sung task matrix, không train lại

Input là đúng output 7.1 (hoặc latest pause của 7.2), cùng source/runtime và
W6/ViT/data identities. Không dùng checkpoint v3 hoặc run khác.

- Nạp lần lượt 15 checkpoints và evaluate chỉ các tasks đã học: j <= t.
- Tổng 120 task-evaluations; 15 cells của final row đã có ở 7.1 được dùng lại
  nguyên raw scores/metrics. Chỉ tính **105 cells còn thiếu**.
- Mỗi batch score là atomic shard. Resume trong một cell không score lại ảnh
  đã commit; kiểm checkpoint hash/test inventory/hash và shard hashes.
- Kiểm persistent state và RNG không đổi; cuối quá trình restore final state.
- Xuất matrix cho đủ bốn metrics; upper triangle giữ null/unlearned.

Với metric R, T=15:

`Final = mean_j R[T-1,j]`

`BWT = mean_{j<T-1}(R[T-1,j] - R[j,j])`

`F = mean_{j<T-1}(max_{t=j..T-2} R[t,j] - R[T-1,j])`

Giữ forgetting âm nếu có, không clamp. Thiếu cell/undefined metric không được
bỏ category hoặc tuyên bố matrix complete.

A2/A3 bổ sung là paired **read-only interventions**, trên 8 meta-support normals
đã định trước, với terminal M state: reset CMS readout, single-scan readout, bỏ
từng stability guard; so earlier first-task history với terminal history; hai
discarded continuations trên 8 seen normals (active vs frozen CMS weights nhưng
giữ clocks hợp lệ). Chạy đủ 4 score views; giữ failure/unstable artifacts.
Continuation score dùng cùng 8 normals, chỉ là in-sample sensitivity, không phải
normal generalization hoặc forgetting/AD proof.
Không retrain, không dùng GT anomaly, không tuyên bố đây là các methods độc lập
hoặc chứng minh accuracy/latency gain. W6 đã có freeze-level diagnostics.

CLI: `run_w7.py --phase 7.2 --job M_seed0 ...`.
Notebook: `notebooks/hope_cad_w7_7_2_kaggle.ipynb`.
**Dừng và export** sau matrix; không tự chạy toàn suite.

## 5. 7.3 — remaining jobs và full suite

Registry authoritative: `training/hope_w7_contract.py::jobs`.

| Nhóm | Jobs | Điều kiện |
|---|---:|---|
| Main HOPE M/C0/C1/C2/C3/C4/A1 × seeds 0/1/2 | 21 | M seed0 dùng lại 7.1+7.2 |
| Matched CADIC 2.5k/10k × seeds 0/1/2 | 6 | Same data permissions/backbone/preprocessing/metrics |
| Orders 100/200 × M/C4/CADIC2.5k/CADIC10k, seed0 | 8 | Giữ bottle đầu, shuffle 14 tasks còn lại |
| Tổng | 35 | Còn 34 streams sau M seed0, không tương đương 34 sessions |

Mỗi job gồm train một lần + final row + toàn matrix; final row được reuse như
7.1→7.2. Một saved session chạy **một job**, có thể pause/resume nhiều phiên.
Không gom toàn suite multi-GB vào một output hoặc hứa hoàn tất trong một đêm.
Hai T4 là hai GPU riêng; runner mặc định dùng một GPU, không tự chia stream.

Seeds 0 reuse W6 initializations của từng method; order studies seed0 cũng dùng
cùng initialization do first task/prefix không đổi. Source/feature/receipt
checks vẫn bắt buộc. Seeds 1/2 **fit fresh** bằng own seed và fixed W6 recipe:

- Không search beta/LR, không chọn lại step theo normal validation/test.
- Fit đúng 400 steps; giữ **terminal weights**, không dùng shortlist mới do
  helper `fit` ghi cho diagnostics. Long gate chỉ accept/reject.
- Non-finite fit hoặc failed gate được giữ là failure; không đổi seed/rate/step
  để cứu. Thiếu acceptance làm suite incomplete.
- C4 kế thừa W6 beta_M, không tự tune; own seed initialization được fit riêng.
- A1 seed s dùng accepted C1 initialization **cùng seed s**, beta_M đã khóa;
  không thêm optimizer fit. Attach C1 W7 evidence; runner kiểm cùng suite và seed.
- Receipt `hope_w7_fixed_refit_v1` ghi seed thật, parent recipe, terminal weight
  hash, fit/gate/schedule/role hashes. Không giả mạo seed1 thành seed0 selection.

CADIC dùng explicit `cadic_paperfaithful_image_v1`, cùng b=9 full Eq.9 score và
numeric kernel đã khóa. Secondary 2.5k thay đúng ID/budget; không nới validator
10k cũ. CADIC giữ extraction transport batch8 của locked config, coreset vẫn
update theo từng ảnh; HOPE feature extraction/event là B1. Effective transport
được ghi rõ ở run lock. Pause/checkpoint ở completed transport/image boundary,
không phá grouping CADIC khi resume. Compact checkpoint chỉ bỏ frozen backbone bị lặp trên disk: input
checkpoint hash/parameter versions vẫn được kiểm, coreset update/scoring không
đổi. Memory report vẫn tính backbone. Nếu CADIC repeats deterministic thì báo
đúng zero variation, không gọi là ba quan sát độc lập.

ReplayCAD của Khang là comparison bổ sung khi có native continual artifacts và
protocol metadata. Không dựng baseline giả hoặc dùng epoch checkpoints của joint
training để tính forgetting. ReplayCAD chưa sẵn sàng không cản main W7 suite.

CLI: `run_w7.py --phase 7.3 --job C0_seed0 ...`, `M_seed1`, v.v.
Notebook: `notebooks/hope_cad_w7_7_3_kaggle.ipynb`; sửa **JOB** theo registry.
M seed0 trong 7.3 bắt buộc import completed 7.2, không retrain.

## 6. Statistics và acceptance của full W7

CPU report: `report_w7.py --evidence <W7 ZIPs...> --output-dir <dir>`;
notebook `notebooks/hope_cad_w7_report_cpu.ipynb` dùng Accelerator None.
`--inventory-only` báo jobs/matrices còn thiếu. Không cần GPU cho reporting.

Main comparison M−matched CADIC10k: final macro I-AUROC và P-AUPR. Bootstrap
2,000 replicates, seed 20261009, categories fixed; resample **whole images**
trong category × image-label strata. Dùng cùng multiplicities cho mọi method,
seed và checkpoint; average metric deltas qua 3 trained seeds, không ensemble
scores. BWT/forgetting giữ covariance qua checkpoints. Báo seed mean/std riêng.
RNG cụ thể là `numpy.default_rng(SeedSequence([20261009, category_index]))`;
analysis lock giữ Python/NumPy/sklearn versions để resume không trộn RNG/metric
implementations. Order-study tables có actual orders và delta so canonical,
không được gộp vào bootstrap categories/seeds.

Co-primary intervals: percentiles 1.25/98.75 (97.5% mỗi endpoint), chỉ claim
simultaneous improvement khi cả hai lower bounds >0. Secondary comparisons và
continual endpoints dùng 2.5/97.5, ghi đúng multiplicity scope. Order studies
báo riêng, không gộp vào seed uncertainty hoặc chọn order tốt nhất.

Exact point metrics giữ toàn pixels. Pixel bootstrap có thể histogram accelerate
nhưng phải so với exact weighted AP trên ít nhất 32 fixed replicates, absolute
error <=2e-4 **cho mỗi cell**. Không đạt thì refine; vẫn fail thì không có CI
acceptance, có thể chạy explicit exact CPU mode. Không thay score/metric để cứu.
Analysis lưu per-cell arrays/checksums để tiếp tục sau interruption, không refit
models. Notebook CPU mặc định inventory-only; khi đủ jobs mới bật bootstrap.

Engineering acceptance: source/runtime/data/checkpoint identity, atomic resume,
no test mutation/leakage, đủ counts/tasks, finite exact metrics, complete raw
scores và model/resource ledger. Scientific acceptance: kết luận từng hypothesis
theo paired controls/CIs; không cam kết trước thắng CADIC. Negative/inconclusive
results phải giữ, và không được gọi suite complete khi thiếu mandatory jobs.

## 7. Artifact schema và storage

`suite_lock.json`: all jobs/recipes, W6 SHA, baseline lock, source/spec hashes,
runtime, statistical policy và exposure policy. `runs/<job>/` chứa:

- `job_lock.json`, `run_lock.json`, `manifest_train.json`, resolved configs.
- `refit/` cho seeds1/2: full optimizer/RNG resume, fixed recipe và acceptance.
- `states/stream_resume.pt`, `states/task_00.pt` ... `task_14.pt`.
- `test_manifests/<task>.json`: all official identities/image/mask SHA256.
- `scores/<checkpoint>_<task>/`: raw NPZ shards, atomic progress, exact result.
- `metrics/`: final per-task/macro, four task matrices, F/BWT, fixed-rule cases.
- `completion.json`: progress/resource/profile/storage; reportable stays false
  until researcher review. Pauses/failures also export `handoff.json`.

`hope_cad_w7_evidence.zip` manifest SHA-binds every entry; source input is
`hope_cad_w7_source_bundle.zip`. Do not overwrite old source bundles/W6 evidence.
Original shared W6 caches stay outside `/kaggle/working`, so per-job exports do
not repeatedly include a 1GB frozen cache. Keep raw scores for uncertainty;
checkpoint/score storage and ZIP export time are real costs, not model memory.
Keep all saved Kaggle versions and archive each job locally/server as practical.

## 8. Preflight và local verification

`hope_cad_w7_preflight_kaggle.ipynb` runs the v4+W7 fixture suite and **selected M
with the real agreed ViT on 9 permitted normal images**, interrupted at event5,
và simulated task boundary sau event6 (tất cả ảnh vẫn từ permitted prefix).
It checks exact state/accumulator/RNG/score resume across the period8 boundary,
read-only scoring, measured synchronized update/inference and resource usage.
It never discovers official test or changes W6 recipe. This is a short engine
preflight, not another 400-step W6 search.

Local verification:

```powershell
pixi run --locked python -B scripts/hope_cad_v2/verify_w7.py --output results/hope_cad_v2/w7_local/verification_w7.json
pixi run --locked python -B scripts/hope_cad_v2/build_w7_notebooks.py
pixi run --locked python -B scripts/hope_cad_v2/package_w7.py
```

Hai tests UI của notebook v3 lịch sử được loại khỏi **W7 verifier scope** và ghi
rõ tên/lý do trong JSON. File `hope_cad_v2_2_kaggle.ipynb` đã được người dùng sửa
trước W7, có SyntaxError ở backslash literal và thay budget/bootstrap; giữ nguyên
file đó. Full unfiltered workspace run ban đầu được giữ ở `verification.json`
(130 tests, 3 errors trong hai historical notebook tests). Không gọi toàn bộ
workspace là green. W7 source không đưa notebook v3 đó vào handoff; tests W7 mới
kiểm ZIP/extracted source và notebook compilation riêng. Các thuật toán v3/v4
chung, current v4 handoff và W7 contracts vẫn được chạy, không bỏ tests của model.

The generated source bundle is verified entry-by-entry. Remaining external gates
are actual Kaggle W7 preflight, 7.1/7.2 outputs, other job artifacts and final
suite statistics/review. No wall-time estimate from previous W6 is a W7 timing
measurement; use preflight and early actual stream/eval timings.

## 9. Code map

| File | Responsibility |
|---|---|
| `training/hope_w7_contract.py` | Registry, source hashes, immutable locks, seed/order manifests |
| `training/hope_w7_engine.py` | Event resume, per-task states, raw-score shards, exact metrics/matrix |
| `training/hope_w7_refit.py` | Fixed terminal refits, seed-specific receipts and acceptance |
| `training/hope_w7_adapter.py` | Matched CADIC budget registration and compact transport |
| `training/hope_w7_io.py` | Verified W6 import and W7 export/restore |
| `training/hope_w7_statistics.py` | Paired image bootstrap, exact/histogram checks, evidence views |
| `scripts/hope_cad_v2/run_w7.py` | Phase driver, protocol/recipe/runtime lock and failure handoff |
| `scripts/hope_cad_v2/preflight_w7.py` | Real normal-prefix engine/profile/resume gate |
| `scripts/hope_cad_v2/diagnose_w7.py` | Bounded read-only normal interventions, not retraining |
| `scripts/hope_cad_v2/report_w7.py` | Complete-suite inventory, seed/order metrics and uncertainty |
| `scripts/hope_cad_v2/build_w7_notebooks.py`, `package_w7.py` | Thin notebook interface/source ZIP |
| `scripts/hope_cad_v2/verify_w7.py` | Explicit W7/shared-v4 verification scope and historical exclusions |
| `tests/hope_cad_v2/test_w7*.py` | Resume/reuse, identity, refit and bootstrap regression fixtures |
