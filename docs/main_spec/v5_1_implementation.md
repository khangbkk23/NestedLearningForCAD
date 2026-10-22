# Nested Learning for CAD — Implementation v5.1

Method `hope_cad_v5_1_causal_scaled_utility`, worktree/branch `hope_cad_v6`. Đọc full architecture/instruction/planning/revision_contract v5_1 và `implementation_acceptance_v5_1.md`; chúng giữ nguyên bytes. Đây là implementation/handoff record mới, không viết lại trạng thái lịch sử trong frozen specs.

## 1. Phạm vi và trạng thái

Đã triển khai namespace successor I1/I2 và kiểm thử local; real ViT/T4 pilot v5.1, anomaly benchmark và VisA **chưa chạy**. Không có performance gain được chứng minh. V5 pilot FAIL và các evidence v4/v5 giữ nguyên. Không dùng v5 learned weights.

## 2. Code executable

| File | Vai trò |
|---|---|
| `models/hope_cad_v5_1/contract.py` | Đọc frozen contract và hash cả năm full spec/acceptance files; executable identity bao gồm probe/counter semantics. |
| `models/hope_cad_v5_1/learner.py` | UtilityModel, functional fast CMS/readout, `fast_step`, exact/FO `meta_loss`, persistent learner và strict learner restore. |
| `models/hope_cad_v5_1/geometry.py` | Tái dùng raw CADIC fork/risk/projection/statistics; causal FP64 c/gamma và strict anchor validation. |
| `models/hope_cad_v5_1/session.py` | One-image causal event, pending fixed banks, transaction/rollback, counters, read-only scoring, temporary-owner restore và memory accounting. |
| `models/hope_cad_v5_1/validation.py` | Exact keys/topology/dtypes/finite/count checks, gồm non-tensor primitive metadata. |
| `models/hope_cad_v5_1/benchmark_adapter.py` | RGB transport, frozen ViT/checkpoint hash, explicit training/scoring lifecycle; full anomaly evaluator/orchestration vẫn planned. |
| `training/hope_support_v5_1.py` | Normal-only feature/cache receipts, content-locked jobs, checkpoint mỗi8 images, crash replay và fixed probe/gate comparison. |
| `notebook_support/support_v5_1/` | CUDA/runtime guard, train-only extraction, source/resume identities, heartbeat/worker, evidence export, independent builder/verifier. |
| `tests/hope_cad_v5_1/` | Algebra, finite differences, ownership, canonical FP32, state corruption/rollback, CADIC parity và real driver trên synthetic normals. |

Inherited code `models/hope_cad_v2/` và `models/cadic_paperfaithful/` giữ nguyên. Geometry v5 được import như immutable dependency, không import old learner/session vào successor. Package cần giữ geometry/contract v5 cùng original JSON để dependency tự chứa; tên thư mục này không làm method v5.1 trở thành old v5. Full method identity nằm ở contract/source manifests mới.

## 3. Đường học và credit

Readout33+bias thuộc persistent fast phi, zero-init. Fast template registered trong model bị frozen, không vào slow Adam. Neural path dùng visual → geometry affine → CMS_fast → CMS_slow; prediction = stored gamma + linear residual. Risk/banks/labels/projection không gradient. Phi gồm LN/MLP CMS_fast và readout; theta gồm visual/initial maps/geometry/CMS_slow.

`fast_step` tính gradient của plain half-squared loss rồi chia context energy; tính step FP64, cast về parameter dtype differentiably. Exact replay giữ graph của cả numerator và denominator; FO detach whole quotient. Actual updates detach giữa labels. Bốn query cùng phi_after_support4; actual phi_after_all8 được giữ sau slow Adam. Partial blocks không flush tại task/pause/end. Learned modes warmup32 completed labels; GEOMETRIC không warmup.

Session resolve previous pending trước current proposal. Next normal đánh giá hai fixed banks, cộng cached historical anchor risks; target `(Jminus-Jplus)/stored_c`. Pending chứa raw bank hashes, original decision c/gamma/d/prediction và learner hash; kiểm chronology trước học. Terminal finalize censor đúng một pending, không tạo target giả. No-op không sinh label; bootstrap giữ image remainder.

Actual pre-write R và combined J được log riêng. Scalar p/y không bounded; nonfinite/schema error rollback toàn event rồi stop. Không clamp/drop/retry hoặc fallback để cứu gate. Persistent saved episode33–40 chứa pre-commit model, phi_start và observed projected records; response samples dùng fixed theta trước slow commit, chỉ forward diagnostic không extra learning.

## 4. State và vận hành

Session/learner/adapter/evidence schemas mới từ chối predecessor checkpoints. Restore trên temporary owner, kiểm toàn bộ topology/metadata/FP32 bank và context/Adam slots/counters/chronology/anchors/private RNG/pending/replay trước swap. Derived CADIC cache rebuild; không serialize runtime tensor_version. Score chỉ dùng raw actual bank/Eq9; allocation profiler không thuộc scientific state.

Driver khóa source/runtime/manifest/rows/probe, actual init theta/phi/P/anchor RNG của FO và NESTED phải bằng nhau. Fixed reused-normal probe là8 first rows, mỗi ảnh784 raw NN distances; image/feature hashes và order phải khớp, không zip/truncate. Probe lock được tạo sau discarded preflight và trước fresh pilot jobs. Cache chỉ extract ảnh khi get đúng row; image bytes luôn được xác minh lại. Không aggregate future features.

Runtime canonical dùng pinned T4 Python/packages/CUDA/cuDNN, deterministic algorithms, CUBLAS4096:8, TF32/AMP off. Mismatch dừng trước giải nén dataset/learning. Source/identity khác thì không resume. Hashes của package files, không HEAD đơn lẻ, định danh dirty source.

## 5. Kiểm chứng và delivery

Local suite có **29 tests v5.1**; whole-checkout regression đã PASS **121 tests +27 subtests**, không fail/error/skip. Bao gồm canonical28×28 FP32 full4+4 trên CPU và reduced FP64 finite differences. Reference design checks cũ không được tính vào số production tests mới. Các fixtures không thay real-normal T4 preflight.

`local_tests.py` cấu hình deterministic runtime/CPU2threads trước toàn suite, đặt temp trong worktree và warm Python OS probe trước preserved subprocess mock trên Windows. Một lượt chưa cấu hình có native Windows exception0xc0000008 ở CPU convolution của test visual cũ; source primitive giữ bytes. Giữ log lần lỗi, không dùng stale XML để đếm test hoặc gọi lượt crash là PASS. Lượt canonical-setup regression đã PASS; nguyên nhân native crash chưa được chứng minh triệt để, và không suy local Windows thành nghiệm thu T4/Linux.

Sau khi build/verify, đọc `local_verification.json`, `delivery_receipt.json`, `delivery_verification.json` và logs ở shared delivery directory. Hai dạng source ZIP/Kaggle-extracted phải bootstrap trong process riêng và suite v5.1 phải chạy từ packaged source. Delivery được publish chỉ sau các checks ấy; không overwrite revision đã publish.

Commands tại checkout:

```powershell
pixi run python -B -m unittest discover -s tests/hope_cad_v5_1 -v
pixi run python -B -m notebook_support.support_v5_1.build_delivery
pixi run python -B -m notebook_support.support_v5_1.verify_delivery
```

Inventory mới: `conf/delivery/support_v5_1_source.json`; không sửa inventory/delivery r4 của v5. Delivery ở shared `.agents/artifacts/deliveries/hope_support_v5_1_r1/`. `bundle_manifest.json` trong ZIP chứa full payload hashes và origin; `delivery_receipt.json`/`delivery_verification.json` ở delivery directory. Notebook source nằm tại `notebooks/hope_cad_support_v5_1_kaggle.ipynb`.

## 6. Hướng dẫn Kaggle sau bàn giao

1. Import notebook v5.1 mới. Settings GPU T4 x2. Attach **ba inputs**: successor source ZIP hoặc đúng extracted Dataset, MVTec archive và locked ViT checkpoint. Tháo source/evidence v5, W6/W7/recovery cũ; không dùng learned parent.
2. Giữ `PHASE='pilot'` để preflight24 discarded normals rồi bốn fresh jobs256. Chỉ muốn kiểm trước: `PHASE='preflight'`; PASS rồi attach output đó và chuyển pilot.
3. `SESSION_HOURS` là soft execution budget; default3, có thể đặt9 nếu đủ quota. Nó không đổi learning/trial budget, không phải ETA hoặc lời hứa hoàn tất. Dùng Save Version → Save & Run All.
4. Pause thì giữ ZIP/log, attach **latest v5.1 output** cùng ba inputs để resume; tháo phiên cũ hơn. Không đổi source/runtime/protocol/model. Không dùng ZIP bootstrap thất bại không có pipeline lock để resume.
5. Complete/pause/failure đều giữ Output và tải ZIP/log về shared `.agents/artifacts/runs/support_v5_1_pilot_01/session_01/`; lần resume tăng session_02, session_03… Không overwrite session cũ.
6. Review artifacts trước benchmark. `pilot_complete_scientific_gate_failed` là scientific FAIL, không phải lý do tăng thời gian/rate hoặc đổi thresholds. `pilot_complete_pending_review` chưa là acceptance, AD gain hoặc quyền tự mở benchmark.

Full successor benchmark/valid ablations/seeds/metrics uncertainty/memory fairness/VisA và thesis còn theo planning; không hoàn tất bởi I1/I2 hoặc một pilot normal PASS.
