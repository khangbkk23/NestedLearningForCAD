# Chặng 1 — Nested Learning for Continual Anomaly Detection

Ngày 2026-10-10. Chỉ thao tác `hope_cad_v2_1`, bắt đầu từ `5187e4d3ba5fe756520e2499ec6b90a4b6f91507`. W3 `363634b` đã có trong lịch sử nhánh này; không merge/cherry-pick và không sửa nhánh khác. Code mới chưa commit/push. Artifact ghi HEAD cộng SHA256 từng file và dirty status, **không gán code mới vào commit 5187e4d**.

Quyết định thuật toán giữ nguyên spec v3: `hope_cad_v3_masked_teacher_cms`, `prefix_cms_stress_v3`. Không đổi recurrence, init, teacher, masks, clock, loss, rate grid hoặc gate thresholds. Các nhận định “clone chưa có W3”, “W4 proposed” trong snapshot freeze cũ là lịch sử; bảng dưới là trạng thái implementation hiện tại. Đây không phải phương pháp mới hoặc bản tóm tắt thay thế spec.

## 1. Trạng thái và bằng chứng

| Phần | Implemented | Tested | Experimentally validated |
|---|---|---|---|
| W1 protocol/registry | `dataset/benchmark_protocol_v2.py::{build_prefix_manifest,PrefixContinualProtocol}`, `training/benchmark_config_v2.py::validate_configs`, `models/benchmark_registry_v2.py` | Prefix roles, byte tampering, forbidden reads, canonical schedules, explicit route, real per-image coreset updates, Eq.9, mask QA và mount-portable lock bằng fixtures | Chưa có matched CADIC 10k run theo prefix mới; kết quả Khang vẫn giữ là real historical/paper-control |
| W3 visual operator | Source từ `363634b`, không đổi các file W3 cũ | Suite W3 24 tests chạy lại; full-grid init diagnostics; pinned upstream CUDA parity PASS trên T4 cho forward/state/first-gradient low-level contract | Chưa chứng minh visual dynamics hữu ích cho CAD |
| W4 CMS | `models/hope_cad_v2/cms.py::{ContinuumMemory,CMSLevel}`, `state.py`, `teacher_head.py`, `predictor.py` | FP64 central finite differences sau 16 support; gradient tới input/decoder/hai initials; pre-event mean oracle; FP32/FP64 component tests | Chưa chạy normal-prefix thật |
| W4 state lifecycle | `training/hope_cms_session_v2.py::CMSSession` | Resume tại event 5, `(16,2)` commits, batch transport equivalence, read-only scoring, atomic state/cursor/RNG rollback | Full W5 RGB adapter chưa có |
| W4 outer/selection | `training/hope_outer_learning_v2.py`, `hope_selection_v3.py`, `hope_cms_stress_v3.py` | Optimizer/RNG resume khớp; failed fit không cứu bằng checkpoint cũ; drift bị từ chối; long gate đủ 1024 events trên fixture nhỏ; CLI synthetic selection/export chạy được | Chưa có β/step được chọn cho method khoa học; không dùng β từ smoke |
| Resources | `profile_stage1.py`, notebook Kaggle | Canonical 784×768 / 256 / 128 / 8 heads ×16, `create_graph=True`, forward/backward PASS trên Tesla T4; peak allocated ≈1.890 GB | Đây là synthetic resource/correctness gate, không phải CAD performance experiment |

Các trạng thái tách biệt: **implemented** có source; **tested** có test/artifact cùng source hashes; **experimentally validated** cần dữ liệu/recipe đúng và kết quả thật. Kaggle technical acceptance của Chặng 1 đã đạt; real-prefix/matched-CADIC evidence vẫn thuộc các chặng thí nghiệm tiếp theo và chưa được gọi là scientific validation.

Artifact local trong `results/hope_cad_v2/` (thư mục gitignored, cần export):

- `stage1_verification.json`: suite W1/W3/W4; source/spec hashes, runtime, fail/skip, số tests thực tế trong JSON.
- Regression harness CADIC v1: **16/16 pass**, JUnit tại `stage1_legacy_regression.xml`. Môi trường Pixi không có pytest/pip; đã dùng một venv tạm kế thừa packages Pixi để chạy pytest, không đổi `pixi.toml`/`pixi.lock`.
- `stage1_w3_verification.json`: W3 24/24 pass trên checkout hiện tại. Không thay artifact Kaggle cũ.
- `stage1_contraction.json`: full-grid canonical init, seed 103, raw features Gaussian, CPU FP32. Có chunk/direction/head norms, gain norms, alpha/eta/cap activity, restored spatial RMS, zero/subnormal và visual-on/off deltas.
- `stage1_reduced_profile.json`: exact 16 support, reduced config `(input=8, dim=8, memory=8, head=4, H=W=4)`, RTX 3050 Ti. Forward ≈1.30 s, backward ≈0.41 s, peak allocated ≈19.0 MB trong lần đo đầu; chỉ là fixture, không ngoại suy lên canonical/T4.
- `stage1_canonical_profile.json`: `pending_external_gpu`; local GPU không đáp ứng điều kiện chạy gate canonical. Chưa có số canonical runtime/VRAM.
- `stage1_upstream_parity.json`: `pending_linux_cuda_toolchain`; source pin/hash đã xác minh, máy Windows không có `nvcc`.
- `utils/stage1_evidence.zip`: Kaggle Tesla T4 evidence, SHA256 `3b2cc3144e6dd0d3fb2002932e772c753485b5c557254b96a8b545c2c6265ad8`. Bundle có 111-file manifest; verifier 56/56 PASS, không fail/error/skip; bốn subprocess đều return code 0.
- Trong evidence Kaggle, canonical exact-16 chạy `create_graph=True`: forward ≈4.827 s, backward ≈7.713 s, peak allocated ≈1,889,753,600 bytes, peak reserved ≈1,906,311,168 bytes. Upstream parity có 19/19 fixtures PASS tại pin `98cfb5d6004472a0ba3d212d3cf884393e39c134`; phạm vi chỉ là normalized-query FP32 forward/state/first-gradient low-level, không phải full-backbone hoặc Hessian parity.
- `stage1_selection_smoke/`: CLI M finite search, từng gate, optimizer resume checkpoints, selection payload và deployment initials trên dữ liệu giả/reduced/1 outer step. Không hợp lệ để chọn hyperparameters cho MVTec/VisA.
- `stage1_outer_diagnostics_reduced.json`: exact 16-support unroll trong 25 outer steps trên fixture reduced; callbacks ghi visual diagnostics ở 0/1/5/25 và group gradient norms. Không thay post-calibration canonical/real-feature diagnostics còn pending.

Init warning mới: `RMS(Pout(V))/RMS(X) ≈ 5.5672e-4`; RMS phần tử content memory cuối ≈`9.3748e-39`, khoảng 76.98% subnormal. Đây là fixture wrapper mới, **không so như cùng fixture** với con số standalone memory trong audit cũ. Không đổi retention/init để xử lý. Cần post-calibration diagnostics và final-score interventions W5/W6 trước claim H3.

## 2. W1: contract thực thi

`conf/benchmarks/protocols/{mvtec_1x15,visa_1x12}_prefix_cal_v2.yaml` giữ prefix 100, minimum 32, 50/20/15/remainder, 16 distinct support. Manifest lưu SHA256 bytes mỗi ảnh train; roles chỉ thuộc task đầu. Online stream loại validation/probe, giữ support/query và normals ngoài prefix; matched CADIC đọc cùng stream. `PrefixFeatures` không cấp quyền probe, future task hoặc official test.

V1 validator/runners/adapters không sửa. Registry mới phân biệt:

| Adapter ID | Effective implementation | Unit |
|---|---|---|
| `cadic_loader_batch_v1` | `models.cadic_benchmark_adapter_v1.CADICBenchmarkAdapterV1`, Euclidean cdist | loader batch; không được gắn nhãn image |
| `cadic_paperfaithful_image_v1` | `models.cadic_paperfaithful.cadic_benchmark_adapter.CADICBenchmarkAdapterV1`, clamp matmul rồi sqrt | image |
| `cadic_paperfaithful_loader_batch_v1` | Cùng paperfaithful implementation | loader batch |

Main matched config khóa **paperfaithful image / budget 10000 / b=9 / raw ViT block index 8 / RGB224 ImageNet normalization / full Eq.9 score**. Checkpoint thật phải có SHA256 và identity; không fallback random backbone. Template chưa tự chứng minh đúng author checkpoint hoặc exact paper parity. B0 paper-control của Khang không bị sửa hoặc chạy lại tự động.

`run_matched_cadic.py --phase prepare` tạo `baseline_lock.json`, manifest/roles, resolved configs và provenance trước khi xem HOPE anomaly metrics. Route/kernel/extractor/metric source hashes nằm trong lock. Resume không chấp nhận đổi data bytes, config hoặc effective route. Root mount/checkpoint file path có thể đổi nếu identities/hashes giữ nguyên. Thứ tự main alphabetic; order study chỉ seeds 100/200, cố định task đầu và shuffle phần còn lại. Validator không tự mở quyền calibration sang category khác.

VisA parser đọc official `split_csv/1cls.csv`, kiểm labels/split/category, duplicate/overlap và full counts 8659 train /2162 test /1200 anomalous test. Evaluator-only mask QA kiểm original geometry, nonempty anomaly masks, raw nonzero encoding (kể cả 0/1, palette), nearest resize và danh sách defects bị xóa sau resize. Giữ image label và pixel zeros. Category không còn positive pixel phải báo co-primary incomplete, không drop category. QA không đi vào calibration provider.

Chuẩn bị khi đã có assets (PowerShell, sau `pixi shell`):

```powershell
$env:MVTEC_ROOT = 'D:/data/mvtec_ad'
$env:VIT_B8_CHECKPOINT = 'D:/weights/vit_b8_in21k.pt'
$env:VIT_B8_SHA256 = (Get-FileHash -LiteralPath $env:VIT_B8_CHECKPOINT -Algorithm SHA256).Hash.ToLower()
python scripts/hope_cad_v2/run_matched_cadic.py --run-dir results/cadic_matched_prefix --phase prepare
```

`train/eval/fm` là các phase riêng; dùng cùng arguments, `--resume` khi có task checkpoints. Caps `--max-tasks/--max-train-images` chỉ cho `--smoke`; phải giữ cùng caps trong lock. Runner dùng exact metrics/triangular evaluator hiện có, không sửa metric theo HOPE. Metadata `reportable:false` được giữ cho tới khi suite/review đủ evidence, không tự biến một phase thành báo cáo khoa học hoàn chỉnh.

**W1 evidence ngoài code chưa đóng:** repo hiện chỉ có thư mục results HOPE, chưa có bộ raw CADIC provenance/output để chứng minh historical run là matched-prefix; chưa có matched final run; chưa có xác nhận schema/task-continuation artifacts ReplayCAD từ Khang. Giao diện adapter chung v1 vẫn được giữ để tích hợp baseline khi Khang sẵn sàng. Không thay thế việc phối hợp đó bằng một claim “ReplayCAD đã matched”.

## 3. W4: graph, clocks, selection và checkpoint

`ContinuumMemory.event` tính cả hai outputs/local losses từ state trước event. `autograd.grad` chỉ lấy derivative theo current parameters của level tương ứng, gồm LayerNorm affine. Mỗi level có accumulator/count, commit mean gradient khi due. Validate mọi proposal hữu hạn trước trả state mới; không mutate input state. Meta dùng `create_graph=True` và giữ dependencies qua input/decoder/history; deployment detach sau event. Không có task-reset/flush API.

`HOPEFeaturePredictor` ghép W3→CMS→decoder ở **mức frozen patch features**, không phải RGB detector hoàn chỉnh. M/C0/C1/C2/C3/C4/A1 có dynamics metadata riêng. Static visual control vẫn đọc learned initial content maps; key/value/eta/alpha inactive được freeze và khai báo số active parameters. C4 bắt buộc nhận selected M beta, rates `(β_M,β_M/8)`, periods `(1,1)`; không có independent beta grid.

`fit` thực hiện exact 16 distinct support + một read-only query; reset learned CMS initials mỗi episode. Adam/clip/checkpoints giữ recipe. C0/C1 dùng mean 17 image losses, không inner update. Checkpoint callback lưu model, optimizer, RNG, complete schedule, step và current shortlist; resume kiểm metadata/schedule/recipe. Candidate có nonfinite train/validation bị loại cả fit. Shortlist chỉ một checkpoint/rate; gate fail không thử checkpoint khác cùng rate.

`run_stress_gate` lặp đúng canonical support+query 1024 events, validation ở 0/16/64/128/256/512/1024. Norm FP64 trên FP32 update, `R_a` đo trước reset kể cả chưa due. Pass thresholds giữ 10/1/2/1.25. Gate chỉ lọc; ranking vẫn short loss→step→beta. Clone terminal không xuất sang deployment. `search` reset chung RNG/schedule giữa candidates; C4 nhận selection payload của M.

`CMSSession` deep-copy/freeze slow parameters, zero current CMS một lần. Mỗi factory nhận view theo global clock. Failure khôi phục RNG và giữ cursor/state; checkpoint giữ pending gradients/counts, current weights, learned initials, slow weights, RNG và identity. Query read-only không sửa persistent state. Transport batch là danh sách ordered image transactions; failure ở ảnh thứ k giữ các ảnh đã commit trước đó và không commit ảnh lỗi.

CLI `calibrate_features.py` chạy selection/export trên **explicit frozen prefix bundle**, hoặc `--synthetic-smoke` được đánh dấu không reportable. Real bundle phải có role manifest, source/backbone/preprocess/mask identity và tensor hashes; features `[1,784,768]` FP32, clean teacher unit-normalized, masks áp RGB trước ViT ở producer W2/W5. Bundle chứa đúng support/query/validation; probe bị loại. Cache producer thực và full RGB adapter vẫn là W2/W5. CLI lưu optimizer checkpoints nhưng orchestration resume toàn search qua nhiều session và full-run evidence ledger sẽ ghép trong W2/W5; API `fit(resume=...)` đã kiểm thử.

Causal diagnostics sau selection, locked-seed refit/VisA transfer of recipe và final anomaly evaluation vẫn thuộc W5/W6/W7/W8. Không lấy một synthetic shortlist làm selected scientific recipe. Các hooks `diagnostic_callback(step, model)` có ở 0/1/5/25; post-fit full canonical diagnostics và selected-checkpoint diagnostics phải chạy khi có GPU/features đúng, chưa gọi completed từ init trace.

## 4. W3 upstream và resource handoff

Upstream `PSRben/VisionHOPE@98cfb5d6004472a0ba3d212d3cf884393e39c134`, MIT. `upstream_pin.json` lưu hashes operator/CUDA/utils/license tải từ commit đó. `verify_upstream.py` chỉ import low-level modules đã kiểm hash, không cài upstream training package hoặc thay backend W4.

Harness copy exact tensors cho local/upstream; seeds 103/104/105, head dims 4/16, chunks 1/7/28, multi-image/direction và sequence 784. Gọi normalized-query API với order content/value/key/eta/alpha đúng upstream. Main fixtures kiểm bound đảm bảo clamp inactive; ngoài vùng clamp ghi expected divergence. Forward/state tolerances `1e-5/1e-4`; first derivatives `1e-5/1e-3`. Hook extension output buffers `save_g=1` quan sát chunk gains để dựng cả năm boundary/final states, không patch CUDA source. Partial tail so valid output, state riêng theo padded terminal. Numerical gate đã chạy trên upstream CUDA/T4 và 19/19 fixtures PASS; phạm vi fidelity vẫn giới hạn như khai báo ở trên.

Notebook: [hope_cad_stage1_kaggle.ipynb](../../notebooks/hope_cad_stage1_kaggle.ipynb).

1. Upload `results/hope_cad_v2/stage1_source_bundle.zip` thành Kaggle Dataset, attach vào notebook. Notebook kiểm toàn bộ source hashes sau giải nén.
2. Bật một GPU T4 và Internet, chạy các cells. Chưa cần MVTec/ViT checkpoint, chưa dùng anomaly metrics.
3. Tests→canonical init diagnostics→canonical exact16 resource spike→pinned upstream CUDA parity. Chạy independent parity kể cả resource gate fail; không tự giảm dimensions/unroll hoặc detach history.
4. Tải `/kaggle/working/stage1_evidence.zip`, đặt lại vào repo/gửi kết quả. Giữ failed/OOM artifacts và command log.
5. Rà kết quả trước scientific pilot. Nếu lỗi code/numerical thì sửa và kiểm lại; nếu cần đổi thuật toán thì báo evidence trước method revision. Có thể triển khai W2/W5 độc lập theo scope tiếp theo, nhưng không vượt acceptance thành full MVTec.

Local reproduce:

```powershell
pixi run python -B scripts/hope_cad_v2/verify_stage1.py --output results/hope_cad_v2/stage1_verification.json
pixi run python -B scripts/hope_cad_v2/diagnose_visual.py --output results/hope_cad_v2/stage1_contraction.json
pixi run python -B scripts/hope_cad_v2/profile_stage1.py --device cuda --reduced --output results/hope_cad_v2/stage1_reduced_profile.json
pixi run python -B scripts/hope_cad_v2/package_stage1.py --output results/hope_cad_v2/stage1_source_bundle.zip
```

Lệnh reduced là smoke có nhãn rõ. Lệnh canonical không có `--reduced`; CLI trả pending khi local GPU dưới 12GiB. Đây là guard tài nguyên của harness, không đổi method. Notebook dùng subprocess để OOM có thể được lưu/tiếp tục đóng gói evidence. Source-only bundle loại data, weights, results, `.git` và mọi thư mục ngoài allowlist.
