# Nested Learning for CAD — Audit và kế hoạch v3

Ngày chốt revision v3: 2026-10-10. Trạng thái: **BƯỚC 3–4 HOÀN TẤT Ở MỨC SPEC; ARCHITECTURE/SELECTION FROZEN FOR IMPLEMENTATION; gate mới pending**.

Tài liệu này giữ đầy đủ planning A/B và contract v3 trong [architecture_decision_v3.md](architecture_decision_v3.md); [instruction_CAD_v3.md](instruction_CAD_v3.md) là guide A–Z. [Báo cáo chặng 1](stage1_implementation.md) cập nhật implementation W1/W3/W4 trên `hope_cad_v2_1`; không thay code v1, spec v2 hoặc các quyết định toán học đã freeze.

**Quy ước phiên bản.** Đây là bản đầy đủ kế thừa v2, tích hợp các sửa đổi sau audit; ba tài liệu v2 và báo cáo W3 cũ được giữ nguyên. Design/method ID mới là `hope_cad_v3_masked_teacher_cms`, selection-rule ID là `prefix_cms_stress_v3`, vì quy tắc chọn cấu hình đã thay đổi. Topology, recurrence, initialization, teacher, masks, final score, clocks và quyền dữ liệu cốt lõi giữ nguyên. Namespace code `models/hope_cad_v2/`, `scripts/hope_cad_v2/`, các tên module `_v2` và protocol IDs `mvtec_1x15_prefix_cal_v2` / `visa_1x12_prefix_cal_v2` vẫn là các tên kế thừa; không tạo bản sao code chỉ để đổi số. Full run phải ghi method ID v3 và selection-rule ID riêng, không lấy component metadata W3 v2 làm identity của full method.

**Checkout hiện hành — chặng 1.** Theo chỉ dẫn mới nhất của người dùng, chỉ làm việc trên `hope_cad_v2_1`, bắt đầu từ `5187e4d3ba5fe756520e2499ec6b90a4b6f91507`; W3 `363634b` đã có sẵn. Code mới W1/W4 ở working tree, chưa commit/push. [Báo cáo chặng 1](stage1_implementation.md) xác định source/function, tests/artifacts và các gate còn pending. Snapshot `a89fd48`/`hope_cad_v2_clone` và các nhãn “proposed/not implemented” trong nội dung freeze bên dưới được giữ làm lịch sử tại thời điểm soạn spec; trạng thái implementation hiện tại lấy theo báo cáo chặng 1. Không thay hợp đồng toán học hoặc gán code mới cho commit cũ.

## 0. Mục tiêu và các ràng buộc đã biết

Đề tài là **Nested Learning for Continual Anomaly Detection (CAD)**. Câu hỏi nghiên cứu chính: một hệ học lồng nhau, với các trạng thái và mục tiêu học ở nhiều thang thời gian, có thể học đặc trưng bình thường của các task mới mà duy trì khả năng phát hiện/localize bất thường trên task cũ hay không?

HOPE là cấu trúc phương pháp chính. VisionHOPE cung cấp cơ chế tự cập nhật phù hợp với ảnh và kiểm soát ổn định. Việc chỉ thêm một visual adapter, chỉ tối ưu anomaly score, hoặc chỉ chứng minh memory không NaN chưa trả lời câu hỏi CAD.

- Backbone đối chứng chung: frozen ViT-B/8 ImageNet-21k, cùng checkpoint, preprocessing và layer trích xuất.
- MVTec AD trước; VisA là phần bắt buộc tiếp theo. Xây hạ tầng dùng chung từ đầu, nhưng chỉ chuyển sang thí nghiệm VisA sau khi pipeline MVTec đã qua nghiệm thu.
- Bạn phụ trách HOPE; Khang phụ trách ReplayCAD. Baseline, dataset protocol và result schema là giao diện phối hợp.
- Kaggle T4×2 là môi trường phát triển/thí nghiệm chính, với quota GPU 30h mà người dùng cung cấp. TPU v5e-8 có quota 20h nhưng chưa nằm trên đường chạy bắt buộc. GPU khoảng 50GB của Khang chưa có model/runtime xác nhận.
- CADIC trong ảnh người dùng cung cấp là **kết quả chạy thật**. Metadata còn thiếu chỉ ảnh hưởng việc tái lập và so sánh đúng protocol.

## A. Báo cáo tiến độ dựa trên bằng chứng

### A1. Snapshot Git và nhánh khởi đầu

**Bảng dưới là snapshot audit v2 ngày 2026-10-09, không phải danh sách refs hiện hành.** Ngày 2026-10-10 đã xác minh `hope_cad_v2_clone@a89fd48` là nhánh duy nhất được sửa tài liệu; `hope_cad_v2@363634b` có hai commit W3 phía sau. Người dùng yêu cầu giữ clone ở commit cũ. Mọi câu về việc tạo nhánh/working tree trong phần lịch sử dưới đây được đọc theo thời điểm audit, không là lệnh merge hoặc trạng thái HEAD hiện tại.

| Ref tại thời điểm audit | Commit | Ý nghĩa |
|---|---|---|
| `taitrn_2` | `440da401466d30531d804c487a0e93adc77866a7` | Cùng snapshot với `origin/main` |
| `origin/main` | `440da40` | Nhánh tích hợp đã có baseline CADIC |
| `main` local | `9b032b1` | Sau `origin/main` 13 commits; không dùng HEAD local này làm nền mới |
| `dikhang_space`, `origin/dikhang_space` | `a89fd48` | Snapshot mới nhất đã pull, có HOPE core và các pilot |
| Snapshot HOPE trước hai commit mới | `44994f3` | Audit cũ về thiếu anomaly/outer-learning chỉ áp dụng ở phạm vi snapshot này |
| Commit mới | `4df4358`, `a89fd48` | Thêm anomaly/readout, image-synchronous memory, normality audit và outer-learning pilot |

Đã tạo và chuyển sang nhánh riêng **`hope_cad_v2`**, từ `dikhang_space@a89fd48f94ed9c5deb26041151ea63b79911342f`, ngày 2026-10-09 theo yêu cầu người dùng. Không tiếp tục phát triển HOPE trực tiếp trên nhánh Khang dùng cho ReplayCAD. `main` tiếp tục là nhánh tích hợp; chưa commit/push các tài liệu v2 trong lượt này.

Lý do: `dikhang_space` có phần HOPE có thể kế thừa; diff trực tiếp giữa `origin/main` và snapshot này không thay đổi các file trong `models/cadic_paperfaithful/`, `models/cadic_patch_coreset_v1.py`, `models/feature_extractors/cadic_vit_v1.py`, `training/benchmark_engine_v1.py`, `training/benchmark_metrics_v1.py`. Hai nhánh có lịch sử merge khác nhau, nên đây không phải khẳng định quan hệ fast-forward. Khi tích hợp sẽ merge/rebase có kiểm tra, không thay lịch sử của Khang.

Lệnh đã thực hiện: `git switch -c hope_cad_v2 dikhang_space`. Đã xác minh HEAD bằng commit nguồn; giữ nguyên các tài liệu v2 và VisionHOPE PDF đang untracked. Không kéo các thay đổi ReplayCAD về liên tục nếu chúng không cần cho giao diện benchmark.

### A2. Cách dùng nhãn trạng thái

- **implemented**: có implementation cụ thể tại ref đã ghi.
- **tested**: có kết quả chạy test, commit và môi trường tương ứng. Có file test chưa đủ để gắn nhãn này.
- **experimentally validated**: có thí nghiệm thực trên dữ liệu liên quan, với phạm vi kết luận được nêu. Chạy được không đồng nghĩa chứng minh component tạo gain.
- **not implemented**: phần yêu cầu còn thiếu trong phạm vi code được kiểm tra.

Phiên audit ban đầu tại `a89fd48` đọc source và metadata Git; không chạy pytest của repository, không load weights và không chạy benchmark GPU. Lượt rà soát cuối có kiểm tra số học in-memory cho công thức meta-gradient, clocks và masks của spec; kết quả/phạm vi ghi ở architecture decision §9, không chuyển nhãn implementation v2 thành tested. Không tìm thấy thư mục kết quả/checkpoint cần thiết của các pilot trong workspace hiện tại. Các chuỗi `PASS` hay số lượng test được ghi cứng trong script không được coi là log test đã chạy.

### A3. Bảng implementation và khoảng cách tới mục tiêu

Các đường dẫn là tương đối từ repository root. Các hàng HOPE/pilot/ReplayCAD áp dụng ở `dikhang_space@a89fd48`; các hàng baseline được đối chiếu thêm với `taitrn_2@440da40`.

| Hạng mục | Evidence: file và symbol | Trạng thái, giới hạn |
|---|---|---|
| Backbone CADIC | `models/feature_extractors/cadic_vit_v1.py::CADICViTFeatureExtractor.extract_patch_features` | **implemented**: strict local checkpoint, frozen ViT, block 8/layer 9, bỏ CLS, 784×768 raw patches. `protocol_metadata` vẫn khai báo các giả định tái lập |
| Patch memory CADIC | `models/cadic_patch_coreset_v1.py::CADICPatchCoresetV1.update/score`; `models/cadic_paperfaithful/cadic_patch_coreset.py::CADICPatchCoresetV1` | **implemented**: patch là memory unit; hai đường tính khoảng cách/adapter cần phân biệt |
| CADIC theo từng ảnh | `models/cadic_paperfaithful/cadic_benchmark_adapter.py::CADICBenchmarkAdapterV1.fit_task` | **implemented**. Adapter thường `models/cadic_benchmark_adapter_v1.py::fit_task` vẫn đưa cả loader batch vào update; cấu hình `update_unit=image` một mình chưa bảo đảm semantics |
| Launcher paperfaithful | `scripts/benchmarks/run_cadic_paperfaithful.sh::run_phase` | **implemented**: thay import adapter bằng `sys.modules`, seed 0, image update, budget truyền từ CLI; chạy train/eval, chưa gọi phase forgetting |
| Kết quả CADIC người dùng cung cấp | Hai bảng đính kèm; người dùng xác nhận thực nghiệm thật | **experimentally validated** ở phạm vi kết quả aggregate báo cáo. Chưa gắn được các bảng với commit/config/checkpoint/run directory cụ thể; không phủ nhận kết quả |
| Benchmark CADIC/CAD | `training/benchmark_engine_v1.py::BenchmarkEngineV1.train/evaluate_final/evaluate_forgetting`; `training/benchmark_metrics_v1.py::compute_metrics/forgetting_matrix` | **implemented**: checkpoint theo task, final macro, all-pixel metrics, ma trận forgetting. Kiểm tra mutation hiện là tùy chọn; cần bắt buộc cho HOPE |
| Test baseline/harness | `tests/test_benchmark_harness_v1.py` | Test source tồn tại; **tested: chưa xác minh ở snapshot này** |
| HOPE self-modifying core | `models/hope_cad/self_modifying_titans.py::SelfModifyingTitans` | **implemented**: nhiều map tự cập nhật, fixed-query mặc định, chunk state. Khác single-matrix TITANS cũ. Chưa phải operator VisionHOPE |
| CMS | `models/hope_cad/continuum_memory.py::ContinuumMemorySystem.commit_image` | **implemented**: residual MLP nhiều level, period theo ảnh, tích lũy gradient, state/counters/checkpoint. Objective CAD vẫn do caller cung cấp |
| HOPE composition | `models/hope_cad/hope_block.py::HopeBlock.commit_image/evaluate_image` | **implemented**: self-modifying core → CMS; atomic rollback, test clone. Chưa có production anomaly adapter hoàn chỉnh |
| Tests core | `tests/test_self_modifying_titans.py`, `tests/test_continuum_memory.py`, `tests/test_hope_block.py` | Có test lifecycle, state và composition; **tested: chưa xác minh log chạy** |
| Image-synchronous P1/P2 | `exps/hope_image_synchronous_memory.py::aggregate_image_statistics/propose_transition/ImageSynchronousMemory` | **implemented**, thêm ở `4df4358`: P1 dùng cùng right-action theo ảnh cho 5 map; P2 chiếu SVD cả transition. Đây không phải stability control token-wise của VisionHOPE |
| Anomaly readouts pilot | `exps/hope_anomaly_signal.py::residual_scores/evaluate_metrics`; `exps/hope_anomaly_score_ablation.py::angular_scores/affinity_scores` | **implemented**, thêm ở `4df4358`: residual, cosine, spatial affinity, covariance/CADIC controls. Các runner này tắt CMS; không chứng minh gain của full HOPE |
| Normality diagnostics | `exps/hope_normality_audit.py::relational_alignment/direction_overlap/fixed_teacher` | **implemented**, thêm ở `a89fd48`: alignment, fixed-teacher và gradient diagnostics; không phải final CAD benchmark |
| Functional outer-learning | `exps/hope_outer_learning_p1_v1.py::functional_support_sequence`; `exps/hope_outer_learning_pilot_v1.py::support_sequence` | **implemented**: gradient tới initial content và query; transition/auxiliary graph-free. Không phải meta-learning toàn bộ update policy |
| Real-data outer-learning runner | `scripts/exps/hope_outer_learning_pilot_v1.py::fit_arm/run/evaluate_all` | **implemented**: Adam trên `A0,Wq`, bottle-only 170 normals chia vai trò, masked RGB teacher, static/history controls. `CMS=DISABLED`; không chạy anomaly evaluation. **experimentally validated: chưa xác minh artifact** |
| Full VisionHOPE-CAD | Kiểm tra `models/hope_cad/`, adapter registry/CLI và các runner trên | **not implemented**: chưa có 4 scans + VisionHOPE guards + CMS CAD objective + final head + full benchmark thống nhất |
| ReplayCAD integration | `scripts/benchmarks/replaycad/run.py::main/run_detector`; `training/replaycad_normalizer.py`; gitlink `refs/ReplayCAD@acc619597c3d14aa35f20f42ac2ed9d1c012678b` | **implemented** ở mức scaffold/native pipeline; mode `continual_boundaries` còn unsupported. Chưa đủ evidence cho continual forgetting; Khang đang phát triển |
| VisA theo benchmark mới | `dataset/benchmark_protocol_v1.py`; protocol files trong `conf/benchmarks/protocols/` | **not implemented** cho giao thức VisA chính thức tương đương MVTec. Config/loader VisA cũ không thay thế phần này |
| Môi trường tái lập | `pixi.toml`, `requirements.txt`, các script mới dùng pandas/parquet/timm | **implemented một phần**: Pixi có win/linux nhưng Python 3.14 và dependency set chưa bao phủ đường thí nghiệm mới; cần lock môi trường Linux/Kaggle đã chạy thử |

### A3.1. Cập nhật W3 sau snapshot audit

Ngày 2026-10-09, source W3 được thêm trên `hope_cad_v2`, sau đó commit thành `6dbe4f1`: `models/hope_cad_v2/{scan_geometry,visual_memory,reference,visual_block}.py`. `FourScanVisualMemory`, `scan_chunked`, `HOPECADVisualBlock` thực hiện visual contract; `scan_reference` là independent FP64 oracle. Artifact local `results/hope_cad_v2/w3_verification.json` ghi 23 tests pass, 7/7 hashes khớp `6dbe4f1`. Theo transcript Kaggle do người dùng cung cấp, source này sau đó fail canonical full-grid CPU backward với `DivBackward0` ở spectral guard. Commit `363634b` sửa expression gần norm² floor, thêm regression test thứ 24. **Implemented/tested tại source mới:** [artifact Kaggle](../../results/hope_cad_v2/w3_verification_kaggle.json) ghi 24/24 pass, không failures/errors/skips, `2026-10-09T15:46:02Z`; đã đối chiếu 7/7 hashes khớp `363634b18d98eefc6da28660a99b967988077ef3`. Runtime Linux/Python 3.13.15/PyTorch 2.11.0+cu128, CUDA 12.8, Tesla T4. Full-grid test chạy CPU; test CUDA dùng reduced fixture. [Báo cáo W3 v2](w3_implementation_v2.md) giữ cả hai artifacts và lịch sử failure, không gộp thành một run. **Checkout clone vẫn ở `a89fd48`, không có các W3 source files này**; chưa experimentally validated CAD, chưa profile full-grid/16-event meta-unroll trên T4.

### A3.2. Audit bổ sung và phần chưa được chứng minh

**Bằng chứng và phạm vi.** Audit trước đã chạy `FourScanVisualMemory.inspect` với seed 103, input Gaussian `[1,784,128]`, FP32, chưa outer calibration. Norm tính FP64: mean initial content norm ≈`1.40e-1`, final ≈`1.36e-37`; direction-0 output RMS ở chunk đầu ≈`8.73e-3`, chunk thứ 14 ≈`1.42e-19`, chunk cuối ≈`1.63e-37`. Đây là synthetic measurement trong transcript audit trên source W3 trước fix NaN, **không nằm trong artifact 23-test và không phải CAD experiment**. Cần lưu artifact tái lập khi làm diagnostics mới; không gán các số này cho `a89fd48` hoặc coi artifact Kaggle 24-test là phép đo lại contraction sau fix.

Ở initialization, zero α-map cho retention 0.9; `0.9^784` giải thích một bậc độ lớn khả dĩ, **không phải định luật co chính xác cho mọi input** vì injection, normalized key và boundaries cùng tham gia recurrence. Zero control maps giữ zero qua inner right-action trước outer learning không có nghĩa outer gradient tới initialization bằng zero. Không suy ra mọi outer gradient giảm đúng `0.9^t`, hoặc outer learning không thể phục hồi utility.

**Thứ tự thực hiện:** ngay sau W3, trước đầu tư đầy đủ W4, đo init trên fixtures cố định; thêm real features chỉ từ `meta_support`/`meta_query` khi W2 có extractor/weights. Theo từng direction/head/chunk và restored spatial position, ghi memory/gain norms, output RMS, zero/subnormal fractions, α/η distributions và cap activity. Với full spatial wrapper, ghi `RMS(P_out(V(Z)))/max(RMS(X),1e-12)` và visual-on/off output deltas. Khi minimal W4 đủ chạy vài exact outer steps, đo lại ở steps 0/1/5/25 và selected checkpoint; chạy gradient diagnostics theo nhóm initial maps/q/projections/decoder/CMS initials, rồi final-score interventions ở W5/W6. Không chờ tới W7 mới nhìn contraction.

**Acceptance:** numerical NaN/Inf, state mutation sai hoặc gradient bị disconnect do implementation là correctness failure. Biên độ nhỏ ở init là **warning cần đo tiếp**, không tự động fail kiến trúc hoặc ép đổi initialization. Diagnostics completeness là gate sớm; utility được đánh giá sau calibration bằng trained static/reset/freeze controls và final-score sensitivity. Không đặt một cutoff dựa vào fixture đã xem rồi gọi là preregistered pass. Nếu chưa có bằng chứng utility thì ghi inconclusive/no-go cho claim H3, không bịa threshold đảm bảo gain. Mọi sửa init/retention/guards để xử lý quan sát phải thành method revision trước anomaly-test evaluation.

Upstream core đã được đối chiếu source/đại số với recurrence và numerical parity trực tiếp đã PASS 19/19 fixtures trên Tesla T4 tại pin `98cfb5d6004472a0ba3d212d3cf884393e39c134`. Gate dùng explicit weights/queries/controls và tránh vùng clamp ±10 không chung giữa implementations; first-order upstream không thay higher-order oracle. Phạm vi và tolerances đầy đủ ở [decision §3.3](architecture_decision_v3.md#33-fidelity-với-paperupstream-và-gate-so-số). Đây là low-level normalized-query forward/state/first-gradient parity, không phải full-backbone/Hessian hoặc CAD utility validation.

### A4. Kết quả CADIC đã có

| Patch budget | CADIC paper I-AUROC | Run thật I-AUROC | CADIC paper P-AUPR | Run thật P-AUPR |
|---|---:|---:|---:|---:|
| 2,500 | 95.50% | 84.27% | 55.90% | 40.40% |
| 10,000 | 97.20% | 87.72% | 58.40% | 46.71% |

Run thật còn có I-AP 93.37%/94.77% và P-AUROC 86.65%/90.02% tương ứng 2.5k/10k, theo ảnh người dùng cung cấp. Số paper là tham chiếu Table VII CADIC, không phải kết quả chạy trong repository.

Các run này là baseline thực tế của nhóm. Khoảng cách với paper cần ghi nhận và truy vết checkpoint, preprocessing, adapter, initialization/update unit, task order và metric resolution; không biến việc đạt đúng số paper thành điều kiện chặn mọi nghiên cứu HOPE.

Thông tin hữu ích để liên kết run: đường dẫn output hoặc notebook/lệnh chạy, `config_resolved`, hash checkpoint và commit. Nếu artifact đang ở máy/Kaggle của Khang, chỉ cần lấy một lần khi đồng bộ benchmark. Script `run_cadic_paperfaithful.sh` là một ứng viên đã tìm thấy, chưa được xác nhận chính là launcher của hai bảng.

#### A4.1. Rà config CADIC: những điểm bất thường đã xác định

Người dùng cho biết config được nhắc tới có thể là nguồn của hai bảng, nhưng tin nhắn chưa định danh một đường dẫn/file mới. Audit đã đọc cả ba config có liên quan và toàn bộ đường wrapper → setup → runner → adapter → score. Chưa gán chắc hai ảnh vào một launcher chỉ từ tên YAML.

| Finding | Evidence tại `a89fd48` | Kết luận và xử lý trong W1 |
|---|---|---|
| Hai method YAML chỉ khác `id` | `conf/benchmarks/methods/cadic_compatible_v1.yaml`, `cadic.yaml` | Cùng ViT-B/8, block 8, raw features, 224, budget mặc định 2500, b=9, float32, batch 8. Tên khác không có nghĩa thuật toán khác |
| Config ghi image update nhưng runner thường dùng batch update | `models/cadic_benchmark_adapter_v1.py::fit_task` gọi `coreset.update(patches)` cho cả batch; `scripts/benchmarks/1_run_benchmark.py::main` import adapter này; wrapper `scripts/benchmarks/cadic/run.py::main` chỉ delegate | **Lệch semantics thật giữa declaration và execution.** Cần explicit adapter ID và ghi `effective_update_unit`, không chỉ hash YAML |
| Launcher paperfaithful thay adapter lúc runtime | `scripts/benchmarks/run_cadic_paperfaithful.sh::run_phase` monkeypatch import; `models/cadic_paperfaithful/cadic_benchmark_adapter.py::fit_task` loop từng ảnh | Nếu run dùng launcher này thì image-update được thực hiện; không quy lỗi của ordinary route cho run này. W1 thay monkeypatch bằng registry rõ ràng |
| Hai distance kernels khác nhau | `models/cadic_patch_coreset_v1.py` dùng `torch.cdist` không MM trong nearest/pair search; bản `cadic_paperfaithful/...::euclidean_distance_mm` dùng công thức GEMM | Có thể khác sai số/tie behavior. Ghi kernel/source hash và kiểm chứng, chưa kết luận đây là nguyên nhân gap paper |
| YAML mô tả image score chỉ có reweight factor | `scoring.image_score` ghi `1-exp(s_star)/sum_exp(...)`; cả hai `CADICPatchCoresetV1.pixel_scores` thực tế append `weight * star_score` | **Lệch metadata**; code hiện có nhân `s*`. W1 sửa mô tả/schema, không “sửa” code thành bỏ `s*` theo string YAML |
| Source YAML có checkpoint trống, budget mặc định 2.5k | Hai method YAML; `0_setup_benchmark.py::main`, `training/benchmark_config_v1.py::apply_overrides`; launcher paperfaithful truyền checkpoint/budget | Đây là template, không phải resolved config của run. 10k phải có override. `checkpoint_identity=vit_b8_imagenet21k_local` là label, không thay SHA256 weights |
| Combined config cũ không phải entry point | `conf/cadic_compatible_v1.yaml` ghi `deprecated: true`; có budget 10000, batch 32, seed 42, dev settings riêng | Không dùng file này để suy ra parameters của canonical harness run. Nếu người dùng đang chỉ file này, cần đối chiếu launcher/resolved artifacts trước so sánh |
| Launcher paperfaithful chưa chạy forgetting | Script chỉ gọi `run_phase train`, `run_phase eval`; runner có `--phase fm` | Không phủ nhận final metrics đã có; cần tính task-matrix từ checkpoints bằng đúng paperfaithful route để có bằng chứng CAD/retention |
| Exact paper parity chưa xác minh | `exact_parity_claim:false`, `unresolved_assumptions`; `training/benchmark_config_v1.py::validate_configs` không cho claim true | Giữ nhãn reproduction/compatible với assumptions rõ. Tên folder `paperfaithful` không tự chứng minh mọi chi tiết bằng implementation tác giả |

Provenance tối thiểu cần thu hồi trước bảng matched final: `method_resolved.yaml`, `protocol_resolved.yaml`, `run.json`, Git commit, effective adapter/source hash, kernel, actual update unit, weights SHA256 và final metrics. Không cần gửi lại/chứng minh tính có thật của số CADIC. Những finding trên không đủ để quy kết nguyên nhân các khoảng cách 9–16 điểm phần trăm với paper.

### A5. Research checkpoint: CADIC, HOPE, VisionHOPE

| Khía cạnh | CADIC | HOPE / Nested Learning | VisionHOPE | Hệ quả cho đồ án |
|---|---|---|---|---|
| Bài toán gốc | Continual anomaly detection | Hệ học lồng nhau; HOPE là kiến trúc cụ thể | Visual backbone học thích nghi trong ảnh | CAD là bài toán đích; không coi benchmark classification là evidence CAD |
| Memory | Coreset các patch vector | Self-modifying associative memories + CMS | 5 loại memory tuyến tính: content, key, value, learning-rate, retention | Không gọi một ma trận CLS là full HOPE |
| Prediction | Patch NN, image score có reweight | Output self-modifying memory đi qua CMS | Fusion các hướng trong visual block | Final anomaly head phải phụ thuộc output HOPE/CMS |
| Thời gian | Stream các normal task | Nhiều tần suất học | Token/chunk trong ảnh; reset từ learned initial states | Phải chỉ rõ clock trong ảnh và clock xuyên ảnh/task |
| Không gian | Patches từ frozen ViT | Không đặc tả đầy đủ cách scan ảnh CAD | 4 hướng, chunk theo hàng/cột | 4-scan và geometry là phần adaptation cần kiểm thử |
| CMS | Không | Thành phần của HOPE, objective theo bài toán | Không dùng CMS | CMS objective cho CAD là đóng góp/giả thuyết của đồ án |
| Outer learning | Không cần train backbone/head | Học các tham số chậm/khởi tạo trong thiết lập của paper | Supervised visual tasks | Phải thiết kế outer objective normal-only, không nhìn future tasks |
| Stability | Budget và quy tắc coreset | Self-referential update cần mapping cụ thể | Soft injection cap + spectral clamp | Không đánh đồng P2 SVD projection hiện có với công thức VisionHOPE |
| Evidence CAD | Có MVTec AD/VisA | Không tự cung cấp kết quả CAD của đồ án | Không có CAD benchmark | Mọi gain/forgetting claim phải đo trên pipeline chung |

Nguồn: [CADIC](https://arxiv.org/html/2511.08634v1), [Nested Learning](https://arxiv.org/html/2512.24695v1), [VisionHOPE](https://arxiv.org/html/2609.33325v1). Các PDF tương ứng nằm trong `docs/papers/`; VisionHOPE local là `VISIONHOPE_2609.33325v1.pdf`. Implementation reference bổ sung: [repository chính thức VisionHOPE](https://github.com/PSRben/VisionHOPE). Chưa coi code upstream là đã audit toàn bộ hay đã tương thích môi trường dự án.

### A6. Đối chiếu mâu thuẫn và quyết định sau freeze

| ID | Mâu thuẫn / điểm mở | Quyết định kế thừa và revision v3 / evidence còn cần |
|---|---|---|
| F1 | `docs/hope_cad/architecture_decision_v1.md` giữ fast state xuyên task; VisionHOPE reset theo ảnh | **Chốt:** reset visual fast state theo ảnh/view; CMS persistent xuyên normal images/tasks; v1 vẫn được lưu nguyên nghĩa |
| F2 | VisionHOPE bỏ CMS; user yêu cầu full HOPE cho Nested Learning CAD | Giữ self-modifying visual core → sequential CMS; thiết kế objective normal-only riêng. Gọi đây là HOPE-CAD adaptation, không nguyên bản VisionHOPE |
| F3 | Chưa có target CAD cho CMS; self-target biến động có thể làm residual giảm do co-collapse | Fixed frozen-ViT teacher từ clean RGB, predictor chỉ nhận masked RGB; kiểm chứng nontriviality và causal controls |
| F4 | Coreset của HOPE-transformed patches sẽ stale khi representation thay đổi | Primary head dùng teacher-prediction residual sau CMS; không tạo bank trong moving feature space. HOPE+NN chỉ là nghiên cứu bổ sung nếu có thiết kế refresh/storage hợp lệ |
| F5 | Core v1 sigmoid/Xavier/1D conv/chunk16 khác VisionHOPE; pilot P1 bỏ alpha, cùng image transition | Chốt operator v2 theo equation map §3 của decision; `head_dim=16`, `heads=8` khi internal width 128. Giữ v1/P1/P2 làm reference |
| F6 | Đã có anomaly/outer pilot nhưng docs v1 vẫn ghi deferred | Tách rõ production core và experimental runners; cập nhật tài liệu sau freeze, giữ lịch sử v1 |
| F7 | `development_manifests()` lấy DEVELOPMENT từ official test | Lập ledger các ảnh/metrics đã xem; không gọi toàn bộ MVTec test là untouched. Khóa confirmation riêng và dùng VisA cho kiểm chứng trên dataset chưa dùng để chọn thiết kế |
| F8 | ReplayCAD native dùng InvAD; yêu cầu benchmark chung ViT-B/8 | Khang cung cấp native reproduction và, nếu triển khai, bản shared-backbone adaptation có tên riêng. Không gán số native thành đối chứng cùng backbone |
| F9 | Outer calibration có thể vô tình nhìn tất cả categories trước continual stream | Chốt prefix `min(100,N_first)`, roles support/query/validation/probe; chỉ task đầu, matched-data track riêng. Không gọi toàn pipeline strict single-pass |
| F10 | Metric pilot upsample native mask, harness hiện dùng 224×224 | Main track dùng cùng resolution 224×224 như harness đã khóa; native-resolution chỉ bảng phụ hoặc một protocol version riêng áp dụng cho mọi method |
| F11 | C4 phải match β của M nhưng v2 cho mọi CMS control tự search | C4 chính kế thừa β_M; fit/chọn own step, không tự chọn β. Tuned variant là ID phụ |
| F12 | Validation 16 events không sàng lọc long-horizon drift | `prefix_cms_stress_v3`: 1,024 support/query-repeat events, normal-validation loss + state/accumulator gates; không mở quyền dữ liệu |
| F13 | Oracle nội bộ cùng theo spec chưa tự đủ làm upstream fidelity | Giữ recurrence; numerical upstream-core parity pinned SHA đã PASS 19/19 low-level fixtures; differences và higher-order scope vẫn khai báo |
| F14 | Init contraction có thể làm visual branch ít tác dụng | Diagnostics ngay sau W3; post-calibration sau minimal W4; tiny init không tự fail design |
| F15 | VisA nearest resize có thể mất defect nhỏ | Preflight counts và giữ official image labels/all images; shared 224-pixel policy, không sửa riêng method |

F1–F5, F9, F10 đã đóng thành contract trong [architecture decision v3](architecture_decision_v3.md). F6 được xử lý bằng versioned docs, không sửa lịch sử v1. F7/F8 còn cần artifacts/phần việc của Khang để xác minh exposure và baseline comparability; chúng không làm thuật toán HOPE v3 bỏ ngỏ.

Rà soát cuối còn đóng hai lỗi tiềm tàng của bản nháp: support length phải là 16 để clock CMS 8 có hai lần update; functional outer loop phải giữ meta-gradient qua inputs và accumulators, dù local inner-update ownership chỉ thuộc level đang học. V1 `models/hope_cad/continuum_memory.py::_CMSLevel.objective_gradient` detach state/input phù hợp deployment, không thể tái dùng nguyên trạng cho exact meta-unroll.

## B. Kế hoạch kỹ thuật hoàn thành đồ án

### B1. Định nghĩa bài toán và quyền truy cập dữ liệu

Có T task theo thứ tự đã khóa; mỗi task cung cấp tập train chỉ gồm ảnh bình thường. Một model duy nhất xử lý lần lượt các task. Không reset CMS khi chuyển task, không dùng task ID để chọn head lúc test. Task ID chỉ phục vụ logging/evaluation. Sau task t lưu checkpoint rồi đánh giá các task đã thấy; test không thay đổi trạng thái persistent.

MVTec main order giữ 15 category alphabetic của `mvtec_1x15_v1.yaml`. VisA có 12 task theo danh sách/CSV chính thức được version hóa. Các thứ tự khác chỉ dùng cho robustness study đã đăng ký trước; không chọn order tốt nhất.

Trong mỗi task, main online stream dùng lexicographic relative-path order của manifest như v1 (`shuffle=False`); seed không tự làm randomized image order cho CADIC. Episode schedules và initialization mới có seed-dependent randomness. VisA main alphabetical order và official split được nêu đầy đủ trong instruction §5.

Hai nhánh báo cáo cần giữ rõ:

1. **Paper-control track:** giữ CADIC 2.5k/10k, protocol full-train và các run thật hiện có. Ghi nguyên metadata; không hồi tố đổi score/metric của run.
2. **Matched-method track:** HOPE, static controls và CADIC dùng cùng training identities, held-out normal validation, thứ tự online và quyền truy cập dữ liệu. Nếu normal calibration/dev split làm thay đổi training pool so với track 1, chạy CADIC matched riêng; không gọi hai track tương đương tuyệt đối.

Calibration đã chốt lấy prefix `P=min(100,N_first)` ảnh normal của task đầu theo canonical path order, yêu cầu P≥32. Chia `floor(0.50P)` support, `floor(0.20P)` query, `floor(0.15P)` validation, phần còn lại normal probe. Support/query dùng để học; validation chọn cấu hình/checkpoint bằng normal loss và sàng lọc long gate B5.1; probe chỉ diagnostic sau selection. Validation/probe không vào online-training pool. Bottle 170-image pilot là evidence về code thí nghiệm, không là split mặc định v2. Không dùng anomalies/test để chọn checkpoint, mask hoặc frequency; P=32 vẫn cho 16/6/4/6, validation nhỏ cần disclosure; long gate chỉ lặp support/query, không dùng normals ngoài prefix hoặc probe. Chi tiết [decision §§6–7](architecture_decision_v3.md).

Episode/export rule đã chốt: mỗi episode có 16 support-image events từ reset, một query không commit; functional outer differentiation qua đủ support history. Sau outer training đóng băng slow parameters, reset CMS về learned initialization đúng một lần, rồi xử lý normal training stream theo manifest. Calibration support/query identities được gặp lại khi chạy deployment stream; số lượt đọc phải công khai và các learned controls có cùng episode schedule/cap. Toàn pipeline mang tên prefix-calibrated continual learning, không strict single-pass streaming từ ảnh đầu tiên.

Harness v1 đang khóa `dev_mode=disabled`, chỉ MVTec, alphabetic order và adapter registry cũ tại `training/benchmark_config_v1.py::validate_configs`. Vì vậy W1 phải tạo protocol/validator v2 cho matched prefix-calibration, VisA và order study; không sửa âm thầm immutable rules v1 rồi dùng cùng protocol ID cho run khác điều kiện.

### B2. Kiến trúc đã freeze — implementation còn proposed

```text
clean RGB ── frozen ViT-B/8 ── stop-gradient teacher patches u

masked RGB ── cùng frozen ViT-B/8 ── projection
           ── VisionHOPE-style visual self-modifying operator
              (4 independent scans; reset mỗi ảnh/view)
           ── CMS level 1 ── CMS level 2 ── fixed decoder sau calibration
           ── predicted teacher patches û

anomaly map = residual(û, u) trên vị trí bị mask, phủ toàn ảnh
image score = max patch residual

normal train image: score/read trước → CMS update event sau
test image: visual adaptation cục bộ; CMS/slow params/counters không đổi
```

Kiến trúc nghiên cứu đã được chốt trong [architecture_decision_v3.md](architecture_decision_v3.md); full implementation vẫn **proposed/not implemented**. Clean teacher chỉ đi vào target, loss và phép tính score; không được đi vào predictor, q/k/v, fusion hay residual shortcut. Đường prediction bắt buộc đi qua final CMS output. Decision là nguồn chuẩn cho equations và tensor/update semantics nếu bản tóm tắt này không mô tả hết chi tiết.

**Backbone contract.** Cùng checkpoint SHA, layer 9/block 8, 224×224, raw 784×768 patches như CADIC. Teacher target được chuẩn hóa L2 (eps 1e-8); prediction không normalize. Đây là phép biến đổi của HOPE head, không sửa extractor/baseline CADIC. Feature cache phân biệt clean/masked views và chứa image hash, mask hash, checkpoint hash, dtype và preprocessing hash; không cache learned visual/CMS outputs qua các lần update.

**Visual core.** Một block, projection width 256, internal memory width 128, **head dimension 16 và 8 heads mỗi direction**. Minimal spatial wrapper theo structural Eq.27, local depthwise 3×3, FFN expansion 4, dropout 0; shape/operator details khóa ở decision §3. Đây là CAD adaptation cụ thể, không tự nhận trùng mọi auxiliary branch của full upstream backbone.

- 5 learned initial maps: content/key/value/eta/alpha; query projection là slow parameter, cố định trong scan.
- Hàng xuôi/ngược và cột xuôi/ngược có state độc lập, inverse permutation đúng tọa độ rồi learned fusion.
- Với 28×28 grid, chunk theo hàng/cột có độ dài 28; đọc/update từ đúng chunk-boundary snapshot theo equation map.
- Dùng retention parametrization, injection soft cap và spectral clamp của VisionHOPE. Viết reference token loop trước implementation tăng tốc.
- Reset từ learned initial states ở mỗi ảnh và mỗi masked view. Không kế thừa state từ ảnh test trước.
- Các initial maps và projection/fusion phải có meta-gradient thực nếu khai báo learned; không tái dùng restriction chỉ `A0,Wq` rồi mô tả là learned update policy đầy đủ.

**CMS.** Hai level sequential residual MLP `256→512→256`, pre-LayerNorm, GELU; periods `(1,8)` tính bằng số ảnh normal đã hoàn thành. Một level và equal-period là ablation. Decoder shared `256→768→768`, GELU, học lúc calibration rồi frozen. Fixed frequencies đã là multi-timescale; không thêm learned scheduler, novelty gate hoặc ACC vào main method.

Một mapping CAD cụ thể để kiểm chứng:

\[
h_0=V_\theta(F(\tilde x)),\quad
h_\ell=h_{\ell-1}+f_{\phi_\ell}(h_{\ell-1}),\quad
\hat u=D_\theta(h_L),\quad
u=\operatorname{sg}(\operatorname{normalize}(F(x))).
\]

Local objective của level l dùng **cùng frozen decoder sau calibration**:

\[
\mathcal L_{l,t}=\frac{1}{2|\Omega_t|}
\sum_{j\in\Omega_t}\|D_\theta(h_{l,t,j})-u_{t,j}\|_2^2.
\]

Khi deployment: input local có thể detach; state/accumulators detach sau event. Khi functional meta-training: lấy gradient chỉ đối với parameters của level đó nhưng **giữ dependencies của input và update history**, `create_graph=True`; không dùng detach của v1 làm đứt outer gradient. Mọi local gradient dùng snapshot trước event, mọi proposal validate trước atomic commit. Chuẩn hóa accumulator theo số image events đã tích lũy; đây là thay đổi so với v1 cộng gradient. Chưa tới period thì serialize accumulator, không flush riêng ở task boundary. Plain SGD, không momentum/weight decay/CMS clipping trong reference.

Local objectives trên từng prefix của chain là **CAD adaptation**, không phải objective anomaly được HOPE paper chứng minh. Outer loss dùng final output sau support updates, không dùng average của các auxiliary heads làm final score. Nếu local mục tiêu làm các tầng học trùng nhau, equal-period/single-level/static controls phải phát hiện được điều đó.

**Outer loop.** 16 normal support events → CMS update counts `(16,2)` → masked normal query → final teacher loss; query không commit. Học đủ visual initial states/q/projection/fusion/spatial params, decoder và CMS initial parameters; backbone frozen. Functional training state riêng giữ exact reference unroll; first-order/truncated approximation phải đổi method ID. Reference Adam lr 1e-4, cap 200 steps; chọn checkpoint ở 0/25/50/100/200 trên normal validation. CMS rate reference 1e-3, finite search `{1e-4,1e-3,1e-2}` tối đa ba fits; seed 0 dùng chọn rate/step count rồi khóa cho final seeds 0/1/2 và recipe VisA. C2/C3 tự search như M; C4 lấy β_M và chọn own step. Selection cuối phải qua gate B5.1; chi tiết graph/export/selection ở decision §7.

**Mask và dense score.** Chốt 4 deterministic views: chia grid thành các block 2×2 patches rồi tô 4 màu theo tọa độ block modulo 2; mỗi view che một màu trên RGB trước ViT. Mỗi patch thuộc đúng một masked set, coverage 100%, 196 patches/view. Mask fill dùng ImageNet mean RGB. Giữ mask family cố định; nếu normal-only nontriviality check thất bại thì đánh dấu failed gate và tạo design revision trước evaluation, không âm thầm đổi mask theo anomaly test.

\[
a_j(x)=\|\hat u_j(x;\Omega_{k(j)})-u_j(x)\|_2^2,
\qquad s(x)=\max_j a_j(x).
\]

Map chính reshape 28×28, bilinear lên 224×224, metric theo mask 224×224 của protocol. Không smoothing/test score calibration. Mỗi test image có 1 clean + 4 masked backbone passes; tính toàn bộ vào latency. Normal train event t dùng view `(t-1) mod 4`; giữ teacher clean và view clock trong checkpoint. Test dùng cùng CMS snapshot cho cả 4 views; không áp teacher-loss gradient lúc test.

**Vì sao không lấy current residual `M k-v` làm final ngay?** Cả predictor và target có thể biến đổi cùng memory; error giảm chưa chắc là normality learning. Pilot fixed teacher là nền có thể kế thừa về code, mới cài đặt objective ở một phần vị trí và chưa có artifact validation tại workspace. Việc chốt fixed target loại bỏ một nguồn nhập nhằng, không chứng minh trước anomaly separation hoặc thắng CADIC.

### B3. State lifecycle bắt buộc

| State | Outer calibration | Normal stream | Chuyển task | Official test |
|---|---|---|---|---|
| ViT weights | Frozen | Frozen | Giữ | Frozen |
| Visual initial maps, q, projections, fusion, decoder | Outer optimizer được phép cập nhật | Frozen | Giữ | Frozen |
| Visual working memory 4 scans | Reset mỗi view; inner updates | Reset mỗi view; inner updates | Không state xuyên task | Reset mỗi view, discard sau inference |
| CMS initial parameters | Meta-learn theo episode | Frozen reset source | Giữ | Frozen |
| CMS current weights | Functional state của episode | Persistent online learning | Không reset | Read-only |
| CMS gradient accumulators/counters | Theo episode | Persistent và checkpoint | Không reset/không partial flush | Không đổi |
| Outer optimizer/RNG/sampler | Checkpoint đầy đủ | Không optimizer outer | Không ảnh hưởng online | Không đổi |
| Train cursor/mask-view schedule | Theo manifest | Atomic event boundary | Tiếp tục | Không đổi |

Batch là đơn vị vận chuyển, ảnh là đơn vị update CMS. B=8 phải cho cùng state cuối và scores như 8 ảnh B=1 theo cùng thứ tự. Test permutation phải cho cùng per-image scores và bitwise unchanged persistent state ở chế độ deterministic.

### B4. Giả thuyết và phép thử bác bỏ

| ID | Giả thuyết | Đối chứng bắt buộc | Evidence đủ để ủng hộ |
|---|---|---|---|
| H1 | Learning dynamics tạo giá trị ngoài static feature fitting | Static trained predictor; frozen visual-update; reset/freeze CMS; history controls chỉ từ tasks đã thấy | Paired normal-query improvement và thay đổi final scores; CAD gain có CI. Hai history samples cùng phân phối không bắt buộc phải cho chất lượng khác nhau |
| H2 | CMS nhiều timescale cải thiện retention/acquisition tradeoff | No CMS; một level; cùng số level nhưng cùng period | Final metrics và forgetting/BWT, acquisition trên task mới; không chỉ norms/counters |
| H3 | Cơ chế VisionHOPE ổn định visual adaptation | 4-scan so với 1-scan; full guards so với bỏ từng guard ở pilot kiểm soát | Numeric oracle; kiểm tra divergence và contraction/collapse theo vị trí scan, contribution so với residual bypass; accuracy/cost báo riêng |
| H4 | Kết luận không chỉ là hiện tượng của một category/dataset/order | 15 task MVTec, 12 task VisA; seed/order robustness | Full protocol, per-class results, cùng hyperparameter-selection rule |

Non-expansion của visual memory không được diễn giải thành định lý chống catastrophic forgetting của toàn HOPE-CAD. CMS learning và AD performance vẫn là giả thuyết thực nghiệm. VisA main run reset toàn bộ method và áp cùng recipe normal calibration trên dữ liệu được phép của VisA; đó là kiểm chứng trên dataset thứ hai, không tự gọi zero-shot transfer.

### B5. Baseline và ablation registry đã chốt

| ID | Method cần triển khai/chạy | Vai trò |
|---|---|---|
| B0 | CADIC paperfaithful 2.5k và 10k | Baseline chính; score gốc CADIC được giữ nguyên |
| B1 | ReplayCAD native do Khang phụ trách | So sánh literature/native; ghi InvAD và pretraining/data khác biệt |
| B1-ViT | ReplayCAD shared-backbone adaptation, nếu Khang triển khai | Cùng ViT-B/8; method ID riêng, không thay tên native |
| C0 | Cùng topology, visual updates OFF, CMS online updates OFF; giữ hai MLP học trong calibration | Static trained control giữ depth/width; ghi active parameters và freeze rõ những control maps không dùng |
| C1 | Visual updates ON, CMS online updates OFF; giữ hai MLP như static predictor | Tách adaptation trong ảnh khỏi continual knowledge, không giảm depth chỉ để bỏ CMS updates |
| C2 | CMS predictor, visual update tắt với cùng operator/params | Kiểm tra visual self-modification có cần thiết không |
| C3 | Một CMS level period 1, hidden width 1024, gần match parameters của hai level hidden 512 | Tách multi-level khỏi một adaptive process; báo exact parameter count và depth |
| C4 | Hai level periods `(1,1)`, rates `(β_M,β_M/8)` với β_M kế thừa selected M; M periods `(1,8)`, rates `(β_M,β_M)` | Giữ capacity và match tổng first-order step size tầng chậm trên 8 ảnh khi gradient cố định; kiểm tra tác dụng gom update theo thời gian |
| M | HOPE-CAD v3 theo frozen decision, full implementation proposed | 4 scans + guards + sequential CMS `(1,8)` + normal-only outer loop + final residual |
| A1 | Full online dynamics như M nhưng initialization lấy từ ordinary normal calibration của C1 | Tách meta-through-update khỏi normal representation fitting; random untrained decoder chỉ là sanity phụ |
| A2 | M 1 scan; M tắt từng stability guard | Mechanism ablation, thử nhỏ trước; unstable run là kết quả phải báo |
| A3 | Reset CMS / wrong history / freeze CMS tại checkpoint | Paired causal interventions, không phải các full retraining runs |

Main table tối thiểu B0, B1 khi Khang sẵn sàng, C0, C1, C2, C3, C4, A1, M. C0/C1 ordinary calibration dùng mean loss trên toàn support + query của cùng episode schedule, tránh static control chỉ được học query còn M nhìn cả support. A1 dùng lại initialization của C1 rồi bật CMS online với rate đã chọn cho M; không chọn initialization theo test. C2/C3 dùng finite rate search + long gate như M; C4 không search β, kế thừa β_M nhưng fit/chọn own step và chạy gate; static controls dùng tối đa ba outer learning rates `{3e-5,1e-4,3e-4}` trên normal validation, cùng cap 200 steps. Chọn settings trên pilot seed 0 rồi khóa cho final seeds và recipe VisA.

C4 tự tune nếu làm phải có ID `C4_tuned_aux` riêng và không thay C4 chính. C4 không có trajectory hoặc optimizer update count giống M; báo rõ compute và số updates. Equal-period rates `(β,β)` là control phụ có ID riêng nếu làm thêm. Paired intervention tại cùng checkpoint khác retraining ablation. Không bỏ ablation bất lợi sau khi xem test; giới hạn compute được quyết định trước run.

Không đưa ACC hay offline Phase 3 vào main candidate. Không có evidence hiện tại rằng chúng cần thiết để kiểm chứng Nested Learning CAD. Learned/adaptive update scheduler là mở rộng sau khi main mechanism có bằng chứng, không phải việc bắt buộc để đủ tên đề tài.

### B5.1. Long-horizon CMS acceptance và model selection

**Contract `prefix_cms_stress_v3` — proposed, bắt buộc trước scientific pilot.** Gate này bổ sung sàng lọc numerical/normal-prediction stability dài hơn 16 support events; không là định lý chống quên, không chứng minh anomaly detection, và không mô phỏng một lượt đi qua các task khác nhau. Các số 1,024 / 10 / 1 / 2 / 1.25 dưới đây là **policy kỹ thuật đăng ký trước các CMS runs**, chưa được thực nghiệm xác nhận là tối ưu; thay chúng để cứu một run phải tạo selection-rule/design revision, không tune bằng probe/test.

**Dữ liệu và lịch.** Nối hai manifests `meta_support` rồi `meta_query`, giữ canonical order nội bộ, lặp nguyên danh sách và cắt đúng **H=1,024 image events**. Lưu toàn bộ identity/view schedule và hash. Event t dùng view `(t-1)%4`, clock tiếp tục qua lần lặp danh sách; không reset ở điểm lặp. Chỉ support/query được update. `normal_validation` dùng read-only để đo loss; `normal_probe`, normals ngoài prefix, future tasks và official test không tham gia gate hoặc lựa chọn β. P≥32 giữ nguyên; với P=32 có 16/6/4/6 ảnh, validation nhỏ phải được disclosure, không tự nâng P để lấy thêm dữ liệu. Lặp ảnh là calibration/stress data reuse, phải tính vào read/compute ledger; không cộng thành thêm unique images.

**State.** Với từng checkpoint ứng viên đã shortlist, clone learned initialization, zero CMS accumulators/counters một lần, freeze slow θ/decoder và chạy đúng deployment numerical update: pre-event snapshots, mean accumulation, detach qua image events, không outer optimizer. Reference FP32. M có 1,024/128 commits; C4 có 1,024/1,024. Không thay period, flush, clipping hoặc warm-up để qua gate. Cuối gate discard terminal state; main deployment bắt đầu lại từ learned initialization, không nhận state đã đọc lặp 1,024 ảnh.

**Đo lường.** Tại `t∈{0,16,64,128,256,512,1024}`, tính `L_t` bằng mean masked teacher loss của **tất cả normal-validation images, đủ bốn views**, đọc cùng CMS snapshot, không commit hoặc đổi RNG/state/cursor. Dùng loss `0.5 * (pred-target).square().sum(-1).mean()` trên masked positions; không mean thêm channel 768. Mọi event đều kiểm finite outputs, local losses/gradients, proposed/current weights và accumulators; ghi norms, update counts và first failure. Đo loss cả khi weights hữu hạn để phát hiện predictor drift.

Đặt `D=max(L_0,L_16,1e-8)` và `E=max(L_16,1e-8)`. `L_16` ở đây đo sau 16 events đầu của stress schedule canonical, không được giả định bằng short-validation loss dùng fixed support schedule riêng để ranking. Với mỗi CMS level l, vectorize **mọi current parameters kể cả LayerNorm affine** thành w_l; w_l,0 là learned initial vector của chính checkpoint. Đo `R_w(l,t)=||w_l,t||₂/max(||w_l,0||₂,1e-12)`. Ngay sau cộng gradient nhưng **trước commit/reset**, gọi accumulator/count là `A_l^+,n_l^+` và đo `R_a(l,t)=||β_l A_l^+/n_l^+||₂/max(||w_l,0||₂,1e-12)`, kể cả event chưa due. Log thêm raw accumulator/gradient norms; không chỉ nhìn accumulator sau reset bằng zero. Norm diagnostics tính FP64 trên tensors FP32 để tránh norm reduction tự underflow/overflow; không đổi update precision.

**Pass iff:** tất cả kiểm finite đạt; mọi event và level có `R_w≤10`, `R_a≤1`; mọi checkpoint đo từ t=16 có `L_t≤2D`; và `L_1024≤1.25E`. Non-finite hoặc bất kỳ điều kiện nào fail thì candidate không eligible. Có thể dừng sớm sau failure, nhưng phải lưu event/reason và gọi run failed, không complete. Gate không sửa/clamp state để tạo pass. Validation t=0 cũng phải finite.

**Thứ tự chọn.** Với M, C2, C3: giữ search β∈{1e-4,1e-3,1e-2}, cùng initialization/episode schedules, tối đa 200 outer steps và short-validation ở 0/25/50/100/200. Fit có non-finite train state/loss/gradient hoặc short-validation loss bị đánh dấu failed và không eligible, lưu first failure; không chọn một checkpoint cũ để che failure của fit. Trong mỗi fit hoàn tất hợp lệ, shortlist **duy nhất checkpoint có short-validation loss thấp nhất**, tie earlier step. Chạy một long gate cho checkpoint đó, tối đa ba gate/method. Không thử lại checkpoint khác của β đã fail. Trong các candidates pass, chọn theo **short-validation loss ban đầu**, tie earlier step rồi smaller β; long-gate loss là bộ lọc, không là tiêu chí ranking mới. Nếu không có candidate pass, ghi `selection_failed`, không mở search, lấy probe hoặc thêm normals ngoài prefix để chọn lại.

**C4 chính:** kế thừa `β_M` đã chọn trên M; rates `(β_M,β_M/8)`, periods `(1,1)`. Fit initialization riêng với exact unroll của C4, chọn own step ở 0/25/50/100/200 bằng short validation, rồi chạy **một** long gate như trên. Không search β riêng, không bắt C4 dùng cùng số outer steps của M. C4 fail được báo unstable; không retune M để cứu C4. `C4_tuned_aux` nếu làm là control phụ có ID/search ledger riêng, không thay C4 chính. First-order matching `8×β_M/8=β_M` chỉ đúng khi coi gradient không đổi, không nói trajectories hoặc commit counts giống nhau.

**Controls và final runs.** C0/C1 không có CMS adaptation nên giữ ordinary calibration/outer-rate search đã chốt, không chạy CMS stress gate. A1 dùng lại C1 initialization + β_M; chạy gate như acceptance, không fit/search thêm. Final seeds 0/1/2 refit đúng selected β và own step count của từng method; recipe VisA khóa từ MVTec, fit fresh trên prefix VisA. Với các refits này, gate chỉ là acceptance đã đăng ký, không chọn lại rate/step/seed theo validation hoặc test. Fail phải có artifact và làm suite thiếu acceptance; không thay seed, bỏ control bất lợi hoặc báo complete matched suite. Seed 0 chỉ được tái dùng fit/gate nếu toàn bộ config, schedules, source và initialization hashes trùng, phải khai báo reuse.

Artifact cần có `selection_rule_id`, constants, schedule/role hashes, shortlisted/rejected candidates, short losses, trace L_t/R_w/R_a, counters, first failure, selected β/step hoặc failure, reads/time/memory, và hash learned initial state trước/sau gate. Gate dùng clone và không mutate checkpoint gốc. Normal probe chỉ chạy **sau khi selection hoàn tất**; không dùng nó chọn lại β. CLI/runner của policy này chưa được triển khai.

**Khóa baseline trước HOPE test feedback.** W1 phải chốt effective CADIC adapter route, update unit, score/kernel/preprocessing/metric resolution và resolved hashes cho matched 10k trước khi xem anomaly-test metrics của HOPE. Historical CADIC numbers vẫn được giữ là real results; matched run riêng chỉ cần khi chưa có artifacts cùng protocol/identities để dùng lại. Không đổi route hoặc chữa baseline dựa trên metric có lợi cho HOPE; thay đổi có căn cứ phải version hóa và rerun mọi comparison bị ảnh hưởng. Việc này không chặn W3/W4 correctness hoặc normal-only diagnostics.

### B6. Metrics, thống kê và báo cáo

**Primary:** final macro I-AUROC và P-AUPR trên mọi task. P-AUPR main dùng `sklearn.average_precision_score` như harness, không âm thầm đổi sang trapezoidal integration. Báo per-task song song macro. I-AP và P-AUROC là secondary. Không pool các class rồi gắn nhãn macro.

Với metric m, lưu `R_m[t,j]`: performance task j sau khi train task t, j≤t. Báo final average `mean_j R_m[T,j]`, BWT `mean_{j<T}(R_m[T,j]-R_m[j,j])`, và forgetting theo formula đã khóa của harness:

\[
F_m=\frac1{T-1}\sum_{j=1}^{T-1}
\left(\max_{t=j,\ldots,T-1}R_m[t,j]-R_m[T,j]\right).
\]

Giá trị âm có thể xảy ra khi cuối stream cải thiện task cũ; không clamp về 0. Nếu thêm positive-only forgetting phải có tên khác. Không trộn normalized F1 forgetting của paper khác với AUROC/AP forgetting này.

- Final runs: tối thiểu 3 seeds đã đăng ký; cùng seed pairing giữa methods. CADIC có thể deterministic theo order: báo đúng nguồn biến thiên, không coi chạy lặp identical là 3 quan sát độc lập.
- Ngoài seed, order robustness là yếu tố riêng. Alphabetic là main; thêm hai orders giữ first task cố định và shuffle phần còn lại với seeds 100/200 cho B0/M/C4. Không trộn thay đổi calibration category với order effect. Full permutations đổi first task là phân tích warm-start riêng nếu làm thêm.
- Báo mean/std và paired deltas; CI resampling unit là image (và giữ cấu trúc category/seed), không lấy hàng triệu pixels làm mẫu độc lập. Nêu chính xác đơn vị CI; 3-seed CI không phải bằng chứng mạnh cho mọi data order.
- Primary point estimates dùng toàn bộ pixels. Histogram approximation nếu dùng cho bootstrap phải ghi tolerance; không thay exact point AP.
- Memory: static learned parameters, mutable weights, reset copies, gradient accumulators, optimizer state, coreset, replay raw/generated data, peak GPU và host RAM. Disk cache tách riêng, nhưng không giấu trong total storage.
- Runtime: cold/cache feature extraction, outer fit, online update, full image inference, checkpoint/eval. Đồng bộ GPU khi đo; báo cả end-to-end và cached-feature, số views/scans và precision.
- Lưu normal/anomaly maps, failure cases theo quy tắc chọn cố định trước; không chỉ đưa ảnh đẹp nhất.

So sánh confirmatory chính: **M vs matched CADIC 10k**, hai co-primary endpoints. Báo paired 97.5% intervals cho từng endpoint (Bonferroni cho hai endpoints, family coverage danh định 95%); chỉ claim đồng thời tốt hơn khi cả hai lower bounds >0. Nếu một endpoint tốt hơn còn endpoint kia chưa rõ thì báo từng endpoint/tradeoff. CADIC 2.5k và các ablations là secondary/explanatory, interval level ghi riêng. Bootstrap main giữ categories cố định và resample paired whole images theo category/image label; seed variability báo riêng hoặc nested bootstrap có label rõ. Với forgetting, dùng cùng image resampling weights qua mọi task checkpoint để giữ covariance. Ba seeds không chứng minh cho mọi data order.

Bootstrap reference khóa 2,000 replicates, RNG seed 20261009; paired image weights chung across methods/checkpoints/seeds, mỗi replicate average metric deltas qua seeds (không ensemble scores). Primary percentile endpoints 1.25/98.75%, secondary 2.5/97.5%. Đây là image-sampling uncertainty conditional trên trained seeds; report seed std riêng. Histogram acceleration chỉ cho bootstrap, kiểm ít nhất 32 fixed replicates so exact weighted metric với absolute error ≤2e-4, không đạt thì refine/exact hoặc chưa claim từ CI. Chi tiết ở instruction §9; exact point metrics không đổi.

### B7. Các gói công việc và acceptance criteria

| Gói | Công việc cụ thể | Đầu ra/điều kiện chuyển giai đoạn |
|---|---|---|
| W0 — Freeze, **DONE ở mức tài liệu** | Architecture/data access, clone/source scope, hypotheses/ablations và selection v3 | [Architecture decision v3](architecture_decision_v3.md), instruction v3; gate mới pending, không claim implementation |
| W1 — Baseline/protocol | Thu hồi A4.1 provenance; explicit adapter IDs; protocol/validator v2 cho prefix split/VisA/order study; ReplayCAD schema agreement | Per-image update, full Eq.9 score, exact metric; unknown adapter fail rõ, không fallback legacy; smoke/resume đúng; v1 immutable |
| W2 — Kaggle runtime, **partial evidence** | Lock dependency/Python/PyTorch/timm/parquet; notebook chỉ bootstrap/CLI; weights/data preflight, mounts/config overrides, resume/export | Kaggle W3 synthetic suite đã pass; chưa đóng W2. Còn clean-session extractor + tiny harness, đúng assets/manifests/masks và crash-resume |
| W3 — Visual operator, **implemented/tested trên `hope_cad_v2_1`** | Token oracle, rank-one chunks, four scans/guards/init/geometry; upstream-core parity và diagnostics A3.2 | W1/W3/W4 verifier 56/56 PASS; upstream parity 19/19 fixtures PASS; canonical exact-16 forward/backward PASS trên T4. Real features và CAD utility vẫn chưa được xác nhận |
| W4 — CMS objective/outer loop | Functional state; local teacher losses; averaged accumulators; 16-support exact unroll; prefix pools/selection/export, inherited-β C4 và long gate B5.1 | Finite differences và counts `(16,2)`; batch transport equivalence trong tolerance; atomic rollback; resume mid-period; meta graph giữ đúng dependencies; resource spike sớm và trace long-gate loss/norm/finite |
| W5 — Dense CAD integration | Full coverage masks, final CMS residual head, adapter `fit_task/score_batch/state_dict/memory_stats`, scoring trước update | 100% patches scored; intervention vào CMS làm đổi final score; test permutation không đổi output; persistent state/counters/RNG không đổi |
| W6 — Mechanism pilot | C0/C1/C2/M; validation chọn cấu hình; probe diagnostic; reset/freeze/static mean controls; stability/contraction/compute | Tách acquisition khỏi static fitting; no NaN/Inf, không chỉ norms; fit một T4; gate fail báo failed/inconclusive theo scope; init warning không tự đổi design; thay recipe phải revision trước test |
| W7 — MVTec full | Chạy main/ablations, per-task checkpoints, metrics matrix, multiple seeds; ledger prior test exposure | 15 task đủ count, exact final metrics + forgetting/BWT, complete artifacts và uncertainty; kết luận rõ giả thuyết nào được/không được ủng hộ |
| W8 — VisA | Official split adapter; cùng benchmark schema, recipe đã khóa; reset method độc lập | 12 task, mask QA B10, no split overlap, final metrics/forgetting/seeds; không chỉnh hyperparameters theo VisA anomaly test |
| W9 — Final packaging | GPU Khang khi biết cấu hình, reproduce representative runs, shared tables với ReplayCAD, thesis figures, demo, docs | Từ clean environment tái lập một complete run; mọi bảng truy được run IDs; instruction phản ánh final code và phần proposed/limitations còn lại |

W1/W2 có thể triển khai song song về mặt công việc; W3→W4→W5→W6 có phụ thuộc correctness. Không dùng từ “hoàn thành HOPE” ở W3 khi chưa có CAD objective/head/benchmark.

**Cập nhật chặng 1:** W3 đã có trên `hope_cad_v2_1`; yêu cầu tích hợp vào clone cũ chỉ còn là lịch sử. W4 reference đã triển khai và kiểm bằng fixtures; canonical-T4 và upstream gates đã PASS theo [báo cáo chặng 1](stage1_implementation.md). Real-prefix gate và CAD utility vẫn pending; init/reduced diagnostics không thay canonical real-feature validation.

**Engineering acceptance bắt buộc:** deterministic state lifecycle, no data leakage trong protocol mới, no hidden task routing, scores đi qua CMS, bounded resource tracking, exact resume, artifact provenance. Đây là pass/fail độc lập với model accuracy.

**Scientific acceptance:** không cam kết trước thắng CADIC. Claim một component cải thiện metric chỉ khi paired evaluation/CI và ablation hỗ trợ; nếu CI chứa 0 thì ghi inconclusive. Claim retention phải đồng thời xem forgetting và khả năng học task mới, tránh giảm forgetting bằng cách không học. Claim full method tốt hơn baseline phải công bố cả hai primary metrics và compute, không chỉ chọn metric có lợi. Negative result có đủ bằng chứng vẫn là kết quả nghiên cứu; nó không được đổi tên thành validation thành công.

Primary claim rule đã khóa ở B6. Practical-effect threshold bổ sung, nếu có, phải khai báo trước evaluation dưới nhãn secondary; không thay co-primary table sau khi thấy test. Numerical/data/state/resource criteria ở decision §9. Không đặt điều kiện phải hơn paper một tỷ lệ tùy ý.

### B8. Tổ chức codespace dự kiến

Giữ v1 và `exps/` làm evidence/reference. W3 visual files/tests/verifier tồn tại tại `hope_cad_v2@6dbe4f1`/`363634b`, chưa nằm trong checkout clone theo quyết định giữ HEAD cũ. Bảng dưới giữ nguyên namespace `_v2` để kế thừa implementation; full method/config/selection v3 là contract mới, các module CMS/head/protocol/runner khác vẫn **proposed**.

| Phạm vi | File/module đề xuất | Trách nhiệm |
|---|---|---|
| Visual memory | `models/hope_cad_v2/visual_memory.py`, `scan_geometry.py` | 5 memories, control formulas, token/chunk execution, scan/fusion |
| CMS | `models/hope_cad_v2/cms.py`, `state.py` | Functional/meta state và deployment state; periods, atomic commits, counters |
| Full method | `models/hope_cad_v2/model.py`, `teacher_head.py` | Visual → CMS → decoder; target ownership và final score |
| Masks/data | `dataset/hope_mask_views.py`, `dataset/visa_benchmark.py` | RGB masks/coverage; official VisA split; common manifest |
| Calibration | `training/hope_outer_learning_v2.py` | Episodes, support/query split, outer optimizer, selection bằng normal validation |
| Benchmark adapter | `models/hope_benchmark_adapter_v2.py` | Common harness contract, fit/score/checkpoint/profile |
| Configuration | `conf/benchmarks/methods/hope_cad_v3.yaml`, `conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml`, `visa_1x12_prefix_cal_v2.yaml` | Explicit IDs/hashes; giữ historical full-train v1 |
| Protocol/registry | `training/benchmark_config_v2.py`, common adapter registry | Validate dataset/split/order/method/calibration permissions; compare protocol/backbone hashes |
| Entry points | `scripts/hope_cad_v2/`, `notebooks/hope_cad_v2_kaggle.ipynb` | Cache/calibrate/train/eval/report; notebook không chứa business logic |
| Tests | `tests/hope_cad_v2/` | Oracle, causality, gradients, geometry, lifecycle, integration; reuse fixture baseline |
| Documentation | `docs/hope_cad/architecture_decision_v3.md`, `docs/instruction_CAD_v3.md` | Frozen scientific/software contract; paper-to-code map; onboarding |

Một experiment directory phải chứa: Git commit + dirty diff hash, resolved config, environment lock/version report, backbone hash, raw-data manifest/hash, split/episode/order/mask manifests, seed/RNG states, event cursor, method state, metrics per task/matrix, timing/memory, exact commands và status. Checkpoint ghi atomically ở event boundary; lưu pending CMS gradients và counts. Kaggle output phải được export trước khi session kết thúc; lần chạy tiếp xác minh hash rồi resume.

**Provenance v3 — proposed schema.** `run.json` phải lưu SHA256 bytes của `docs/instruction_CAD_v3.md`, `docs/hope_cad/architecture_decision_v3.md`, `docs/hope_cad/planning_CAD_v3.md`; `design_id`, `selection_rule_id`, code namespace/schema, actual source commit và source hashes, dirty/untracked bundle hash, backbone/data/role/schedule hashes. Không tự ghi `a89fd48` như thể commit đó chứa W3. Trước chuyển máy, dùng source commit hoặc archive có manifest hashes bao phủ cả untracked files; branch name/tag một mình không đủ tái lập. Lượt soạn tài liệu này không commit/push/tag. Khi tag freeze ở bước đóng gói, tag phải trỏ commit thực sự chứa đúng tài liệu/source được dùng, không hồi tố thay historical run metadata.

### B9. Dùng Kaggle và tài nguyên final

Mỗi NVIDIA T4 có 16GB GPU memory theo [datasheet NVIDIA](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf). Hai T4 là hai thiết bị riêng. Main run phải fit trên một GPU; ưu tiên chạy hai seed/ablation độc lập trên hai GPU. Không chia một continual stream thành hai worker update CMS không đồng bộ.

- FP32 cho recurrent state, guard calculations, loss/updates và parity baseline. Mixed precision của frozen ViT chỉ bật sau khi kiểm tra sai khác feature/score; ghi precision thành một phần protocol.
- FP32 cache một ảnh 784×768 chiếm 2,408,448 bytes, khoảng 2.30MiB. Clean + 4 views tăng xấp xỉ 5 lần storage nếu cache toàn bộ. Dùng shards/lazy extraction và cache đúng working set, không mặc định giữ tất cả trên GPU.
- Pilot profile đo seconds/image, seconds/update và full evaluation; từ đó dự báo toàn bộ suite, kể cả ma trận forgetting. T task cần T(T+1)/2 task-evaluations cho lower-triangular matrix, không phải chỉ T final evaluations.
- Quota 30h là ràng buộc do người dùng cung cấp; đọc usage thực trong Kaggle khi chạy để xác định cách tính hai GPU. Không giả định 30h này đủ toàn bộ 3-seed/2-dataset/ablation suite.
- Phân bổ vòng quota đầu theo tỷ lệ: 10% setup/profile, 30% mechanism/gradient pilots, 40% integration + MVTec pilot, 20% reserve/debug/resume. Chỉ đặt số full runs sau khi có runtime projection; đây không phải cam kết thời lượng chạy.
- TPU không nằm trên critical path vì pipeline đang dùng PyTorch/CUDA, stateful custom updates và upstream cần kiểm tra compatibility. Không port sang TPU chỉ để dùng hết quota. Có thể dùng sau nếu một workload stateless cụ thể cho lợi ích đã đo.
- GPU final của Khang: khi xác nhận model/VRAM/runtime, chạy preflight cùng config trước; batch transport có thể tăng nhưng image update semantics không đổi. Không tự đổi backbone, precision hoặc scoring giữa Kaggle và final server.

Runtime W3 đã được kiểm trên Kaggle: Linux/Python 3.13.15/PyTorch 2.11.0+cu128, Tesla T4, 24/24 tests. Elapsed 6.706 s của verifier là thời gian chạy suite synthetic trên CPU và reduced CUDA, **không phải throughput của full HOPE-CAD hoặc projection cho quota**. W2 còn cần extractor/weights/data, dependency lock cho full pipeline và harness/resume; không gọi toàn môi trường đã nghiệm thu từ W3 smoke.

16-image exact higher-order graph chưa được profile trên T4. Tối ưu vectorization/recomputation phải giữ gradient parity; OOM không cho phép tự rút support xuống 4, detach graph hoặc đổi periods. Nếu vẫn không fit sau tối ưu đúng semantics, revision phải có evidence tài nguyên trước khi chạy scientific benchmark.

**Resource spike sớm — proposed.** Ngay khi minimal functional CMS có thể unroll, profile một exact 16-event `create_graph=True` episode với random weights/synthetic features trên một T4: forward, backward, peak allocated/reserved VRAM, wall time, CPU RAM và backward tới mọi nhóm trainable. Tách synthetic feasibility khỏi real-feature learning evidence; W2/W4 chưa có measurement này. Sau đó đo real-feature cached/end-to-end, five-view score và đủ 120/78 task-evaluations cho MVTec/VisA, cả calibration stress gate và bootstrap. Không suy runtime toàn hệ bằng đúng 5× CADIC từ số lượt backbone.

Compute ledger seed-0 tối đa: M/C2/C3 mỗi method 3 fits, C4 1 fit, C0/C1 mỗi method 3 fits, tổng 16 fits; A1 reuse C1. Tối đa 10 long gates cho M/C2/C3/C4 và 1 acceptance gate A1; counts thực tế ghi cả failed fits. Final refits/seeds/orders/datasets là chi phí bổ sung, không nằm trong 16 fits. `C4_tuned_aux` thêm search riêng nếu được đăng ký. Trước anomaly evaluation, ưu tiên correctness/profile → M/C0/C1/C2 mechanism pilot → C3/C4/A1 → suite main/seeds → order studies/VisA theo W7/W8. Khi thiếu quota, hoãn optional auxiliary variants trước; không bỏ mandatory controls, co-primary metrics, intermediate pixel metrics, seeds hoặc VisA rồi tuyên bố hoàn thành 100%. Chuyển phiên/máy bằng checkpoint đã verify, không đổi toán học để vừa quota.

### B10. VisA và giao diện làm việc với Khang

Hạ tầng dữ liệu/manifest/metrics viết chung ngay từ W1. Adapter VisA được viết sau khi MVTec smoke đúng và có thể unit-test sớm bằng fixtures; full VisA experiments đi sau W7. Cách này tránh nhân đôi công sức mà vẫn có một pipeline đã ổn định để kiểm chứng trên dataset thứ hai.

VisA dùng split chính thức, đối chiếu [script chuẩn bị dữ liệu của tác giả](https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py) và `split_csv/1cls.csv`. Không dùng loader cũ tự chia 80/20 rồi gọi là CADIC-compatible. Kiểm tra category list, train-normal-only, mask geometry, path/hash uniqueness và full image counts.

**VisA geometry/mask QA — proposed.** RGB được resize **trực tiếp về 224×224 bằng PIL bilinear**, không giữ aspect ratio bằng padding/crop; policy này áp dụng mọi method trong matched table. Ground-truth mask dùng nearest, binarize raw mask bằng `nonzero`, không mặc định threshold `>127`. Xác minh encoding/geometry từ official preparation và CSV; nếu mask annotation đa kênh phải xác minh kênh nội dung trước khi quy về binary, không tính alpha channel thành defect. Raw anomaly mask thiếu/rỗng hoặc geometry không hợp lệ là lỗi integrity cần giải quyết, không tự suy diễn/sửa nhãn.

Evaluator preflight lưu theo category: số official normal/anomaly images, positive pixels trước/sau resize, identity/hash và số anomaly masks trở thành rỗng ở 224. **Giữ nguyên các ảnh này**: image label lấy từ official CSV, không suy lại từ resized mask; pixel metric dùng đúng binary mask 224 kể cả all-zero. Không drop ảnh, tô lại defect hoặc đổi interpolation cho riêng HOPE. Nếu một category mất toàn bộ positive pixels khiến pixel metric undefined, ghi undefined và suite chưa đủ co-primary acceptance, không loại category khỏi macro để claim thắng. Native-resolution analysis nếu làm phải là protocol/table riêng, áp cho mọi method và công khai geometry.

Chưa có phép đếm local xác nhận tỷ lệ mất mask trên VisA; đây là rủi ro cần QA, không lỗi đã chứng minh. Mask QA nằm trong evaluator/data-integrity tooling, không cấp masks/labels/test metrics cho calibration hoặc dùng để chọn β/architecture. Recipe MVTec phải khóa trước VisA anomaly-test feedback.

| Hạng mục | Chủ trì | Giao diện cần thống nhất |
|---|---|---|
| HOPE/VisionHOPE/CMS/outer objective | Bạn | Method adapter và artifact schema |
| ReplayCAD/LDM/SAM/native detector | Khang | Data access, backbone identity, task checkpoints, final metrics và resource accounting |
| CADIC baseline provenance | Nhóm, lấy run metadata từ nơi đã chạy | Checkpoint/config/adapter hashes; budgets 2.5k/10k |
| Dataset protocol/metrics | Dùng chung | Immutable manifest, order, resolution, metric formulas, evaluation-state policy |
| Final comparison | Dùng chung | Native vs shared-backbone labels; same-protocol table và bảng khác điều kiện tách rõ |

ReplayCAD chưa sẵn sàng không chặn correctness/pilot HOPE. Nhưng bảng so sánh cuối chỉ ghi ReplayCAD continual khi có checkpoint sau từng task và continuation semantics được Khang xác nhận/kiểm thử; không lấy epoch checkpoints của joint training để tính continual forgetting.

### B11. Nội dung `instruction_CAD_v3.md` sau khi freeze

Các quyết định F1–F5/F9/F10 đã chốt. [instruction_CAD_v3.md](instruction_CAD_v3.md) là tài liệu bước 5, bao gồm abstract/contribution scope, CAD formalization, toán học, equation-to-code map, algorithms/shapes/state, data/metrics, codespace, môi trường/commands, reproducibility và limitations. Đây là entry point tự chứa cho người/AI tiếp quản; không yêu cầu đọc tuần tự mọi docs lịch sử.

Mỗi module ghi `implemented`, `tested`, `experimentally validated` hoặc `proposed/not implemented` kèm evidence. Phần kế thừa paper, lựa chọn implementation và giả thuyết mới tách nhãn rõ. Khi code thay đổi, update bảng evidence và snapshot; không biến kế hoạch thành mô tả hệ thống đã tồn tại.

## C. Các quyết định đã chốt

Sau rà soát chuyên môn và theo quyền quyết định người dùng giao, chốt:

1. **Kiến trúc main:** visual fast state reset theo ảnh/view; CMS persistent xuyên ảnh/task; thay đổi versioned so với v1.
2. **Objective/head:** normal-only masked fixed-teacher prediction, squared residual từ decoded output cuối CMS; không moving-space NN bank.
3. **Dữ liệu:** prefix task đầu, 4 pools đã định nghĩa, không future-task calibration; paper-control và matched-method tracks riêng.

Revision tài liệu ban đầu được tạo trên `hope_cad_v2_clone@a89fd48`; đó là lịch sử freeze. Checkout triển khai hiện hành là `hope_cad_v2_1`, bắt đầu tại `5187e4d3ba5fe756520e2499ec6b90a4b6f91507`, và chỉ nhánh này nằm trong scope thao tác. Lịch sử nguồn `dikhang_space@a89fd48` giữ nguyên; MVTec trước VisA; Kaggle là môi trường chính; bạn làm HOPE, Khang làm ReplayCAD. Không còn câu hỏi về ba quyết định mục C.

Bước 3–4 hoàn tất ở mức tài liệu v3. Trên `hope_cad_v2_1`, W3 và W4 feature-level đã implemented/tested; Chặng 1 technical acceptance đã PASS với verifier 56/56, canonical exact-16 T4 và upstream parity 19/19 fixtures. Cần thu thập tiếp CADIC matched-prefix provenance, full RGB pipeline, real-prefix selection/gate/selected β và benchmark của từng method. Chúng là evidence/đầu ra W2/W5–W7, không điền giả. Chưa có benchmark CAD xác nhận hiệu quả của full method v3.
