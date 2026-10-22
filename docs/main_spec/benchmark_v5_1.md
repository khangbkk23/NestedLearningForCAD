# Nested Learning for CAD — benchmark v5.1, study v1

Ngày 20/10/2026. Method `hope_cad_v5_1_causal_scaled_utility` đã qua canonical normal pilot; source/model/spec cũ giữ bytes. Tài liệu này và `conf/benchmarks/support_v5_1_study.json` bổ sung protocol/orchestration benchmark, không đổi phương pháp. MVTec đã development-exposed qua v4/v5/pilot; đây không là prospective confirmation độc lập. Chưa có anomaly metrics v5.1 khi viết tài liệu. Code/source hash và tests trong delivery receipt quyết định trạng thái implementation; local/preflight PASS không là AD gain.

## 1. Câu hỏi, đối chứng và phạm vi

So NESTED với matched CADIC K10000 trên macro final I-AUROC/P-AP; FIRST_ORDER kiểm contribution của exact meta. GEOMETRIC kiểm geometry prior; FAST_ONLY loại slow Adam; NO_SELF_MODIFICATION giữ static visual boundary reads, initial K/V/eta/alpha inactive và frozen, còn initial content/query/wrapper hoạt động; NO_ANCHOR_OBJECTIVE đặt anchor weight0 trên trajectory riêng; NO_FAST_ADAPTATION không fast/slow optimizer, zero readout nên geometry policy ngoài warmup32 trên trajectory riêng. Không gọi nó independent slow baseline. CADIC_TOTAL_BYTES K24576 chỉ đối chứng payload lớn, **không equal-total-memory claim**. Đo persistent bank/pending/anchors/model/Adam/replay/cache và runtime peaks riêng; baseline được thêm memory không chứng minh cùng total capacity.

Chín methods×ba seeds0/1/2 =27registered jobs, mỗi job fresh state. Cùng seed learned initial theta/phi/projection/anchor RNG của NESTED/FO phải khớp. Không đưa pilot weights, v4/v5 trained weights hoặc output của method khác vào learner. Không lựa chọn checkpoint, learning rate, threshold, score direction, smoothing hoặc controls dựa trên AD metrics. User có thể dừng compute vì quota/kết quả, nhưng phải công bố subset và không báo study27 jobs đã hoàn tất. Kết quả âm vẫn giữ.

## 2. Dữ liệu và inference

Giữ frozen ViT-B/8 ImageNet21k checkpoint SHA, block8 zero-based/raw784×768FP32, RGB direct224bilinear/ImageNet normalization. Cùng fixed lexicographic15 tasks và matched partition3599online train/good: 30normals validation/probe taskđầu bị loại như baseline. Không offline calibration/search. Các role names v4 trong partition không cấp quyền learning mới. Task IDs chỉ evaluator bookkeeping; cả train và meta clocks liên tục. Current pending được resolve bằng ảnh normal thực tiếp theo, kể cả qua boundary task. Không flush partial replay hoặc reset fast state ở boundary. True end chỉ sau3599images: censor pending một lần, giữ partial replay1–7records và không tạo target giả.

Official evaluator khóa1725test image paths/imageSHA/GTmaskSHA/labels theo15 tasks. Metadata/hash discovery không cấp features/labels cho learner. Kiểm raw masks không rỗng, encoding/channels và image-mask geometry. Mask raw nonzero→direct224nearest; giữ image label khi defect nhỏ bị erase, ghi QA identity/count. Nếu primary pixel metric không xác định cho cả category thì stop, không bỏ category. Test image/mask bytes phải tái kiểm trước load; mọi methods/seeds dùng cùng row ordering, hashes và interpolation.

Inference là actual bank raw NN Eq8/full Eq9 b9. Score càng cao càng anomaly. Pixel map28×28→224bilinear `align_corners=False`, **canonical interpolation CPU FP32 cho mọi method**, không smoothing/L2/normalization mới. Không controller/optimizer/pending resolve trong score. Kiểm exact scientific state trước/sau cell. Không đánh giá future-task cells. State checkpoint sau task cuối được finalize; checkpoint tasktrước vẫn giữ pending/partial để tiếp học chính xác.

## 3. Ba chặng, không lặp đường học

Preflight mới là technical gate benchmark/IO/evaluator: tests và numerical checks bằng synthetic fixtures; discarded real train-normal clone để kiểm full raw shape, transport/state/read-only score/resume. Không truy cập official test hoặc học pilot thêm; không thay accepted normal-pilot gate. Source/runtime/study/design/asset/manifest locks phải hợp lệ trước run.

**7.1:** một selected seed0 job, mặc định `NESTED_seed0`, học đầy đủ15 tasks từ đầu, save tất cả task checkpoints rồi evaluate15final cells. Muốn matched quick view, chạy thêm `CADIC_seed0`, `FIRST_ORDER_seed0` thành hai versions/jobs độc lập. Không gọi ảnh baseline Khang hoặc metrics v4 là matched v5.1 evidence. Cả ba giữ cùng chuẩn; mỗi job có ZIP riêng để dung lượng không tăng thành27 jobs trong một phiên.

**7.2:** chọn cùng job seed0 và attach completed7.1 output. Không gọi training loop; nạp task checkpoints để thêm 105 cells còn thiếu, tổng120triangular cells. Không học lại, không final metrics gain từ việc thêm matrix. Final15cells, checkpoints và events cũ được giữ bytes. Có thể áp dụng cả CADIC/FO seed0 nếu đã có7.1 để tránh train lại.

**7.3:** một registered job mỗi invocation, mặc định `NESTED_seed1`; chạy remaining modes/seeds và full120 cells. Với job đã có7.1/7.2, reuse output cùng job; không khởi tạo lại. Không tự chọn best seed hoặc bỏ control. Full study completion cần đủ 27matrices. Runner không tự launch GPU jobs hoặc VisA.

Mỗi ZIP gồm common locks/preflight và **một job**. Resume chỉ giữ latest output cùng job; khi đổi job, bootstrap chỉ import common preflight locks, không nhập learned state của job khác. Nếu muốn bổ sung phases cho job cũ, phải attach latest output của đúng job đó; thiếu parent mà cố chạy lại cùng job được disclosure là rerun, không lén coi là resume. Notebook không thể biết các runs đã tồn tại ngoài inputs, user phải giữ run ledger. Dùng Save Version → Save & Run All; sau pause attach latest ZIP/NotebookOutput cùng source/assets và giữ phase/job. `SESSION_HOURS` là soft execution budget .1–11h, trừ180s để export; không learning budget/ETA. Hard platform kill vẫn có thể ngăn export. Heartbeat30s và progress mỗi8images.

## 4. State, integrity và dung lượng

Active checkpoint mỗi8images/task boundary/pause; atomic same-filesystem replacement, hashes/clock/job identity và event prefix được kiểm. Hard crash replay tối đa7events; sẵn trace phải exact. Task checkpoints immutable với hash receipts. Tránh metadata tuple/list mismatch bằng canonical JSON identity trước lock. Source/dataset/checkpoint/runtime/method/study khác thì refuse, không bỏ guard để resume.

Evaluation chunks8images có image scores/raw patch distances/labels/resized masks; NPZ không pickle, không lưu lossily pixel maps. Reconstruct bằng đúng FP32 interpolation để tính độc lập. NPZ/rows/state/checkpoint/hash binding trước reuse; orphan partial chunks chưa có receipt được tính lại read-only, không ảnh hưởng train. Không dùng full raw feature cache3599×784×768 trong working disk; train/eval extract causal từngảnh và lưu hash receipts thay cho cache lớn. Giữ15 task states và perimage events, raw score chunks, metrics, locks, resource receipts. Không claim total GPU memory chỉ từ sum modeltensors.

## 5. Metrics và uncertainty

Mỗi category: image AUROC/AP từ Eq9; pixel AP=`sklearn.average_precision_score` trên toàn pixels224×224 của category; pixel AUROC tương tự. Primary macro là arithmetic mean đủ15category, không pooled hoặc anomalyonly pixels. `p_ap` và historical alias `p_aupr` là cùng average precision, không trapezoidal PR-AUC. Metric không finite/missing thì fail, không `nanmean`.

Matrix a[i,j] = metric taskj sau khi học taski, j≤i. Với j<14: F_j=max(a[j:14,j])−a[14,j], BWT_j=a[14,j]−a[j,j]; average14oldtasks, không clamp F về0 hoặc đọc future cells. Single-task fixture F/BWT null.

So NESTED−CADIC (primary), NESTED−FO và các ablations là secondary. Whole-image paired stratified bootstrap theo category×image label,2000replicates/private NumPy SeedSequence([20261020,task_id]); cùng image multiplicities cho methods và cùng finite seeds. Resample normal/anomaly ảnh trong từngstratum có replacement, giữ mọi pixels của ảnh với multiplicity tương ứng, không bootstrap pixels độc lập. Average finite three seed endpoints, không resample seed population. CI percentile1.25/98.75 cho hai co-primary endpoints; NI margin−2 pp cho cảhai; superiority chỉ khi cảhai lowerbounds>0. Nếu thiếu3 seeds thì descriptive subset, không report full-study CI. Bootstrap draw có zero positive pixels được AP=0 theo fixed degenerate convention, không redraw hoặc drop replicate; official full-category undefined metrics vẫn technical fail. Deterministic bootstrap seed/draw receipts khóa trước AD metrics; CPU statistical aggregation tách khỏi GPU training và có thể tốn đáng kể. Không đổi sampler/CI sau kết quả. Cùng sorted pixel scores có thể tái dùng để tính weighted AP; sorting/ties phải match sklearn.

## 6. Nghiệm thu và bàn giao

Local tests cần meaningful interrupted/uninterrupted session/event/state parity, cross-task pending, truefinalize, partial optimizer windows; readonly evaluator and compactmaps metric equivalence; score chunk/taskcheckpoint/data/hash corruption rejection; 7.2train forbidden/finalrow preservation; allcontrol actual optimizer clocks/gradients; exactinit pairing; paired bootstrap weights/ties and missing-coverage failures; notebook ZIP/extracted sourcebootstrap; runtime mismatch fail trước data/model.

Source r1 benchmark có schema/inventory riêng và giữ original144 pilotpayloads byte-exact. Fullmethod specs/acceptance cũ giữ nguyên; additive study/spec này được sourcehash bao gồm. New benchmark source digest khác pilot delivery, không resume pilot weights. Preserved core namespaces thay đổi ngoài spec thì stop/revision, không lặng lẽ overwrite bundle. Code/tests/docs/build/verify có receipt riêng. Run acceptance còn cần user chạy realT4 preflight và benchmark rồi cung cấp ZIP/log. Không fabricated metrics, no guarantee≥CADIC.

MVTecbenchmark là development evidence. VisA/replayCAD/orderstudies không runnable bởi notebook này; cần protocol và freeze/authority riêng. Thesis phải giữ negativev4/v5, pilotv5.1 và eventualADresults với đúng lineage, không biến passnormal risk thành theoremAD.
