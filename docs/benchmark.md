# Benchmark và acceptance

Study ID `hope_support_v5_1_mvtec_development_benchmark_v1`; toàn bộ jobs, seeds, task order và ngưỡng nằm trong `src/hope_cad/resources/study.json`. Đây là study MVTec đã development-exposed; không là protocol VisA đã được nghiệm thu.

## Cách thực thi hiện hành

Một CLI: `hope-cad preflight`, `train`, `evaluate`, `matrix`, `aggregate`, `export`. Trên máy thuê dùng [runbook](operations.md). Mỗi job bắt đầu fresh trên runtime mới; source/preflight/data/backbone/runtime locks phải khớp. Không resume bank/checkpoint T4 lịch sử hoặc v7 r2 qua source layout mới. Không cần chạy lại các evidence lịch sử để coi chúng tồn tại.

`train` chỉ học 3.599 normals, `evaluate` tính final 15 cells, `matrix` bổ sung 105 cells mà không train. Chín methods × ba seeds là 27 jobs theo study; chạy NESTED/CADIC seed0 không tự hoàn thành full study. Test chỉ vào evaluator, không vào learner/selection. Acceptance vận hành/kiểm context không tự là superiority.

Các mục dưới giữ protocol/metrics/claim rules của study. Tên chặng 7.1/7.2/7.3 là thuật ngữ lịch sử để đối chiếu kế hoạch; thao tác hiện tại dùng CLI ở trên. Các chỉ dẫn notebook/Kaggle cũ đã được thay bằng runbook SSH.

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

Trong CLI hiện hành, một cohort/run-dir có thể giữ nhiều jobs nhưng mỗi invocation chỉ chạy job được chỉ định. Resume là cùng run-dir/job/source/runtime/inputs, không import state của job khác. `evaluate` chỉ chạy sau train complete; `matrix` chỉ chạy sau final complete. Soft `--session-hours` là execution budget, trừ 180 giây để export; không phải learning budget hoặc ETA. Train progress mỗi 8 ảnh, evaluator mỗi cell. Giữ log và latest export; xem operations để bảo vệ phiên SSH và backup.

## 4. State, integrity và dung lượng

Active checkpoint mỗi8images/task boundary/pause; atomic same-filesystem replacement, hashes/clock/job identity và event prefix được kiểm. Hard crash replay tối đa7events; sẵn trace phải exact. Task checkpoints immutable với hash receipts. Tránh metadata tuple/list mismatch bằng canonical JSON identity trước lock. Source/dataset/checkpoint/runtime/method/study khác thì refuse, không bỏ guard để resume.

Evaluation chunks8images có image scores/raw patch distances/labels/resized masks; NPZ không pickle, không lưu lossily pixel maps. Reconstruct bằng đúng FP32 interpolation để tính độc lập. NPZ/rows/state/checkpoint/hash binding trước reuse; orphan partial chunks chưa có receipt được tính lại read-only, không ảnh hưởng train. Không dùng full raw feature cache3599×784×768 trong working disk; train/eval extract causal từngảnh và lưu hash receipts thay cho cache lớn. Giữ15 task states và perimage events, raw score chunks, metrics, locks, resource receipts. Không claim total GPU memory chỉ từ sum modeltensors.

## 5. Metrics và uncertainty

Mỗi category: image AUROC/AP từ Eq9; pixel AP=`sklearn.average_precision_score` trên toàn pixels224×224 của category; pixel AUROC tương tự. Primary macro là arithmetic mean đủ15category, không pooled hoặc anomalyonly pixels. `p_ap` và historical alias `p_aupr` là cùng average precision, không trapezoidal PR-AUC. Metric không finite/missing thì fail, không `nanmean`.

Matrix a[i,j] = metric taskj sau khi học taski, j≤i. Với j<14: F_j=max(a[j:14,j])−a[14,j], BWT_j=a[14,j]−a[j,j]; average14oldtasks, không clamp F về0 hoặc đọc future cells. Single-task fixture F/BWT null.

So NESTED−CADIC (primary), NESTED−FO và các ablations là secondary. Whole-image paired stratified bootstrap theo category×image label,2000replicates/private NumPy SeedSequence([20261020,task_id]); cùng image multiplicities cho methods và cùng finite seeds. Resample normal/anomaly ảnh trong từngstratum có replacement, giữ mọi pixels của ảnh với multiplicity tương ứng, không bootstrap pixels độc lập. Average finite three seed endpoints, không resample seed population. CI percentile1.25/98.75 cho hai co-primary endpoints; NI margin−2 pp cho cảhai; superiority chỉ khi cảhai lowerbounds>0. Nếu thiếu3 seeds thì descriptive subset, không report full-study CI. Bootstrap draw có zero positive pixels được AP=0 theo fixed degenerate convention, không redraw hoặc drop replicate; official full-category undefined metrics vẫn technical fail. Deterministic bootstrap seed/draw receipts khóa trước AD metrics; CPU statistical aggregation tách khỏi GPU training và có thể tốn đáng kể. Không đổi sampler/CI sau kết quả. Cùng sorted pixel scores có thể tái dùng để tính weighted AP; sorting/ties phải match sklearn.

## 6. Nghiệm thu và bàn giao

Local tests cần meaningful interrupted/uninterrupted session/event/state parity, cross-task pending, truefinalize, partial optimizer windows; readonly evaluator and compactmaps metric equivalence; score chunk/taskcheckpoint/data/hash corruption rejection; 7.2train forbidden/finalrow preservation; allcontrol actual optimizer clocks/gradients; exactinit pairing; paired bootstrap weights/ties and missing-coverage failures; source ZIP/extracted checkout installation; runtime mismatch fail trước data/model.

Source hiện hành có inventory toàn bộ package/config/docs/tests, thiết kế lịch sử có lineage riêng. Code refactor thay execution/source identity, không thay method ID hoặc study; không resume source cũ bằng cách bỏ guard. Local installation/tests/parity có receipt riêng, nhưng remote GPU preflight và run acceptance cần artifacts thật. Không fabricated metrics hoặc guarantee≥CADIC.

MVTecbenchmark là development evidence. VisA/order studies chưa là study được triển khai; ReplayCAD chỉ có integration riêng, chưa là matched result; cần protocol và freeze/authority riêng. Thesis phải giữ negativev4/v5, pilotv5.1 và eventualADresults với đúng lineage, không biến passnormal risk thành theoremAD.
