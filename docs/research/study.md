# Study — attribution theo giai đoạn và candidate H32, bản 1

Study ID: **hq32_mvtec_fulltrain_development_v1**.

**Trạng thái: runner/controls đã implement và test local; chưa remote preflight/pilot hoặc AD results, chưa tự được cấp quyền chạy GPU.** Study này không sửa study lịch sử gồm 9 methods × 3 seeds. Nó khóa candidate, các đối chứng, thứ tự thực hiện và quy tắc quyết định chi tiếp. Quyền sử dụng GPU, tài khoản, SSH và ngân sách tiền thực tế vẫn cần người dùng giao trong nhiệm vụ vận hành sau.

## 1. Cohort chung và các lớp identity

Mọi arm được so sánh trực tiếp phải dùng cùng source bundle, Linux/Python/dependencies, GPU, numerical flags, raw feature producer, checkpoint, preprocessing, danh sách ảnh train/test, thứ tự dữ liệu, K, Eq9 và evaluator. Runtime RTX 5090 kế thừa các pins và yêu cầu preflight thực của v7. Có pins chuẩn bị sẵn không đồng nghĩa đã nghiệm thu môi trường GPU thuê.

Không nhập trạng thái T4 cũ hoặc trọng số pilot vào các lượt fresh của cohort mới. Đối chiếu số đo lịch sử chỉ là tham khảo, không phải paired comparison của cohort này.

| Lớp identity | Nội dung bắt buộc |
|---|---|
| Cohort chung | Digests của source, runtime, raw producer, checkpoint, data, protocol, metrics và study |
| Job/phương pháp | Parent design được kế thừa; method/variant ID; gradient/history policy; parameter và state schema; seed; job digest |

Aggregator nhận các **method digests khác nhau đã đăng ký trong cùng study**, đồng thời kiểm các identity chung. Không ép mọi arm mang cùng design ID bằng cách gắn nhãn H32 thành v5.1; cũng không bỏ kiểm source/runtime/data để gộp run tùy ý.

R_EXACT giữ semantics v5.1; history bổ sung chỉ là bookkeeping phục vụ đối chứng và không được đổi đường học/scoring. R_FO giữ first-order semantics tương ứng của parent. H32_EXACT, H32_FO và H32_DIRECT có identity phản ánh đúng candidate và variant. Parent-design reference không thay identity của thuật toán thực thi.

contract.py, study.py, runtime preflight và statistics hiện khóa một accepted design. Implementation mới cần factory/manifest phân biệt revisions trong src/hope_cad, đồng thời giữ strict guards của API, JSON và trạng thái cũ. Không thêm tên vào METHODS lịch sử hoặc ghi đè method.json/study.json. Candidate contract là resource riêng, được package, lập inventory và hash. Mọi đặc tả cần cho implementation nằm trong repo; không cần kho context riêng.

Training dùng canonical MVTec order gồm 15 tasks, **3.629 train/good normals, Bottle 209 và không loại ảnh nào**. Protocol `mvtec_1x15_fulltrain_v7_v1` phải được triển khai bằng successor manifest factory riêng; không dùng partition factory 3.599 hiện tại rồi chỉ đổi metadata. Mọi learning seed 0/1/2 dùng cùng IDs/order; không có partition RNG hoặc calibration. Learning/model/private RNG seeds chỉ dùng cho replication. Khóa full path, file SHA và mask policy trước thực thi. D0 riêng vẫn đọc trace lịch sử 3.599; không sửa historical counts. Giữ preprocessing hiện hành theo [pipeline spec](../pipeline_alignment_spec.md).

## 2. Các variants đã đăng ký

| ID | Thuật toán | History bookkeeping | Slow query và gradient |
|---|---|---|---|
| CADIC | Matched CADIC hiện hành | Không | Không utility learner |
| R_EXACT | v5.1, bốn recent queries | 32 records; khớp payload/sampling với H32 | Bốn support steps → recent4, exact |
| R_FO | First-order semantics của v5.1 | 32 records như R_EXACT | Bốn support steps → recent4, detach toàn increment |
| H32_EXACT | Candidate chính | 32 records | Bốn support steps → recent2 + history2, exact |
| H32_FO | First-order counterpart của H32 | 32 records | Cùng query rule, detach toàn increment |
| H32_DIRECT | Direct replay control | 32 records | Cùng query rule tại phi_start; không functional support step |
| FAST_ONLY | Online fast learning, không slow Adam | Giữ bookkeeping/budget 32 records theo candidate contract; công khai allocation | Không slow update |
| BIAS_VETO | Scalar fast bias | Không projected history | Không slow update |
| PILOT_RATE_RANDOM_VETO | Admission ngẫu nhiên, rate cố định | Không learner history | Không slow update |

Đây là **9 variants của study mới**, không phải 9 methods đã hoàn thành hoặc quyền tự chạy 27 jobs. GEOMETRIC không nằm trong pilot tối thiểu mới; nó vẫn giữ nguyên trong study v5.1. Không đưa vào batch này không có nghĩa GEOMETRIC luôn tương đương CADIC.

### 2.1. BIAS_VETO

Prediction p = gamma + b, b0 = 0. Sau completed label: **b ← b − 0.05 × (p_at_decision − y)**, denominator bằng 1. Không neural context/Adam. Actual fast-update count bằng số completed labels L; slow-update count bằng 0. Causal pending, warmup 32, no-op, censor, bank và anchors giữ semantics của baseline. Pending record lưu prediction tại quyết định để credit đúng label; không dùng lại model của arm khác.

Control có thể thu được lợi ích từ utility learning ít chiều mà không cần NL đầy đủ. Cùng learning rate không đồng nghĩa cùng effective optimization với full model.

### 2.2. PILOT_RATE_RANDOM_VETO

Reject probability cố định **145/211**, accept probability **66/211**, lấy từ normal pilot cũ đã review; không phải tỷ lệ theo task của full benchmark. Warmup 32 completed labels force PASS. Ngoài warmup, mỗi full-bank changed proposal draw đúng một lần từ private random.Random(seed + 95537); chấp nhận nếu random() < 66/211. RNG là owner liên tục, không khởi tạo lại mỗi proposal.

Không task ID, future counts hoặc learned prediction. No-op không draw. Giữ pending counterfactual pair và label clock để warmup/diagnostics; không optimizer giả. RNG, clocks, bank, transaction và resume phải strict.

Đây là **pilot-informed fixed-rate control**, không full-stream rate-matched. Báo realized proposal rejects, vector writes và chi phí. Không lấy completed NESTED rates theo task để chỉnh probability; hindsight oracle thuộc study khác.

## 3. Stage A — D0 và nghiệm thu local

D0 theo [attribution v2](attribution.md), không GPU. Output hoàn thành phải được review trước khi quyết định thời điểm chạy candidate. Không dùng D0 để đổi capacity, query ratio, optimizer hoặc stat 12. Kết quả bias/clock hỗ trợ diễn giải và quyết định **chi tiếp hay dừng**, không tạo candidate thứ hai dưới cùng ID.

Implementation local đáp ứng [candidate acceptance](candidate.md) và regressions cũ. Kiểm clean installation, source inventory và Git bytes. Baseline chính thức `89df9a21658b265731f98a7c2014afb1876f1924` đã chứa source và bộ docs/spec/plan công khai; whole-source identity khác archive r3. Khi ship candidate code, tạo **delivery mới**; không overwrite r3 hoặc gọi current checkout byte-exact r3.

D0 inconclusive được báo đúng như vậy; không bắt buộc tìm root cause tuyệt đối mới code candidate có phép bác bỏ. Nếu người dùng giao implementation bundle, có thể code/test local theo spec; benchmark vẫn khóa đến khi có quyền chạy và đạt gates. Spec hiện chưa là lệnh implementation/execution.

## 4. Stage B — preflight thật và normal pilot

### 4.1. Preflight trên discarded clones

Trên RTX 5090 thật, kiểm GPU/driver/Python/runtime/source/data/checkpoint/numerical flags; full-grid exact/FO historical meta-learning, cached NN/Eq9 parity, scoring read-only, atomicity/resume, RNG và corruption rejection.

Dùng 24 allowed normals đầu cùng **bốn normal events kế tiếp** dùng kiểm resume trên discarded clones của CADIC, R_EXACT, H32_EXACT và H32_DIRECT. Không giữ learner/bank state cho pilot. Historical-gradient checks dùng completed-history synthetic fixtures đã khóa với full token grid; không bịa completed labels cho 28 ảnh thật để ép history đủ. Controls còn lại phải qua local state/gradient tests trước pilot.

Chỉ đọc train/good, không test/GT. Có thể cache temporary features nếu đã bind source/runtime/producer/file SHA và không chứa learned state. Nonfinite hoặc device-parity failure: export, dừng và review; không đổi precision/LR để vượt guard. Technical patch giữ equations nhưng đổi source/producer phải preflight lại đúng identity mới.

### 4.2. Normal pilot cố định

Chạy **9 arms × 256 normals**, seed 0. Canonical 256 allowed online IDs đầu là **209 Bottle + 47 Cable**. Mỗi arm fresh state; cùng initial theta/phi khi topology tương đương, cùng dữ liệu và private RNG tách theo contract.

Job order cố định: **CADIC → R_EXACT → FAST_ONLY → BIAS_VETO → PILOT_RATE_RANDOM_VETO → R_FO → H32_EXACT → H32_FO → H32_DIRECT**.

Không chọn winner giữa arms, bỏ arm âm hoặc chuyển pilot state sang full AD. Pilot kiểm cơ chế/safety ở một transition, **không xác nhận generalization hoặc retention qua 15 tasks**. Không lặp pilot để train thêm hoặc chọn checkpoint.

Gates đã khóa:

1. Cả chín arms đủ 256 IDs; binds/counters/state/score/RNG/finite checks đạt. Pause chưa là PASS.
2. Mỗi arm duy trì completed-label clock, trừ CADIC, có ít nhất 64 completed labels. R/H có ít nhất tám scheduled slow steps; FAST_ONLY/BIAS/RANDOM có đúng 0 slow steps. Traces khớp selected history/query IDs, Algorithm R và startup fallback.
3. H32_EXACT có ít nhất một slow episode sau startup mà historical-query loss đóng góp gradient theta hữu hạn, khác 0. Kiểm bằng fixture và gradient trace có provenance; không dùng step-norm ratio để claim CMS benefit. Theta gradient bằng 0 ở block 0 của DIRECT là dự kiến, không phải lỗi.
4. H32_EXACT có post-warmup reject fraction trong [0.01, 0.99], bank khác CADIC. **Sau khi đủ 256 normals**, score read-only đúng **24 allowed train IDs đầu** bằng H32_EXACT và fresh CADIC; so pixel maps 224×224 theo cùng evaluator bilinear, không normalize thêm. Max absolute map difference trên toàn bộ 24 ảnh phải > 1e-6. Không chọn subset/checkpoint/loại score sau khi thấy kết quả. Đây là reused-normal diagnostic kiểm can thiệp cơ chế, không chứng minh reject đúng hoặc AD gain; outputs không đưa vào learner hoặc sampler.
5. Safety: cửa sổ 128 normals cuối gồm **81 Bottle + 47 Cable**. Mean prequential R của H32_EXACT trên 128 normals cuối / mean R của fresh CADIC trên đúng tập đó **<= 1.10**. Margin 10% cùng pilot pool/risk giới hạn suy giảm normal coverage, **không phải theorem hoặc AD acceptance**. Không bắt H32 thắng R_EXACT về normal MSE mới được test AD.
6. Báo mọi arm/gate, functional delta-p, query ages, clip frequency, memory và timing. Không đổi 256→128, history 32→16 hoặc gate để cứu failure.

### 4.3. Budget Stage B

Cap **12 GPU wall-hours cộng dồn**, từ bắt đầu preflight đầu đến export pilot cuối. Tính setup trên máy GPU, feature extraction, hashing, I/O và pause mà GPU vẫn billed; không tính thời gian máy đã Stop. Mỗi session soft limit tối đa **4 giờ**, reserve **180 giây** để checkpoint/export. Cumulative ledger qua sessions và atomic resume cùng identity là bắt buộc.

Cap là giới hạn chi tiêu, không ETA. Hết cap chưa đủ arms/gates → **budget_unresolved**, không lấy completed subset làm PASS. Chỉ người dùng quyết định cấp thêm budget qua record vận hành; không retune method.

Cost cap USD dựa actual offer/user budget trước SSH; không gửi API key hoặc tự thuê. Nếu cost cap thấp hơn giá thực tế × 12 giờ thì cost cap ưu tiên. Người dùng có thể không chạy phase; đó là lựa chọn tài nguyên, không phải thiếu thuật toán.

## 5. Stage C — một development batch chính

Sau khi pilot được nghiệm thu kỹ thuật và người dùng giao chạy, thực hiện đúng **CADIC_seed0, R_EXACT_seed0, H32_EXACT_seed0**. Mỗi job fresh train 3.629 normals, đánh giá 15 final cells + 105 matrix cells còn lại = **120 cells**. Discard chín pilot states. Common cohort/runtime/producer/IDs/evaluator; scoring read-only; raw outputs đủ recompute. Báo đủ ba jobs, không chỉ winner.

Batch C có cap riêng **24 GPU wall-hours cộng dồn**, cùng cách tính thời gian/reserve như B. Hết cap chưa đủ jobs/matrix → budget_unresolved, không NI/superiority claim. Đây là cap quản trị, không dự báo runtime hoặc quyền mở thêm jobs.

Primary metrics: macro_final_i_auroc và macro_final_p_ap. Giữ image Eq9 b=9; raw patch grid 28→224 bilinear CPU; không Gaussian/L2 bổ sung. Raw mask nonzero, direct nearest resize, tiny-defect QA. Undefined metric fail, không drop task. Task matrix, forgetting/BWT và diagonal acquisition giữ definitions. Score lớn hơn nghĩa bất thường hơn; không flip theo task.

### 5.1. Spend-screen để mở Stage D

Các điều kiện chỉ quyết định chi tiếp cho attribution, **không phải statistical NI, final selection hoặc meaningful-gain proof**:

- Delta P-AP(H32_EXACT − R_EXACT) > 0.
- Delta I-AUROC(H32_EXACT − R_EXACT) >= −0.02.
- Delta I-AUROC(H32_EXACT − CADIC) >= −0.02 và delta P-AP(H32_EXACT − CADIC) >= −0.02.

Metrics ở thang [0,1]; 0.02 là 2 điểm phần trăm. H32−R_EXACT đo incremental history benefit; H32−CADIC đo toàn hệ thống. Gain P-AP 2,53 điểm đã có của v5.1 không được gán lại cho H32. AP dương chỉ là trigger đầu tư attribution, không chứng minh vượt nhiễu hoặc đáng chi phí.

Không đạt screen: dừng vòng H32, giữ v5.1 làm method chính; báo all metrics/negative. Không sửa normalization/anchors/query ratio để chạy lại cùng ID. Đạt: chưa adopt H32 hoặc mở VisA; chỉ mở D đã định nghĩa khi người dùng cấp budget.

### 5.2. Diễn giải retention

Báo H32−R_EXACT về pixel FM/BWT, diagonal P-AP, image FM/BWT và per-task metrics. Nếu AP tăng qua diagonal acquisition nhưng FM không tốt, gọi acquisition benefit, không proof consolidation retention. Không gộp selected tasks; pixel-retention advantage không tự là image advantage.

Uncertainty/CI phải đúng development scope. Không dùng 80% gain như theorem bác mọi revision; không coi cells/ảnh phụ thuộc là nhiều independent seeds.

## 6. Stage D — attribution và replication có điều kiện

Danh sách cố định; không chọn chỉ controls có vẻ sẽ thắng sau C:

1. Hoàn tất sáu seed-0 variants: R_FO, H32_FO, H32_DIRECT, FAST_ONLY, BIAS_VETO, PILOT_RATE_RANDOM_VETO. Mỗi job cùng cohort, 3.629 train normals + 120 cells.
2. Reuse ba completed C jobs nguyên trạng, không retrain để thay seed-0 result.
3. Seeds 1/2 cho CADIC, R_EXACT, H32_EXACT: sáu fresh jobs, mỗi job 120 cells.

Tối đa **15 full jobs**: ba C + sáu controls + sáu replications; không tự chạy old 27-job study. Đây là conditional schedule, không prerequisite của D0/local code.

### 6.1. Budget thực hiện Stage D

Số jobs, thuật toán và thứ tự của D đã đặc tả. **Budget thực tế D chưa được cấp**: người dùng chọn cap GPU wall-hours/USD dựa measured B/C cost trước execution, lưu allocation record. Không chạy không giới hạn chỉ vì danh sách 15 jobs đã có.

Allocation là quyết định tài nguyên, không quyền đổi history/optimizer/controls/seeds/trial cap. Có thể dừng vì budget; completion/claims thu hẹp tương ứng. Không suy ETA 10–100 giờ hoặc RTX speedup từ core spec. Budget extension chỉ bởi người dùng và record mới.

### 6.2. Statistical reporting

Kế thừa paired whole-image bootstrap, category/image-label strata, cùng draws giữa methods. Khi đủ ba seeds hữu hạn, báo finite-three-seed average; **2.000 replicates**, bootstrap seed **20261020**, percentiles **1.25/98.75**. Không bỏ seed/task lỗi để có CI đẹp.

Với một contrast ghi rõ:

- NI margin −0.02: cả hai primary lower bounds phải > −0.02 để thỏa NI criterion đã định nghĩa.
- Cả hai lower bounds > 0 để thỏa superiority criterion đã định nghĩa.

CI trên finite-seed average không là population-seed CI. H32−CADIC và H32−R_EXACT là hai contrasts riêng. Two-endpoint percentiles không tự điều chỉnh mọi contrast/control/quyết định mở D. **MVTec đã development-exposed và chương trình có conditional screening; inference tổng thể là exploratory nếu chưa có family-wise analysis plan thích hợp.** Không gọi confirmatory MVTec chỉ vì đạt bounds.

CI chứa 0: chưa establish superiority; không proof equivalence hoặc buộc phải bỏ NL.

### 6.3. Claims và giới hạn attribution

| Claim | Bằng chứng và giới hạn |
|---|---|
| Toàn hệ thống hơn CADIC | Matched metrics/CI, disclosure tổng memory/cost, development scope |
| Incremental historical-query benefit | H32−R_EXACT, cùng topology/query count/storage/numeric/data; R_FO/H32_FO interaction report descriptive |
| Exact higher-order gain | H32_EXACT−H32_FO và R_EXACT−R_FO; không từ action counts/norms. Controls chỉ seed 0 vẫn exploratory; broad exact claim cần replication controls riêng trước claim |
| Hơn ordinary direct replay | H32_EXACT−H32_DIRECT; DIRECT không equal compute và vẫn có actual fast learning |
| Multi-frequency causal benefit | Chưa chứng minh bởi bundle; không claim nếu thiếu matched synchronous/frequency control |
| Bias/random giải thích gain | Báo khả năng giải thích bằng policy đơn giản; không dùng 80% như theorem hoặc tự chọn winner mới trên VisA |

Controls/replications ngoài 15 jobs không tự được mở để cứu claim; cần scope/budget riêng. Controls giải thích gain → hạ claim hoặc để simplification cho quyết định sau, không giấu kết quả.

## 7. VisA và phần hoàn tất luận văn

VisA chưa runnable trong study hiện có. Trước confirmation cần protocol/implementation riêng: official split_csv/1cls.csv, 12 categories, fresh learner/bank, train-normal rights/exclusions, mask/resize QA, metrics/seed/order/runtime/checkpoint/compute và exposure ledger toàn nhóm. Không chỉ đổi data path hoặc nhập MVTec learned bank. Với cùng evaluation design, ma trận tam giác 12 tasks có **78 cells: 12 final + 66 extra**.

Các phần phụ thuộc dataset chưa khóa ở đây. Đây là **handoff boundary có chủ định**, không cam kết VisA đã sẵn chạy. Method/recipe freeze trước VisA test; confirmation là suite có multiple jobs/seeds/resume, không chỉ một process. Technical reruns có receipt không tự là leakage; retune sau test là development exposure, phải đổi/thu hẹp claim.

Phần replication/ablations còn thiếu của study baseline, external ReplayCAD, VisA, statistics và luận văn không tự hoàn thành từ ba seed-0 jobs hoặc H32 pilot PASS. Không gọi publication-ready/100% đề tài chỉ từ validation tài liệu.

## 8. Kỷ luật giữ spec trong quá trình chạy

Chỉ **một candidate H32, một normal pilot, một first development batch**; không hyperparameter grid/candidate thứ hai. Pause/resume cùng source/runtime/state là continuation, không trial mới. Budget extension chỉ bởi người dùng, có record; không đổi weights/clocks/order để dùng budget mới.

Thiếu logs/artifacts nghĩa chưa nghiệm thu. Technical bug phải có amendment/version nếu sửa semantics; không gắn nhãn result bất lợi là bugged khi chưa chứng minh bug. Sửa typo hoặc cấp budget vận hành không được dùng làm cớ đổi thuật toán.

Mục tiêu là khóa đủ quyết định để implementation hữu hạn, không tạo ảo tưởng spec không bao giờ sai. Audit kỹ cho phép bắt đầu code; empirical validation vẫn phải làm, không thay bằng đồng thuận AI.
