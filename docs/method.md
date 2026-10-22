# Phương pháp — Nested Learning for CAD

Method ID: **`hope_cad_v5_1_causal_scaled_utility`**. Workspace/package v7 là cách tổ chức source, không phải thuật toán mới. HOPE/CMS học utility của proposal để quản lý raw normal-support bank; ViT và CADIC Eq9 giữ nguyên.

## 1. Phạm vi và nguồn chuẩn

Method đã implement; normal pilot và matched MVTec seed0 final/matrices đã được review trong workflow trước. Đó là single-seed development evidence, chưa là bằng chứng tổng quát hoặc VisA confirmation. Runtime Linux/5090 mới vẫn cần preflight thật. Không dùng status “chưa implement” ở spec lịch sử làm mô tả hiện tại.

Các mục 2–10 dưới đây kế thừa đầy đủ phần định nghĩa toán học và contracts của kiến trúc đã khóa. `resources/method.json` là contract máy đọc nguyên byte. `resources/lineage.json` giữ hashes của bộ spec gốc và design digest lịch sử; các bản gốc nằm trong source archive trước refactor. Docs hợp nhất và source hiện hành được khóa độc lập bởi source digest mới. Không tuyên bố docs mới còn byte-exact với spec cũ.

Đọc [kiến trúc code](architecture.md), [benchmark](benchmark.md), [vận hành](operations.md). Các clock/target/precision/bank/score không đổi trong refactor này. Đề xuất historical-query consolidation chưa được triển khai. Chỉ thiết kế revision mới khi có nhiệm vụ và contract riêng; không thay thuật toán sau khi xem anomaly metrics rồi gọi cùng frozen benchmark.

## 2. Dữ liệu và raw geometry không đổi

Stream gồm các normal images thực sự đã đến, một ảnh/event. Frozen backbone F là `vit_base_patch8_224.augreg_in21k`, checkpoint SHA256 `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`. RGB PIL resize trực tiếp 224×224 bilinear, không giữ aspect ratio; mean (.485,.456,.406), std (.229,.224,.225). Trích block 8 zero-based / layer 9 one-based, bỏ CLS: raw FP32 features [1,784,768]. Không thêm LN/L2 vào raw patches.

MVTec giữ canonical 15-task order trong contract và đúng 3,599 online identities: 3,629 normals trừ 15 validation + 15 probe của Bottle theo matched protocol. Tên protocol có `prefix_cal` là lịch sử phân vùng, **không cho v5.1 offline calibration**. V5.1 không dùng các excluded roles để train/search. Các controls có cùng identities/order/checkpoint/preprocessing. VisA chưa có runner/protocol được nghiệm thu ở repo này. Official `split_csv/1cls.csv`,12tasks và fresh state là yêu cầu hướng tới; data rights/exclusions/order/evaluator phải được freeze riêng trước confirmation, không mặc định mang partition Bottle sang dataset khác.

Manifest/hash kiểm identity không cấp quyền đọc future features. Feature cache có thể dùng chung frozen producer nhưng learner chỉ nhận một ảnh theo cursor; không tính aggregate future features, normalizers hoặc sampling weights từ các ảnh chưa đến. Task/category ID chỉ thuộc evaluator, không input, reset hoặc routing của learner. Test/GT masks không vào optimizer hoặc selection.

Bank C chứa tối đa K=10,000 **raw observed patch vectors**. Proposer là đúng CADIC image update, gồm repeated replacements, deterministic ties, cache và remainder của ảnh làm đầy bank. Veto nhận/từ chối toàn **proposal**, không thay memory unit thành image và không nhận mọi 784 patches vô điều kiện.

Giữ numeric route cached CADIC đang có tại baseline source `84cccbae...`; không lén sửa pair-distance cancellation trong revision học này. Audit đã tái hiện hazard nhưng chưa đo prevalence/AD impact. Nếu sửa kernel sau này phải có numeric route/source mới, áp dụng cho mọi matched controls và review riêng trước benchmark. `exact_paper_parity_claim=false` tiếp tục có hiệu lực.

## 3. Scoring và cơ sở normal-support

Patch score `d(q,C)=min_c ||q-c||_2`. Lấy q* có NN distance s* lớn nhất, c* là matched bank row và N9(c*) là 9 bank rows gần c*. Image score giữ full CADIC Eq9:

`s_img=s*·[1-exp(s*)/sum_{c in N9(c*)} exp(||q*-c||_2)]`.

Dùng logsumexp ổn định, distance tiles 256; map 28×28 bilinear lên 224×224, align_corners=false, không Gaussian mới. Inference chỉ đọc actual bank. Không chạy utility learner, không resolve pending bằng test, không cập nhật bank/weights/RNG từ test. Controller ảnh hưởng prediction bằng **bank thật mà nó đã chọn**.

Với finite observed normal support S và C⊆S, epsilon=max_s d(s,C), triangle inequality cho `d(q,S) <= d(q,C) <= d(q,S)+epsilon`. Nếu normals gần S trong radius r0 và anomalies cách S ít nhất r1>r0+epsilon, patch NN có separation. Đây là conditional bound trong số học khoảng cách lý tưởng; không chứng minh các assumptions đúng, Eq9 ranking, AUROC, hoặc numerical kernel không có sai số.

Normal risk của một image/reference patch set X:

`R(X,C)=.5 mean_j d(x_j,C)^2 + .5 max_j d(x_j,C)^2`.

Anchor risk a là mean R của reservoir 32 image records, mỗi record 8 uniform sampled patches. Sampling 8 không bằng full 784 max; retention chỉ được regularize hữu hạn. Không claim global coverage/forgetting bound từ reservoir này.

## 4. Causal credit và scale dự đoán

Tại proposal image t, sau khi resolve label cũ, đặt C− là bank trước write, C+ là clone sau đúng proposer. A_t là reservoir **trước khi thêm image t**. Tính và cache:

`r_t^- = R(F(x_t),C-)`, `r_t^+ = R(F(x_t),C+)`, `a_t^- = R(A_t,C-)`, `a_t^+ = R(A_t,C+)`.

Với lambda=1, kappa=1 đơn vị của squared-distance risk:

`c_t = kappa + r_t^- + r_t^+ + lambda(a_t^-+a_t^+) >= 1`,

`gamma_t = [(r_t^-+lambda a_t^-)-(r_t^++lambda a_t^+)]/c_t`.

Kappa là design constant khóa trước run, tránh scale gần 0. Nó không được fit từ test hoặc đổi theo dataset, và không có scale-invariance claim khi đổi raw backbone geometry. Gamma/c đều known tại decision; gamma nằm (-1,1) trong exact real arithmetic. FP64 có thể round thành ±1 khi floor nhỏ so với risk; không clip hoặc fail chỉ vì đạt biên đó. GEOMETRIC của revision dùng **cùng numerator FP64** và sign gamma, tránh suy sign từ stats[-1] đã round FP32. Mathematical geometry policy không đổi; historical v5 control/receipt vẫn mang numeric identity cũ.

Commit bank theo policy rồi giữ một alternative full bank và cached costs đến image kế tiếp. Khi **normal x_(t+1) thực sự đến**, trước write của nó:

`J_t^- = R(F(x_(t+1)),C-) + lambda a_t^-`,

`J_t^+ = R(F(x_(t+1)),C+) + lambda a_t^+`,

`Delta_t = J_t^- - J_t^+`, `y_t = Delta_t / stored_c_t`.

Y là detached **causally scaled combined advantage**, không bị chặn [-1,1], không probability và không AD label. Không chia y bằng future sum S và không nhân loss bằng S/c. Nếu ghi legacy relative advantage `Delta/(J^-+J^++1e-8)` thì chỉ là diagnostic với key riêng, tuyệt đối không dùng làm target.

Cả accepted/rejected proposal đều có label từ hai fixed banks. Label là one-step fixed-bank contrast, không unbiased reward của full counterfactual rollout. Không backprop qua hard bank choice/raw risks. Terminal pending bị censor một lần; task boundary/pause không tạo label hoặc flush. No-op proposal không tạo label giả; bootstrap không so hai banks khác kích thước.

## 5. Context, visual self-modification và CMS

Fixed private CPU Rademacher matrix P[768,32], seed 20261017, entries ±1/sqrt32. U=F(x)P, projected context `X=U/sqrt(mean(U²)+1e-6)` theo toàn image. Context [1,784,32] không thay raw bank/score. Không claim 32D đáp ứng JL guarantee cho mọi patches.

Visual block cấu hình input/dim/memory 32, head_dim 8, four directions × four heads/direction, grid 28×28, chunk 28 theo hàng/cột. Wrapper: affine 32→32, pre-LN eps 1e-6, in-projection 32, depthwise 3×3, four-scan memory, out-projection/residual, pre-LN FFN 32→128→32 (GELU)/residual. Working maps reset cho mỗi image/context; slow learned initials được giữ giữa images.

Five memories gồm content M, key Mk, value Mv, rate Meta và retention Malpha. Trong mỗi chunk boundary b, với normalized visual input z:

`k=Mb_k z/sqrt(||Mb_k z||²+1e-6)`, `v=Mb_v z`, `eta=.025 softplus(Mb_eta z)`, `alpha=sigmoid(Mb_alpha z+logit(.9^(1/28)))`.

`delta=k-v`, `r=sqrt(||delta||²+1e-12)+1e-6`, `eta_inj=(1-alpha)(1-1e-3)/r`. Soft injection step `bar_eta=eta_inj*(-expm1(-eta/eta_safe))`, eta_safe=eta_inj nếu dương, ngược lại 1. Spectral cap `2alpha(1-1e-6)/max(||k||²,1e-20)`, FP32 forward theo rsqrt rồi bình phương và nextafter về 0. `eta_exec=min(bar_eta,spectral_cap_rounded)`.

`A=alpha I-eta_exec kk^T`, `B=-eta_exec delta k^T`, `M_t=M_(t-1)A+M_b B`. Five maps dùng chung ordered gain recurrence của operator kế thừa. Readout từ **boundary content map**, không post-token state; inverse scan/fusion giữ nguyên. Existing ULP straight-through và FP64 derivative bridge không được bỏ. Đây là tested primitive được tái dùng, không claim toàn successor đã tested hoặc full upstream VisionHOPE reproduced. Non-expansive visual recurrence không là stability theorem cho outer utility learner.

Meanpool visual output thành [1,1,32], cộng affine 16→32 của geometry statistics. Stats 16 theo thứ tự: current NN mean/p95/max/min/std chia mean raw norm; candidate NN mean/max cùng scale; relative current-risk gain; relative anchor-risk gain; bank fill; unique changed-row fraction; replaced/incoming; log1p(cursor)/log1p(1e6); reservoir fill; incoming/bank mean raw-norm ratio; relative immediate combined gain. Tính hoàn toàn từ ảnh hiện tại, fixed banks và past anchors. Stats[-1] vẫn là context feature lịch sử, không nguồn numerator chuẩn của prior.

Token 32 đi qua `CMS_fast(phi)` rồi `CMS_slow(theta)`. Mỗi CMS là residual `h + Linear32→64→32(GELU(LN(h)))`, LN eps 1e-6. Gọi kết quả là h_t∈R32. Không reconstruct teacher features, không masked RGB objective.

## 6. Fast readout, loss và update rule duy nhất

Đặt `u_t=[h_t, float32(log1p(c_t))]` gồm 33 neural features, `a_t=[u_t,1]` gồm 34 coordinates. Dùng cùng cast log feature trong readout và normalizer. Prediction scalar:

`p_t=gamma_t + w_fast^T u_t + b_fast`.

Readout w33/b1 khởi tạo **zero**; output affine residual không tanh/atanh/clipping. P và y có thể vượt [-1,1]. Bias derivative trực tiếp bằng 1. Gamma/c/risks/labels không nhận gradient; h_t, readout và d trong replay có gradient theo ownership.

Fast phi gồm tất cả CMS_fast parameters (cả LN) và readout w/b. Slow theta gồm visual wrapper+learned memory initials, affine geometry và CMS_slow. Fast template chỉ để fresh initialization, không nằm trong slow Adam; không có parameter vừa fast vừa slow. Backbone/projection/banks frozen.

Sau một completed label, recompute p và a từ **stored decision context** dưới pre-update theta/phi, đặt:

`ell_t=.5(p_t-y_t)^2`, `g_t=partial_phi ell_t`, `d_t=a_t^T a_t >= 1`,

`phi_next = phi - .05*(g_t/d_t)`.

**Tính gradient của ell trước rồi chia d. Không tính `grad(ell/d)`**: d phụ thuộc h/theta/phi, công thức đó thêm term ngoài thiết kế. D không dùng y, Delta, next image, current gradient norm hoặc một EMA vừa cập nhật. Không có optimizer accumulator mới cho fast state.

Actual fast update detach toàn state sau record để không giữ graph xuyên stream. Tính current partial derivative bằng functional fast parameters; không đưa fast template registered weights vào slow optimizer.

### 6.1. Hai kết quả lý thuyết và phạm vi

Gọi F_t là toàn bộ causal information sau resolve label cũ tại decision t: past/current normals, banks/anchors, theta/phi, RNG, context, c/gamma/a/d; **chưa chứa next normal**. Với conditional second moments hữu hạn, trong lớp predictor không hạn chế:

`p*(F_t)=E[y_t|F_t]=E[Delta_t|F_t]/c_t`.

Vì c_t>0, sign optimum khớp expected combined advantage. Tại parameters/context được giữ cố định khi lấy expectation:

`E[phi_next-phi|F_t]=-(.05/d_t) partial_phi E[ell_t|F_t]`.

Đây là **conditional surrogate-gradient alignment**, không policy gradient, không gradient của long-horizon bank risk, không convergence theorem. Across contexts, 1/d và 1/c² vẫn reweight learning; restricted capacity/shared parameters và compressed context không kế thừa optimum tự động. Outer Adam/clipping không có theorem alignment này.

Tại **w=b=0**, gamma/c detached, không slow commit: gradient CMS_fast bằng 0, h giữ nguyên; chỉ augmented readout cập nhật `v_next=-.05(p-y)a/(a^Ta)`. Đánh giá lại **cùng context** cho:

`p_after-y=(1-.05)(p_before-y)=.95(p_before-y)`.

Đây là 5% sửa sai prediction của **bước zero-head đầu tiên** trong exact arithmetic, không 5% risk/AD gain. Khi head khác 0, CMS Jacobian tham gia; d không là full `||grad_phi p||²`. Không có contraction hoặc fast-step norm bound chung. Floor c>=1 không bound target; scalar loss curvature 1 không bound full parameter Hessian.

## 7. Slow meta-learning và các clocks

Image clock tăng sau committed image event. Completed-label clock chỉ tăng khi một full-bank non-noop pending có next normal thật. Theta giữ nguyên suốt block tám labels cho đến slow commit.

Trước fast update label đầu block, lưu detached phi_start. Actual phi học **cả tám** labels theo §6. Tại label 8, sau actual fast update, slow objective clone detached phi_start; unroll đúng bốn chronological support records với cùng pre-commit theta. Recompute projected-context encoder, h, d cho từng replay state. `g=autograd(ell,phi,create_graph=True)`; **giữ derivative của cả g và d trong quotient g/d**. Không thay d bằng saved detached factual d.

Query records 5–8 đều tính loss dưới **cùng phi_after_support4**, không tiếp tục fast-adapt query trong meta objective. Slow loss là arithmetic mean bốn plain half-squared losses, không weight normalization. Exact derivative gồm term `partial(g/d)=(partial g)/d-g(partial d)/d²`; không differentiate hard bank trajectories, detached labels/coefficients hoặc history trước phi_start.

FIRST_ORDER giữ cùng inner forward nhưng detach **toàn quotient g/d**; chỉ detach g mà giữ derivative d là một control khác, bị cấm gắn cùng ID. Exact/FO cùng starting state+records phải có cùng forward values; khác backward là có chủ đích.

Slow Adam LR 1e-4, betas (.9,.999), eps 1e-8, weight_decay 0; clip global slow gradient norm 1 trước Adam. Record gradient norm và actual parameter-step norm riêng. Sau slow commit giữ **actual phi_after_all8**, không meta clone/phi4, không reset/rebase. Next decision dùng new theta với retained phi8: representation drift là limitation công khai. Partial block 1–7 giữ qua task boundary/pause/end, không flush để tạo thêm slow update.

## 8. Admission, initialization và ablation semantics

Khi bank chưa đầy, chạy full original CADIC image update, kể cả remainder. Khi bank full và proposal không đổi raw rows, commit no-op bookkeeping nhưng không label/learning giả. Với full-bank changed proposal, **learned modes** trong 32 completed labels đầu vẫn force PASS và học bình thường; sau đó accept iff p>=0, zero tie accept. CADIC luôn nhận proposal; GEOMETRIC dùng sign gamma và **không có warmup override**. Không drift warmup hoặc category reset.

Zero residual có cùng **sign GEOMETRIC** ngoài forced override, hoặc khi gamma>=0 trong warmup. Không claim all-stream parity với GEOMETRIC, càng không claim GEOMETRIC≡CADIC. Forced-PASS test riêng mới đòi full bank/counters/Eq9 parity CADIC khi admission luôn nhận.

Initial visual content/key/value truncated normal std .01 trong [-.02,.02], eta/alpha initial maps 0; wrapper/geometry/CMS affine Xavier, biases 0, LN scale/bias 1/0. Row retention .9; fusion .25 mỗi direction theo inherited operator. Fast CMS khởi tạo như v5 nhưng readout zero, bỏ slow head bias +.05 / weights ×.01. Init dùng private seed từng job; Rademacher seed riêng 20261017. Không học initialization từ v4/v5 hoặc prior pilot.

`NO_FAST_ADAPTATION` với zero readout làm p=gamma mãi; theta không có functional gradient. Nó dùng geometry admission **trên trajectory riêng ngoài learned warmup**, không là independent trainable slow baseline. Forced-PASS có thể làm bank khác GEOMETRIC từ warmup, nên không gọi hai full policies tương đương. Không fail control này vì missing slow gradients hoặc mượn nó chứng minh CMS coupling. `FAST_ONLY`, `NO_SELF_MODIFICATION`, `NO_ANCHOR_OBJECTIVE` là controls được định nghĩa trong study/code hiện hành nhưng chưa có completed full benchmark ở bảng kết quả; không lén chuyển head sang slow để cứu ablation.

## 9. Event, numerical và checkpoint contracts

Event: validate identity/raw input → resolve old pending+learning → pre-write R actual bank → fork proposer → current/anchor context+c/gamma+p/d → admission/commit chosen bank+alternative → update reservoir → trace/cursor. D của actual credited record có cùng giá trị với decision state vì không learning chen giữa decision và next resolve; lưu decision_d để audit, nhưng exact replay phải recompute d.

Model/features/context/fast parameters FP32; scalar r-values sau original FP32 risk, c/gamma/y, log, d accumulation, final p/loss và step arithmetic FP64. Neural log feature cast FP32 như §6; cast đó cũng đi vào a/d. **Promote cả w và u đã cast FP32 lên FP64 trước residual dot/reduction**, không dot FP32 rồi mới cộng prior FP64. D dùng chính u ấy promoted FP64 và bias coordinate1. Compute next parameter trong FP64 rồi cast FP32 differentiably. Exact meta là derivative theo declared autograd/derivative-bridge semantics, gồm differentiable casts; không phải đạo hàm của phép machine-rounding rời rạc. TF32/autocast off, highest FP32, deterministic algorithms/CUBLAS policy theo canonical T4 lock. Reduced FP64 fixtures chỉ kiểm đạo hàm; không thay canonical FP32 acceptance.

Bất kỳ nonfinite scalar/tensor/gradient/norm/next parameter/Adam state hoặc c<1/d<1/schema/chronology sai: **rollback toàn image event, export failure, dừng trial**. Không clip target/prediction/fast increment, drop record, retry rate, fallback GEOMETRIC hoặc candidate khác để hoàn tất run. No general bounded-fast-step claim. Outer clip 1 là rule riêng đã công khai, không được gọi như guarantee risk.

Schema mới `hope_support_session_v5_1`/`hope_support_learner_v5_1`. Lưu phi gồm readout, theta/Adam, counters, pending banks/hashes/risks/c/gamma/decision_d/context, anchors + private RNG, partial records + phi_start, IDs và configs/source/runtime/protocol hashes. Không lưu detached learned h để thay recomputation. Reject v4/v5 checkpoints kể cả shapes trùng.

Load trên temporary owner: validate exact keys/shapes/dtypes/finiteness, integer clocks, unique chronological IDs, c>=1/d>=1, finite **unbounded y/p**, pending cursor=cursor−1, replay len=labels%8 và phi_start iff replay nonempty; slow count theo mode; anchor 32×8×768 và RNG hợp lệ; banks geometry/count/hash. Sau mọi validation mới commit. Derived cache invalidate/rebuild; runtime tensor_version không là portable identity. Observe rollback gồm cả optimizer/anchors/RNG/episode/pending/counters. Score read-only scientific state; diagnostic allocation counters tách riêng.

Atomic checkpoint mỗi 8 images và khi pause/end; hard crash replay tối đa 7 events và existing traces phải exact-match. Pause không finalize; terminal finalize censor pending một lần, không slow flush. Failure cố export nhưng hard platform kill vẫn có thể chặn export.

## 10. Pilot, controls, acceptance và claim

Một fresh normal-development trial: discarded preflight 24 real normals trên T4; rồi CADIC/GEOMETRIC/FIRST_ORDER/NESTED, seed 0, mỗi job 256 canonical images = 179 Bottle + 77 Cable. Same raw producer/cache/data/K/numeric route; learned pair matched initialization. Không grid, extra seven-control search hoặc tăng budget sau FAIL. Soft session budget chỉ chia execution; pause/resume cùng identity không là trial mới. Đây là protocol pilot lịch sử đã được review, không phải yêu cầu tự chạy lại trên mỗi refactor. Preflight máy thuê hiện hành nằm trong operations; không nhập learned pilot state vào benchmark.

Technical gates trước pilot: source/runtime/data locks; partial-vs-divided-loss fixture; zero-head .95 contraction; full 4+4 meta finite differences gồm denominator derivative; exact/FO cùng forward; real ViT shapes/preprocessing; forced-PASS full-bank/Eq9 parity; malformed state rejection, full rollback/resume, read-only score và resource profile. Cùng-context toy có encoder meta-gradient 0 do đúng cancellation; không đòi mọi group khác 0 trên mọi sample. Varied-context fixture mới kiểm nondegenerate dependency.

Giữ mechanism thresholds: NESTED ≥64 completed labels, ≥8 slow updates; reject fraction **postwarm eligible changed proposals** trong [.01,.99]; final actual bank khác CADIC, max abs patch-score difference trên reused-normal probe >1e-6; fast/slow/meta path finite/active theo mode. No-effect phải báo riêng, không biến parity baseline thành NL contribution.

Giữ safety gate: mean **128 pre-write actual-bank image risks R cuối** <=1.10×CADIC. Không cộng anchor, không chỉ accepted images, không đổi window/threshold vì transition. Toàn pilot 256 images có 255 finite risks vì bank đầu rỗng; terminal image có risk dù cuối credit bị censor. Expected label counts có thể khác v5 do no-op trajectories, không ép số 242. Minimum gates vẫn khóa như trên.

Trong cùng pilot log current/next/anchor decomposition; c/gamma/y/pred/residual/d; counts/action/warmup; replaced operations và changed rows riêng; gradient và actual fast/slow step norms; bank hashes; memory/timing scope. Full functional response sample ở completed labels 1,33,65,97,129,161,193,225: cùng stored context/prior, fixed theta, actual fast state trước/sau đúng một update, không extra learning. Ghi `response_gain=(abs(p_before-y)-abs(p_after-y))/abs(p_before-y)`; nếu mẫu số<=1e-12 thì null và small_error=true, không NaN hoặc loại event khỏi risk. Lưu một actual meta episode labels 33–40 theo clock, không chọn sau khi thấy đẹp. Không nhân đôi mọi unroll hoặc mở GPU diagnostics trial riêng.

FAIL hoặc no-effect: giữ evidence và stop review, không đổi method trong cùng run. PASS chỉ đủ review đề xuất benchmark. Exact/FO cùng actions không tự là failure kỹ thuật nhưng không chứng minh higher-order benefit. Normal-risk PASS không là AD accuracy validation.

Full benchmark được định nghĩa trong study (paired seed0 đã có evidence, phần còn lại chưa hoàn tất): matched CADIC/GEOMETRIC/FO/NESTED, ablations hợp lệ, seeds 0/1/2, final metrics + task matrix/F/BWT và uncertainty. Primary keys `macro_final_i_auroc`, `macro_final_p_ap`; P-AP=sklearn average_precision_score, historical p_aupr alias phải map rõ. Paired whole-image resampling trong category và image-label strata, cùng draws cho methods, 2000 replicates, percentile 1.25/98.75, average tasks/seeds đã khóa. Đây là paired finite-seed evaluation uncertainty, không thay population-seed CI. NI margin 2 pp cả hai endpoints; superiority chỉ khi cả lower bounds >0; NL-specific claim cần controls tương ứng. Executable evaluator hiện được source-bound; xem benchmark.md. Không đổi statistical rule sau khi nhìn kết quả.

Fairness gồm equal K10k và measured memory/compute categories: main/alternative banks, anchors, contexts, phi_start, fast/slow/readout, Adam, caches, shared backbone, Python/allocator/transactional GPU+CPU/IO peaks. K24576 chỉ là conservative planned payload comparator, chưa exact total-memory match. Cache warming pilot không là fair latency benchmark.

VisA confirmation chỉ sau method/code/scoring/data/transfer/metric freeze, fresh causal state, cùng constants, không retune từ test. QA raw/resized masks, image/mask hashes và undefined-metric policy phải khóa cho mọi method trước evaluation. Giữ image label nếu tiny defect biến mất sau resize; không drop/relabel để cải thiện AP. Thesis báo historical negatives, assumptions, trial/exposure ledger và limits. Không có guarantee ngang/hơn CADIC hoặc “tận dụng tối đa” mọi khả năng Nested Learning.


## 11. Nguồn thực thi và thay đổi chưa triển khai

Các phương trình trên mô tả v5.1. Đường thực thi: [learner](../src/hope_cad/models/learner.py), [session](../src/hope_cad/models/session.py), [geometry](../src/hope_cad/models/geometry.py), [runtime manifest](../src/hope_cad/runtime/execution.py), cùng [method contract](../src/hope_cad/resources/method.json). Fast/slow counters phản ánh actual successful optimizer calls; rollback giữ nguyên counters khi fail; scoring chỉ đọc bank.

[Pipeline spec](pipeline_alignment_spec.md) quy định successor full-train3.629; [H32](research/candidate.md) thay slow-query context và thêm history state. Cả hai chưa được runtime hiện tại triển khai, không thay phương pháp hoặc số liệu đang mô tả ở đây. [Plan](research/implementation_plan.md) ghi acceptance trước khi chạy candidate.
