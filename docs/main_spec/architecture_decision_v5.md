# Nested Learning for CAD — Quyết định kiến trúc v5

Ngày 17/10/2026. Design ID `hope_cad_v5_causal_support_controller`. **Frozen for implementation; chưa experimentally validated.** Người dùng giao Stage2 gồm chọn revision/spec/implementation; lựa chọn dưới đây là quyết định triển khai trong phạm vi đó, không phải kết quả benchmark. Contract máy đọc: `revision_contract_v5.json`. V4 và mọi evidence cũ giữ nguyên.

## 1. Research checkpoint, evidence và lựa chọn

| Nguồn | Điều kế thừa | Điều không suy ra |
|---|---|---|
|CADIC §III, Eq1–9|Raw patch support, incremental diversity proposer, Euclidean NN và image weighting|Không phải mọi gate hoặc coreset khác đều tốt hơn|
|Nested Learning §3,7,8|Nhiều quá trình học, context/clock riêng và cross-level gradients|Hai timer đơn thuần không chứng minh continual learning tốt|
|VisionHOPE §2, Appendix A|Self-referential visual memories, four scans, bounded chunk recurrence|Paper không cung cấp objective CAD hoặc utility controller này|
|Recovery seed 0|Matched CADIC 88,3571/46,0137% so M 66,8010/17,0163%|Không sửa lại frozen v4 hoặc gọi historical baseline hoàn toàn paper-identical|
|Normal causal diagnostic|Decoder/joint có normal fitting headroom nhưng joint past loss tăng 8–14%; cap 400 còn headroom|Normal loss gain không là AUROC gain, latent oracle không là lower-bound certificate|

Paper nguồn: [CADIC](https://arxiv.org/html/2511.08634v1), [Nested Learning](https://arxiv.org/html/2512.24695v1), [VisionHOPE](https://arxiv.org/html/2609.33325v1); [official VisionHOPE](https://github.com/PSRben/VisionHOPE). Bản local nằm docs/papers; pinned operator common-regime đã được kiểm ở lịch sử upstream 98cfb5d6004472a0ba3d212d3cf884393e39c134. V5 tái dùng operator PyTorch đã kiểm của repo, không tự nâng dependency theo upstream hiện tại. Đây là **HOPE-derived CAD adaptation**, không tái lập toàn bộ HOPE language model/VisionHOPE backbone.

Chọn hướng B: **học quản lý normal-support memory trong raw ViT geometry**. Learner dự đoán lợi ích tương lai của một proposal cập nhật coreset, thay vì reconstruct 768 chiều từ mask. Lý do: objective mới gắn trực tiếp với approximation của observed normal support; loại frozen reconstruction decoder khỏi score; retention được đưa vào utility label; vẫn giữ self-modification và nested fast/slow credit.

Loại các alternatives trong revision này: (A) chỉ unfreeze decoder—interference đã quan sát, predictability/normality mismatch còn nguyên; (C) exact CADIC + learned search—hợp lệ cho efficiency nhưng chưa có geometric certificate đủ mạnh ở 768D và không nhắm accuracy; fusion rawNN/reconstruction—tăng degrees of freedom và không có AUROC floor. Không thêm NSP2/CBP/TITANS/generative replay. Không sửa vì một category test xấu.

Điểm chưa giải quyết bằng lý thuyết: true anomaly separation chưa biết; future utility nonstationary; finite anchors và finite proposal set có bias; gate có thể không hữu ích. Đây là hypotheses và failure conditions, không được che bằng một narrative chắc thắng.

## 2. Bài toán, hình học và score

Causal stream normal images x_t; inference không có task routing. Frozen F là ViT-B/8 ImageNet21k, block 8 zero-based/layer 9 one-based,224×224 direct PIL bilinear/ImageNet mean/std, bỏ CLS, **không thêm LN/L2 trên raw patches**. F(x)∈R[1,784,768]. SHA checkpoint và distance tiles 256 / b9 đúng matched CADIC. V4 learned weights không được reuse.

Bank C_t chứa tối đa K=10000 **raw observed patch vectors**, không synthetic prototypes. Score patch d(q,C)=min_c||q-c||2. Image score giữ **đầy đủ Eq9**: lấy patch q* có NN distance s* lớn nhất và c* là matched bank row; N9(c*) là9 bank rows gần c*, rồi `s_img=s*·[1−exp(s*)/Σ_{c∈N9(c*)}exp(||q*−c||)]`, thực hiện logsumexp ổn định. Map28×28 bilinear lên224; không gaussian smoothing mới. Inference chỉ đọc bank, không học bằng test hoặc teacher của test. Controller không cần chạy trên test: ảnh hưởng của nó nằm ở **actual bank** đã được chọn, bank đó quyết định score.

Dùng cùng 3599 online train identities của matched MVTec prefix protocol (3629 normals trừ15validation/15probe của bottle). Các quyền giữ lại để so sánh; v5 **không offline calibration hoặc search trên prefix**. Nó học causal từ normal observations đã đến, kể cả trên VisA. Không label/mask/taskID vào controller.

## 3. Support objective và các bảo đảm có điều kiện

Cho finite observed normal set S và C⊆S, epsilon=max_s d(s,C). Triangle inequality cho mọi q:

`d(q,S) ≤ d(q,C) ≤ d(q,S)+epsilon`.

Proof: bound trái do C⊆S. Với s* gần q nhất, chọn c* gần s* nhất; ||q−c*||≤||q−s*||+epsilon. Nếu normal query cáchS≤r0 và anomalies cáchS≥r1>r0+epsilon thì patch distance có separation. Assumptions này không được chứng minh cho dataset; không tự chuyển thành theorem cho Eq9 hoặc AUROC.

Một ảnh/reference patch setX có risk `R(X,C)=.5 mean_j d(x_j,C)^2+.5 max_j d(x_j,C)^2`. Tail term được chốt trước run; không chọn theo anomaly masks. Anchor risk là trung bình R trên 32 historical image records, mỗi record 8 sampled patches. Sampling 8 không phải max 784 thật; limitation phải báo.

Tại proposal t: C− là bank trướcimage, C+ là **đúng một CADIC image-update proposal** trên clone, cùng initializer/tie/kernel/cache semantics. Hai banks fixed đến normal image kế tiếp. A_t là anchor reservoir trước x_t. Cache a−=R(A_t,C−),a+=R(A_t,C+) lúc proposal; không đưa x_t vào A_t rồi tự đánh giá nó.

Khi x_(t+1) thực sự đến, trước write của nó:

`J−=R(F(x_(t+1)),C−)+lambda·a−`, `J+=R(F(x_(t+1)),C+)+lambda·a+`, lambda=1.

`y_t=(J−−J+)/(J−+J++1e−8)` là detached supervised utility, nằm[-1,1] vì J±≥0. Nó đo bounded one-step fixed-bank realized advantage; không unbiased value của toàn counterfactual policy trajectory và không AD reward. Cả accepted/rejected proposals đều nhận label từ hai banks đã khóa. Candidate x_t không là future reward. Terminal pending proposal bị **censor** vì chưa có next image; không tự tạo label hoặc flush tại task boundary.

Nếu utility predictor có uniform error≤e với true expected declared utility trên finite action set, argmax regret≤2e: Δ(a*)≤Δhat(a*)+e≤Δhat(ahat)+e≤Δ(ahat)+2e. Training MSE không chứng minh uniform error hoặc anomaly regret. Gate decision không có AUROC guarantee; fallback chỉ giữ được baseline khi bank và score **thực sự identical**.

## 4. Context, HOPE mapping và topology duy nhất

Để không giữ 8 raw 768D images cho meta replay, context dùng fixed Rademacher projection P∈R[768,32], seed 20261017, entries±1/sqrt32, sinh bằng private CPU generator. Với U=F(x)P, X=U/sqrt(mean(U²)+1e−6) theo toàn image. P không học, không dùng dữ liệu tương lai. Đây chỉ là context compression, **không đổi bank/score geometry**; không nhận32D là JL guarantee cho mọi patch.

X[1,784,32] qua tested visual block cấu hình input_dim=32,dim=32,memory_dim=32,head_dim=8,grid28×28. Có 4 heads/direction và 4 scans; canonical v5 khác canonical v4. Row retention rho=.9; initial alpha=.9^(1/28). Tái dùng đầy đủ 5 self-modifying maps, stable soft injection/spectral caps và FP32 recurrence. Trong mỗi chunk boundary b:

`k=Mb_k z /sqrt(||Mb_k z||²+1e−6)`, `v=Mb_v z`, `eta=.025 softplus(Mb_eta z)`, `alpha=sigmoid(Mb_alpha z+logit(.9^(1/C)))`.

`delta=k−v`, `r=sqrt(||delta||²+1e−12)+1e−6`, `eta_inj=(1−alpha)(1−1e−3)/r`, `bar_eta=eta_inj·[−expm1(−eta/eta_safe)]`, với `eta_safe=eta_inj` nếu dương và bằng 1 khi injection bằng 0. Spectral cap liên tục là `2alpha(1−1e−6)/max(||k||²,1e−20)`; forward FP32 dùng thứ tự rsqrt rồi bình phương như code và làm tròn nextafter về 0. `eta_exec=min(bar_eta,spectral_cap_rounded)`. Existing ULP straight-through và FP64 derivative bridge được giữ nguyên từ v4, bao gồm đạo hàm bậc cao hữu hạn; không được bỏ các margins 1e−3/1e−6 khi tái lập. `A=alpha I−eta_exec kkT`, `B=−eta_exec delta kT`, `M_t=M_(t−1)A+M_b B`. Readout dùng **boundary content map**, không post-token state. Working maps reset mỗi image/context; learned initials thuộc slow theta. Spatial inverse/fusion/residual/FFN giữ operator đã kiểm; không gọi đây full upstream VisionHOPE reproduction.

Meanpool visual output thành [1,1,32]. Cộng affine projection của 16 geometry statistics đã biết trước quyết định: current NN mean/p95/max/min/std chia mean raw norm; candidate NN mean/max cùng scale; relative current risk gain; relative anchor risk gain; fill ratio; unique changed row fraction; replacement count / incoming patches; log1p(stream cursor)/log1p(1e6); anchor fill ratio; incoming/bank mean raw norm ratio; relative combined immediate current+anchor gain. Các stats finite và không dùng future reward.

Token 32 này đi qua **CMS_fast(phi)** rồi **CMS_slow(theta_s)**, mỗi module residual `h+Linear32→64→32(GELU(LN(h)))`, sau đó scalar affine32→1 và tanh. Utility head bias=.05, Xavier weights×.01. Không reconstruct teacher features, không masked RGB. Self-referential image memory +persistent CMS fast/slow +cross-level optimization là HOPE-derived mapping, không gắn tên HOPE vào scalar gate không học.

## 5. Objectives, clocks và exact nested gradient

**Ownership.** Phi là current fast CMS weights, bao gồm LayerNorm, persistent xuyên images/tasks. Theta gồm visual encoder và learned initial maps, geometry projection, slow CMS và utility head. Fast template chỉ cung cấp initialization, không thuộc slow optimizer. Backbone, fixed projection và raw bank không nhận gradient. Scalar loss là `ell = .5·(predicted_utility − y)^2` với prediction qua tanh.

Sau mỗi completed delayed label r, tính `g = partial_phi ell(f_theta,phi(r),y)` rồi `phi_next = phi − .05·g/sqrt(1+Σ||g||²)`. Smooth transform có norm nhỏ hơn 1; actual fast step detach qua các records, không giữ graph dài vô hạn. Nó không dùng task label hoặc future reward chưa đến.

Mỗi block tám completed labels lưu phi_start trước fast update đầu tiên. Slow objective clone phi_start detached, unroll đúng bốn smooth fast steps trên bốn chronological support labels với `create_graph=True`, sau đó lấy mean loss trên bốn later query labels với cùng adapted phi. Recompute encoder từ fixed projected tokens để gradient đi qua inner gradients tới theta. Replay không lưu detached learned embeddings. Slow Adam LR=1e−4, betas=(.9,.999), epsilon=1e−8, weight_decay=0; global gradient clip norm=1. Hard bank decisions và utility labels detached.

Actual phi đã học cả tám records được giữ sau slow commit; không reset sang meta clone. Slow commit có thể gây representation drift — đây là failure mode phải đo, không có hidden rebase. Partial block giữ qua task boundary/pause; không dùng task ID làm clock. Cuối stream không flush partial block. Receipts ghi actual fast/slow counts, support/query identities, losses và gradient groups.

Admission chỉ áp dụng khi bank đầy. Trong 32 completed labels đầu, warmup ép PASS; sau warmup accept iff predicted utility ≥0. Proposal không đổi raw bank không tạo utility label giả. Khi bank chưa đầy, chạy original CADIC full-image update, bao gồm remainder của image vừa làm đầy bank; không học reward giữa banks khác kích thước. Anchor reservoir được cập nhật cuối event. Một alternative full bank được giữ cho delayed reward và tính vào memory budget.

**Initialization và topology.** Visual input/output/memory dim đều 32, head dim 8, bốn directions, chunk theo hàng 28 tokens. Five-map initialization và wrapper giữ implementation đã kiểm: content/key/value truncated normal std=.01 trong [−.02,.02], eta/alpha initial maps bằng 0; affine/convolution dùng Xavier, bias 0 và LayerNorm scale/bias=1/0. Wrapper có depthwise 3×3 và FFN 32→128→32. Geometry affine bias 0, Xavier weights. CMS dùng residual pre-LayerNorm MLP 32→64→32 của `CMSLevel`. Utility head Xavier weights nhân .01, bias .05. Private RNG theo seed, fixed projection seed 20261017; không học initialization từ v4 hoặc từ pilot khác. Giá trị này là thiết kế đã khóa trước real pilot, chưa được chứng minh tối ưu.

## 6. Causal event, checkpoint và inference contracts

Một event xử lý đúng thứ tự:

1. Xác minh stable identity, cursor, raw feature shape và finiteness.
2. Resolve pending bằng normal image hiện tại; cập nhật fast learner và slow learner nếu đủ labels.
3. Đo prequential normal risk trước write hiện tại.
4. Bootstrap hoặc fork đúng CADIC image proposal.
5. Tính context đã biết, dự đoán utility và admission.
6. Commit chosen bank; giữ alternative bank cho reward ở image kế tiếp.
7. Đưa image vào anchor reservoir, ghi event và tăng cursor.

Ngoài pilot, cùng rule này áp dụng cho toàn stream; task boundary chỉ bookkeeping. Inference không resolve pending bằng test, không chạy controller learning và không thay bank/weights/RNG. `finalize_stream` chỉ censor pending khi stream thật sự kết thúc; không dùng ở pause.

Observe là transaction: snapshot learner/optimizer/episode/anchors/private RNG/pending và bank owner; proposer chỉ sửa clone. Failure restore state trước event và báo lỗi, không tìm một candidate khác để cứu kết quả. Checkpoint atomic khóa source, protocol, runtime, data và config; lưu model, actual fast weights, Adam, replay episode/phi_start, banks, counters, anchor reservoir/RNG, pending context/cached risks và event identities. Projection được tái dựng đúng từ seed/shape đã khóa. Derived pair cache không serialize; load phải invalidate/rebuild đúng. Uninterrupted/resume phải khớp bank, state và decisions, không chỉ metrics xấp xỉ.

Schema mới tách khỏi v4; không dùng selected v4 weights hoặc old evidence làm learned state v5. Checkpoint mỗi tám images và khi pause/end; sau hard crash có thể replay tối đa bảy events từ checkpoint trước. Event trace đã có phải khớp khi replay. Failure/pause cố xuất evidence; hard platform kill vẫn có thể chặn export.

## 7. Controls, fairness và acceptance trước pilot

**Normal pilot:** một trial, seed 0, first 256 online images theo canonical order: 179 bottle và 77 cable. Preflight riêng trên 24 real normals bằng discarded clone; không reuse learned state. Bốn jobs cùng data/K/geometry:

| Job | Cơ chế |
|---|---|
| CADIC | Luôn nhận exact proposal; không controller learning |
| GEOMETRIC | Sign của combined immediate current+anchor risk advantage đã biết |
| FIRST_ORDER | Cùng bốn support/bốn query forward; detach inner-gradient dependence |
| NESTED | Exact nested meta-gradient qua bốn fast updates |

Private initialization/context RNG matched cho learned pair. Không grid hoặc hyperparameter search. Technical gates gồm source/data/runtime, finiteness, forced-PASS bank/full-Eq9 equivalence, finite-difference meta-gradient, ownership, atomic failure/resume, score read-only và resource profile.

Mechanism gates cho NESTED: ít nhất 64 completed labels, tám slow updates; reject fraction sau warmup trong [.01,.99]; actual bank và normal score khác CADIC; fast/slow/meta paths có gradient hữu hạn và hoạt động. Normal-risk gate: mean prequential risk của last 128 images không vượt 1,10 lần CADIC. Failure hoặc no-effect phải được giữ; không đổi threshold/budget rồi gọi cùng method. PASS chỉ đủ để review đề xuất benchmark, không tự launch anomaly evaluation. Pilot ngắn với một category transition không chứng minh long-horizon retention.

**Full benchmark dự kiến sau pilot:** NESTED/CADIC/GEOMETRIC/FIRST_ORDER và FAST_ONLY/NO_SELF_MODIFICATION/NO_ANCHOR_OBJECTIVE/NO_FAST_ADAPTATION, seeds 0/1/2. Core modes bổ sung đã có thể thực thi, nhưng orchestration, full matrices và uncertainty còn phải hoàn thiện trước claim. Mỗi seed/dataset bắt đầu fresh state.

Memory fairness cần cả equal-main-K10000 và total persistent bytes: two banks, 32×8 anchors, tám projected contexts, phi_start/fast/slow parameters, Adam, RNG/counters và pair cache. CADIC K24576 được đăng ký như conservative payload-budget control; chỉ được claim resource fairness sau đo đủ bytes. Current counter chưa tính mọi Python object, shared backbone, allocator/IO peak; phải công khai giới hạn. Runtime phải tính proposer, NN, delayed credit, learning và IO. Pilot cache warmup khác nhau nên elapsed không là fair latency comparison; không hứa nhanh hơn CADIC chỉ vì cùng một ViT forward.

Primary endpoints: macro final I-AUROC và P-AP (`sklearn.average_precision`; historical label P-AUPR). Secondary: I-AP/P-AUROC, task matrix, forgetting/BWT, latency/memory. Ba seeds; paired whole-image/category-label-stratified bootstrap 2000 replicates, primary CI percentiles 1.25/98.75. Noninferiority margin 2 pp trên cả hai primary endpoints; superiority chỉ khi cả hai lower bounds >0. NL-specific claim cần controls hỗ trợ. Duplicate deterministic baseline seeds không là independent replicates.

MVTec đã development-exposed; mọi revision gain trên nó là development evidence. Trước VisA khóa source/spec, metrics, data/preprocessing, transfer/init, rates/windows/anchors và trial budget. VisA dùng fresh causal state, không offline learning từ future categories hoặc tune theo test. Không có guarantee CADIC parity: nếu gate không hữu ích hoặc NL kém, phải báo negative result/tradeoff.
