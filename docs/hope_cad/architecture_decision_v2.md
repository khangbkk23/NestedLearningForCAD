# Architecture decision v2 — Nested Learning for CAD

Ngày chốt: 2026-10-09. Design ID: **`hope_cad_v2_masked_teacher_cms`**.

**Trạng thái quyết định: FROZEN FOR IMPLEMENTATION. W3 visual operator: IMPLEMENTED / TESTED local; full HOPE-CAD: chưa hoàn chỉnh, chưa experimentally validated.** Người dùng đã giao quyền chốt các quyết định mục C sau khi rà soát chuyên môn. Freeze xác định phương pháp sẽ triển khai và kiểm chứng; không xác nhận trước hiệu quả thực nghiệm. [Báo cáo W3](w3_implementation.md) ghi file/symbol, source hashes, 23 tests và phạm vi kiểm chứng; W4/W5 vẫn proposed.

Tài liệu này là nguồn chuẩn cho kiến trúc v2, cùng [kế hoạch A/B](planning_CAD_v2.md). `architecture_decision_v1.md` và `paper_to_code_spec_v1.md` tiếp tục mô tả implementation v1 tại `dikhang_space@a89fd48`. V2 không hồi tố thay đổi semantics hoặc tên của các run v1/P0/P1/P2/OL-04.

## 1. Ba quyết định đã chốt và cơ sở

1. **Visual self-modifying memory reset theo mỗi ảnh/view; CMS persistent xuyên normal images và task boundaries.** Như vậy visual context học trong ảnh, còn continual knowledge nằm trong CMS. Outer learning là thang chậm nhất, kết thúc sau calibration prefix.
2. **Normal-only masked teacher prediction.** Frozen ViT tạo target từ clean RGB; predictor chỉ nhận masked RGB qua cùng frozen ViT. Anomaly score dùng prediction được giải mã từ **output cuối CMS** so với target cố định.
3. **Outer calibration chỉ dùng prefix normal của task đầu.** Không meta-train trên tập hợp tất cả category. Sau calibration đóng băng slow parameters; chỉ CMS tiếp tục học trên normal stream. Không cập nhật persistent state bằng official test.

Đây là một **HOPE-CAD instantiation**: self-modifying visual memory → sequential CMS, có nhiều clock và objective được định nghĩa. Không phải bản tái lập nguyên xi HOPE language model hay toàn bộ backbone VisionHOPE. Không dùng checkpoint VisionHOPE/ImageNet-1k để thay backbone ViT-B/8 của benchmark chung.

Về học thuật, feature discrepancy với frozen teacher là một hướng AD có tiền lệ, ví dụ [Reverse Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html). Nguy cơ mạng tái tạo học đường tắt được phân tích trong [UniAD](https://papers.nips.cc/paper/2022/hash/1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html). Hai nguồn này hỗ trợ lựa chọn objective và nhu cầu chống sao chép đầu vào; chúng không chứng minh masked HOPE-CAD sẽ phát hiện anomaly hoặc chống quên. Việc kết hợp visual self-modification, CMS và objective dưới đây là giả thuyết cần ablation của đồ án.

## 2. Forward path và tensor contract

Với ảnh đã resize `x ∈ R[1,3,224,224]`, extractor `F` là frozen ViT-B/8 ImageNet-21k, block index 8/layer 9 one-based, bỏ CLS, không final LayerNorm/L2 thêm vào raw extractor. Output có shape `[1,784,768]` theo row-major 28×28. Cùng checkpoint SHA và preprocessing với CADIC.

```text
clean x ── F ── normalize ── stop-gradient u ──────────────────┐
                                                            │ target/score only
masked x_k ── F ── P_768→256 ── visual block ── CMS1 ── CMS2 ── D ── û
                                                            │
                      final anomaly map ← squared discrepancy(û,u)
```

- Teacher: `u_j = F(x)_j / max(||F(x)_j||₂, 1e-8)`, detached, shape `[1,784,768]`.
- Input projection P: learned affine `768→256`.
- Một visual block có output width `D=256`, internal width `D_m=128`, **head dimension `d=16`, 8 heads/direction**, 4 directions. Không nhầm head dimension 16 với 16 heads.
- Hai CMS residual MLP, mỗi level `256→512→256`, pre-LayerNorm eps `1e-6`, GELU, biases; current parameters gồm cả affine LayerNorm.
- Decoder D: shared across CMS levels, `256→768→768`, GELU giữa hai affine layers; không normalize prediction. Decoder được học trong calibration và frozen trong normal deployment stream.
- Teacher normalization là một phần head v2; không thay raw features hoặc score của CADIC.

Không clean feature, label anomaly, mask ground truth hoặc task ID nào được đưa vào predictor. Target clean chỉ được dùng để tính loss khi train normal và discrepancy khi score. Decoder đọc final CMS output; không cộng thêm CADIC score, raw-feature NN score hoặc intermediate-memory residual trong main method.

Decoder nonlinear được chọn để tránh ép mọi teacher prediction vào một affine subspace 256 chiều cố định của linear `256→768`. Nó vẫn có thể generalize kém sau calibration chỉ trên task đầu; đó là rủi ro phải đo bằng acquisition/retention controls, không được giải quyết bằng cách nhìn future tasks.

## 3. Visual self-modifying operator

### 3.1 Spatial block: phần kế thừa và mapping của đồ án

Với X=P(F(masked x)), dùng dạng cấu trúc Eq.27 của VisionHOPE:

\[
Z=\mathrm{DW}_{3\times3}(P_{\rm in}(\mathrm{LN}(X))),\quad
X'=X+P_{\rm out}(\mathcal V(Z)),\quad
h_0=X'+\mathrm{FFN}(\mathrm{LN}(X')).
\]

`P_in:256→128`, `P_out:128→256`; depthwise convolution kernel 3, padding 1, bias; FFN `256→1024→256`, GELU. LayerNorm theo channel, eps `1e-6`; dropout/stochastic-depth bằng 0. Không thêm CPE, per-route input skip hoặc positional branch khác vào bản reference v2. Đây là mapping CAD tối giản từ structural block, **không tuyên bố trùng toàn bộ implementation backbone ở Appendix B.5**. Nếu dùng upstream block có các branch bổ sung thì phải đổi config/method version, không thay ngầm.

`V` là 4 scans row-major forward/reverse và column-major forward/reverse. Reverse đảo toàn sequence. Row chunk dài W=28; column chunk dài H=28. Inverse permutation trả mỗi output về đúng vị trí ban đầu; fusion là tổng theo direction với learned channel scales `λ_r ∈ R[128]`, khởi tạo `1/4`.

Shared slow query projection `Wq ∈ R[128,128]` áp một lần lên Z, sau đó reorder/split heads và L2-normalize từng head với eps `1e-6`. K/V/control generation dùng head tokens của Z. Mỗi direction/head có initial states riêng, không share working memory giữa directions, images hoặc views.

### 3.2 State và phương trình cập nhật

Mỗi head giữ ba matrices `M^m,M^k,M^v ∈ R[16,16]` và hai row maps `M^η,M^α ∈ R[1,16]`. Initial matrices dùng truncated normal std `0.01`, bounds `[-0.02,0.02]` (±2 std); hai control maps bắt đầu bằng zero, sau đó được outer-learn. Other affine/conv layers dùng Xavier-uniform weights, zero bias; LayerNorm scale 1/bias 0. Initial CMS parameters cũng dùng quy tắc affine này, không zero cả residual branch làm mất gradient input ở initialization.

Với chunk bắt đầu ở boundary b, **mọi token trong chunk** tạo quantities từ `M_b`:

\[
k_t^{raw}=M_b^k z_t,\quad k_t=\frac{k_t^{raw}}{\sqrt{\|k_t^{raw}\|^2+10^{-6}}},\quad
v_t=M_b^v z_t,
\]
\[
\eta_t=0.025\,\mathrm{softplus}(M_b^\eta z_t),\qquad
\alpha_t=\sigma(M_b^\alpha z_t+\operatorname{logit}(0.9)).
\]

Với `δ_t=k_t-v_t`, `ε=1e-6`:

\[
r_t=\sqrt{\|\delta_t\|^2+\epsilon^2}+\epsilon,\quad
\eta_t^{inj}=(1-\alpha_t)/r_t,\quad
\bar\eta_t=\eta_t^{inj}\bigl(1-e^{-\eta_t/\eta_t^{inj}}\bigr),
\]
\[
\tilde\eta_t=\min\!\left(\bar\eta_t,\frac{2\alpha_t}{\|k_t\|^2}\right),\quad
A_t=\alpha_t I-\tilde\eta_t k_tk_t^T,\quad B_t=-\tilde\eta_t\delta_tk_t^T.
\]

Working state thực thi `M_t^□ = M_{t-1}^□ A_t + M_b^□ B_t`, tương đương shared gain `G_b=I; G_t=G_{t-1}A_t+B_t; M_t^□=M_b^□G_t`. Readout của token trong chunk là `y_t=M_b^m q_t`, **không** đọc từ state sau token hiện tại. Cuối chunk dùng state mới làm boundary cho chunk tiếp theo. Chunk một token là special case đúng; không thay chunk update bằng P1 image-average hoặc một phép `M@T` trung bình cả ảnh.

Trong FP32 áp safeguards ở Appendix B.3: nhân injection limit với `1-1e-3` trước soft cap; nhân spectral limit với `1-1e-6`, mẫu số key norm² lower-bound `1e-20`, round spectral limit về next representable value hướng zero. Dùng `-expm1(-η/η_inj)` để tính soft cap ổn định. Nếu retention saturate tới 1 do roundoff thì injection limit bằng 0, executed step bằng 0; nhánh này phải có xử lý finite rõ ràng, không chia 0. Các guards là một phần math contract, không phải global gradient clipping.

Local associative target được giữ cố định khi suy ra inner update; điều đó không có nghĩa detach mọi boundary state khỏi outer graph. Outer gradients đi qua recurrence, quantities và initial states. Với nextafter, backward biểu diễn hàm safeguarded real-arithmetic trước phép làm tròn ULP; phải ghi mapping này trong kernel và so với reference, không stop-gradient cả learning-rate/retention branch.

Theorem của VisionHOPE ràng buộc gain/memory norm dưới các giả định của operator; không ràng buộc full residual block, CMS, anomaly ranking hoặc forgetting. Key norm≤1 phải được giữ. Numerical tests kiểm tra gain theo chunk-boundary, không yêu cầu memory norm giảm đơn điệu so với token ngay trước trong chunk.

Nguồn phương trình: [VisionHOPE §§2.1–2.3, Appendix B.3–B.5, C.1](https://arxiv.org/html/2609.33325v1). Dùng toàn ảnh và frozen ViT có global receptive field: đây là thứ tự update có boundary rõ ràng, **không phải pixel-causal inference**.

## 4. CMS: objective, clock và gradient ownership

Với current weights trước normal event t:

\[
h_{l,t}=h_{l-1,t}+W_{2,l}\,\mathrm{GELU}(W_{1,l}\mathrm{LN}_l(h_{l-1,t})+b_{1,l})+b_{2,l},\qquad l=1,2.
\]

Một event dùng một masked view. Local objective áp trên masked positions Ω của view đó:

\[
\mathcal L_{l,t}=\frac1{2|\Omega|}\sum_{j\in\Omega}\|D_\theta(h_{l,t,j})-u_{t,j}\|_2^2,\qquad
g_{l,t}=\partial\mathcal L_{l,t}/\partial\phi_{l,t^-}.
\]

Local update ownership chỉ gồm weights của level l. Không dùng `L_2.backward()` để vô tình update level 1, decoder hay visual slow parameters trong deployment. Các inputs/outputs/losses của cả hai level được tạo từ **cùng pre-event snapshot** trước bất kỳ commit nào.

Chu kỳ `p=(1,8)`. Level l giữ accumulator A_l và pending count n_l. Mỗi event: `A_l += g_l; n_l += 1`. Nếu `t mod p_l == 0`, commit `φ_l ← φ_l - β_l A_l/n_l`, rồi zero A_l,n_l; nếu chưa due, giữ weights. Dùng plain SGD, không momentum/weight decay/gradient clipping cho CMS reference. Validate all proposals finite trước atomic commit. Không flush khi chuyển task, hết session hoặc evaluation; pending gradients luôn nằm trong checkpoint. Global event clock không reset giữa tasks.

Đây là schedule trung bình gradient theo ảnh, khác code v1 cộng gradient không chia count. Local prefix losses là CAD adaptation của objective-choice trong [HOPE/CMS Eq.70–71](https://arxiv.org/html/2512.24695v1); không claim mọi biến thể nested/head-wise của Eq.72–74 đã được dùng.

**Hai chế độ graph cùng một numerical update:**

- **Functional meta-training:** local partial derivative chỉ lấy với parameters của level hiện tại, `create_graph=True`; giữ computational dependencies qua visual inputs, decoder, state transitions và accumulators để outer differentiation đúng. “Không inner-update module khác” không đồng nghĩa detach inputs khỏi meta graph.
- **Deployment:** slow θ frozen; được detach inputs phục vụ local gradient và detach toàn bộ state/accumulators sau mỗi event. State triển khai không giữ graph từ các ảnh trước. Forward/update values phải khớp functional reference.

B=8 là 8 image events theo thứ tự, không một event trung bình 8 ảnh. Với 16 events từ reset, update counts của main M phải là `(16,2)`; equal-period C4 có counts khác. Fixed periods đã biểu diễn nhiều timescale; không cần ACC, learned scheduler hoặc offline consolidation để định nghĩa hệ này. Không bọc online update bằng `inference_mode/no_grad` vì local CMS loss cần gradient; chỉ frozen extraction/read-only score được dùng no-grad thích hợp.

## 5. Masked views và final score

Ảnh được resize về 224 trước khi mask. Trên grid 28×28, tại patch `(r,c)` định nghĩa `k(r,c)=2*(floor(r/2) mod 2)+(floor(c/2) mod 2)`. View k che toàn bộ RGB footprints của các patches mang màu k bằng ImageNet mean RGB; sau normalization các pixels bị che bằng 0.

Bốn sets Ω_k không giao nhau, union bằng toàn grid, mỗi set 196 patches. Mỗi block che là 2×2 patches = 16×16 RGB pixels. Giữ token positions; không drop tokens. Mask phải áp **trước** ViT: zero các features của clean ViT sau global attention không đáp ứng contract.

Normal deployment event t dùng view `(t-1) mod 4`. Cùng ảnh xuất hiện lại trong calibration/deployment là hai lượt đọc được ledger ghi nhận, không thêm nhiều CMS events chỉ vì inference có nhiều views.

Official scoring đọc cùng một CMS snapshot cho cả 4 views. Mỗi view reset visual states; không CMS commit. Với j∈Ω_k:

\[
\hat u_j=D_\theta(h_{2,j}(F(x_k);S)),\qquad a_j=\|\hat u_j-u_j\|_2^2,\qquad s(x)=\max_j a_j.
\]

Ghép a theo tọa độ, reshape 28×28 rồi bilinear lên 224×224, `align_corners=False`. Không Gaussian smoothing, test-derived thresholds, per-category normalization hoặc fusion head trong main run. Prediction không được normalize âm thầm; target unit-norm và squared residual là lựa chọn đã khóa.

Tổng chi phí inference: 1 clean + 4 masked ViT passes, 4 visual passes, 4 read-only CMS passes. Có thể batch các views nếu không share recurrent state; latency vẫn tính tất cả công việc. Teacher target từ clean test image là thành phần score, không được dùng để gradient-update CMS/model lúc test.

## 6. Calibration prefix và quyền dữ liệu

Mỗi dataset/run độc lập có first task c₁. Canonical path order của `train/good` trong c₁ định nghĩa prefix; đây là thứ tự protocol, không khẳng định timestamp thu thập thực tế. Lấy `P=min(100,N_c₁)` ảnh đầu; yêu cầu P≥32. Dataset/order không đủ điều kiện phải fail preflight, không mượn ảnh task tương lai.

Trong prefix, theo canonical path order:

| Vai trò | Số ảnh | Dùng |
|---|---:|---|
| `meta_support` | floor(0.50P) | Inner support episodes, online train sau calibration |
| `meta_query` | floor(0.20P) | Outer query loss, online train sau calibration |
| `normal_validation` | floor(0.15P) | Chọn hyperparameter/checkpoint bằng normal loss; không training |
| `normal_probe` | Phần còn lại | Causal/nontriviality diagnostics sau selection; không training/selection |

Với P=100: 50/20/15/15. Với P=60: 30/12/9/9, vẫn đủ support length 16. Không dùng cố định split 170 ảnh của OL-04 làm protocol mới.

Tập online-train của task đầu là toàn bộ normal training trừ validation/probe; bao gồm support/query prefix. Các task sau dùng toàn bộ official normal train. Tại deployment mỗi identity đi qua một lần; outer calibration đã dùng support/query nhiều lần nên **toàn pipeline là prefix-calibrated continual learning, không strict single-pass streaming từ ảnh đầu tiên**.

Matched CADIC và các controls dùng cùng online-training identities/order. Paper-control CADIC dùng full training giữ nguyên ở bảng reproduction. Không trộn hai track mà bỏ qua khác biệt data access. Tất cả learned controls dùng cùng pools, episode schedule và cap optimization; công khai số cập nhật/feature reads thay vì giả định compute bằng nhau.

Chỉ first-task prefix được cấp cho calibration loader. Không load feature/mask/label từ later tasks để tune; metadata dataset/order phục vụ provenance không được đi vào learner. Test filenames/labels/masks chỉ vào evaluator; evaluation không chọn checkpoint. Normal validation/probe vốn thuộc official train, không lấy từ official test.

Các DEVELOPMENT images mà `exps/hope_anomaly_signal.py::development_manifests` lấy từ official test được ghi trong exposure ledger của nghiên cứu cũ. Không xóa lịch sử này để gọi toàn bộ MVTec final test “untouched”. Full canonical table vẫn báo để so sánh literature, kèm disclosure; confirmation subset còn chưa dùng phải được xác minh từ artifacts, không suy ra từ tên `CONFIRMATION_UNEVALUATED`. VisA dùng recipe đã khóa trước khi xem test metrics.

## 7. Outer optimization và export

Mỗi episode reset CMS về learned initialization và zero accumulators/counters. Chọn **16 distinct support images** từ `meta_support` theo schedule có seed; image t dùng view `(t-1) mod 4`. Như vậy CMS chậm có **hai update boundaries**, không dùng 4-support pilot làm bằng chứng đã meta-learn clock 8.

Sau support, lấy 1 image từ `meta_query`, chọn view theo outer step modulo 4, tính final-CMS teacher loss **trước mọi query update**. Query không commit. Backprop qua cả 16 support events và visual recurrences. Không detach CMS support history hoặc gradient accumulator trong functional graph. Sau outer optimizer step, discard toàn episode; episode sau bắt đầu từ updated learned initial parameters, không từ state cuối support trước.

Outer trainables: P; visual block projection/conv/FFN/norm/fusion/q; đủ năm initial memory maps mỗi direction/head; CMS initial parameters; decoder. Frozen ViT không nhận gradient. Learning-rate/retention initial maps nằm trong nhóm trainable, khác pilot hiện tại chỉ học `A0,Wq`.

Reference recipe: Adam lr `1e-4`, betas `(0.9,0.999)`, eps `1e-8`, weight decay 0, max 200 steps; outer global gradient-norm clip 1.0 có logging. Đánh giá normal-validation ở steps 0/25/50/100/200. Mỗi validation evaluation dùng fixed 16-image support schedule từ `meta_support`, reset CMS và score mọi validation image đủ 4 views, CMS read-only. Chọn mean teacher error thấp nhất; tie chọn checkpoint sớm hơn rồi rate nhỏ hơn. Nếu step 0 được chọn, ghi đúng 0 optimizer steps, không claim thực nghiệm này chứng minh lợi ích outer training.

Reference CMS learning rates β₁=β₂=1e-3; finite search `{1e-4,1e-3,1e-2}` dùng cùng initialization/episodes và normal-validation, tối đa 3 fits. Không tune periods, mask family, teacher layer hay score theo official anomaly test. Scientific pilot/reference seed 0; sau selection khóa rate và số outer steps cho final seeds 0/1/2. Các final seeds fit số bước đã khóa, không chọn best test seed. VisA dùng chính các selected hyperparameters/step count của recipe MVTec, fit fresh initial parameters từ prefix VisA; không lấy learned MVTec weights/CMS state sang main VisA run.

Static controls C0/C1 giữ hai MLP, dùng ordinary normal calibration với mean loss trên toàn bộ 16 support + 1 query của cùng episode schedule, nhưng không CMS inner commit. Chọn outer learning rate trong `{3e-5,1e-4,3e-4}`, cùng cap 200 và các validation steps; validation của static control không bật CMS adaptation. Các controls có CMS dùng finite CMS-rate search như M. A1 lấy initialization đã fit của C1 rồi bật full online dynamics với rate đã chọn của M. “No outer learning” với random decoder chỉ là sanity phụ; tổng search/training compute của mọi method được công khai. Registry chi tiết và C4 rate matching theo planning B5.

`normal_probe` chỉ dùng sau selection; không dùng nó để chọn trong ba rates/checkpoints. Nếu nhiều lần đổi thiết kế dựa trên cùng probe, đổi trạng thái của tập đó thành development và ghi exposure; không tiếp tục gọi là held-out confirmation.

Export: đóng băng slow θ, clone learned CMS initialization thành current deployed CMS, zero accumulators/counters và reset deployment cursor đúng một lần. Bắt đầu normal stream theo manifest. Không export episode-terminal CMS hay outer optimizer vào deployed state. Checkpoint training có optimizer/RNG; deployment checkpoint có slow θ, initial/current CMS, pending gradients/counts, cursor/view clock và hashes. Không chạy outer optimizer lại ở task boundaries.

## 8. Evaluation, clocks và fairness

- Normal update là phép biến đổi `S_(t-1),x_t → prediction_before_update,S_t`; score/test là phép `S,x → score` không sửa S.
- Test image/view order không đổi per-image result; state, counters, accumulators và relevant RNG phải unchanged. Visual adaptation chỉ sống trong ảnh/view; target error không được điều khiển update CMS lúc inference.
- Không per-task heads/routing hoặc task-conditioned normalization. Task boundaries dùng để checkpoint/evaluate, không reset/hyperparameter change.
- Cache chỉ frozen-ViT features/teacher targets theo checkpoint+image+mask+preprocess+dtype hashes. Không cache visual/CMS outputs qua outer steps hoặc across changing CMS states.
- Các local losses dùng normal clean target; mask là corruption phục vụ self-supervision, không phải labels bất thường lấy từ test.
- Shared backbone nghĩa là cùng frozen feature extractor; HOPE có thêm trainable capacity và nhiều views. Phải báo cả extra pretraining/data access và compute; không claim matched total model/compute chỉ vì cùng ViT.

Main metrics/ablation matrix theo [planning B4–B6](planning_CAD_v2.md#b4-giả-thuyết-và-phép-thử-bác-bỏ). Final macro I-AUROC và all-pixel P-AUPR là co-primary; I-AP/P-AUROC secondary; forgetting/BWT tính từ task-checkpoint matrix, không epoch matrix. Điểm pixel main ở 224×224; native-resolution analysis phải là protocol/table riêng áp dụng cho mọi method.

Order robustness chính giữ first task cố định, shuffle các task còn lại với order seeds 100/200 để không trộn thay đổi calibration category với thay đổi thứ tự stream. Full permutations đổi first task là phân tích riêng, ghi rõ warm-start identity thay đổi.

Không yêu cầu một history sample cùng phân phối phải thua sample history khác: đó không phải tiêu chuẩn memory learning hợp lệ. Reset/freeze interventions kiểm tra adaptation; khác category history chỉ dùng sau khi các category đó đã xuất hiện, trong evaluator riêng. “Wrong history” cùng class là sensitivity diagnostic, không chứng minh biết đúng danh tính ảnh lịch sử. Replay trong diagnostic evaluator không được giấu thành memory/data access của deployed method.

## 9. Acceptance trước khi chạy thí nghiệm lớn

**Correctness:** shapes/coverage/inverse permutations đúng; pre-event snapshot; per-view reset; 16 events → `(16,2)` updates; partial-period resume; batch transport equivalence; immutable test state; strict target/input separation; feature/manifest/checkpoint hashes; no future-task calibration; no fallback backbone.

**Numerical reference:** toy FP64 token/chunk loop vs optimized reference, atol 1e-9/rtol 1e-7; FP32 normalized-scale reference atol 1e-5/rtol 1e-4 trước khi chấp nhận tăng tốc. Sai số raw feature có scale khác phải báo relative và absolute riêng, không nới tolerance âm thầm. Gain-bound checks theo assumptions; NaN/Inf abort event trước commit. Finite differences trên nondegenerate small fixtures, tránh đúng kink của `min`/clipping; verify derivatives tới mọi nhóm outer trainables, gồm initial CMS và η/α maps. Optimized autograd chưa đạt parity thì chưa được dùng cho scientific run.

**Learning signal:** teacher không đổi và có variance; mask làm thay đổi target-relevant information; so với frozen masked-feature predictor, global/position-wise teacher mean, fitted static predictor. Normal-query improvement đơn lẻ chưa chứng minh AD. Intervention phải tới final score, không chỉ norms của hidden memory. Nếu full predictor không hơn static/reset controls trên normal probe, ghi inconclusive/no-go cho claim memory acquisition; không dùng test anomalies để cứu selection.

**CAD claim:** chỉ kết luận gain khi complete paired seeds/per-class metrics và ablation hỗ trợ. Không coi thấp forgetting là thành công nếu acquisition task mới bị mất. Không ép threshold để phải đạt một số paper. Mọi run thất bại/unstable cũng nằm trong run registry.

**Resource gate:** chạy sạch một T4, profile cả five-view extraction và task-matrix evaluation; không ước lượng quota từ forward một view. Có checkpoint atomic và recovery. Không port TPU hoặc thêm distributed online updates trước khi đường một GPU đúng.

Exact 16-image higher-order graph chưa có measurement xác nhận fit T4. Được vectorize/recompute/checkpoint nếu giữ gradient parity; không tự giảm support length, detach graph hoặc đổi clocks khi OOM. Nếu tối ưu giữ semantics vẫn không fit thì cần resource-backed design revision trước scientific run.

### Kiểm tra số học thực hiện trong lượt rà soát này

Đã chạy một script in-memory bằng `pixi run python`, PyTorch `2.12.0+cu126`, CPU float64, không dữ liệu/weights thật:

- Toy two-level residual CMS, 16 events, periods 1/8: update counts `(16,2)`.
- Derivative outer theo scalar visual parameter: autograd `0.11131695360015488`; central finite difference `0.1113169535833791`; absolute error `1.68e-11`.
- Cùng numerical forward nhưng detach sai local inputs cho gradient `0.1294478921025762`, xác nhận cần phân biệt local ownership với meta graph.
- Four masks trên grid 28×28: 196 patches/view, union 784, không overlap.
- 20 random FP64 fixtures của shared-gain recurrence, 28 tokens/chunk: observed max gain norm `0.829776978685745`, không vi phạm bound trên các fixtures đó.

Đây là **sanity check các công thức ở thời điểm freeze**, không phải proof thay paper hoặc experiment CAD. Sau đó W3 đã có regression suite riêng: 23 tests pass ngày 2026-10-09, gồm FP64 oracle, FP32 parity, first/second derivatives và state lifecycle; xem [báo cáo implementation](w3_implementation.md). Regression tests CMS/outer loop/full CAD vẫn thuộc W4/W5 chưa triển khai.

## 10. Giới hạn được chấp nhận và change control

Rủi ro nghiên cứu: masked teacher discrepancy có thể phản ứng với mask hoặc reconstruction difficulty hơn anomaly; prefix một category có thể học representation không generalize; slow CMS có thể chỉ giảm learning rate hiệu dụng hoặc thêm capacity, chưa chắc giữ tri thức tốt hơn; multi-direction context có thể làm bất thường được reconstruct tốt; ổn định có thể đi kèm memory contraction/collapse. Static/matched-capacity/equal-period controls, norms/variance diagnostics và multi-task/VisA evaluation được dùng để phân biệt các khả năng này.

Những bất định này là đối tượng của thí nghiệm, không phải mâu thuẫn bỏ ngỏ trong thuật toán. Topology, data-access, target, scoring, reset, gradient/clock semantics và finite hyperparameter-selection rule đã chốt. Các giá trị selected rate/step, GPU final cụ thể, baseline provenance và measured throughput là kết quả cần thu thập, không giả vờ đã có.

Đổi teacher layer/normalization, mask, score, memory persistence, CMS objective/periods, gradient approximation, input privileges hoặc thêm upstream bypass phải tạo design revision/method ID và cập nhật ablations trước evaluation. Không đổi thuật toán sau khi xem test rồi giữ nguyên tên/method claim. Các file/module v2 chưa triển khai phải tiếp tục mang nhãn `proposed/not implemented` trong báo cáo và instruction.

## 11. Quan hệ với bước 3–5

Bước 3–4 hoàn tất bằng decision này cùng planning A/B và config audit trong planning. Đã tạo và chuyển sang nhánh riêng `hope_cad_v2` từ `dikhang_space@a89fd48f94ed9c5deb26041151ea63b79911342f` ngày 2026-10-09; chưa commit/push các tài liệu v2.

[instruction_CAD_v2.md](../../instruction_CAD_v2.md) là tài liệu tiếp quản A–Z của bước 5: toán học, protocol, lifecycle, codespace, kế hoạch và giới hạn bằng chứng. W3 visual front end đã implement/test; full v2 chưa hoàn chỉnh/validated. Không cần hỏi lại ba quyết định mục C.
