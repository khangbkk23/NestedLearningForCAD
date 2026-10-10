# Architecture decision v3 — Nested Learning for CAD

Ngày chốt revision v3: 2026-10-10. Design ID: **`hope_cad_v3_masked_teacher_cms`**. Selection-rule ID: **`prefix_cms_stress_v3`**.

**Trạng thái quyết định: FROZEN FOR IMPLEMENTATION v3.** Chặng 1 trên `hope_cad_v2_1` đã triển khai W1/W4 và kiểm lại W3; correctness fixtures không thay nghiệm thu canonical-T4/upstream hoặc CAD validation. [Báo cáo chặng 1](stage1_implementation.md) là bảng trạng thái/evidence hiện hành. Full method vẫn chưa end-to-end/experimentally validated.

Tài liệu này là nguồn chuẩn cho kiến trúc/selection v3, cùng [kế hoạch A/B](planning_CAD_v3.md). `architecture_decision_v1.md` và `paper_to_code_spec_v1.md` tiếp tục mô tả implementation v1 tại `dikhang_space@a89fd48`; v2 giữ nguyên làm lịch sử. V3 không hồi tố thay semantics hoặc tên của các run v1/P0/P1/P2/OL-04.

**Quy ước phiên bản.** Đây là bản đầy đủ kế thừa v2, tích hợp các sửa đổi sau audit; ba tài liệu v2 và báo cáo W3 cũ được giữ nguyên. Design/method ID mới là `hope_cad_v3_masked_teacher_cms`, selection-rule ID là `prefix_cms_stress_v3`, vì quy tắc chọn cấu hình đã thay đổi. Topology, recurrence, initialization, teacher, masks, final score, clocks và quyền dữ liệu cốt lõi giữ nguyên. Namespace code `models/hope_cad_v2/`, `scripts/hope_cad_v2/`, các tên module `_v2` và protocol IDs `mvtec_1x15_prefix_cal_v2` / `visa_1x12_prefix_cal_v2` vẫn là các tên kế thừa; không tạo bản sao code chỉ để đổi số. Full run phải ghi method ID v3 và selection-rule ID riêng, không lấy component metadata W3 v2 làm identity của full method.

**Checkout hiện hành — chặng 1.** Theo chỉ dẫn mới nhất của người dùng, chỉ làm việc trên `hope_cad_v2_1`, bắt đầu từ `5187e4d3ba5fe756520e2499ec6b90a4b6f91507`; W3 `363634b` đã có sẵn. Code mới W1/W4 ở working tree, chưa commit/push. [Báo cáo chặng 1](stage1_implementation.md) xác định source/function, tests/artifacts và các gate còn pending. Snapshot `a89fd48`/`hope_cad_v2_clone` và các nhãn “proposed/not implemented” trong nội dung freeze bên dưới được giữ làm lịch sử tại thời điểm soạn spec; trạng thái implementation hiện tại lấy theo báo cáo chặng 1. Không thay hợp đồng toán học hoặc gán code mới cho commit cũ.

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
- Teacher normalization là một phần head v3; không thay raw features hoặc score của CADIC.

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

`P_in:256→128`, `P_out:128→256`; depthwise convolution kernel 3, padding 1, bias; FFN `256→1024→256`, GELU. LayerNorm theo channel, eps `1e-6`; dropout/stochastic-depth bằng 0. Không thêm CPE, per-route input skip hoặc positional branch khác vào bản reference v3. Đây là mapping CAD tối giản từ structural block, **không tuyên bố trùng toàn bộ implementation backbone ở Appendix B.5**. Nếu dùng upstream block có các branch bổ sung thì phải đổi config/method version, không thay ngầm.

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

**Spectral-guard implementation đã kiểm thử tại `363634b`.** Với `d=key.square().sum(-1).clamp_min(1e-20)`, tính `inverse_key_norm=torch.rsqrt(d)` rồi `spectral=(2*alpha*(1-1e-6))*inverse_key_norm.square()` trước `nextafter` hướng zero và backward mapping hiện có. Dạng này tương đương bound trong real arithmetic, nhưng FP32/autograd có thể khác phép chia trực tiếp; phải kiểm numerical/gradient parity, không suy từ đẳng thức đại số rằng mọi expression có cùng backward ổn định. Traceback Kaggle trước fix chỉ tới `DivBackward0` tại phép chia spectral limit. Không nâng denominator floor, đổi retention/init, bỏ guard hoặc nới tolerance để xử lý lỗi. Regression `test_spectral_limit_backward_is_finite_near_denominator_floor` tại `tests/hope_cad_v2/test_visual_w3.py` dùng key FP32 `1.01e-10`, kiểm first-order gradients hữu hạn gần floor; zero-key/saturated-alpha và full-grid backward cũng được suite kiểm. Gradgrad tests hiện có dùng fixtures riêng, không tự chứng nhận higher-order gần floor hoặc exact 16-event W4. Đây là numerical bug fix có source hash riêng; topology, phương trình, design/method ID và selection rule v3 giữ nguyên.

Local associative target được giữ cố định khi suy ra inner update; điều đó không có nghĩa detach mọi boundary state khỏi outer graph. Outer gradients đi qua recurrence, quantities và initial states. Với nextafter, backward biểu diễn hàm safeguarded real-arithmetic trước phép làm tròn ULP; phải ghi mapping này trong kernel và so với reference, không stop-gradient cả learning-rate/retention branch.

Theorem của VisionHOPE ràng buộc gain/memory norm dưới các giả định của operator; không ràng buộc full residual block, CMS, anomaly ranking hoặc forgetting. Key norm≤1 phải được giữ. Numerical tests kiểm tra gain theo chunk-boundary, không yêu cầu memory norm giảm đơn điệu so với token ngay trước trong chunk.

Nguồn phương trình: [VisionHOPE §§2.1–2.3, Appendix B.3–B.5, C.1](https://arxiv.org/html/2609.33325v1). Dùng toàn ảnh và frozen ViT có global receptive field: đây là thứ tự update có boundary rõ ràng, **không phải pixel-causal inference**.

### 3.3 Fidelity với paper/upstream và gate so số

Recurrence ở trên giữ nguyên: `M_t = M_(t-1) A_t + M_b B_t`, tương đương `G_t=α_t G_(t-1)-η̃_t(G_(t-1)k_t+δ_t)k_tᵀ`. Đối chiếu đại số với Eq.45–46 Appendix B.4 không cho thấy lỗi “đếm đôi”; token oracle và optimized path cùng khớp spec chưa đủ để chứng minh numerical parity với tác giả.

Nguồn upstream ghim `98cfb5d6004472a0ba3d212d3cf884393e39c134`, MIT: [SRNL wrapper](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/models/srnl.py), [chunk CUDA](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/models/srnl_core/csrc/chunk.cu). Source audit chỉ bao phủ các operator paths liên quan, không toàn repository. Upstream có clamp pre-activations η/α ở ±10, query-normalization conventions cần đồng nhất, và custom operator hỗ trợ first-order gradients. W3 giữ unclamped paper-form controls và query eps 1e-6; không tự thêm clamp hoặc thay wrapper/init theo upstream trong revision tài liệu này.

**Gate bổ sung W3 — proposed/not executed:** gọi trực tiếp low-level upstream core và local reference trên cùng explicit tensors/initial maps/normalized queries, cùng layout, guards, dtype và chunk boundaries. Không dùng “cùng seed” thay việc copy chính xác weights; map states theo tên content/key/value/eta/alpha, không giả định positional argument order của hai APIs giống nhau. Chốt fixtures seeds 103/104/105, head dimensions 4/16, chunk lengths 1/7/28, nhiều chunks và ít nhất một sequence 784 tokens; bao gồm multi-image isolation và boundary readout. Main state-parity fixtures có sequence length chia hết chunk length. Với partial tail, [upstream normalized-query path](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/models/srnl_core/chunk.py) padding tới đủ chunk: so valid-token outputs trước, ghi riêng padded terminal state, không nhầm nó với state ngay sau last real token. Fixtures parity chính giữ logits trong vùng clamp upstream không hoạt động. Fixtures ngoài vùng này ghi expected divergence do contract khác, không nới tolerance để gọi khớp.

So output FP32 và boundary/final states mà API thực sự expose: `atol=1e-5, rtol=1e-4`. So first-order gradients với inputs, normalized queries và đủ năm initial maps trên fixtures không tại clamp/min kink: `atol=1e-5, rtol=1e-3`; ghi max absolute/relative errors theo nhóm. Nếu cần instrumentation để expose states, lưu patch/hash harness; nếu chưa quan sát được state thì đánh dấu pending, không claim state parity. Ghi upstream SHA/license, build/runtime, source hashes và inputs hash vào artifact riêng.

Higher-order/gradgrad vẫn kiểm bằng local FP64 oracle và finite differences; không dùng upstream first-order kernel để chứng nhận Hessian hoặc thay vào exact 16-event outer unroll. Chưa chạy gate trực tiếp thì nhãn là **paper/operator mapping audited; upstream numerical parity pending**. W4 reference có thể được dựng tiếp sau diagnostics W3; fidelity claim và scientific use của backend cần gate phù hợp đã đạt, không giả định wrapper/full-backbone parity.

### 3.4 Contraction: diagnostics sớm và ranh giới kết luận

**Bằng chứng và phạm vi.** Audit trước đã chạy `FourScanVisualMemory.inspect` với seed 103, input Gaussian `[1,784,128]`, FP32, chưa outer calibration. Norm tính FP64: mean initial content norm ≈`1.40e-1`, final ≈`1.36e-37`; direction-0 output RMS ở chunk đầu ≈`8.73e-3`, chunk thứ 14 ≈`1.42e-19`, chunk cuối ≈`1.63e-37`. Đây là synthetic measurement trong transcript audit trên source W3 trước fix NaN, **không nằm trong artifact 23-test và không phải CAD experiment**. Cần lưu artifact tái lập khi làm diagnostics mới; không gán các số này cho `a89fd48` hoặc coi artifact Kaggle 24-test là phép đo lại contraction sau fix.

Ở initialization, zero α-map cho retention 0.9; `0.9^784` giải thích một bậc độ lớn khả dĩ, **không phải định luật co chính xác cho mọi input** vì injection, normalized key và boundaries cùng tham gia recurrence. Zero control maps giữ zero qua inner right-action trước outer learning không có nghĩa outer gradient tới initialization bằng zero. Không suy ra mọi outer gradient giảm đúng `0.9^t`, hoặc outer learning không thể phục hồi utility.

**Thứ tự thực hiện:** ngay sau W3, trước đầu tư đầy đủ W4, đo init trên fixtures cố định; thêm real features chỉ từ `meta_support`/`meta_query` khi W2 có extractor/weights. Theo từng direction/head/chunk và restored spatial position, ghi memory/gain norms, output RMS, zero/subnormal fractions, α/η distributions và cap activity. Với full spatial wrapper, ghi `RMS(P_out(V(Z)))/max(RMS(X),1e-12)` và visual-on/off output deltas. Khi minimal W4 đủ chạy vài exact outer steps, đo lại ở steps 0/1/5/25 và selected checkpoint; chạy gradient diagnostics theo nhóm initial maps/q/projections/decoder/CMS initials, rồi final-score interventions ở W5/W6. Không chờ tới W7 mới nhìn contraction.

**Acceptance:** numerical NaN/Inf, state mutation sai hoặc gradient bị disconnect do implementation là correctness failure. Biên độ nhỏ ở init là **warning cần đo tiếp**, không tự động fail kiến trúc hoặc ép đổi initialization. Diagnostics completeness là gate sớm; utility được đánh giá sau calibration bằng trained static/reset/freeze controls và final-score sensitivity. Không đặt một cutoff dựa vào fixture đã xem rồi gọi là preregistered pass. Nếu chưa có bằng chứng utility thì ghi inconclusive/no-go cho claim H3, không bịa threshold đảm bảo gain. Mọi sửa init/retention/guards để xử lý quan sát phải thành method revision trước anomaly-test evaluation.

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

Loss reduction tương đương `0.5 * (pred-target).square().sum(-1).mean()` trên masked positions của event. Không mean theo cả 768 channels làm learning rate hiệu dụng đổi 768 lần.

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

Với P=100: 50/20/15/15. Với P=60: 30/12/9/9, vẫn đủ support length 16. Với P=32: 16/6/4/6; giữ ngưỡng 32, disclosure validation chỉ 4 ảnh và không suy ra selection đáng tin như pool lớn. Không dùng cố định split 170 ảnh của OL-04 làm protocol mới.

Tập online-train của task đầu là toàn bộ normal training trừ validation/probe; bao gồm support/query prefix. Các task sau dùng toàn bộ official normal train. Tại deployment mỗi identity đi qua một lần; outer calibration đã dùng support/query nhiều lần nên **toàn pipeline là prefix-calibrated continual learning, không strict single-pass streaming từ ảnh đầu tiên**.

Matched CADIC và các controls dùng cùng online-training identities/order. Paper-control CADIC dùng full training giữ nguyên ở bảng reproduction. Không trộn hai track mà bỏ qua khác biệt data access. Tất cả learned controls dùng cùng pools, episode schedule và cap optimization; công khai số cập nhật/feature reads thay vì giả định compute bằng nhau.

Chỉ first-task prefix được cấp cho calibration loader. Không load feature/mask/label từ later tasks để tune; metadata dataset/order phục vụ provenance không được đi vào learner. Test filenames/labels/masks chỉ vào evaluator; evaluation không chọn checkpoint. Normal validation/probe vốn thuộc official train, không lấy từ official test.

Các DEVELOPMENT images mà `exps/hope_anomaly_signal.py::development_manifests` lấy từ official test được ghi trong exposure ledger của nghiên cứu cũ. Không xóa lịch sử này để gọi toàn bộ MVTec final test “untouched”. Full canonical table vẫn báo để so sánh literature, kèm disclosure; confirmation subset còn chưa dùng phải được xác minh từ artifacts, không suy ra từ tên `CONFIRMATION_UNEVALUATED`. VisA dùng recipe đã khóa trước khi xem test metrics.

## 7. Outer optimization và export

Mỗi episode reset CMS về learned initialization và zero accumulators/counters. Chọn **16 distinct support images** từ `meta_support` theo schedule có seed; image t dùng view `(t-1) mod 4`. Như vậy CMS chậm có **hai update boundaries**, không dùng 4-support pilot làm bằng chứng đã meta-learn clock 8.

Sau support, lấy 1 image từ `meta_query`, chọn view theo outer step modulo 4, tính final-CMS teacher loss **trước mọi query update**. Query không commit. Backprop qua cả 16 support events và visual recurrences. Không detach CMS support history hoặc gradient accumulator trong functional graph. Sau outer optimizer step, discard toàn episode; episode sau bắt đầu từ updated learned initial parameters, không từ state cuối support trước.

Outer trainables: P; visual block projection/conv/FFN/norm/fusion/q; đủ năm initial memory maps mỗi direction/head; CMS initial parameters; decoder. Frozen ViT không nhận gradient. Learning-rate/retention initial maps nằm trong nhóm trainable, khác pilot hiện tại chỉ học `A0,Wq`.

Reference recipe: Adam lr `1e-4`, betas `(0.9,0.999)`, eps `1e-8`, weight decay 0, max 200 steps; outer global gradient-norm clip 1.0 có logging. Đánh giá normal-validation ở steps 0/25/50/100/200. Mỗi validation evaluation dùng fixed 16-image support schedule từ `meta_support`, reset CMS và score mọi validation image đủ 4 views, CMS read-only. Trong mỗi β fit, shortlist checkpoint có mean teacher error thấp nhất, tie checkpoint sớm hơn; chỉ chọn cuối cùng giữa các candidates pass long gate §7.1, tie earlier step rồi smaller β. Nếu step 0 được chọn, ghi đúng 0 optimizer steps, không claim thực nghiệm này chứng minh lợi ích outer training.

Reference CMS learning rates β₁=β₂=1e-3; finite search `{1e-4,1e-3,1e-2}` dùng cùng initialization/episodes và normal-validation, tối đa 3 fits. Không tune periods, mask family, teacher layer hay score theo official anomaly test. Scientific pilot/reference seed 0; sau selection khóa rate và số outer steps cho final seeds 0/1/2. Các final seeds fit số bước đã khóa, không chọn best test seed. VisA dùng chính các selected hyperparameters/step count của recipe MVTec, fit fresh initial parameters từ prefix VisA; không lấy learned MVTec weights/CMS state sang main VisA run.

Static controls C0/C1 giữ hai MLP, dùng ordinary normal calibration với mean loss trên toàn bộ 16 support + 1 query của cùng episode schedule, nhưng không CMS inner commit. Chọn outer learning rate trong `{3e-5,1e-4,3e-4}`, cùng cap 200 và các validation steps; validation của static control không bật CMS adaptation. C2/C3 dùng finite CMS-rate search và long gate như M. C4 chính kế thừa selected β của M, fit initialization/chọn own step riêng và phải qua cùng gate; không search β độc lập. C4 tự tune nếu có mang ID `C4_tuned_aux` riêng. A1 lấy initialization đã fit của C1 rồi bật full online dynamics với rate đã chọn của M. “No outer learning” với random decoder chỉ là sanity phụ; tổng search/training compute của mọi method được công khai. Registry chi tiết và C4 rate matching theo planning B5.

`normal_probe` chỉ dùng sau selection; không dùng nó để chọn trong ba rates/checkpoints. Nếu nhiều lần đổi thiết kế dựa trên cùng probe, đổi trạng thái của tập đó thành development và ghi exposure; không tiếp tục gọi là held-out confirmation.

Export: đóng băng slow θ, clone learned CMS initialization thành current deployed CMS, zero accumulators/counters và reset deployment cursor đúng một lần. Bắt đầu normal stream theo manifest. Không export episode-terminal CMS hay outer optimizer vào deployed state. Checkpoint training có optimizer/RNG; deployment checkpoint có slow θ, initial/current CMS, pending gradients/counts, cursor/view clock và hashes. Không chạy outer optimizer lại ở task boundaries.

### 7.1 Gate CMS dài hạn và selection rule

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

## 8. Evaluation, clocks và fairness

- Normal update là phép biến đổi `S_(t-1),x_t → prediction_before_update,S_t`; score/test là phép `S,x → score` không sửa S.
- Test image/view order không đổi per-image result; state, counters, accumulators và relevant RNG phải unchanged. Visual adaptation chỉ sống trong ảnh/view; target error không được điều khiển update CMS lúc inference.
- Không per-task heads/routing hoặc task-conditioned normalization. Task boundaries dùng để checkpoint/evaluate, không reset/hyperparameter change.
- Cache chỉ frozen-ViT features/teacher targets theo checkpoint+image+mask+preprocess+dtype hashes. Không cache visual/CMS outputs qua outer steps hoặc across changing CMS states.
- Các local losses dùng normal clean target; mask là corruption phục vụ self-supervision, không phải labels bất thường lấy từ test.
- Shared backbone nghĩa là cùng frozen feature extractor; HOPE có thêm trainable capacity và nhiều views. Phải báo cả extra pretraining/data access và compute; không claim matched total model/compute chỉ vì cùng ViT.

Main metrics/ablation matrix theo [planning B4–B6](planning_CAD_v3.md#b4-giả-thuyết-và-phép-thử-bác-bỏ). Final macro I-AUROC và all-pixel P-AUPR là co-primary; I-AP/P-AUROC secondary; forgetting/BWT tính từ task-checkpoint matrix, không epoch matrix. Điểm pixel main ở 224×224; native-resolution analysis phải là protocol/table riêng áp dụng cho mọi method.

**VisA geometry/mask QA — proposed.** RGB được resize **trực tiếp về 224×224 bằng PIL bilinear**, không giữ aspect ratio bằng padding/crop; policy này áp dụng mọi method trong matched table. Ground-truth mask dùng nearest, binarize raw mask bằng `nonzero`, không mặc định threshold `>127`. Xác minh encoding/geometry từ official preparation và CSV; nếu mask annotation đa kênh phải xác minh kênh nội dung trước khi quy về binary, không tính alpha channel thành defect. Raw anomaly mask thiếu/rỗng hoặc geometry không hợp lệ là lỗi integrity cần giải quyết, không tự suy diễn/sửa nhãn.

Evaluator preflight lưu theo category: số official normal/anomaly images, positive pixels trước/sau resize, identity/hash và số anomaly masks trở thành rỗng ở 224. **Giữ nguyên các ảnh này**: image label lấy từ official CSV, không suy lại từ resized mask; pixel metric dùng đúng binary mask 224 kể cả all-zero. Không drop ảnh, tô lại defect hoặc đổi interpolation cho riêng HOPE. Nếu một category mất toàn bộ positive pixels khiến pixel metric undefined, ghi undefined và suite chưa đủ co-primary acceptance, không loại category khỏi macro để claim thắng. Native-resolution analysis nếu làm phải là protocol/table riêng, áp cho mọi method và công khai geometry.

Chưa có phép đếm local xác nhận tỷ lệ mất mask trên VisA; đây là rủi ro cần QA, không lỗi đã chứng minh. Mask QA nằm trong evaluator/data-integrity tooling, không cấp masks/labels/test metrics cho calibration hoặc dùng để chọn β/architecture. Recipe MVTec phải khóa trước VisA anomaly-test feedback.

**Provenance v3 — proposed schema.** `run.json` phải lưu SHA256 bytes của `docs/instruction_CAD_v3.md`, `docs/hope_cad/architecture_decision_v3.md`, `docs/hope_cad/planning_CAD_v3.md`; `design_id`, `selection_rule_id`, code namespace/schema, actual source commit và source hashes, dirty/untracked bundle hash, backbone/data/role/schedule hashes. Không tự ghi `a89fd48` như thể commit đó chứa W3. Trước chuyển máy, dùng source commit hoặc archive có manifest hashes bao phủ cả untracked files; branch name/tag một mình không đủ tái lập. Lượt soạn tài liệu này không commit/push/tag. Khi tag freeze ở bước đóng gói, tag phải trỏ commit thực sự chứa đúng tài liệu/source được dùng, không hồi tố thay historical run metadata.

Order robustness chính giữ first task cố định, shuffle các task còn lại với order seeds 100/200 để không trộn thay đổi calibration category với thay đổi thứ tự stream. Full permutations đổi first task là phân tích riêng, ghi rõ warm-start identity thay đổi.

Không yêu cầu một history sample cùng phân phối phải thua sample history khác: đó không phải tiêu chuẩn memory learning hợp lệ. Reset/freeze interventions kiểm tra adaptation; khác category history chỉ dùng sau khi các category đó đã xuất hiện, trong evaluator riêng. “Wrong history” cùng class là sensitivity diagnostic, không chứng minh biết đúng danh tính ảnh lịch sử. Replay trong diagnostic evaluator không được giấu thành memory/data access của deployed method.

**Khóa baseline trước HOPE test feedback.** W1 phải chốt effective CADIC adapter route, update unit, score/kernel/preprocessing/metric resolution và resolved hashes cho matched 10k trước khi xem anomaly-test metrics của HOPE. Historical CADIC numbers vẫn được giữ là real results; matched run riêng chỉ cần khi chưa có artifacts cùng protocol/identities để dùng lại. Không đổi route hoặc chữa baseline dựa trên metric có lợi cho HOPE; thay đổi có căn cứ phải version hóa và rerun mọi comparison bị ảnh hưởng. Việc này không chặn W3/W4 correctness hoặc normal-only diagnostics.

## 9. Acceptance trước khi chạy thí nghiệm lớn

**Correctness:** shapes/coverage/inverse permutations đúng; pre-event snapshot; per-view reset; 16 events → `(16,2)` updates; partial-period resume; batch transport equivalence; immutable test state; strict target/input separation; feature/manifest/checkpoint hashes; no future-task calibration; no fallback backbone.

**Numerical reference, gồm upstream gate §3.3 còn pending:** toy FP64 token/chunk loop vs optimized reference, atol 1e-9/rtol 1e-7; FP32 normalized-scale reference atol 1e-5/rtol 1e-4 trước khi chấp nhận tăng tốc. Sai số raw feature có scale khác phải báo relative và absolute riêng, không nới tolerance âm thầm. Gain-bound checks theo assumptions; kiểm finite cả forward/backward ở zero key, gần denominator floor và saturated alpha, gồm regression §3.2. NaN/Inf abort event trước commit. Finite differences trên nondegenerate small fixtures, tránh đúng kink của `min`/clipping; verify derivatives tới mọi nhóm outer trainables, gồm initial CMS và η/α maps. Optimized autograd chưa đạt parity thì chưa được dùng cho scientific run.

**Learning signal, diagnostics init ngay sau W3 và post-calibration theo §3.4:** teacher không đổi và có variance; mask làm thay đổi target-relevant information; so với frozen masked-feature predictor, global/position-wise teacher mean, fitted static predictor. Normal-query improvement đơn lẻ chưa chứng minh AD. Intervention phải tới final score, không chỉ norms của hidden memory. Nếu full predictor không hơn static/reset controls trên normal probe, ghi inconclusive/no-go cho claim memory acquisition; không dùng test anomalies để cứu selection.

**CMS long-horizon acceptance:** thực hiện §7.1 trước scientific pilot; validation-loss drift, weights/accumulator norms và finite-state checks đều bắt buộc. Đạt 16-event meta-gradient test không thay gate 1,024 events; đạt gate lặp prefix không chứng minh deployment ổn định qua 15/12 tasks.

**CAD claim:** chỉ kết luận gain khi complete paired seeds/per-class metrics và ablation hỗ trợ. Không coi thấp forgetting là thành công nếu acquisition task mới bị mất. Không ép threshold để phải đạt một số paper. Mọi run thất bại/unstable cũng nằm trong run registry.

**Resource gate:** chạy sạch một T4, profile cả five-view extraction và task-matrix evaluation; không ước lượng quota từ forward một view. Có checkpoint atomic và recovery. Không port TPU hoặc thêm distributed online updates trước khi đường một GPU đúng.

Exact 16-image higher-order graph chưa có measurement xác nhận fit T4. Được vectorize/recompute/checkpoint nếu giữ gradient parity; không tự giảm support length, detach graph hoặc đổi clocks khi OOM. Nếu tối ưu giữ semantics vẫn không fit thì cần resource-backed design revision trước scientific run.

**Resource spike sớm — proposed.** Ngay khi minimal functional CMS có thể unroll, profile một exact 16-event `create_graph=True` episode với random weights/synthetic features trên một T4: forward, backward, peak allocated/reserved VRAM, wall time, CPU RAM và backward tới mọi nhóm trainable. Tách synthetic feasibility khỏi real-feature learning evidence; W2/W4 chưa có measurement này. Sau đó đo real-feature cached/end-to-end, five-view score và đủ 120/78 task-evaluations cho MVTec/VisA, cả calibration stress gate và bootstrap. Không suy runtime toàn hệ bằng đúng 5× CADIC từ số lượt backbone.

Compute ledger seed-0 tối đa: M/C2/C3 mỗi method 3 fits, C4 1 fit, C0/C1 mỗi method 3 fits, tổng 16 fits; A1 reuse C1. Tối đa 10 long gates cho M/C2/C3/C4 và 1 acceptance gate A1; counts thực tế ghi cả failed fits. Final refits/seeds/orders/datasets là chi phí bổ sung, không nằm trong 16 fits. `C4_tuned_aux` thêm search riêng nếu được đăng ký. Trước anomaly evaluation, ưu tiên correctness/profile → M/C0/C1/C2 mechanism pilot → C3/C4/A1 → suite main/seeds → order studies/VisA theo W7/W8. Khi thiếu quota, hoãn optional auxiliary variants trước; không bỏ mandatory controls, co-primary metrics, intermediate pixel metrics, seeds hoặc VisA rồi tuyên bố hoàn thành 100%. Chuyển phiên/máy bằng checkpoint đã verify, không đổi toán học để vừa quota.

### Kiểm tra số học lịch sử khi freeze v2 và evidence W3

Đã chạy một script in-memory bằng `pixi run python`, PyTorch `2.12.0+cu126`, CPU float64, không dữ liệu/weights thật:

- Toy two-level residual CMS, 16 events, periods 1/8: update counts `(16,2)`.
- Derivative outer theo scalar visual parameter: autograd `0.11131695360015488`; central finite difference `0.1113169535833791`; absolute error `1.68e-11`.
- Cùng numerical forward nhưng detach sai local inputs cho gradient `0.1294478921025762`, xác nhận cần phân biệt local ownership với meta graph.
- Four masks trên grid 28×28: 196 patches/view, union 784, không overlap.
- 20 random FP64 fixtures của shared-gain recurrence, 28 tokens/chunk: observed max gain norm `0.829776978685745`, không vi phạm bound trên các fixtures đó.

Đây là **sanity check các công thức ở thời điểm freeze**, không phải proof thay paper hoặc experiment CAD. W3 có artifact local 23 tests pass ngày 2026-10-09 tại source `6dbe4f1`, gồm FP64 oracle, FP32 parity, first/second derivatives và state lifecycle. Source này sau đó fail full-grid CPU backward trên Kaggle theo transcript người dùng; fix `363634b` đã qua suite 24 tests trong [artifact Kaggle](../../results/hope_cad_v2/w3_verification_kaggle.json) (`2026-10-09T15:46:02Z`, không failures/errors/skips), có 7/7 source hashes khớp commit. Runtime: Linux, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA runtime 12.8, Tesla T4. Canonical full-grid test chạy CPU; CUDA parity/backward dùng reduced fixture. Đây chưa là full-grid T4 profile, upstream numeric parity, long CMS gate hoặc CAD validation. W2 có evidence runtime một phần, W4/W5 tests còn chưa triển khai. Xem [báo cáo implementation](w3_implementation_v2.md) cho lịch sử, hashes và phạm vi.

## 10. Giới hạn được chấp nhận và change control

Rủi ro nghiên cứu: masked teacher discrepancy có thể phản ứng với mask hoặc reconstruction difficulty hơn anomaly; prefix một category có thể học representation không generalize; slow CMS có thể chỉ giảm learning rate hiệu dụng hoặc thêm capacity, chưa chắc giữ tri thức tốt hơn; multi-direction context có thể làm bất thường được reconstruct tốt; ổn định có thể đi kèm memory contraction/collapse. Static/matched-capacity/equal-period controls, norms/variance diagnostics và multi-task/VisA evaluation được dùng để phân biệt các khả năng này.

Những bất định này là đối tượng của thí nghiệm, không phải mâu thuẫn bỏ ngỏ trong thuật toán. Topology, data-access, target, scoring, reset, gradient/clock semantics và finite hyperparameter-selection rule đã chốt. Các giá trị selected rate/step, GPU final cụ thể, baseline provenance và measured throughput là kết quả cần thu thập, không giả vờ đã có.

Đổi teacher layer/normalization, mask, score, memory persistence, initialization/retention/guards, CMS objective/periods, gradient approximation, input privileges, selection/gate constants hoặc thêm upstream bypass phải tạo design revision/method ID và cập nhật ablations trước evaluation. Không đổi thuật toán sau khi xem test rồi giữ nguyên tên/method claim. Các file/module phục vụ v3 chưa triển khai phải tiếp tục mang nhãn `proposed/not implemented` trong báo cáo và instruction.

## 11. Quan hệ với bước 3–5

Bước 3–4 là planning/spec đã chốt; revision v3 giữ architecture/selection contracts. Task tài liệu trước chỉ soạn trên clone cũ. Hiện người dùng đã chuyển công việc sang `hope_cad_v2_1`, có W3 sẵn và cho phép triển khai chặng 1; code mới chưa commit/push. Xem [báo cáo chặng 1](stage1_implementation.md) cho provenance hiện hành.

[instruction_CAD_v3.md](instruction_CAD_v3.md) là tài liệu tiếp quản A–Z: toán học, protocol, lifecycle, codespace, kế hoạch và giới hạn bằng chứng. W3 implementation/reference nằm trên nhánh riêng như đã ghi; full method v3 chưa hoàn chỉnh/validated. Không cần hỏi lại ba quyết định mục C; trước W4 trên clone phải có bước tích hợp source W3 được cho phép và kiểm hash/test tương ứng.
