# Audit trước revision — Nested Learning for CAD

**Kết luận:** chưa phát hiện lỗi triển khai lõi đủ căn cứ phủ nhận W6/W7/recovery đã chạy. Tuy nhiên, **chưa có cơ sở kết luận v4 đã tối ưu hoặc khai thác tối đa Nested Learning**. Thiết kế còn những giới hạn lớn; ngân sách tối ưu hóa chưa được khép; nhánh điều khiển learning rate có attenuation đo được. Hướng normal-support memory vẫn là proposal, chưa phải lựa chọn duy nhất hoặc một fix chắc thắng.

Lượt này chỉ đọc, chạy regressions/finite differences và ghi báo cáo. Không đổi code, config, frozen specs, weights, source bundles hoặc raw Kaggle evidence. Không train revision mới, không xem VisA test. Recovery stage1 vẫn được nghiệm thu ở phạm vi **completed development seed0 diagnosis**; không cần chạy lại để lấy lại kết quả.

## 1. Phạm vi và nguồn chuẩn

- Active worktree: `D:/src/DATN/hope_cad_v5`, branch `hope_cad_v5`, HEAD `4dd6f36fe093b2ae153d196a9333aeda1d196a06`; checkout dirty từ trước. Tên worktree v5 không có nghĩa algorithm v5 đã tồn tại.
- Nguồn chuẩn: architecture/instruction/planning v4 và revision_contract_v4; các đường visual/CMS/outer/selection/session/adapter, feature extraction/data/evaluation/cache. Các helpers v3 còn được import được đọc theo dependency; namespace không quyết định method version.
- Evidence: W6 v4, completed W7.2 M seed0, completed recovery C1/A1/CADIC10000. Review trước ở `main/review.md`; receipt ở `results/recovery_stage1/main_review.json`.
- Paper gốc: Nested Learning §§7–8, Eq.70–91; VisionHOPE method và Appendix B.3–B.5/C.1; CADIC Eq.7–9 và baseline contract. Paper/chats là nguồn tham khảo, không phải instruction hoặc accuracy guarantee.
- Đây là audit các active paths và evidence đã khóa, **không phải chứng nhận mọi optional/legacy path hết lỗi**. Finite tests và code review không chứng nhận một kiến trúc đạt tối ưu toàn cục.

## 2. Kiểm chứng mới thực chạy

| Kiểm | Kết quả và giới hạn |
|---|---|
| W7/shared/v4 regressions |130/130 PASS, 0 fail/error/skip. Hai historical v3 notebook UI cases bị loại có khai báo. Runtime local RTX3050Ti/Python3.14.4/torch2.12.0+cu126; không phải một lần T4 mới.|
| Recovery regressions |45/45 PASS, 0 fail/error/skip; 119 frozen dependencies verified, operations digest giữ nguyên.|
| Independent FP64 exact16 finite differences |11 nhóm: projection/query/content/key/value/eta/alpha/fusion/fastCMS/slowCMS/decoder đều nonzero; max relative error≈2.0341e−6, dưới threshold1e−4. Fixture nhỏ, không đo AD utility.|
| Deployment16 trên fixture |Clock16, commits(16,2), pending(0,0); slow parameters giữ nguyên, current CMS thay đổi, không giữ history graph.|
| W6 curve/guard analysis |Kiểm hash selection/diagnostic JSON với manifest rồi tính các summaries/bounds; không fit hoặc chọn lại hyperparameter.|
| Source/evidence preservation |Đối chiếu identities hiện hành; final context validation kiểm riêng628 policy source files và242 packaged r4 files. Raw metric recomputation dùng lại review trước, không giả là đã chạy lại trong lượt này.|

Bốn tests mà reviewer learning chạy riêng và24 targeted data/evaluator tests đều là subset của130; không cộng trùng. Tổng hai suite chính là175 tests. Receipts và script tái lập được lưu trong shared review pack. Engineering PASS không đồng nghĩa AD gain.

## 3. Paper → thiết kế v4 → code

| Khái niệm | Nguồn và mapping | Phán quyết |
|---|---|---|
| Sequential CMS, task loss tùy bài toán, fixed frequencies |NL tr.27 Eq.70–71, tr.28 Eq.73; architecture§4; `cms.py:47–57,94–150`|Hợp lệ trong NL. V4 không implement fully recursive nested variant Eq.72.|
| Five-map self-modification |VisionHOPE tr.4–5 Eq.8–17; `visual_memory.py:126–188`|Có content/key/value/eta/alpha, boundary readout và shared gain; khác prototype CLS fast-weight legacy.|
| Query projection cố định trong scan |VisionHOPE tr.5 Eq.14; NL tr.31 Eq.83; architecture§3.1|Đúng route paper đã chọn, không phải thiếu Mq. Query vẫn được outer-learn trong calibration.|
| Recurrence có current và boundary contributions |VisionHOPE tr.18 Eq.45–46; `visual_memory.py:165–188`|Chưa thấy chép sai hoặc đếm đôi; numerical evidence có scope rõ.|
| Stability bound |VisionHOPE tr.17–19 Prop.1/Theorem1; `visual_memory.py:94–122`|Chỉ bound gain/memory matrices của operator; không bound cả decoder/CMS hoặc bảo đảm chống quên/AUROC.|
| Reset mỗi ảnh/view |VisionHOPE tr.7 Eq.25–26; architecture§1/§3|Hợp visual setting. Tri thức liên ảnh nằm ở persistent CMS.|
| Full VisionHOPE block/backbone |Appendix B.5, tr.19–20 có CPE, gated DW, local branches, nhiều blocks; architecture§3.1 công khai minimal adaptation|One-block v4 bỏ nhiều nhánh có chủ đích. Không lấy accuracy của full paper bảo chứng head hiện tại.|
| Full HOPE memories/optimizer |NL tr.31–32 Eq.83–91, tr.39 Table6 có deep residual MLP memories/optimizer variants|V4 chọn linear visual memory + plain SGD CMS. Không tự thêm momentum/DGD rồi gọi đó là bug fix; cần mapping và revision riêng.|

V4 có nested optimization thật: outer gradients đi qua inner updates/learned initials và có state ở các clock khác nhau. Nó là một instantiation hạn chế, chưa phải toàn bộ experimental HOPE hoặc một cấu hình được chứng minh tốt nhất cho CAD.

## 4. Findings

### F1 — Cao: decoder và representation cố định hạn chế acquisition

Nguồn: `training/hope_cms_session_v2.py:13–27`; `models/hope_cad_v2/predictor.py:42–46,72–73`; `teacher_head.py:7–15`; architecture§1/§2/§7.

Sau bottle prefix, projection, visual slow weights và decoder đều frozen. Online learner chỉ thay current CMS. Mọi prediction có dạng `û=Dθ(Hφ(x))`, với cùng decoder `Dθ:R^256→R^768` học từ task đầu. Đây **đúng contract**, không phải vô tình quên gọi optimizer.

Với `e=Dθ(h)−u`, có `∇φL=J_HᵀJ_Dᵀe` và `||Dθ(h)−u||²≥inf(v∈ImDθ)||v−u||²`. Thành phần residual vuông góc column-space của Jacobian decoder không tạo tín hiệu bậc một cục bộ để CMS sửa. Rank Jacobian≤256 không chứng minh nonlinear output nằm trong affine256D, cũng không chứng minh teacher targets thực tế vượt reachable manifold. Cần đo trên checkpoint thật để quy causal share.

Recovery cho thấy acquisition yếu: M macro normal loss0.604614 ở diagonal và0.743455 ở cuối stream; zero-predictor anchor với unit teacher≈0.5. Nhưng chưa có bằng chứng chỉ mở decoder là đủ khép gap với CADIC.

### F2 — Cao: meta task/horizon khác deployment

Nguồn: `hope_outer_learning_v4.py:75–84,101–127`; `hope_cms_stress_v4.py:15–18`; architecture§7.1.

Mỗi meta episode dùng16 support từ50 bottle normals, query từ20 bottle normals và reset CMS. Slow level chỉ có2 commits trong horizon16. Deployment có3.599 events/449 slow commits qua15 categories, không reset. Gate1.024 lặp cùng70 bottle normals là stability screen, không tạo category diversity hoặc huấn luyện retention qua domain shifts.

Warm objective dành50% weight cho clock0 và50% cho16. Đây là policy khóa trước run, chưa được chứng minh tối ưu. Các episodes cùng category có thể ưu tiên predictor tĩnh hơn adaptation qua domains; đó còn là giả thuyết. Tăng số episodes không tự tạo distribution shifts. Không được lấy future-task normals vào calibration rồi giữ cùng protocol.

### F3 — Cao: clock chậm chưa bảo đảm retention

Nguồn: `cms.py:94–150`; `predictor.py::dynamics`; architecture§4.

Hai CMS đều tối ưu own partial của current final teacher loss. Sau prefix chưa có explicit past-normal risk, replay, anchors hoặc consolidation objective. Với một bước current gradient, `ΔL_old≈−β⟨g_old,g_current⟩`; period8 không bảo đảm tích vô hướng này≥0. Giảm tần suất/average gradient có thể giảm tốc hoặc variance, nhưng không tự ngăn quên.

14/14 task cũ tăng normal loss là quan sát thật. Tuy nhiên C3/C4 chưa có matched full-stream AD để tách riêng tác dụng nhiều timescale. M−A1 cũng không phải pure timescale contrast. Fixed frequencies vẫn hợp khung NL; vấn đề là chưa có bằng chứng clocks hiện tại học giữ tri thức tốt.

### F4 — Cao: prediction objective khác anomaly membership

Nguồn: `cms.py::masked_teacher_loss`; `benchmark_adapter_v4.py:173–200`; architecture§4/§5; recovery review§4.

Nghiệm population MSE là conditional mean `E[U|F(mask(X))]`. Residual có thể phản ánh uncertainty của corruption/context, transfer error và anomaly. Normal khó dự đoán có thể có score cao; anomaly dễ dự đoán có thể có score thấp.

Joint-final loss đã sửa mismatch giữa intermediate head và final head. Nó chưa biến mean masked MSE thành objective trực tiếp tối ưu AUROC/pixel AP. Train mean196 masked patches còn khác image-max784 patches; teacher L2 normalization bỏ radial magnitude của raw features mà Euclidean CADIC khai thác.

Frozen ViT có global attention:75% visible pixels có thể chứa thông tin vùng anomaly lân cận. Đây là shortcut risk của representation, không phải train/test label leakage. Chưa có intervention xác định mức tác động.

A1 normal loss thấp hơn M nhưng AD cũng thấp hơn M. M có angular share≈98.20% tổng residual trên sampled normals; radial share tại image-max≈18.69%. Vì vậy normal MSE nhỏ hơn hoặc normalize output đều chưa bảo đảm benchmark tốt hơn.

### F5 — Trung bình/Cao: chưa khép optimization budget

Nguồn: W6 ZIP `selection/<method>/selection.json`; sidecar `optimization_curves.json`; `hope_outer_learning_v4.py:17–19,169–172`; revision contract.

**Cả16 fits đều shortlist step400**, đúng cap đã khóa. M chọnβ=.01, ở mép trên grid. Validation loss của M giảm từ0.15745845 tại300 xuống0.14490038 tại400: giảm7.9755%; last outer gradient norm≈0.501285. Static C1 chọn LR=.0003 cũng ở mép trên; loss300→400 giảm10.3223%.

Do đó chưa có convergence certificate hoặc optimum được bracket. M warm-0 loss=.14538758, warm-16=.14441317: short adaptation gain≈0.6702%. CMS không chết—state thay đổi và A1−C1 có gain—nhưng chưa có bằng chứng đã tận dụng hết adaptation. Clip69/400 của M không có nghĩa mọi bước bị clip.

Continuation có thể chỉ giảm bottle loss mà không giúp cross-category AD. Không tự tăng steps/β hoặc chạy lại toàn W6 để hy vọng. Nếu cần kiểm cap, chỉ nên có một bounded normal diagnostic riêng, rule/budget khóa trước và không export thành selected v4.

### F6 — Trung bình: η-control bị attenuation do safeguard

Nguồn: `visual_memory.py:94–122`; W6 `selection/M/selected_diagnostics.json`, SHA `006c3e415fb0f347e9e12a76c9fb0a90cd83b82ddf0e73f10686b10c52ef2667`; bottle/train/good/000.png, view0, selected400.

Khi spectral cap không active, `ηexec=c(1−exp(−η/c))`; direct partial giữ key/delta/alpha cố định là `exp(−η/c)`.

Chunk0: rawη_min=.01732167415, executedη_max=.00371779059, spectral-cap fraction0. Bound `ηexec/η≤.21463229` suy `η/c≥4.61289858`, nên **direct sensitivity≤.00992302** cho mọi tokens/heads/directions trong chunk đó. Upper bounds ở chunks1/7/13 lần lượt≈.01020/.01264/.03648.

Đây là attenuation đo được, không hidden detach. Nhánh visual vẫn hoạt động: branch/input RMS=.0569783; on/off delta RMS=.128921; center/border ratios=.2813–.3063. Không còn cơ sở dùng collapse v3 như lời giải thích chính hiện tại.

Một ảnh/view không đại diện toàn stream; direct partial không là total meta-gradient. Initialη gradient nhỏ cũng chưa chứng minh utility bằng0 vì Adam rescaling và các đường alpha/key/value còn hoạt động. **Không bỏ guard để boost**; đổi parameterization/init phải giữ stability bound và dùng revision.

### F7 — Trung bình: fairness và causal claim cần đúng phạm vi

Các method có cùng backbone/checkpoint/preprocessing/online3.599/test1.725; không có cùng memory hoặc total compute. M persistent state4.214784MB, CADIC30.720140MB; tổng deployment356.038144MB và373.951956MB. Không giảm CADIC budget sau test để cứu headline; resource curve nếu nghiên cứu cần track riêng.

A1−C1 hỗ trợ giá trị online CMS với cùng initialization. M−A1=+2.4795pp/+0.7108pp hỗ trợ **package initialization/training recipe**, vì objectives, LR grids và selected weights khác. Không quy riêng gain này cho exact meta-gradient, visual self-modification hoặc slow clock. C2/C3/C4, multi-seed và paired uncertainty vẫn thiếu để claim cơ chế.

CADIC `exact_parity_claim:false` đối với author pipeline còn nguyên. Matched local baseline có giá trị đối chiếu, chưa phải hoàn hảo paper reproduction.

### F8 — Thấp: runtime inefficiency và profiling

`models/feature_extractors/cadic_vit_v1.py:121,142–151` hook block8 nhưng forward đủ12 blocks, bỏ output3 blocks cuối. HOPE scoring dùng5 passes/ảnh. Early-exit có thể tiết kiệm runtime nhưng cần numerical parity/source identity mới; nó không tự tăng AUROC vì prediction đã lấy block8.

CADIC feature timer không synchronize CUDA như HOPE; HOPE update timer bao cả feature extraction. Không cộng/so các timer categories như benchmark latency matched. Cần wall/resource đúng scope và tính calibration cost riêng. Đây là reporting/runtime risk, không phải metric corruption.

## 5. Những nghi ngờ đã được kiểm và chưa xác nhận thành bug

- `torch.func.grad` dùng independent formal weight dictionaries đúng own partial; outer graph giữ input/decoder/history. Finite differences11 nhóm và functional/deployment tests khớp.
- Mean accumulation, pre-event snapshots, clocks, no task-boundary flush/reset và atomic rollback đúng contract. Export learned initials/counters0, không dùng episode-terminal hoặc stress-terminal state.
- Final score đọc output CMS cuối, four RGB masks coverage1. Không đưa clean teacher/label/task ID vào predictor hoặc cộng raw NN score vào M.
- ViT strict checkpoint, block8, no CLS, no extra final LN, raw dimension768 khớp CADIC. GT masks nearest/nonzero/label policy, cùng224 evaluation, Eq.9 distance multiplier/cstar support đúng locked implementation.
- Prefix roles và30 validation/probe normals bị loại khỏi online cho cả hai. Evaluator từ chối future tasks, bảo vệ state/RNG, kiểm manifests/shards/hash/resume. Không thấy direct leakage ở paths đã audit; không forensic mọi thao tác ngoài artifacts.
- Recovery metrics giữ nguyên: M66.8010/17.0163%, C1 56.8491/12.3667%, A1 64.3216/16.3055%, CADIC88.3571/46.0137%. Gap lớn chưa chỉ ra một causal solution duy nhất.

## 6. Bước quyết định trước revision

Không thể khép mọi giả thuyết bằng review giấy. Recovery reportA8.3 đã đề xuất capacity fits nhưng **gói recovery vừa chạy chưa thực hiện các fits ấy**: nó đo norms/decomposition/discarded single-event proposals, không train các arms CMS-only/decoder-only.

Khuyến nghị **một gói chẩn đoán normal-only có budget hữu hạn**, nếu muốn loại nghi vấn trước khi thay family. Gói này vẫn **PROPOSED, chưa triển khai/chạy**:

1. Khóa data rights, checkpoints, arms, steps/LR và stop rules trước. Chỉ dùng normals đã đến tại checkpoint; clones bị discard, không đọc thêm anomaly test/VisA. Reused normals phải ghi in-sample; auxiliary query split không làm nghiên cứu trở lại untouched.
2. Trên cùng inputs/targets hợp lệ, đo best-found latent fit với frozen decoder, CMS-only, decoder-only, CMS+slow decoder; thêm past-anchor arm nếu xét retention. Cân bằng current-fit budget, đo current/past riêng. Latent fit là nonconvex best-found, không chứng minh global minimum.
3. Nếu kiểm cap400, thêm đúng một continuation prefix-normal có budget khóa trước. Không mở grid vô hạn, tune head theo AD hoặc bỏ stability. Có thể log η sensitivity/relative updates trong cùng gói.
4. Latent/decoder fit tốt mà CMS-only không tốt sẽ ưu tiên acquisition/controllability. Nếu mở slow head+anchors giúp current/past trade-off, cần ordinary online learner+replay control trước claim NL. Nếu gains chỉ ở bottle hoặc surrogate vẫn yếu, hạ ưu tiên predictor trong budget hiện có và xem xét hướng B. Kết quả âm không chứng minh mọi predictor bất khả thi.
5. Sau đó chốt **một revision** và freeze trước VisA. Hướng B cần learned normal utility/credit assignment, fast/slow ownership, retention, raw geometry, matched memory/compute và ablations. Learned writes có thể làm bank kém; **không bảo đảm baseline floor** trừ khi giữ exact baseline output.

Nếu nhóm chọn chuyển B ngay vì budget, cần ghi rõ chưa loại hết causal hypotheses của hướng A. Không tuyên bố v4 đã tối ưu tối đa hoặc Nested Learning không hợp CAD. Giữ negative/development evidence và không sửa history.

## 7. Phán quyết

| Câu hỏi | Kết luận |
|---|---|
| Có confirmed core bug buộc hủy/rerun benchmark? |Chưa tìm thấy trong phạm vi đã audit; không bảo đảm mọi optional path hết lỗi.|
| Completed recovery/benchmark dùng được? |Có, đúng development seed0 đã review; chưa full W7/CI/VisA.|
| V4 đã tối ưu/tận dụng max NL? |Chưa có bằng chứng; F1–F6 là restrictions hoặc optimization questions thật.|
| B đã được chứng minh tốt nhất/thắng CADIC? |Chưa, vẫn proposal.|
| Cần full W6/full7.3 ngay? |Không có lý do từ audit này. Nếu muốn khép causal questions, khóa một gói normal diagnostic riêng.|
| Có sửa source/spec/weights? |Không; chỉ thêm audit/receipts/context notes.|

Các core correctness nghi ngờ đã được đối chiếu và có reproducer. Các giới hạn còn lại có nguồn nhưng causal share chưa được đo hết. Không thể bảo đảm “không còn thiết kế chưa tối ưu” hoặc “sửa sẽ tăng bao nhiêu điểm”.
