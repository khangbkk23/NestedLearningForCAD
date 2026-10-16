# Nested Learning for CAD — research audit after W7.1

Ngày: 2026-10-16, Asia/Bangkok. Scope: worktree `D:/src/DATN/hope_cad_v2_2`, branch registry `hope_cad_v2_2`, HEAD `b49a1036b3b79e8e28baf673dcb9da57476785f9` cộng source đã hash trong bundles. Đây là **báo cáo audit và đề xuất nghiên cứu**, không phải architecture freeze, không sửa hoặc thay thế các spec v4, không cấp nhãn implemented cho các phương án bên dưới.

Snapshot: W7.1 đã hoàn tất; W7.2 đang chạy theo thông tin người dùng, **chưa nghiệm thu matrix**. Bản transport fix JSON tuple/list được ghi nhận riêng, không là method revision hoặc lời giải thích cho performance thấp của W7.1. Không chạy thêm official anomaly test trong lượt audit này; không đọc local VisA images, labels, masks hoặc kết quả thực nghiệm để chọn phương pháp.

## 1. Kết luận điều hành

1. Chưa tìm thấy lỗi chia split, đảo nhãn, tính metrics hoặc bỏ task đủ giải thích khoảng cách performance. Có bug vận hành resume đã sửa độc lập, và có thiếu sót observability: loss online tính rồi bị bỏ không lưu.
2. Các đường học chính bám spec v4. Vấn đề lớn nhất là **giả định hữu ích của phương pháp chưa được chứng minh**, đặc biệt head được calibration chỉ trên bottle rồi đóng băng trong khi CMS phải học các category mới.
3. NL/HOPE/VisionHOPE không có định lý bảo đảm masked teacher residual của instantiation này bằng hoặc hơn CADIC. Có công thức đúng và numerical stability không đồng nghĩa có anomaly ranking tốt.
4. W7.1 là experimental AD evidence thật nhưng bất lợi. Không dùng engineering PASS hoặc việc hoàn tất notebook để mô tả thành scientific success.
5. Nên hoàn tất W7.2 không đổi method, rồi ưu tiên matched CADIC seed0 cùng một nhóm controls nhỏ đã đăng ký. Chưa có căn cứ đổ quota vào toàn bộ suite 35 stream với hy vọng chọn được seed đẹp.
6. Để tăng cơ hội đạt CADIC trên VisA, hướng đáng ưu tiên nghiên cứu là **giữ hình học frozen patch features và đưa Nested Learning vào cơ chế học/lưu/giữ normal support**. Đây là proposal, còn phải giải quyết objective và credit assignment; không được đổi tên CADIC hoặc thêm nhánh vô tác dụng rồi gọi là đóng góp.
7. Nếu tiếp tục pure HOPE predictor, cần kiểm tra và xử lý acquisition/decoder bottleneck trước; chỉ tăng steps, tăng số tầng hoặc đổi beta là chưa có lý do đủ mạnh.
8. Không có quy trình normal-only nào bảo đảm AUROC/AUPR không kém CADIC với mọi phân phối anomaly tương lai. Có thể bảo đảm bằng score nếu giữ exact CADIC scoring và chứng minh retrieval exact, nhưng lúc đó đóng góp phải là efficiency thực đo, không giả làm accuracy gain.

## 2. Phạm vi bằng chứng và trạng thái W1–W9

Các trạng thái được tách: `implemented` là có code; `tested` là có verification đúng scope; `experimentally validated` phải ghi rõ property nào được đo; `pending` không được thay bằng kết quả dự kiến.

| Gói | Implementation/test/evidence hiện có | Điều chưa hoàn thành hoặc chưa chứng minh |
|---|---|---|
| W1 protocol/baseline | Explicit registry, per-image memory update, full Eq.9, prefix manifests, validators, normal-only access và mask QA đã implement/test | Lock/prepare không phải baseline benchmark. Historical Khang numbers thật nhưng chưa là matched comparison; author parity còn assumptions |
| W2 runtime/repro | Kaggle T4 W6/W7 preflight, source/runtime/data hashes, cache checks, export và resume có evidence | Một notebook chạy hết cells không đủ; xem handoff/status. Bản sửa transport có receipt riêng |
| W3 visual operator | Independent FP64 oracle, FP32/autograd tests, upstream-core pin parity; stage1 report ghi 56 tests, 19 upstream fixtures và canonical exact16 T4 profile | Không là full VisionHOPE backbone reproduction hoặc evidence AD gain |
| W4 CMS/outer | Exact16 unroll, joint-final objective, clocks1/8, mean accumulators, C4 rate inheritance, failure atomicity và resume đã implement/test | Chưa có proof cross-category transfer, retention hoặc AD superiority |
| W5 RGB CAD | Frozen ViT, RGB masking, final CMS decoder score, read-only evaluation đã chạy thật | Objective/scoring vẫn có thể không phù hợp anomaly ranking dù code đúng |
| W6 v4 | Completed real-prefix fits, long stability gates, selected zero-clock initializations và mechanism diagnostics đã có | Gate chỉ repeated first-task normals; mechanism evidence mixed, không phải AD validation |
| W7.1 | Đủ 3599 online normals, 15 checkpoints, 1725 test images; raw scores/maps và exact metrics đã kiểm | M seed0 performance thấp; chưa có matched baseline hoặc seed robustness |
| W7.2 | Code và transport fix đã có; đang chạy ngoài local theo người dùng | Chưa nghiệm thu task matrix/F/BWT hoặc acquisition-vs-forgetting |
| W7.3 | Registered jobs/runner/refits/reporting đã implement và fixture-test | Chưa hoàn tất full benchmark, controls, seeds, orders hoặc paired CIs |
| W8 VisA | Official split/prefix config, loader và mask QA có code/tests | W7 runner và report còn hard-code MVTec/15 tasks; chưa có complete W8 runner/evidence. Không chỉ đổi dataset path |
| W9 thesis/reproduction | Có specs và artifact/provenance infrastructure | Còn final scientific results, reproducibility, report/demo và kết luận |

Lưu ý tên: C0 trong v3/v4 hiện hành là static trained HOPE-topology control; CADIC là baseline riêng. Không mang ý nghĩa C0 của tài liệu legacy sang bảng hiện hành.

Nguồn trạng thái: `stage1_implementation.md`, `v4_implementation.md`, `w1_w6_handoff.md`, `w7_implementation.md`, source liên quan và ZIP thực tế bên dưới. Nhiều đoạn status lịch sử trong instruction/planning vẫn ghi pending hoặc nhánh clone cũ; không coi chúng là snapshot implementation hiện hành. Chúng cần status-index riêng khi cập nhật tài liệu, không sửa âm thầm frozen dependencies của run.

## 3. Evidence hiện có và kết quả W7.1

| Evidence | SHA256 | Scope |
|---|---|---|
| `docs/w6_kaggle_v4.1/hope_cad_v4_normal_evidence.zip` | `ae9d257ecf1ced738df651fa07887b838a96da5fe4cdda84e46c51533c9bcd71` | Completed W6 v4 normal-only |
| W6 artifact manifest | `5892af759a096a6f2ba658eff5a3095aadc94c05bdb7c5849bfb6b122f37d4cd` | Parent identity pinned by W7 |
| `docs/w7_kaggle_preflight/hope_cad_w7_evidence.zip` | `ce11eae03c2b6b9d3ccb5212af6425626c1147d1dc06e1f09880541054bed0fd` | 130 regressions; real normal-RGB exact state/RNG/score resume |
| `docs/w7_kaggle_7_1/hope_cad_w7_evidence.zip` | `9bdf8089155e2c325e96057892d52db4d641391546680bc09d0bc765fd831e65` | 303 files; final AD results, 15 task checkpoints |

W7.1 suite digest: `17cd635b35e62484c5a000b4be4af503284e7d00da9aec79d64576c8d5055240`.
Run digest: `7bc6e394bd9849e45dc357559bcd2fb369f6a5535f4dddfca76a25fe3c312227`.

Đã kiểm: archive inventory và mọi entry hash; đúng online/test counts; unique identities; validation/probe loại khỏi stream; đủ task checkpoints; CMS pending giữ xuyên boundaries; final state bằng stream-resume state; state tensors finite; labels khớp paths; maps/masks shape224; không anomaly mask rỗng; tính lại đủ bốn metrics từ raw arrays của cả15 categories khớp dưới1e-12. Đây là kiểm tra integrity và arithmetic, không proof rằng mọi giả định thiết kế đều tốt.

| Metric | M seed0 final | Historical CADIC10k của Khang | Chênh lệch exploratory |
|---|---:|---:|---:|
| I-AUROC | 66.8010% | 87.72% | -20.9190 pp |
| P-AUPR (AP) | 17.0163% | 46.71% | -29.6937 pp |
| I-AP | 83.2417% | 94.77% | -11.5283 pp |
| P-AUROC | 81.1590% | 90.02% | -8.8610 pp |

Không gọi đây là paired/matched causal comparison. Tuy nhiên khác protocol không phải lý do đủ để phủ nhận độ yếu hiện tại.

Dấu hiệu: bottle93.97 I-AUROC; carpet95.55; leather99.08; wood96.67. Ngược lại capsule45.59, metal_nut43.70, pill44.00, screw6.89 và last-task zipper66.49. Zipper không có subsequent task nên **không thể giải thích toàn bộ bằng quên do các task sau**. Tại screw, median image residual của normal10.5791 cao hơn anomaly4.7077; đây là lỗi ranking thực đo, chưa định danh cơ chế gây ra.

Timing: train1043.23s; final cell scoring/IO1426.84s; notebook đến handoff2694.89s trước export. W7.2 missing-cell projection theo từng category là10126s (~2h49), chưa gồm toàn bộ metrics/loading/export overhead. Không dùng timing một M stream nhân ba rồi gọi là full35-stream suite.

## 4. Paper/repository checkpoint

| Nguồn | Điều hỗ trợ | Điều không được suy ra |
|---|---|---|
| [CADIC, III-B–D, Eq.1–9](https://arxiv.org/html/2511.08634v1#S3) | Frozen raw ViT patch geometry; shared incremental normal coreset; NN patch score và weighted image score | Paper numbers tự động là matched local result hoặc mọi triển khai tên CADIC đều identical |
| [NL/HOPE, §7–8](https://arxiv.org/html/2512.24695v1) | CMS có context/objective/frequency; task loss phải lựa chọn; cross-level initialization có vai trò | Clock separation tự chứng minh retention; bottle-only calibration đủ học mọi future category |
| [VisionHOPE, Appendix A.4](https://arxiv.org/html/2609.33325v1#A1.SS4) | Five within-image self-referential memories và stability control; thêm CMS trong vision cần objective bổ sung | Non-expansion là AD guarantee hoặc paper đã giải quyết chính objective CAD hiện tại |
| [Pinned VisionHOPE source](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/models/srnl.py#L18-L31) | Within-image state reset; numerical reference được pin | Upstream first-order core parity thay kiểm exact16 meta-gradient hoặc full CAD benchmark |
| [Reverse Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html) / [UniAD](https://arxiv.org/html/2206.03687) | Teacher discrepancy có tiền lệ; reconstruction shortcut là rủi ro đã được nghiên cứu | Normal MSE giảm luôn làm AUROC tăng; masking bất kỳ đủ chống shortcut |
| [ReplayCAD paper](https://www.ijcai.org/proceedings/2025/0328.pdf) / [official repo](https://github.com/HULEI7/ReplayCAD) | Replay có vai trò cụ thể đưa thông tin cũ vào learner | Hai clock loss current-task tương đương replay |

Local primary PDFs đã kiểm phần CMS §7.1–7.3 của NL (pp27–30) và VisionHOPE Appendix A.4 (p16), kể cả render trang phương trình. `refs/ReplayCAD` trong checkout đang rỗng: không tuyên bố đã audit code local reference không tồn tại. Chưa xác minh được một official released CADIC repository từ primary paper/search; repo baseline của dự án là implementation có compatibility assumptions, không gán nhãn author implementation.

Đã có published-literature exposure: trang paper/search có hiển thị các số VisA công khai. Không dùng chúng để chọn candidate hoặc threshold. Điều này khác với giữ local VisA test và model outputs chưa được dùng để chỉnh phương pháp; không tuyên bố chưa từng thấy bất kỳ con số VisA nào.

## 5. Findings có thể kiểm chứng

### F1 — Khả năng học category mới bị giới hạn ở CMS phía sau encoder/decoder đóng băng

Evidence: `training/hope_cms_session_v2.py::CMSSession.__init__/update` (lines13–27); `teacher_head.py::TeacherDecoder` (7–10); `hope_outer_learning_v4.py` role/episode validation và `episode_loss`; architecture v4 §1–2.

Đã đối chiếu mọi tensor `adapter.session.model` trong cả15 task checkpoints với initialization W6: tất cả bằng nhau. Chỉ current CMS state thay đổi. Vì vậy đây là intentional contract, không phải accidental missing optimizer.

Một decoder D đã cố định chỉ sinh được tập `R_D={D(h):h∈R^256}`. Với teacher target u, mọi CMS đều chịu `||D(h)-u||² ≥ inf_v∈R_D ||v-u||²`. Local Jacobian rank≤256; **không** được gọi nonlinear decoder này là một affine subspace256 chiều. Observation về chiều không chứng minh bất kỳ target thực tế nào không thể biểu diễn; nó xác định giới hạn CMS không thể tự sửa bằng cập nhật decoder.

Diagnosis cần: đánh giá normal reconstruction theo checkpoint/task đã thấy; so CMS-only với decoder-adaptation auxiliary trên normal train. Một local optimizer không tìm được latent tốt chỉ cho best-found error, **không** chứng minh global irreducible floor.

### F2 — Masked teacher prediction và normal-support discrimination là hai mục tiêu khác nhau

Evidence: `cms.py::masked_teacher_loss`; `benchmark_adapter_v4.py::score_batch`; `architecture_decision_v4.md` §5. Predictor dùng masked RGB; teacher là clean unit patch features.

Đặt Z là masked representation, U là clean teacher. Population squared-loss optimum là `f*(Z)=E[U|Z]` và:

`E||U-f(Z)||² = E tr Cov(U|Z) + E||E[U|Z]-f(Z)||²`.

Residual gồm ambiguity do corruption cùng model error, không chỉ khoảng cách tới normal support. Một normal phức tạp có thể khó dự đoán hơn anomaly có cấu trúc đơn giản. Vì vậy normal loss giảm không suy ra ranking AD cải thiện. Đây là suy luận toán học, không phải benchmark mới.

W6 normal probe: M L70≈0.145497; static C1≈0.090956; position-wise teacher mean≈0.043015. Position mean thấp hơn M trên aligned bottle không biến nó thành AD winner; bottle AD của M vẫn khá tốt. Các endpoint này không thể thay thế nhau.

### F3 — Norm của prediction và max pooling có thể chi phối image score

Evidence: teacher unit norm nhưng decoder output không normalize (`teacher_head.py:7–15`); image=max patch residual (`benchmark_adapter_v4.py::score_batch`). Đúng spec, không là implementation mismatch.

Với ||u||=1, r=||u_hat||, `a=r²+1-2r cos(theta)`. Score phụ thuộc cả norm và angle. Một patch normal output norm lớn đủ nâng score toàn ảnh. Screw high pixel-AUROC nhưng low image-AUROC tương thích với nghi vấn này; **chưa chứng minh attribution**.

Không đảo dấu score riêng category, search pooling hoặc normalize theo anomaly test để cứu bảng rồi gọi là unchanged v4.

### F4 — Hai timescale không bảo đảm hai loại kiến thức hoặc chống quên

Evidence: `cms.py` joint-final partials và mean accumulation; `predictor.py::dynamics` periods/rates. Cả hai levels đi theo final loss của current normal stream, không có old-task replay/retention constraint.

Nếu g gần cố định, period8 dùng mean gradient cho một step-beta mỗi8 events, tương đương first-order rate beta/8 mỗi ảnh. Vì vậy C4 có vai trò thực, không chỉ cho đủ ablation.

Fresh checkpoint measurement: relative current-weight drift fast/slow từ initialization cuối stream≈9.7653%/0.9718%; parameter norms quanh30.6, không gross norm explosion. Nhỏ drift không chứng minh function retention hoặc vô dụng của slow level.

W6 L70 full≈0.145497; freeze-slow≈0.145568; freeze-fast≈0.146432. Đó là first-category mechanism evidence, không cross-category theorem. Không hứa tăng số tầng hoặc scale sẽ tự làm contribution tốt lên.

### F5 — W6 gate kiểm một distribution, không kiểm deployment shift

Evidence: `training/hope_cms_stress_v4.py` repeats70 support/query normals đến1024 events; validation cùng bottle. Spec §7.1 công khai giới hạn này.

Gate PASS hợp lệ về finite/bounded same-domain behavior. Nó không đo acquisition trên cable/screw, hỗ trợ mọi category, hoặc duy trì performance qua15 tasks. Việc dùng gate này như bằng chứng “đã khá chắc sẽ ngang CADIC” là diễn giải quá mạnh, không phải kết quả paper.

### F6 — Concrete observability defect: online normal loss không được lưu

`benchmark_adapter_v4.py::fit_task` trả `mean_level_losses`, nhưng `training/hope_w7_engine.py:129` bỏ return value. Các records lines140–148 chỉ có task/count/cursor. ZIP thực tế xác nhận không có per-category normal learning curves của3599 events.

Cần một diagnostic sidecar ở revision vận hành riêng để ghi loss/gradient/update magnitudes; không đổi updates hoặc score. Không bịa lại đường loss đã không lưu và không sửa ZIP lịch sử thành evidence nguyên bản.

### F7 — Static/dynamic optimization khác nhau, không quy toàn bộ gap cho CMS

`hope_outer_learning_v4.py:119–135`: M tối ưu một query tại clocks0/16; C0/C1 mean17 support/query images. Lines185–191: M outer LR fixed1e-4, static search ba rates. Đây là thiết kế đã khai báo, không code bug.

W6: M selected short loss≈0.144900; C1 at selected3e-4≈0.088757; C1 at1e-4≈0.121505. Không thể kết luận khoảng60% normal-loss gap chỉ do meta-learning có hại; optimizer và signal budget là confounds. A1 (C1 initialization + online CMS) cùng các matched-recipe controls giúp phân biệt. Không sửa control hồi tố để M đẹp hơn.

### F8 — Baseline fairness có giới hạn nhưng không cứu được performance hiện tại

`conf/benchmarks/methods/cadic_matched_prefix_v2.yaml`: `exact_parity_claim:false`; checkpoint/preprocess/b/init/AP assumptions còn unresolved. Registry có explicit routes, unknown fail. `cadic_benchmark_adapter.py::fit_task` phân biệt extractionB8 với per-image write; `benchmark_metrics_v1.py:39–51` dùng sklearn AP cho P-AUPR.

Historical Khang numbers không bị phủ nhận. Cần matched CADIC10k để lượng hóa đúng gap; không đổi route, budget hoặc integration vì nhìn score HOPE.

### F9 — Chi phí suite có thể giảm hợp lệ nhờ loại duplication, không nhờ bỏ kết quả xấu

CADIC current algorithm/order/init/tie rules không phụ thuộc seed; ba canonical seeds cho cùng bank budget nên functionally redundant dưới deterministic runtime. Kiểm nguồn: `cadic_patch_coreset.py:127–157,387–449`; fixed loader `benchmark_protocol_v2.py:182–187`.

Proposal: declared experiment-plan revision có thể dùng một deterministic baseline/order/budget để pair với ba HOPE refits. Không giả làm ba baseline độc lập, không tự skip current35-job validator, không chọn đổi sau khi thấy seed nào đẹp.

### F10 — Engineering fix đã ghi nhận, không đổi khoa học

`notebook_support/w7_resume_fix/launcher.py` normalize proposed identity qua JSON rồi gọi original strict lock; verify119 frozen dependencies và archive helper/receipt. `docs/main_spec/transport_fixes/w7_json_lock_fix_v1.md` giải thích scope.

Đã đọc local artifacts:130 original regressions,10 focused tests, đủ15 checkpoint compatibility, original locks và final scores preserved. Không chạy lại suites trong audit này hoặc tự gán local fixture thành completed GPU matrix. Failure export/status cũng đã được sửa ở notebooks. Các lỗi vận hành này không giải thích metrics7.1.

## 6. Không có shortcut toán học để bảo đảm thắng CADIC

Thêm CADIC branch với weight0 cho HOPE chỉ chứng minh CADIC nằm trong hypothesis class, không bảo đảm selection normal-only sẽ chọn một model AD tốt hơn.

Counterexample: baseline scores normal0/anomaly1; auxiliary scores normal2/anomaly0. `max(baseline,auxiliary)` làm normal2>anomaly1, đảo một ranking đúng dù fused scores không nhỏ hơn baseline. Pointwise score lower bound không phải AUROC lower bound.

Normal-only validation hoặc conformal control của normal false-positive rate cũng không cho universal guarantee về ranking/TPR đối với anomaly chưa biết. “CI chứa0” không phải bằng chứng tương đương. Nếu mục tiêu là noninferiority, cần margins có ý nghĩa thực tiễn và quy tắc claim trước VisA; không chọn margin từ gap đã thấy.

## 7. Các hướng cải tiến được phép xem xét — toàn bộ là PROPOSED

### A. Tiếp tục HOPE predictor nhưng cho slow acquisition thực sự

Giữ frozen ViT và fast CMS; decoder (và nếu cần input projection) trở thành một level học chậm theo normal stream, kèm explicit retention bằng bounded past-normal replay/anchors. Không dùng task ID hoặc future categories để route/tune. Nếu encoder thay đổi thì cached inputs phụ thuộc encoder phải được invalidate/recompute.

Lý do: mở khả năng học teacher targets của category mới mà current CMS không đổi được ở head. Phải tính đầy đủ bytes của teacher/masked feature pairs/replay và optimizer, không so với CADIC bằng số entry không cùng đơn vị.

Falsifiable diagnostic: decoder-only auxiliary trên already-seen normal train cải thiện đáng kể normal reconstruction so CMS-only ở cùng budget; nếu không, không có lý do ưu tiên hướng này. Dù đạt, vẫn chưa giải quyết chắc chắn masked-predictability versus anomaly-support mismatch. Không tự gọi đây là fix bảo đảm ngang CADIC.

### B. Ưu tiên theo mục tiêu performance: Nested Learning cho normal-support memory

Giữ nguyên frozen raw ViT geometry thay vì bắt toàn bộ kiến thức đi qua bottle-trained decoder. HOPE/visual/CMS phải có nhiệm vụ cụ thể trong learned write/retention/consolidation; content lưu actual observed normal patch vectors trong common budget. Score dùng NN trên memory được policy đó tạo ra; không trộn một residual yếu vào CADIC để hy vọng tự thắng.

Một objective family có ý nghĩa CAD là causal prequential coverage và past-anchor retention:

`J(C;R) = mean_{r∈R} d(r,C)² + lambda_max * max_{r∈R} d(r,C)²`.

R chỉ gồm legal current/past normal observations; đánh giá observation mới **trước khi ghi nó vào C**. Self-inclusion tạo coverage giả. Anchors/controller/counters/optimizer/shadow states đều phải được tính memory/compute; không có free replay buffer ngoài budget.

Conditional geometric rationale, do audit tự suy ra: với finite historical normal set S, nonempty C⊆S và epsilon=max_{s∈S}d(s,C), triangle inequality cho:

`d(z,S) ≤ d(z,C) ≤ d(z,S)+epsilon`.

Proof: bound trái do C⊆S; lấy nearest s*∈S rồi nearest c*∈C, có ||z-c*||≤||z-s*||+||s*-c*||. Nếu normal nằm trong radius r0 của S và anomaly cách S ít nhất r1>r0+epsilon, patch distances có margin phân biệt. Đây là **giả định support-separation**, không đã được chứng minh trên MVTec/VisA. Coverage tốt hơn giảm bound approximation, không tự chứng minh AUROC hơn CADIC hoặc image Eq.9 hơn.

Trước freeze phải giải quyết: actual loss/targets cho controller; gradients hoặc delayed credit qua discrete insert/evict; nếu dùng soft surrogate thì ghi rõ khác hard execution; roles/frequencies/ownership từng level; causal eligibility của targets; total byte budgets; inference/reset; controls fixed policy/two-bank/reservoir/CADIC và learned-policy OFF. **Chưa có thuật toán đầy đủ hoặc implementation của B.** Một scalar gate cộng hai bank names không đủ Nested Learning. Hướng này giữ đúng đề tài nhưng khác pure predictor contract v4, cần revision rõ ràng.

### C. Phương án bằng CADIC theo construction: exact retrieval với learned scheduling

Giữ identical CADIC bank và Eq.8/9 scores. Learner NL chỉ ưu tiên vùng tìm kiếm; geometric lower bounds certify phần chưa search không thể thay nearest neighbors; chưa đủ certificate thì tiếp tục exact search. Eq.9 neighbor queries cũng phải exact, không chỉ pixel1NN.

Nếu chứng minh và kiểm exact equality, predictions bằng CADIC; câu hỏi nghiên cứu là CMS routing có giảm latency/energy so exact search/static routing không. Không được nhận accuracy gain của CADIC là gain do NL. Extra indexing/controller cost và fallback worst case phải đo. Hướng này là efficiency contribution có điều kiện, chưa implement hoặc có bằng chứng hiệu quả.

## 8. Thứ tự thí nghiệm đề xuất trước khi đụng VisA test

Đây là protocol đề xuất để review, **chưa là executable runner hoặc frozen recipe mới**.

1. **Hoàn tất v4 W7.2 unchanged.** Đọc diagonal acquisition, final row và trajectory. Tốt rồi giảm khác yếu ngay lúc học. Không sửa score theo category.
2. **Matched development controls nhỏ:** CADIC10k seed0; C1 seed0; A1 seed0. C1/A1 cùng ordinary initialization giúp xem online CMS có cải thiện hay làm hại trên cross-category stream; M/A1 xem meta-init, vẫn phải báo optimizer/objective confounds. Đây là registered jobs có thể ưu tiên, không có nghĩa full suite đã hoàn tất. Official MVTec results mới vẫn là development exposure.
3. **Normal-only checkpoint diagnostics:** deterministic identity-based samples lấy từ train normals của category đã đến tại checkpoint đó; lưu mean/tail loss, norm/angular components, per-level update magnitudes, past-normal retention và decoder/CMS sensitivity. Không dùng future category ở early checkpoint. Normals đã tham gia stream chỉ cho in-sample acquisition evidence, không giả làm independent validation.
4. **Bounded auxiliary tests:** CMS-only versus decoder-capacity diagnostic và current-only versus bounded historical-anchor retention; matched observations, optimizer steps, byte/read budget; không export forensic state thành main initialization. Kết quả không hội tụ không chứng minh representational impossibility. Tất cả thresholds/budgets/candidate count phải chốt trước chạy, không tăng liên tục để cứu giả thuyết.
5. **Chọn một revision dựa vào failure mechanism.** Nếu mục tiêu ưu tiên CADIC-level generalization, B đáng ưu tiên thiết kế; nếu evidence chỉ rõ acquisition head là nút thắt và support hypothesis tốt, A là nhánh sửa ít hơn. C chỉ dành mục tiêu equal accuracy + efficiency. Không launch cả ba rồi chọn best VisA.
6. **Freeze một candidate trước confirmation:** equations, modules, actual gradient path, source/runtime/data rights, scoring, seed/order list, selection, byte budgets, failure handling, primary metrics, practical margins, total trial cap và acceptance. Không thiếu các phần này rồi gọi spec ready.
7. **Pilot normal-only mới** đủ kiểm learning mechanism và full causal state, không chỉ finite. Chỉ đi anomaly evaluation sau khi có evidence đúng failure target; normal PASS vẫn không bảo đảm AD.
8. **Một lượt prospective VisA suite:** chạy đúng toàn bộ seeds/baselines đã định trước; không quyết định chạy tiếp chỉ khi seed0 đẹp.

Không cần full35-stream sweep của một phương pháp đang yếu mới được phép làm forensic; đồng thời không được bỏ controls bất lợi rồi báo full positive evidence. Hai yêu cầu này tương thích nếu ghi scope trung thực.

## 9. VisA protection và acceptance

- MVTec đã development-exposed. Revision mới dựa vào kết quả này không nhất thiết là direct training leakage, nhưng là adaptive benchmark development. Đổi method ID không làm sạch lịch sử exposure.
- Không dùng VisA anomaly images/masks/metrics để chọn architecture, rates, masks, score pooling, checkpoint, stopping, category-specific normalization hoặc best seed.
- Frozen v4 rule là reset model độc lập và fresh-prefix calibration trên first VisA task candle với transferred recipe, không tự gọi zero-shot transfer hoặc chuyển MVTec weights. Nếu chọn A/B/C cần ghi lại transfer rule trước test; không âm thầm giữ rule cũ khi thực tế đã khác.
- Cho causal online learner học normal images khi chúng đến là hợp lệ; nhìn tất cả future-task normals để meta-train rồi claim cùng first-task-prefix protocol là vi phạm. External normal pretraining, nếu có, phải độc lập và disclosure với baseline/data/compute fairness.
- Official split, square224 geometry, nearest-mask/nonzero handling, erased-tiny-defect policy và macro/AP integration phải chung mọi method, khóa trước score. QA không được drop category hoặc defect vì bất lợi.
- Định nghĩa cụ thể “bằng hoặc hơn”: exact parity, noninferiority với justified margins, hay joint superiority I-AUROC/P-AUPR. CIs conditional on fixed categories/trained seeds không chứng minh generalization mọi factory. Không coi nonsignificance là equality.
- Cần chứng minh NL có ích bằng learned-policy/dynamics controls tại cùng capacity/data/compute; fallback luôn trả CADIC chỉ là baseline parity, chưa là đóng góp Nested Learning.
- Nếu sau VisA test lại sửa method, VisA trở thành development cho revision đó; muốn confirmation cần dữ liệu chưa dùng để sửa hoặc ghi exploratory scope. Không hứa có thể vừa dùng VisA liên tục vừa giữ nó untouched.

## 10. Quyết định audit hiện tại

**Không khuyên từ bỏ tên đề tài. Không khuyên bảo vệ mọi lựa chọn v4 chỉ vì đã freeze. Không khuyên một lần nữa tăng steps rồi kỳ vọng tự thắng CADIC.**

Giá trị đã có là một implementation/test/provenance nền tảng tốt và một negative AD result có thể truy nguồn. Để đạt mục tiêu performance mới, cần chuyển từ “operator đúng, hy vọng residual hữu ích” sang “learning objective bám normal support, acquisition/retention có bằng chứng, NL tạo ảnh hưởng thật tới quyết định cuối”.

Chưa thay method, learned weights, metric, baseline route, source ZIP hoặc 7.2 đang chạy trong lượt audit. Không có v5 đã chốt hay benchmark gain mới. Các proposal chỉ được thăng trạng thái sau khi có spec hoàn chỉnh, tests và actual evidence theo từng scope.
