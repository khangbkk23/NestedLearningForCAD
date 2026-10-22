# D0 — Đặc tả attribution offline, bản 2

ID: `support_utility_attribution_v2`. Thuộc [bundle v7](README.md). Thay bản D0 v1 cho implementation mới, giữ v1 làm lịch sử. **Analyzer chưa được triển khai hoặc chạy trong repo hiện tại.** D0 không gọi model, GPU hoặc bộ đánh giá anomaly chính thức.

## 1. Input binding và allowlist

Input root do người chạy cung cấp ở ngoài source checkout; không có đường dẫn máy cá nhân bắt buộc. Hai ZIP dưới đây **không được phát hành kèm repo**. Chỉ chạy D0 khi nhận được đúng archives và kiểm SHA; thiếu chúng phải báo thiếu input, không tự dựng số liệu. Input allowlist:

1. NESTED ZIP `<input-root>/hope_cad_matrix_NESTED_seed0_evidence.zip`, SHA256 `668f9e348561d5ec55c4cf51762c5890b19328c2a950c7ba2fb09f037b1bea0d`.
2. CADIC ZIP `<input-root>/hope_cad_matrix_CADIC_seed0_evidence.zip`, SHA256 `035f88639f8faab8bf02fdfbcd5f502677620c3756af53dcf4815dd40e8cc889`; chỉ đọc normal manifest, events và train completion tương ứng khi cần kiểm provenance hoặc số lượng train records, không coi đây là counterfactual trên cùng bank.
3. Số liệu pilot/provenance được tóm tắt trong [kết quả và giới hạn](../results.md), và [cơ sở thiết kế](rationale.md). Đây là ngữ cảnh diễn giải, không là input bắt buộc để fit/scoring D0. Hai archives đã khóa ở trên là input số học chính; không cần truy cập review riêng của người phát triển.

Trong NESTED ZIP, canonical member paths là:

```text
artifact_manifest.json
benchmark/manifest_train.json
benchmark/runs/NESTED_seed0/events/{cursor:05d}.json   # cursor 0..3598
benchmark/runs/NESTED_seed0/train_complete.json
```

CADIC dùng cùng dạng với `CADIC_seed0`. Không glob mọi `/events/`: ZIP NESTED còn **376 legacy events** ở `benchmark/legacy/runs/NESTED_seed0/events/`; chúng không vào bảng phân tích chính. Canonical namespace có 3.599 events; đọc cả hai namespace tạo 3.975 rows là sai. Chỉ được đếm tên legacy members để ghi nhận namespace bị loại, không đọc nội dung hoặc dùng để lấp event thiếu.

Kiểm ZIP SHA bằng streaming; kiểm path safety và duplicate filenames trong central directory; đối chiếu manifest map của **mọi member đã dùng**. Đọc UTF-8 JSON có giới hạn, từ chối duplicate JSON keys và giá trị không hữu hạn. Không `extractall`, pickle, `torch.load` hoặc truy cập test scores/maps, GT, ảnh hay ViT. Chỉ liệt kê tên `.pt` để kiểm kê sự hiện diện của state; không giải mã tensors. Các norm values từ review cũ không thay thế neural probe.

Giới hạn uncompressed: artifact manifest tối đa 32 MiB; mỗi event 1 MiB; train manifest/completion 16 MiB; tổng event bytes được đọc 256 MiB. Vi phạm thì xuất failure, không tự tăng limit hoặc bỏ record. Chỉ đọc member theo allowlist; filesystem input phải resolve tới archive đã khóa hoặc immutable receipts đã chọn. Không download hoặc thay input.

## 2. Event schema và chronology bắt buộc

Đối chiếu mỗi canonical event với `manifest_train.online[cursor].relative_path`; toàn bộ 3.599 cursors phải liên tục, identity duy nhất, không chọn bản đầu hoặc cuối để deduplicate. Lưu member SHA và input manifest SHA. Trường `credit` trong actual event có thể tồn tại ngay cả khi **current event là no_change**.

Event decision lấy: cursor, identity, action, accepted, warmup, prediction, gamma, c, current_minus/plus, anchor_minus/plus, changed_rows và proposal counters. Bootstrap/no_change không có đầy đủ decision fields; giữ `null` đúng nghĩa.

Credit lấy: decision_identity, cursor, reward_observed_at, Jminus/Jplus, next_minus/plus, anchor_minus/plus, c, gamma, target, factual prediction/denominator và accepted. Join bằng identity/cursor với đúng decision; kiểm credit được quan sát tại `decision_cursor + 1` và khớp factual scale/action. Không tạo credit giả cho no-op hoặc pending bị censor.

Thứ tự tại mỗi **image cursor t** phải đúng source `session.observe`:

```text
đọc và kiểm event t
nếu có credit của pending decision cũ:
    yêu cầu cached predictions đúng identity/cursor; chưa từng scored/update
    score các cached predictions với label vừa đến
    update đủ các baseline states bằng credit này
    clear pending predictions
nếu event t có full changed-proposal decision:
    predict các baseline bằng states SAU bước resolve ở trên
    cache predictions/features cần update khi label tương lai tới
    chưa đọc/fit target của chính decision t
ghi tiến độ cursor và metadata
```

`reward_observed_at<=t` là cutoff hợp lệ cho prediction mới; credit tại chính t đã được resolve trước decision t. Không áp “predict trước mọi update trong event” khiến baseline chậm một label. Nếu dùng join retrospective, predictor function phải nhận view có cutoff, không cả bảng targets.

Expected counts đối với archive này: 3.599 events, 3.211 changed decisions, 3.211 completed credits, 32 warmup decisions, 3.179 postwarm decisions/credits và 1.935 rejects. Event 3598 là no_change, nên terminal pending censor = 0. Những con số này là QA identity của dataset đã chọn, không là rule cho mọi run. End state không còn pending; nếu dùng archive khác phải có protocol/version riêng, không lách expected counts.

QA scalar: hữu hạn, c >= 1; `target=(Jminus−Jplus)/c`; J là next risk cộng cached anchor risk. Tolerance để **đối chiếu recomputation**: `abs(a−b)<=1e-12+1e-10*max(abs(a),abs(b))`. Không dùng tolerance để đổi sign/zero của stored targets. Không clip hoặc ghi lại raw values. QA thất bại thì dừng và giữ thông tin lỗi, member, cursor.

## 3. Predictor recipes duy nhất

Mọi prediction là utility scalar, không phải probability. Main metrics chỉ dùng postwarm completed records, với cùng identities cho mọi arm. Chạy updates từ **label 1**, bao gồm 32 warmup labels. Không reset theo task, không lấy category làm feature hoặc fit key. File identity chỉ dùng để join và logging.

| ID | Prediction tại decision t | State/update |
|---|---|---|
| OBS-NESTED | factual `prediction` | Không forward/rerun learner |
| OBS-GAMMA | gamma | Không fit |
| OBS-MEAN | mean past y | sum(y), count |
| OBS-GAMMA-BIAS | gamma+mean past(y−gamma) | sum(y−gamma), count |
| OBS-AFFINE-GAMMA | beta·[1,gamma] | cumulative ridge, phương trình dưới |
| OBS-AFFINE-CLOCK | beta·[1,gamma,q] | q=log1p(t)/log1p(1e6) |
| OBS-NLMS-BIAS | gamma+b | b0=0; update `b←b−.05*(cached_prediction−y)` |

**Fallback khi chưa có 32 past labels chỉ áp dụng cho bốn statistical fitted arms MEAN/GAMMA-BIAS/AFFINE-GAMMA/AFFINE-CLOCK**: output gamma nhưng vẫn cập nhật sufficient statistics từ mọi label hợp lệ. OBS-NLMS-BIAS output gamma+b **ngay từ decision đầu**, không fallback và không có hai prediction khác nhau cho nội bộ/output. Khi credit đến, cached prediction dùng cùng b chưa nhận thêm label khác; update không dùng factual NESTED p.

Ridge khóa trên **sum loss**, không mean loss:

`S=Σ x xᵀ`, `v=Σ x y`, `beta=solve(S+1e-6*diag(0,1[,1]),v)`.

S/v chỉ dùng labels đã quan sát; x là feature tại decision gốc, không dùng current gamma/clock để update label cũ. Float64, intercept không penalty; không normalization fit từ tương lai, không hyperparameter search. Solver failure/nonfinite xuất failure; không đổi lambda hoặc drop columns âm thầm.

Action-only ALWAYS_ACCEPT/ALWAYS_REJECT chỉ vào action metrics, không bịa prediction để tính MSE. Không per-task intercept/main random row split. Các phép oracle hindsight nếu cần thuộc task riêng, không đổi frozen D0 này.

## 4. Metrics đủ định nghĩa

Cho mỗi completed postwarm row: `Delta=Jminus−Jplus`, `a=1[p>=0]`. Prediction zero nhận proposal. Ground-truth zero là **stored Delta==0 chính xác**, không dùng epsilon. Nonzero sign agreement là `1[(p>=0)==(Delta>0)]` chỉ khi Delta != 0. Đặt `prediction_error=p−y` và `learned_residual=p−gamma`; trường `credit.residual` trong source có nghĩa là `prediction_error`, không phải `learned_residual`.

- Target MSE=`mean((p−y)^2)`; MAE=`mean(abs(p−y))`.
- Raw advantage MSE=`mean((c*p−Delta)^2)`; không gọi là full bank risk.
- Realized fixed-pair one-step regret=`max(Delta,0)−a*Delta`, report mean/sum. Oracle biết realized next-image label; khác Bayes oracle hoặc full AD policy.
- Accept fraction, sign agreement, zero target fraction; means và quantiles của target, prediction, gamma, prediction_error và learned_residual.
- Cancellation diagnostic trên gamma != 0: `cancel=abs(p)/abs(gamma)`, report quantiles; gamma == 0 thì report count riêng. Không áp ratio khi denominator = 0 hoặc gọi ratio nhỏ là numerical error. Pearson dùng mô tả cho p/y/gamma/clock, với null nếu n < 2 hoặc variance = 0.

Main aggregate là **micro mean trên 3.179 rows**. Category breakdown và macro-category mean ghi trường riêng, không tráo micro/macro. Annotation category lấy từ manifest chỉ sau predictions, không dùng để reset hoặc selection. Time bins cố định `[256k,256(k+1))` theo image cursor, không chọn biên từ lỗi.

Margin bins |p|: [0,1e-4), [1e-4,1e-3), [1e-3,1e-2), [1e-2,.05), [.05,.1), [.1,+∞). Báo count, mean p, mean y, nonzero count, sign errors, Σ|Delta| và regret. Không gọi utility regression là probability calibration. Mẫu số bằng 0, bin rỗng hoặc correlation không xác định → `null` kèm reason; không NaN hoặc fabricated 0.

Mọi quantiles trong D0 dùng probabilities cố định **[0, .25, .5, .75, 1]**, với linear interpolation của ordered samples (NumPy `method='linear'` hoặc oracle tương đương). Không bootstrap dependent credit rows như independent seeds. Prediction quality trên logged pairs không xác nhận policy AD tốt hơn vì đổi action sẽ đổi bank và distribution.

Proposal components join đúng credit: changed_rows, proposal replacement ops, committed changed_rows=(accepted?changed_rows:0), next advantage, anchor advantage, current advantage, y và Delta. Report counts, quantiles và Pearson theo all/accepted/rejected/warmup/postwarm; các nhóm chỉ dùng để mô tả, không fit. Không gọi rejection rate là fraction writes hoặc runtime savings.

## 5. Budget, output và restart

Một deterministic D0 attempt: tối đa 2 CPU threads, target RAM 2 GiB, soft budget 30 phút **tính từ khi bắt đầu kiểm input, gồm hashing/parse**, dự phòng 30 giây để export. Trước khi bắt đầu cần đủ disk cho output; không copy archives nhiều GB. Limits là guard/cap, không phải ETA. Một pass canonical events cho mỗi input; hash ZIP bằng streaming trong một pass riêng.

D0 nhỏ nên **không true-resume**: nếu pause/interrupt, giữ original partial session và rerun deterministic từ 0 trong session mới với cùng inputs/spec/source. Không append vào partial cũ. Không gọi rerun này là trial/hyperparameter mới. Tối đa 2 attempts cho operational failure của cùng recipe; nếu lỗi lặp lại, báo diagnostic bị chặn, không tự tăng budget hoặc đổi analysis. Scientific negative không được chạy lại bằng filter khác.

Output root do người chạy chỉ định, ví dụ `runs/support_utility_attribution_offline_02/session_NN` (bị Git ignore): input_receipt.json, record_qa.json, prequential_predictions.jsonl, metrics.json, margin_bins.json, proposal_components.json, state_availability.json, resources.json, limitations.md, handoff.json và artifact_manifest.json. Không ghi raw test outputs. Mỗi row có evaluation flag, source member/hash và prediction của từng arm; predictor state không chứa targets của future rows.

Atomic publication: write temp outputs → flush/replace; manifest chứa SHA/bytes của các payload khác manifest; temp files không là authoritative. `handoff` có trạng thái complete_pending_review/paused_preserve_partial/failed_preserve_evidence; complete không tự là acceptance. Tránh self-referencing manifest/handoff hash. Acceptance do task review riêng kiểm QA, counts và output completeness, không chỉ exit code 0.

## 6. D0 acceptance và giới hạn

Tests khi implement: allowlist loại legacy; duplicate/missing cursor; label tại cùng cursor đến trước decision; no-op mang credit; factual-pair join; full-trace future trap; statistical fallback khác bias; sum-ridge oracle; zero tie/empty bins; partial deterministic restart; corrupt member SHA; không truy cập model/test data. Reduced fixture đủ cho correctness, không gọi synthetic result là AD gain.

Pass kỹ thuật: input bindings, QA, counts, toàn bộ predictions và metrics đầy đủ; finite hoặc explicit null; temporal tests đạt. Kết quả predictor thua hoặc inconclusive vẫn là kết quả hợp lệ; không có threshold cần vượt để script “PASS”. D0 không thể bác mọi H32 revision hoặc chứng minh learned context vô ích. D0 không chọn history capacity, query ratio, LR hoặc clock ablation từ test.

Với việc cần neural state/forward — pre/post slow Δp, frozen CMS, exact WNR thiếu provenance — chỉ báo state availability, không được mở rộng ngầm phạm vi D0. Candidate và study đã có spec riêng nên không cần thêm một vòng tìm paper hoặc thiết kế lại sau D0.
