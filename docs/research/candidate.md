# H32 — Candidate học hợp nhất chậm bằng historical queries

Method ID đề xuất: **`hope_cad_hq32_fulltrain_v1`**. Trạng thái: `specified_experimental_candidate_not_implemented_not_adopted`.

Đây là một adaptation của Nested Learning cho CAD, chưa phải phương pháp chính đã được chấp nhận. Spec xác định đủ thuật toán để triển khai thí nghiệm có giới hạn; nó không tái lập đầy đủ HOPE và không bảo đảm thắng CADIC. Các lựa chọn bên dưới được khóa trước thí nghiệm. Phương pháp v5.1, contracts và toàn bộ evidence đã có giữ nguyên.

## Phạm vi dữ liệu của bản công khai

Candidate này dùng fresh full-train **3.629 normals, Bottle 209, không exclusions**, cùng CADIC và recent-only controls; xem [pipeline contract](../pipeline_alignment_contract.json) và [study](study.md). Không nhập checkpoint 3.599 cũ. Đây là tích hợp quyết định full-train đã chốt, không thêm một thay đổi thuật toán H32 mới. Contract nghiên cứu [contract.json](contract.json) chưa được runtime CLI nạp; chỉ các contracts hiện hành trong `src/hope_cad/resources` đang chạy.

## 1. Những thành phần được kế thừa nguyên vẹn

Base design digest của v5.1: `6f6b2c847a815619b6ad5ed899a0859aab1d7b532d62b27ae3217d926695db99`. SHA256 của `method.json` gốc: `b3e5004d618ac1a39ffe3eab36b0b8056fd53d0de611c5b9f3b33f6313f3d3a1`. Không ghi đè hai identity này để hệ thống nhận H32 như thể vẫn là v5.1.

Trừ các thay đổi được định nghĩa rõ trong spec này, H32 kế thừa toàn bộ equations, constants, data rights, numeric routes, failure rules và state semantics của v5.1. Đặc biệt giữ:

- Preprocessing, checkpoint, block trích xuất và không gian raw ViT features.
- Projection 32D, 16 geometry statistics, visual memory bốn hướng, topology của `CMS_fast`, `CMS_slow` và readout.
- Ownership fast/slow, zero initialization của fast head, warmup 32 completed labels; normalized fast learning rate 0,05; slow Adam learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8` và global gradient clipping 1.
- Neural tensors FP32, scalar arithmetic FP64 và các derivative bridges đã khai báo; không AMP, TF32 hoặc precision route mới.
- Admission `p >= 0`, bank K = 10.000, CADIC proposer, full Eq9 với b = 9, interpolation và mask policy.
- Reservoir 32 anchor images × 8 uniform patches, fixed-pair delayed label và causal scale ban đầu.
- Không task reset, không học từ scoring, không backpropagate qua hard bank choices.

Clock tại statistic index 12 được giữ nguyên; candidate này không mask clock hoặc thay nó bằng relative clock. Clock là causal input có rủi ro shortcut, không tự động là data leakage. Normalization mismatch được công bố nhưng không sửa trong cùng revision. Không thêm NSP2, CBP, ContCore, anchor policy mới, width mới, learned learning rate hoặc rate search.

Thay đổi khoa học tập trung vào **ngữ cảnh query của slow objective**. Bookkeeping, schemas và diagnostics mới phục vụ thay đổi đó; các controls được nêu riêng ở §7.

## 2. Objective và fast/slow semantics

Block b, đánh số từ 0, gồm tám completed labels có indices `8b+1 … 8b+8`; gọi tám current records là `r0 … r7`. Fast state ngay trước actual update của `r0` được lưu thành detached `phi_start`. Actual fast learner tiếp tục học cả tám records theo v5.1, detach sau từng update; trạng thái thực cuối block là `phi8`. Theta không đổi trong block cho tới slow commit.

Functional replay bắt đầu từ một clone detached của `phi_start`:

`phi_(i+1) = phi_i - 0.05 * partial_phi[0.5 * (p(theta, phi_i, r_i) - y_i)^2] / d(theta, phi_i, r_i)`, với `i = 0 … 3`.

Support chỉ gồm `r0 … r3`. Query list được định nghĩa chính xác:

- Block `b = 0`: `[r4, r5, r6, r7]`. Startup fallback này giữ nguyên bốn recent queries và reduction order của v5.1.
- Block `b >= 1`: `[r6, r7, h0, h1]`. `h0`, `h1` được chọn từ snapshot lịch sử ở đầu block. **Giữ thứ tự mà private `random.sample` trả về**; không sort theo tuổi, category, utility hoặc loss.

Slow objective:

`L_outer = mean_{q in query_list} 0.5 * (p(theta, phi4, q) - y_q)^2`.

Bốn query losses có trọng số bằng nhau và được stack/reduce theo đúng query order trên. Không tiếp tục fast adaptation trên các query records. Exact meta-gradient giữ derivative của cả numerator gradient và denominator trong quotient `g/d` qua bốn support updates. Các `tokens`, `stats`, `c`, `gamma`, `target` ban đầu là detached data. Encoder output và denominator được tính lại dưới theta và functional phi của replay; không dùng stored factual `decision_d` làm replay denominator.

Adam và clipping chỉ cập nhật theta như v5.1. Sau slow commit, giữ actual `phi8`: không deploy functional `phi4`, không reset/rebase fast state và không meta-learn fast initialization. Historical replay không tăng image clock, completed-label clock hoặc actual fast-update count. Mỗi block tám labels chỉ có một scheduled slow update.

Sự tồn tại của query cũ chưa chứng minh khả năng học qua nhiều category hoặc giữ kiến thức trong bank hiện tại. Historical targets vẫn mô tả original bank pair và có thể chịu covariate shift hoặc context aliasing; chúng không phải reward của bank hiện tại.

## 3. History record và quyền dữ liệu

Mỗi immutable `HistoryRecord` có envelope:

```text
completed_label_index: positive integer
reward_observed_at: image cursor mà delayed label thực sự được quan sát
record: original v5.1 numerical record
payload_sha256: digest của metadata và tensor identity
```

Original record giữ nguyên `tokens[1,784,32]` FP32, `stats[16]` FP32, `c`, `gamma`, `target`, `Jminus`, `Jplus`, `decision_d`, `decision_identity` và `decision_cursor`. Tensor payload được detach và clone một lần khi tạo record, sau đó bất biến. Wrapper không đưa identity, category, label index hoặc observed time vào model features.

Định nghĩa `payload_sha256`: gọi `tensor_digest([{'tokens': tokens, 'stats': stats}])` hiện hành; đưa tensor digest đó cùng toàn bộ non-tensor record fields, `completed_label_index` và `reward_observed_at` vào `contract.digest` theo canonical JSON hiện hành. Không đưa chính trường `payload_sha256` vào dữ liệu cần hash. Không băm pickle, đường dẫn file hoặc một bản JSON có NaN.

Stored `decision_d` chỉ phục vụ provenance và kiểm tra factual update theo v5.1. Không yêu cầu nó bằng denominator mới khi replay record lịch sử.

Chỉ tạo envelope sau khi delayed label thật đã đến. Kiểm target arithmetic, topology, finite values và `reward_observed_at == decision_cursor + 1`. Không tạo fake-zero record cho bootstrap, no-op hoặc censor. Không lưu anomaly labels, GT, test features hoặc future normal data. Các record là duy nhất theo decision identity, decision cursor và completed-label index.

History có tối đa **32 records**, không chia theo task, khởi tạo rỗng. Tám current records chưa commit không thuộc history. Current block và history không có cùng decision IDs. Capacity 32 không bảo đảm đại diện mọi category; không đặt quota category hoặc chọn tuổi record từ evaluator.

## 4. Reservoir và hai private RNG

- Reservoir RNG: Python `random.Random(seed + 96136)`.
- Query RNG: Python `random.Random(seed + 96137)`.
- Hai RNG tách khỏi global Python RNG, Torch RNG, anchor RNG và projection RNG. Query sampling không thay đổi reservoir replacement hoặc model initialization.
- Python/runtime version thuộc identity của cohort; không đổi RNG implementation/version rồi resume dưới cùng identity.

Sau một full block đã slow-update thành công, đưa tám envelopes mới vào reservoir theo completed-label order. Với record mới thứ `n = history_seen + 1`:

```text
n <= 32: append vào slot n-1; không draw reservoir RNG
n > 32:  j = reservoir_rng.randrange(n)
         nếu j < 32, thay slot j bằng record mới
history_seen = n
```

Đây là Algorithm R uniform trên các completed records đã commit, không trên image cursor hoặc proposal count. Không rank theo sign, utility, category hoặc loss; không bỏ các zero targets.

Khi completed label đầu tiên của block tới, **trước actual fast update**, snapshot eligibility của toàn committed reservoir. Với block 0, selection rỗng và dùng startup fallback; không draw query RNG. Với block sau, gọi `query_rng.sample(range(len(buffer)), 2)` đúng một lần. Lưu hai selected record IDs, slot indices tại thời điểm sample, payload hashes và post-draw RNG state. Giữ thứ tự hai indices mà `sample` trả về.

Buffer không thay trong partial block. Selected queries chỉ là **immutable references** tới payload trong buffer; không sao chép thêm token tensors để tạo một selected-query buffer thứ hai. Resume dùng selection đã lưu, không resample. Forward hoặc replay query không advance RNG.

Từ block 1, valid history occupancy luôn ít nhất 8. Occupancy 1–7 không thể xuất hiện trong valid state của chính sách này; phải fail thay vì thêm fallback mới. Eligibility dựa vào thời điểm **label đã được quan sát trước first-credit event của block**, không chỉ decision cursor. Snapshot của block b chỉ chứa completed-label indices không lớn hơn `8b`.

## 5. Event order, transaction và true stream end

Giữ session order của v5.1: tại image t, resolve pending label trước; sau đó mới dựng proposal, predict và commit bank; cuối cùng cập nhật anchors. Trong learner, khi credit mới tới:

1. Validate factual record chronology và factual `decision_d` như actual path cũ.
2. Nếu current block rỗng, lưu detached `phi_start`, completed-label cutoff `8b` và first-credit observed cursor; snapshot/sample history theo §4.
3. Chạy actual fast update trên credit mới; tăng completed-label count L và append current envelope.
4. Nếu đủ tám records, tính slow objective bằng support/query đã định nghĩa; kiểm finite, autograd, clipping và Adam; giữ actual `phi8`. Khi đến lịch probe/debug, đo pre/post theta và chụp bounded query payloads trước khi insert history có thể thay selected slots; selected references còn hợp lệ suốt bước này.
5. Chỉ sau khi slow update thành công mới insert cả tám records bằng Algorithm R, ghi episode provenance rồi clear current records, `phi_start` và active selected-query references.
6. Commit toàn image event. Nếu bất kỳ bước nào fail, rollback cả bank, pending, anchors, observed IDs, cursor, counters, theta, phi, Adam, history, hai RNG và episode metadata.

Không mutate in-place tensor payload cũ hoặc reservoir đang được snapshot trước khi transaction commit. Có thể dùng ownership/reference transaction hoặc copies có kiểm chứng, nhưng không đổi semantics để giảm I/O. Checkpoint/event publication tiếp tục atomic như harness hiện hành. Hard crash sau một update nhưng trước durable checkpoint phải replay từ last valid state; không được insert history hai lần.

Scoring chỉ đọc actual bank; không đọc history để đổi scoring, không advance RNG hoặc update learner. Task boundary chỉ checkpoint/evaluate, không flush block/history hoặc reset clock. True stream end censor pending nếu có, giữ partial block, không insert partial records hoặc chạy thêm slow update. Không invent next image. Repeated finalize phải idempotent.

## 6. State schemas và validation

Successor schemas: session `hope_hq_session_v1`, learner `hope_hq_learner_v1`, history `completed_utility_history_v1`. Method/variant, study/source/design digests và các schemas phải được bind vào job identity. Không nhận predecessor v5.1 hoặc T4 checkpoint bằng một flag bỏ guard. Old v5.1 loading path tiếp tục strict.

`R_EXACT` giữ learning equations của v5.1 nhưng mang successor data contract `hope_cad_v5_1_fulltrain_v1` và execution wrapper có history metadata; nó không được relabel thành H32. Wrapper schemas/variant identity phải công bố rõ. New study tách per-job method identity khỏi common cohort identity theo [study](study.md).

Đặt `L = completed_labels`, `r = L % 8`, `B = floor(L/8)`. Với H32 exact/FO/direct và recent matched variants có Adam mỗi block, validation bắt buộc:

- `actual_fast_updates == L`; `actual_slow_updates == B`.
- `len(current_records) == r`; `history_seen == 8*B`; history occupancy bằng `min(32, 8*B)`.
- History label indices thuộc `[1,8*B]`. History decision IDs, cursors và label indices đều duy nhất. Current partial indices đúng `8*B+1 … L`, theo chronology và không overlap history.
- Khi `r == 0`, active `phi_start`, block cutoff và selected-history references đều absent. Khi `r > 0`, chúng hiện diện, cutoff bằng `8*B`; selected history rỗng iff `B == 0`, ngược lại có đúng hai distinct IDs trong buffer bất biến.
- Reward-observed time của mỗi selected history record nhỏ hơn first-credit observed time của block. Payload hash, topology và target arithmetic phải đúng. Selected IDs tồn tại và không trùng support hoặc current recent queries. Stored slot indices còn trỏ tới đúng immutable IDs trong partial block.
- Hai RNG states được validate bằng temporary owners. Load hoàn chỉnh vào một owner mới rồi mới commit; không suy RNG state chỉ từ seed, redraw selection hoặc sửa state để “khớp”.
- Fast template vẫn frozen; theta/phi ownership, shapes và dtypes giữ theo base. Adam slots, actual step counts và tensor finite checks phải khớp. History/sampler counts không thay thế actual optimizer clocks.

Tại checkpoint, active selected-query references không có tensor copies riêng. Sau block commit, retained debug episode có thể sở hữu bounded copies phục vụ reproducibility; phần đó phải được khai báo riêng trong memory accounting.

Saved meta episode/probe phải ghi **query records thực sự cùng nguồn recent/history/fallback của từng query**, không chỉ tám current records. Giữ một bounded debug episode tại completed-label count 40, gồm pre-commit model theta, `phi_start`, support 4, query 4, selected IDs/RNG metadata và flags. Không dùng debug copies như unbounded history hoặc tự chạy offline probes. Task checkpoints, debug copies và transaction snapshots đều phải được tính bytes.

## 7. Matched controls và trường hợp zero gradient đầu tiên

`R_EXACT` và `R_FO` giữ history capacity 32, Algorithm R, query sampling và bookkeeping như H32 để giảm storage/sampler confounds, nhưng query loss luôn dùng `[r4,r5,r6,r7]`; selected historical references vẫn được ghi là sampled-but-unused. `R_EXACT` phải khớp scientific theta, phi, actions, bank và score của v5.1 trên parity fixtures khi loại metadata mới khỏi phép so. Private Python RNG mới không được làm đổi scientific RNG. Không yêu cầu full state bytes bằng nhau vì schemas/history metadata khác.

`H32_FO` giữ cùng forward values và query loss như H32 exact, nhưng detach **toàn increment `g/d`**, không chỉ g. `H32_DIRECT` giữ cùng query list nhưng evaluate tại detached `phi_start`, với **zero functional support steps**. Actual tám fast updates vẫn diễn ra; đây là direct-query training control, không phải “không fast learning” và không có compute ngang exact meta-learning.

Ở block 0 của DIRECT, zero readout làm slow theta gradient bằng 0. Đây là finite zero gradient đúng dự kiến, không phải disconnected error và không cho phép thay initialization, warmup hoặc head ownership. Nếu thực sự gọi Adam với finite zero gradients, step đó vẫn tăng slow-update count. Từ block sau, actual fast head có thể truyền theta gradients. Không ép exact/FO bốn support steps phải có first-block zero gradient. Mọi variant phải báo zero-gradient frequency; không chế nonzero gradient hoặc cứu control bằng thay thuật toán.

`FAST_ONLY` không có slow optimizer. Nó giữ actual fast updates và completed blocks; nếu được dùng trong study mới, history capacity 32 và sampling bookkeeping vẫn được thực hiện để công bố storage. Với arm này, insert history sau mỗi tám successful actual fast updates, không giả một slow Adam step; `slow_updates` luôn bằng 0. Các state invariants tại §6 phải dùng policy-specific expected optimizer counts, không gán `slow_updates == B` cho FAST_ONLY.

BIAS/RANDOM có recipes riêng trong study, không được gán trainable CMS. `NO_FAST_ADAPTATION` của old study không được gọi là trainable slow-only baseline.

## 8. Resources, diagnostics và failure policy

Token payload của 32 history records là `3,211,264 B`, khoảng `3.0625 MiB`; tám current records là `802,816 B`, khoảng `0.7656 MiB`. Selected immutable ID references không nhân bản token storage. Bất kỳ implementation nào lưu thêm hai selected contexts sẽ tốn thêm `200,704 B` và vi phạm ownership đã chọn; phải sửa implementation, không âm thầm cập nhật resource claim.

Tổng accounting còn gồm stats, metadata, bounded debug episode, pending alternative bank, anchors, Adam, pair caches, transaction backups và CPU/GPU/I/O peaks. Không dùng same K hoặc con số 3,06 MiB để claim equal total memory.

Mỗi meta-update ghi support/query IDs, source của từng query, observed cutoff, query ages, fallback, recent/historical losses, exact/FO/direct flag, theta gradient norm, parameter-step norm, clipping, history count/slot IDs, RNG digests và actual counters. Query age tính theo completed-label index. Category chỉ là evaluator annotation hậu kiểm; không gọi mọi historical query là cross-category.

Tại debug episode completed labels 40 và mỗi 32 successful slow commits có index dương, đo read-only predictions trên **cùng bốn queries** dưới theta trước/sau slow update, với **cùng actual `phi8`**. Báo mean/max absolute prediction change. Probe dùng `no_grad`, không advance RNG hoặc optimizer, không đọc test/GT. Đây là phép đo thay đổi function khi tọa độ representation đổi, không phải tự động alarm hoặc căn cứ chọn learning rate. Các recent/history arms có slow optimizer dùng cùng diagnostic clock; không double-execute nếu hai điều kiện trùng nhau. CADIC không có utility probe. FAST_ONLY không có before/after slow commit, nên trường này là not-applicable có lý do, không fabricated zero; không chạy periodic probe chỉ vì `slow_updates == 0` chia hết cho 32.

Nonfinite values hoặc sai topology/hash/chronology/RNG/counters dẫn tới rollback, export failure và dừng. Không clip target/prediction, skip record, retune learning rate, fallback sang GEOMETRIC hoặc giảm history để vượt gate. Capacity 32 và query ratio 2+2 không phải một grid search. Phân biệt technical failure với negative scientific result.

## 9. Acceptance trước remote execution

Các kiểm tra bắt buộc khi implementation đã được giao:

- Independent reduced-FP64 finite differences cho exact meta-gradient với historical queries, bao gồm denominator derivative; exact/FO cùng forward values và backward khác theo thiết kế.
- Startup fallback parity với v5.1; independent Algorithm R/query RNG oracle; eligibility/future-label trap; distinct/disjoint IDs và immutable payload ownership.
- Mọi partial-block lengths 1–7 load/resume chính xác; failed meta update hoặc reservoir insertion rollback cả RNG/state; hard-crash replay không double-insert.
- Score và RNG read-only; terminal partial block không flush; DIRECT first-block zero theta gradient được nhận là đúng dự kiến; corrupted schema/count/hash/state bị từ chối.
- `R_EXACT` scientific parity với accepted v5.1, đồng thời old v5.1 contract/API/state guards không bị nới lỏng. New schemas từ chối old T4/source checkpoints.
- Clean installation/source inventory; revision-aware factories, preflight bindings và aggregation nhận đúng registered per-job method identities, vẫn từ chối source/runtime/producer/data/protocol mismatch.

Canonical FP32 full `28×28` grid/32-channel exact four-step meta-backward phải finite và mọi active group connected. Không coi reduced FP64 finite differences là full FP32 acceptance. Các mask, CADIC proposer/distance/Eq9/kernel tests và old v5.1 regressions vẫn cần. Đo memory và latency thật; giữ nguyên số query không chứng minh overhead bằng 0 hoặc runtime không đổi.

**Thuật toán đã đủ định nghĩa cho implementation thí nghiệm.** Candidate chưa được chọn làm phương pháp chính. Nghiệm thu anomaly performance/retention, cấp quyền chạy và quyết định tiếp tục phải theo [study](study.md), không theo numerical tests, paper names hoặc sự đồng ý giữa các AI.
