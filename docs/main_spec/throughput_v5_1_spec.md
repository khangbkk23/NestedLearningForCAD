# Nested Learning for CAD — đặc tả đo và tối ưu throughput v5.1

Ngày 20/10/2026. Engineering ID: `hope_support_v5_1_throughput_engineering_v1`.

**Trạng thái khi ban hành: specified; profiler, optimization và migration chưa được implement hoặc chạy T4.** Người dùng giao viết docs/spec trước và yêu cầu giữ đầy đủ cơ chế của phương pháp. Tài liệu này không cấp quyền agent tự chạy GPU, không hứa speedup hoặc performance gain. Kế hoạch thực hiện nằm ở [plan](throughput_v5_1_plan.md), các giá trị khóa máy đọc nằm ở [contract](throughput_v5_1_contract.json).

## 1. Mục đích và quan hệ với frozen specs

Giảm công việc tính toán dư thừa của `hope_cad_v5_1_causal_scaled_utility` mà vẫn thực hiện cùng thuật toán. Chưa có phép đo thành phần trên T4 để coi thời gian hàng chục giờ là chi phí tất yếu của Nested Learning. Cũng chưa có bằng chứng một bản tối ưu sẽ cải thiện AUROC/P-AP: khi tính tương đương được giữ, mục tiêu là cùng kết quả với thời gian/tài nguyên tốt hơn.

Nguồn chuẩn phương pháp vẫn là [architecture](architecture_decision_v5_1.md), [instruction](instruction_CAD_v5_1.md), [planning](planning_CAD_v5_1.md), [revision contract](revision_contract_v5_1.json), [acceptance addendum](implementation_acceptance_v5_1.md). Nguồn chuẩn benchmark vẫn là [study spec](benchmark_v5_1.md). Không sửa bytes các tài liệu đó để đưa tối ưu vào một run đã khóa. Addendum này chỉ điều chỉnh cách triển khai tính toán và yêu cầu kiểm tương đương, không thay objective hoặc protocol.

Nếu một đề xuất chỉ nhanh hơn nhờ giảm số phép học mà method đòi hỏi, bỏ gradient dependence, giảm thông tin hoặc đổi quyết định, nó không thuộc tối ưu tương đương ở đây. Phải ghi thành đề xuất method revision riêng; không tự áp dụng dưới nhãn “tăng tốc”.

## 2. Cơ sở thực nghiệm và điều chưa biết

Hai nguồn gốc của công việc này là [throughput audit](../../../.agents/evidence/reviews/support-v5-1-throughput-audit-20261020/review.md) và [review phiên 7.1 ở 376 ảnh](../../../.agents/evidence/reviews/support-v5-1-benchmark-seed0-session1-20261020/review.md). Đây là quan sát development trên MVTec, không independent confirmation.

| Quan sát đã kiểm | Giới hạn của kết luận |
|---|---|
| Pilot: NESTED 63.941 proposal replacements, CADIC 46.721; NESTED có 30.536 replacements trong proposal bị veto | Số operations, không phần trăm wall time. Hai policy có bank trajectory khác nhau |
| Có duplicate prediction trước actual fast update trong `NestedUtilityLearner.observe` | Chưa đo tỷ trọng T4 hoặc speedup khi bỏ duplicate |
| Full reservoir tạo 64 anchor NN calls cho hai bank; K10k/chunk256 dẫn tới 2.560 distance-block calls | Không phải số CUDA kernels và chưa là tỷ trọng runtime |
| Một hot event điển hình có bốn bank digests, tương đương 122,88 MB logical tensor payload | Chưa đo PCIe bytes, cache reuse hoặc chi phí hash thực tế |
| Pilot exact NESTED chỉ lâu hơn FIRST_ORDER khoảng 0,69% | Timing khác trajectory/cache; không chứng minh higher-order miễn phí hoặc vô ích |
| 7.1 paused tại 376/3.599; bank 10k, 362 fast và 45 slow updates; first256 events khớp pilot | Không có anomaly metrics v5.1, không biết final performance hoặc thời gian fullstream |

Chưa có profile trên T4, optimization implementation, migration approval hoặc AD superiority. Tài liệu không coi normal-risk +2,97% là giảm 2,97 điểm AUROC, và không dùng các số trên để retune phương pháp.

## 3. Những phần bắt buộc giữ nguyên

Các bất biến sau là acceptance requirements, không phải gợi ý có thể bỏ để đạt thời gian mong muốn.

| Thành phần | Hợp đồng giữ nguyên |
|---|---|
| Backbone và raw geometry | Frozen ViT-B/8, checkpoint SHA đã khóa, preprocessing, block8 zero-based, 784×768 FP32; không thêm normalization hoặc giảm patches |
| Memory/proposal | K=10.000 raw patch vectors; full image CADIC proposal, mọi lần replacement, tie/index ordering và no-op semantics; không early-veto trước khi dựng đủ proposal |
| Scoring | Raw NN và đầy đủ CADIC Eq9 b=9; pixel interpolation CPU FP32 như benchmark; score chỉ đọc state |
| Visual/HOPE | Context projection/RMS; 4 directions, 4 heads/direction, head_dim8, 28×28 tokens; năm learned initial maps và recurrence/chunk-boundary reads; giữ row retention và mọi gradient path hoạt động |
| CMS/readout | Fast CMS/readout persistent; slow CMS, geometry và visual slow parameters như contract; không reset ở task boundary |
| Causal target | Ảnh normal kế tiếp thực sự đến mới resolve pending; hai fixed banks, current/anchor risks và stored c/gamma; không relabel, clip target hoặc cập nhật pending ngược thời gian |
| Anchors | Uniform reservoir 32 ảnh ×8 patches, weight1; cùng private RNG và thời điểm update; không giảm số lượng, sampling rate hoặc đổi objective |
| Actual learning | Mỗi completed label có cùng actual fast update, lr0,05, denominator theo context energy; detach actual inter-record graph đúng contract |
| Exact nested learning | Mỗi8 completed labels: replay4 support innersteps, mean4 query losses ở cùng post-support state; giữ derivative của quotient/denominator, `create_graph` đúng exact branch; không chuyển sang FIRST_ORDER |
| Slow update | Adam lr1e-4, betas/eps/weight_decay, clip norm1; actual fast state sau8 nhãn được giữ, không rebase |
| Admission | Warmup32 labels force-PASS, sau đó accept tại p≥0; không quota nhận mẫu, không sửa threshold hoặc curriculum theo category |
| Precision/runtime | FP32 neural/raw tensors, các scalar/step FP64 theo contract; no AMP/TF32, deterministic runtime đúng suite; không thay rounding bridge hoặc GEMM route ngầm |
| Durability | Checkpoint mỗi8 ảnh, task boundary, pause/end; transactional rollback, hashes, rejection của state không hợp lệ; pending và partial replay không bị flush ở pause |
| Dữ liệu/benchmark | 3.599 online normals, task order/split/seed, test evaluator, metrics và control meanings như study; không dùng task ID/GT làm learner input |

Không giảm “sức mạnh” bằng cách tắt meta-learning, tự giới hạn số self-updates, giảm memory, ít anchors hơn, chỉ học một số ảnh, rút ngắn unroll, giảm số heads/tokens hoặc tăng update period. Nếu sau đo còn chi phí nội tại lớn, báo thật tradeoff; không cắt các phần này để cứu runtime.

## 4. Mô hình toán học của tính tương đương

Gọi S_t là scientific state trước ảnh t: bank vectors và counters, theta, phi, Adam, anchors/private RNG, pending, replay/block_initial/saved episode, clock/seen IDs/finalization. F là frozen reference event transition, O là candidate implementation. Với cùng raw feature x_t và cùng input identity:

`F(S_t, x_t) = (S_(t+1), e_t)`.

Mục tiêu engineering là `O(S_t, x_t) = F(S_t, x_t)` cho state và event quan trọng, không chỉ loss gần bằng. Trong số thực, các biến đổi đại số có thể tương đương nhưng cách cộng/nhân FP32 khác có thể đổi ties và p gần0. Vì vậy proof đại số là cần cho ý tưởng, chưa đủ để chấp nhận numerical trajectory.

Nếu S_0 bằng nhau và mỗi transition bằng nhau trên một stream, quy nạp cho state và score bằng nhau trên stream đó. Kiểm một số fixtures không chứng minh mọi input tương lai. Tài liệu chỉ cho phép claim scoped parity, cộng guard tại migration; không viết “universal bit-exact” từ vài tests.

Với actual update, `L=.5*(p-y)^2`, `phi' = cast_FP32(phi - .05*partial_phi(L)/d)`. Phải lấy gradient của L trước rồi chia d. Không đổi sang gradient của `L/d`. Khi exact replay, d được recompute từ pre-step state và vẫn tham gia derivative của increment. FIRST_ORDER control vẫn detach toàn increment; không dùng nó làm implementation thay thế NESTED.

Ví dụ duplicate-forward reuse chỉ hợp lệ trong cùng call, với cùng theta/phi/context/c/gamma và không mutation giữa hai lần. Có thể chia sẻ output/graph của pre-step prediction vào actual fast-step thay vì recompute; không được tái dùng detached h trong meta replay hoặc sau slow commit. Autograd graph lifetime, `.grad`, cached tensors và memory usage phải kiểm riêng. Không cache learned embeddings qua theta version.

## 5. Gói profiling đầu tiên: scope đã khóa

### 5.1 Parent và cửa sổ

Parent là reviewed ZIP session01 của `NESTED_seed0`:

- Archive SHA256 `35f05b523b26ce65d7118fd12aa91c569e73698fcfe64e542ee700ecabc8f257`.
- Active checkpoint cursor376 SHA256 `1be60248204fd4426a5aa9266ac88db79695913f5d317c3f3a53884587cf3e29`. Đây là mốc bảo toàn để có thể tiếp tục benchmark, không phải điểm đầu replay các ảnh đã thấy.
- Điểm đầu profile là **task_00 checkpoint cursor179**, SHA256 `3fb6b1ec31e281539ad16c098d8f77f854504f79c84e2d95cec73204d92c447d`.
- Replay **16 ảnh có index179–194**, tức `cable/train/good/000.png` tới `015.png`, theo manifest/order/hash gốc. Tất cả đã được quan sát trong parent376; không mở normal tương lai để chọn tối ưu.

Không thể load active376 rồi nạp lại identity179 vào chính actor đó: điều này vi phạm seen-ID/chronology. Phải load đúng predecessor179 vào temporary owner. Cửa sổ đi qua pending cuối bottle, bank đủ10k, anchors đủ32, có hai slow commits tại events181 và189 trong reference đã lưu. Labels đầu165→181, slow20→22, vẫn còn partial block5 và pending cuối cửa sổ. Không `finalize_stream`, không censor, không flush vì hết16 ảnh.

Đây là engineering replay trên dữ liệu đã thấy, không sample đánh giá độc lập. Không dùng risk/sign accuracy/loss ở cửa sổ để chọn learning rule; endpoint chọn optimization chỉ là chi phí và parity. Nếu inputs/hash/expected counters khác, stop; không đổi cửa sổ sang nơi chạy nhanh hơn.

### 5.2 Ba passes cố định và budget

Thứ tự là **U1 → P1 → U2**. Mỗi pass bắt đầu từ cùng checkpoint179, cùng deterministic setup và private RNG; không lấy cuối pass trước làm đầu pass sau.

- U1/U2: reference không component instrumentation; có wall timer ngoài pipeline và end-of-pass synchronization. Giữ event/checkpoint/feature-receipt work như driver reference. U1/U2 là hai lần đo tổng để lộ biến động, không đủ statistical CI.
- P1: cùng16 transitions nhưng có component scopes/counters và CUDA events theo §6. Scientific state/events phải bằng U1/U2 và events179–194 của parent. Chỉ profiling overhead khác.
- Tối đa **48 image transitions** ở lượt profiling này, không tính load-only validation. Không thêm24ảnh pilot hoặc refit. Không có thử learning recipes.
- Soft session budget **1,5 giờ**, reserve180s cho export; cumulative budget tính cả resume. Đây là ngân sách engineering, không ETA, epoch limit hoặc giới hạn năng lực của method. Một atomic operation có thể vượt reserve; không hardkill giữa image update chỉ để giữ đúng đồng hồ.
- Hết budget: export completed passes/progress/failure; chưa đủ U1/P1/U2 thì status incomplete, không suy ra bottleneck ranking chắc chắn. Không tự tăng budget, đổi số ảnh, bỏ slow event hoặc kéo dài nhiều sessions không có trần.

Bootstrap/data extraction/restore/load-cache rebuild, measured pipeline, final export được tách thời gian. Mỗi pass có process isolation hoặc equivalent verified reset; imports phải trỏ vào reference source SHA map, không vô tình dùng active candidate qua `sys.path`. Cùng extraction/cold-cache policy giữa passes; không cho P1 dùng features cached sẵn trong khi U1 phải decode/ViT rồi báo speedup.

Trong primary timing, provider và pipeline chạy cùng đường như benchmark, gồm bytes verification/receipt work. Nếu có thêm feature-only microbenchmark để phân rã, phải ghi riêng scope và không thay primary end-to-end result. Một feature lấy ở arrival t không cấp cho learner t-1.

## 6. Đặc tả instrumentation và hạch toán chi phí

Tất cả tools/runner/notebook ở phần này là **proposed modules**, không mô tả chúng đã tồn tại.

| Scope bắt buộc | Móc code reference | Đơn vị/counter |
|---|---|---|
| Provider | `training/support_benchmark_v5_1/data.py` provider | decode/input hashing, ViT, feature hashing/receipt, wall s/bytes |
| Transaction backup | `models/hope_cad_v5_1/session.py::observe` | copy learner/anchors/IDs, CPU/device bytes được đo hoặc logical bytes |
| Resolve delayed credit | `SupportSession._resolve` | next-minus/next-plus NN, fixed-state guards, target construction |
| Actual learner | `NestedUtilityLearner.observe`, `fast_step` | precheck duplicate prediction, actual forward, autograd.grad, update, diagnostics |
| Slow learner | `meta_loss`, slow branch của observe | support4, query4, backward, clipping, Adam; chỉ events có slow commit |
| Prequential/proposal | session risk/fork, CADIC cached operator | prequential NN, fork, candidate NN, closest-pair/cache rebuild/refresh, replacement counts |
| Anchors/context | `models/hope_cad_v5/geometry.py` functions | risk calls, distance blocks, shared-bank guards, stats/projection |
| Admission/pending | session learned branch | predict, comparisons, fixed-bank fingerprints, pending storage |
| Trace/memory/I/O | `training/support_benchmark_v5_1/engine.py`, `training/runtime.py` | events/receipts, memory_profile, serialization, fsync, checkpoint SHA/write, wall bytes |

Mỗi region phải có parent ID/cursor. Host `perf_counter` đo inclusive và exclusive spans, trừ children trước khi tổng hợp; không cộng toàn bộ nested times rồi gọi là total. Dùng wrapper/helper tách riêng để đo, không đổi control flow hoặc arithmetic của reference. Có trường `unattributed_host_seconds`; không ép phần dư vào “GPU compute”.

CUDA events ghi stream-ordered spans quanh coarse regions; không synchronize sau từng NN tile hoặc từng token. Chỉ collect timing sau safe boundaries; synchronization thêm bởi profiler được ghi và chỉ thuộc P1. **CUDA event span không đồng nghĩa kernel-busy time:** có thể gồm host launch gap/idle. Không cộng device span vào host total hoặc tính CPU bottleneck bằng host minus device span. Nếu cần kernel-active timeline, đề xuất riêng bounded `torch.profiler` pass với budget/spec mới; chưa tự thêm trace nặng vào lượt đầu.

Timer code không consume model/anchor RNG, không đổi tensor dtype/order/layout, không detach loss, không thêm no_grad vào learner, không di chuyển tensors của scientific state. Nếu hook giữ tensors/graphs thì ghi GPU peak và gỡ reference sau collect; instrumentation không được làm sai memory estimate hoặc tạo leak. Không dùng `empty_cache()` ở giữa events để làm đẹp peak.

Các counters cần phân biệt:

- `proposal_replacements` / `unique_changed_rows` / committed-vetoed work: không quy operations thành seconds.
- NN calls, distance blocks, pair-cache invalidations/hits: không gọi là CUDA kernel count.
- logical hash/copy payload và actual transferred bytes nếu đo được: không gộp hai loại.
- GPU allocated/reserved peaks, persistent scientific payload và derived caches: không claim equal-total-memory từ bank K.
- normal events, actual completed labels, fast/slow commits, visual encodes chia admission/actual/meta: không nhầm visual32-channel với ViT768.

Tổng unprofiled là U1 và U2; P1 overhead báo dưới dạng `P1 / mean(U1,U2) - 1`, đồng thời xuất riêng hai reference times. Ratio này chỉ là mô tả trên cửa sổ, không CI hoặc correction factor phổ quát. Component ranking từ P1 chỉ để chọn ứng viên; sau tối ưu phải đo unprofiled original/candidate cùng inputs/cache/runtime. Không dự báo speedup từ call count đơn lẻ.

## 7. Chính sách chọn và giới hạn tối ưu

Ưu tiên không cố định theo cảm giác: sau profile, chọn tối đa **ba patches nhỏ** từ bảng sau, mỗi patch có isolated parity và timing. Không yêu cầu phải dùng hết ba nếu không có lợi ích. Lượt này không search phương pháp.

| Ứng viên | Điều kiện hợp lệ | Tình huống phải từ chối/defer |
|---|---|---|
| Reuse actual pre-step prediction/loss/d | Cùng state và graph đúng §4; không đổi exact replay; không tăng memory vô hạn | Reuse detached learned embedding cho meta hoặc reuse qua theta/phi update |
| Reuse fixed-bank finite guards/digest | State bất biến có version/ownership chính xác; giữ external input/load/error checks và cùng hash bytes | Chỉ dựa tensor shape/data_ptr; tensor bị sửa in-place/outsideowner mà cache không invalidate |
| Device-local rollback snapshot | Clone đầy đủ model/phi/Adam/replay/anchors/RNG; rollback byte/value-equal sau fault, GPU memory có headroom đo được | Bỏ rollback, bỏ optimizer snapshot, không chứa pending, hoặc OOM vì thêm copies |
| Aggregate diagnostics transfers | Giữ đủ field/norm/reduction và cùng rounding như reference | Bỏ gradients/steps/NaN checks hoặc đổi reductions làm lệch traces bắt buộc |
| Cache exact anchor results | Key gồm bank content/version, anchor content/version, geometry/numerics; mọi mutation/fork/load đều invalidate đúng | Dùng cached aggregate sau reservoir update hoặc đổi bank; xấp xỉ, stale memo |

Những cách có nguy cơ thay arithmetic/trajectory — grouped NN, GEMM reassociation, pair kernel mới, fullK²cache, truncate ViT sau hook, batching/prefetch — chỉ là ứng viên nghiên cứu engineering sau khi profile chỉ ra cần. Không là patches mặc định. Phải có spec bổ sung về precision, ties, arrival rights và memory rồi mới thử. Full40kcache không thuộc lượt này.

Không thay checkpoint cadence8, event logging/fsync policy, num_threads/runtime, dataset/cache permissions, precision hoặc cơ chế cập nhật trong batch đầu. Chi phí durability có thể lớn nhưng không được “tối ưu” bằng cách bỏ bằng chứng và khả năng phục hồi.

## 8. Acceptance gates

### G0 — Đặc tả và nguồn

Docs/JSON nhất quán; old166packaged/live payloads byte-exact; parent/protocol/runtime assets verified; new modules không có production entrypoint tự bật. Method và study digests giữ nguyên. G0 hiện chỉ có thể được nghiệm thu ở mức specification.

### G1 — Profiler transparency

U1/P1/U2 từ cùng predecessor; đủ16 event IDs; predictions/risks/targets/actions/bank vectors và counters, theta/phi/Adam, replay/pending/anchors/private RNG đều exact; scientific event chain khớp parent đoạn tương ứng, bắt đầu từ hash chain của events0–178. Derived distance caches có thể được rebuild khi load; wall times và profiler counters là ngoại lệ phải liệt kê. Không cho phép wildcard “metadata khác thì bỏ qua”. Timer wrappers không consume RNG và có cleanup.

### G2 — Candidate local parity và robustness

Giữ nguyên existing core/benchmark regressions. Bổ sung tests thật sự liên quan đến patch: full shape forward/backward, raw parameter gradients trước clipping và actual step, exact support/query meta-gradients/denominator derivative, FIRST_ORDER distinction, tie/zero-key/no-op/warmup, stale-cache invalidation, malformed state/fault injection, pending/partial resume. Numeric closeness chỉ chẩn đoán; không dùng tolerance để accept khác action/bank hoặc để tự hạ mode từ exact sang approximate.

### G3 — Canonical T4 paired check

Reference/candidate replay cùng16 observed normals từ179, cùng runtime. Hai unprofiled pairs theo thứ tự đã khóa R,O rồi O,R, reset scientific state từng pass. Gate nhận optimization đòi complete exact trajectory/state parity, không lỗi/OOM, và aggregate candidate end-to-end time không lớn hơn reference. Với hai pairs phải xuất từng time, không gọi statistical significance. Chênh sát noise hoặc memory tăng đáng kể: defer, không gọi accelerated. Budget mặc định của validation này riêng **1,5 giờ/64 transitions**, không gộp với profile90min hoặc tự chi không giới hạn. Chỉ implement/run sau khi profile review chọn patches.

Không gắn heavy gradient-copy hooks vào primary timing rồi gọi đó là unprofiled speedup. Trong validation, lưu hai meta-episodes sắp commit ở events181/189 bằng scope được khai báo ngoài primary measurement; recompute trên disposable owners để so raw gradients trước clipping, support increments/denominators, query losses và actual parameter steps. Đây là gradient validation riêng, không thêm image arrivals hoặc cập nhật benchmark, và mọi chi phí vẫn tính vào cumulative budget90min. Nếu việc capture phải có thêm một pass nhận ảnh thì nó vượt64transitions đã khóa: phải đưa ra spec/budget amendment trước khi chạy, không tự thêm. Có thể capture immutable replay/context/phi-start/theta tại safe boundary ngoài timed event region; receipt phải nói rõ capture/reset overhead. G2 bổ sung kiểm actual fast-gradient trên các pre-step states thu được. Primary timing và diagnostic recomputation được báo tách riêng; scientific parity yêu cầu ở cả hai scope.

Nếu raw gradients/float outputs chỉ close nhưng không exact, không auto migration; ghi `numerical_variant_not_compatible`, phân tích thứ tự arithmetic, rồi giữ reference hoặc đề xuất execution/method lineage mới theo scope thay đổi. Không đổi threshold/tolerance từ kết quả AD để làm gate qua.

### G4 — Checkpoint compatibility

Các tests và known-window parity không tự cho phép source mới đọc source lock cũ. Phải có migration spec/receipt §9, canonical load/pending re-evaluation exact ở active376, và fail-closed shadow continuation. Chưa có G4 ở thời điểm viết.

### G5 — Benchmark completion

Chỉ full3599normal stream và15final test cells đã review mới là7.1 completed. Speed test16ảnh không là normal pilot mới, AD benchmark hoặc proof hơnCADIC. 7.2 và7.3 giữ ý nghĩa cũ; không bỏ controls/seeds/matrix để tuyên bố complete.

## 9. Bảo toàn checkpoint376 và migration

### 9.1 Ba danh tính riêng

1. **Method identity**: scientific contract/design digest.
2. **Study identity**: data/scoring/metrics/control/seed protocol.
3. **Execution identity**: source file map, runtime, orchestration version, performance implementation.

Một optimization có thể giữ1/2 nhưng đổi3. Không viết source cũ vào receipt mới, không sửa digest guard thành “accept mọi source”, không rewrite parent raw ZIP/locks/events/taskstates. Source package mới có SHA map riêng gồm explicit frozen-base reference và extension/candidate payloads.

### 9.2 Quy trình migration dự kiến

- Giữ parent376 bất biến. Trước canonical load, kiểm fullarchive/member/source/study/runtime/backbone/data row hashes. Chỉ allow đúng parent hashes trong contract, không accept theo branch name.
- Tạo **execution lineage mới** tham chiếu parent job/suite/source/checkpoint SHA. Identity/session schema vẫn phải được strict validate; loader tương thích là proposed, không bypass validator gốc. Chưa có receipt hợp lệ thì source mới không được resume output cũ.
- Trên T4, reference/candidate cùng load active376 vào temporary owners; pending prediction/d và allstate phải exact. Không load checkpoint bằng CPU rồi coi khác rounding là đã chứng minh GPU parity.
- Inherited events0–375 và task_00.pt/json giữ bytes, trace chain anchor giữ nguyên. Không sửa wrapper của inherited taskstates để giả chúng được sinh bởi source mới; lineage resolver phải xác minh original job digest khi đọc chúng. Active wrapper mới có thể đổi execution/job digest nhưng session scientific payload bằng parent; record old/new SHA rõ.
- **Shadow bridge8 canonical arrivals** chỉ được chạy khi user cho phép migration sau G3. Hai actors cùng bắt đầu376, mỗi rawnormal đến theo thứ tự chỉ được cấp cho actors tại arrival đó; reference/candidate xử lý xong và so scientific event/state mới đọc ảnh tiếp theo. Không nhìn trước376–383 để tune. Đây là verified canonical continuation, tách biệt các profile clones bị discard.
- Mismatch/nonfinite: không publish candidate checkpoint hoặc completed-migration receipt. Giữ parent376 và failure evidence; phần reference shadow không được tự trở thành benchmark continuation để giấu failed candidate. Không sửa thresholds, tests, identity hoặc candidate rồi lặp vô hạn.
- Khi đủ8events exact, publish atomic candidate checkpoint cursor384 và compatibility receipt, execution segments/counts/timing riêng. Old376 vẫn tồn tại. Receipt chỉ cho exact source pair/runtime/schema/parent đã kiểm; không là general compatibility promise.

### 9.3 Receipt tối thiểu

`engineering_id`, parent archive/manifest/suite/job/active/taskstates SHA, old/new source map/digest, method/study/runtime identities, changedfile list, G1/G2/G3 review hashes, state equivalence policy/coverage/exclusions, pending verification, bridge start/end/IDs/event chain, failure status, checkpoint old/new SHA, inherited task owner references, actual compute ledgers và approval scope. Trạng thái chỉ chuyển `not_implemented`→`eligible_pending_bridge`→`accepted_for_exact_parent_and_source_pair`; thiếu field/gate thì fail.

Nếu không đạt parity hoặc implementation migration an toàn quá phức tạp, **giữ đường source cũ để resume** là lựa chọn hợp lệ. Không tự bắt user rerun W6/fullpilot hoặc toàn3599 chỉ vì build mới; mọi rerun lớn phải được trình chi phí và lý do trước. Nếu thuật toán thật sự đổi, phải method revision riêng và không gắn kết quả mới vào frozen v5.1.

## 10. Artifacts và kết luận có thể báo

Profiler source delivery dự kiến: `.agents/artifacts/deliveries/hope_support_v5_1_throughput_profile_r1/`. Raw runs: `.agents/artifacts/runs/support_v5_1_throughput_profile_01/session_01/` và các session tiếp nếu cumulative budget còn. Đây là **đường dẫn dự kiến**, chưa có notebook/ZIP thật.

Evidence cần có manifest, handoff/execution, contract/source/runtime/parent receipts, fixed window manifest, U1/P1/U2 per-event/region timing và counters, peakmemory, overhead, scientific parity report, error/partial traces và logs. Giữ state bằng hashes/selected small records; không nhân bản parent119MB vào từng profile ZIP. Không cần W&B để nghiệm thu; dashboard bổ sung nếu sau này có, không thay evidence gốc.

Các status là `profile_complete_pending_review`, `profile_incomplete_budget`, `profile_failed_preserve_evidence`; optimization/migration có status riêng. “Complete” profiling không mở benchmark/VisA tự động. Không đặt claim `max sức mạnh Nested Learning`, `tối ưu toàn cục` hoặc `performance>=CADIC` từ gói này.

**Kết quả mong muốn:** biết chi phí thật, bỏ phần thực sự dư thừa nếu kiểm chứng được, giữ nguyên toàn bộ khả năng học đã định nghĩa, và bảo toàn đường tiếp tục từ checkpoint376.

## 11. Chốt triển khai sau kiểm tra chéo code — 20/10/2026

Phần này làm rõ các yêu cầu vận hành ở §5–9; khi một mô tả trước đó chưa đủ cụ thể, dùng quy tắc dưới đây. Không thay method, cửa sổ dữ liệu hoặc ngân sách. Engineering contract revision `2`; profiler vẫn chưa implement.

### 11.1 Bounded driver không được giả lập end-of-stream

`training/support_benchmark_v5_1/engine.py::train` hiện finalize khi `actor.cursor == len(rows)` và tiếp tục tới hết rows. Vì vậy **không truyền `rows[:195]` vào train**, không reset cursor về0, không thay job lock bằng một manifest16ảnh. Cũng không lấy `task_00.pt` làm `active.pt` bằng cách sửa schema trong parent.

Extension cần bounded loop riêng với full registered manifest3599 làm nguồn quyền/identity, nhưng chỉ cấp arrivals179–194. Load nguyên `session` của task checkpoint đã verify vào temporary owner. Bảo toàn prefix events0–178/chain và seen IDs; task wrapper gốc vẫn bất biến. Dùng lại đúng các primitives của frozen driver: provider.get → actor.observe → atomic event publication → chain → memory_profile → cadence checkpoint. Global cursor184 và192 là hai periodic saves trong cửa sổ; cursor195 chỉ có **diagnostic final snapshot ngoài primary timing**, không finalize/censor hoặc xuất train_complete/task-complete giả.

Mỗi pass có thư mục riêng. Copy/verify prefix ở bootstrap ngoài timing; receipts cho16ảnh replay ban đầu phải **chưa tồn tại**, giống arrival đầu của reference. Không copy toàn bộ376 feature receipts vào pass rồi đo thêm nhánh đọc/so sánh mà original arrival không chạy. Reference expected events179–194 nằm ở kho đối chiếu tách riêng, không đặt sẵn vào event output để đổi nhánh save_or_verify. Ghi mọi khác biệt với continuous benchmark: load checkpoint179, cold derived caches, startup và isolated replay; không suy duration16ảnh thành ETA toàn3599.

### 11.2 Ranh giới đo, đối chiếu và bộ nhớ

Mỗi pass dùng process mới với cùng source/runtime, actor/provider được tạo và checkpoint load xong trước đo. Synchronize một lần trước bắt đầu; primary pipeline time bao gồm16 arrivals, provider, event/hash/memory_profile và hai periodic checkpoint writes; synchronize cuối trước dừng timer. Không thêm warmup bằng ảnh, precompute feature hoặc prebuild pair cache. OS/filesystem page cache không thể hứa reset trên Kaggle: ghi trạng thái best-effort, thứ tự U/P/U hoặc R/O/O/R và hạn chế này.

Baseline profiler G1 chỉ thêm coarse timers ở P1. U1/U2 không có full-state copy hay parity validator sau từng ảnh bên trong timed loop. **So toàn bộ emitted events từng ảnh và toàn bộ serialized scientific state cuối pass** sau dừng timer; đây là coverage hữu hạn được báo đúng tên, không khẳng định đã dump toàn state ở mọi transition. Source parent chỉ có task179 và active376, không có full-state oracle cursor195: state195 phải đối chiếu giữa các owners, events/feature receipts đối chiếu parent. Không tạo “expected state195” bằng suy đoán từ events.

G3 dùng cùng primary accounting. Nếu capture meta/actual pre-step fixtures cần dừng giữa pass, tách các interval capture bằng synchronization ở **cả reference lẫn candidate tại cùng boundary**. Báo `pipeline_seconds_excluding_diagnostics`, `diagnostic_capture_seconds` và tổng wall thực; không gọi tổng pipeline ghép interval là continuous production runtime. R/O đều dùng cùng capture schedule; không chỉ trừ chi phí candidate. Recomputation ngoài primary measurement, vẫn tính budget; captures không thêm arrivals và không thay source reference arithmetic.

Reset peak GPU counters sau bootstrap, ngay trước primary start; báo allocated/reserved tại start và peak absolute trong pass, kèm bootstrap/load peak riêng. Không cộng peak của các process thành peak một run, không lấy persistent payload làm peak VRAM. Full state comparisons, export, CUDA event collection và diagnostic capture có cost ledger riêng. Tổng elapsed/budget bao gồm toàn bộ chúng.

### 11.3 Interrupt/resume và giới hạn số lần thực hiện

Một **pass hoàn chỉnh không bị gián đoạn** mới được đưa vào U/P/U overhead hoặc paired-speed gate. Resume chỉ skip các pass đã complete, hash-verified; được bắt đầu pass kế tiếp nếu cumulative budget còn. Không ghép timing trước/sau session hoặc process restart thành một pass chuẩn, vì checkpoint reload làm đổi cache/startup.

Nếu thiếu budget trước pass: pause/export ở pass boundary. Nếu budget tới hạn hoặc lỗi giữa pass: lưu partial diagnostics ở atomic image boundary, đánh dấu pass incomplete; không replay lại pass đó tự động và không xóa attempted work khỏi ledger. Tối đa48/64 **attempted image transitions**, kể cả observe đã bắt đầu rồi rollback; không chỉ đếm thành công. Một pass incomplete làm cả bộ đo incomplete, muốn thử lại phải review nguyên nhân và amend attempt/budget rõ trước run mới. Không giảm sốảnh, bỏ slow event hoặc nối partial pass để cứu gate.

Ledger tối thiểu: session start/end, bootstrap/load, pass_id/status, attempted/succeeded cursor IDs, primary time, diagnostics/compare/export time, cumulative spent, reserve và deadline. Soft budget không bảo đảm export trước hard platform kill. Checkpoint376 vẫn không bị ảnh hưởng.

### 11.4 Phạm vi equality và migration còn phải kiểm

So recursive state theo key/schema, scalar value và tensor dtype/shape/value exact; không so bytes của file `torch.save`, storage address hoặc Python object IDs để kết luận model khác nhau. Scientific state gồm đầy đủ `SupportSession.state_dict()` và event fields của reference, không chỉ bank/action. Serialization container có thể khác bytes nên ghi riêng file hash và logical-state digest.

Các ngoại lệ chỉ gồm: elapsed/timer fields, paths/execution lineage được khai báo, profiler counters, derived pair-cache buffers/counters không nằm trong scientific checkpoint. Cache không được loại trừ khỏi **correctness tests**: cold/warm/fork/in-place invalidation phải cho cùng NN/tie/proposal/scoring. Reference checkpoint không serialize autograd graph hoặc `.grad` buffer; không fabricate chúng khi load. So raw actual/meta gradients bằng diagnostic fixtures G2/G3, không suy từ final-state equality.

Migration G4 hiện là yêu cầu thiết kế, chưa là loader đã chứng minh. Trước khi implement migration phải liệt kê và test mọi consumer lock: train active/event prefix, inherited task-state reader, evaluation cell identity và final/matrix completion. Trong code hiện tại chúng đều kiểm job_digest nghiêm ngặt. Lineage resolver chỉ allow exact original owner/source cho inherited task00; states mới thuộc new execution owner. Không sửa frozen engine để bỏ kiểm tra mọi digest, không rename task wrapper rồi giả là new owner. Kiểm full final15cells và matrix120cells bằng synthetic lineage fixtures trước khi giao bridge; không cần đọc anomaly test thật để kiểm logic này.
