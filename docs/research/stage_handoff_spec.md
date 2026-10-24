# Bàn giao các stage giữa session và máy thuê — proposal v1

Ngày: 23/10/2026. ID proposal: `hq32_stage_handoff_v1`.

**Trạng thái: đặc tả nghiên cứu/vận hành đề xuất; chưa freeze, chưa implement, chưa có lệnh migration chạy được.** Tài liệu này không thay đổi equations, số arms, data rights, gates, ngân sách hoặc identities đã khóa trong [study](study.md), [candidate](candidate.md), [pipeline](../pipeline_alignment_spec.md) và `resources/research.json`. Khi triển khai phải có operational/study amendment được người dùng chấp thuận và source identity mới. Không sửa frozen resources hoặc evidence cũ để giả rằng chúng đã hỗ trợ proposal này.

**Amendment24/10/2026:** user đã giao triển khai subset [handoff v1](stage_handoff_v1.md): one-ZIP + exact-identity gate/restore/bootstrap đã có code/local fixtures. Các phát biểu “chưa implement” trong proposal này áp dụng cho giao thức rộng hơn, đặc biệt changed-runtime/source mapping và segment-rate migration. V1 không nới guards hoặc nhận legacy source bằng lời xác nhận. Actual cross-host GPU acceptance vẫn pending.

## 1. Mục tiêu và câu trả lời vận hành

Tách việc tính toán đã hoàn thành khỏi máy thuê đang giữ nó. Một host mới không được buộc train lại các kết quả đã hoàn thành chỉ để sinh lại file acceptance; đồng thời copy file hoặc thấy chữ PASS không đủ để công nhận runtime mới tương thích.

Hai cách chia được hỗ trợ trong thiết kế:

| Cách chia | Gói cuối stage | Công việc tiếp theo trên host đích |
|---|---|---|
| Hai đợt | B preflight/pilot + C ba seed-0 jobs đủ endpoint/matrix | D các controls/replications còn thiếu, sau compatibility/review/allocation |
| Ba đợt | B preflight + pilot; C main + matrix; D extension | B→C và C→D dùng cùng giao thức bàn giao; C luôn train fresh |

Tên 7.1/7.2 cũ tương ứng phần training/endpoints và task matrix; trong runner research hiện hành chúng nằm trong Stage C (`--phase main`). Phần người dùng gọi 7.3 tương ứng Stage D (`--phase extension`). Không thêm một lượt train chỉ vì đổi tên stage/session.

**Phạm vi ưu tiên là chuyển ở ranh giới stage hoặc job đã hoàn thành.** Không đồng nhất nó với chuyển một learner đang học dở sang GPU/driver khác. Cách ít rủi ro nhất là giữ cùng numerical runtime, backup toàn bộ output và chuyển giữa các jobs.

## 2. Những gì code hiện tại thực sự khóa

Đã lần theo implementation hiện hành thay vì suy từ tên máy:

| Đường code | Ràng buộc hiện tại |
|---|---|
| `runtime/research.py::binding` | Full source digest, research-study digest, runtime, normal-row digest, backbone SHA, protocol, producer/backend |
| `runtime/identity.py::runtime_identity` | Python/packages/CUDA/cuDNN, GPU model/capability/total memory/compiled architectures, driver, device, threads, deterministic/TF32/CUBLAS flags |
| `runtime/research.py::assert_preflight`, `run` | Preflight binding phải bằng suite hiện tại; review gắn receipt/source; main/extension cần accepted pilot; extension cần development screen PASS |
| `benchmark/engine.py::train` | Job lock chứa suite/method/seed/rows/order; active checkpoint/event chain/seen IDs và task states phải khớp; completed train được skip khi hợp lệ |
| `benchmark/engine.py::evaluate`, `cell` | Không train trong evaluation; checkpoint SHA/job/test/cell/chunk bindings; completed cells được reuse nguyên trạng |
| `benchmark/statistics.py::aggregate` | Hiện yêu cầu full suite identity bằng nhau giữa inputs; cùng tên job xuất hiện hai lần bị từ chối |
| `runtime/budget.py::Budget` | Ledger stage B/C/D gắn allocation cố định; đổi rate/cap tại chỗ bị từ chối; chưa có reconciliation xuyên host |
| `runtime/evidence.py` | Archive có payload hashes và safe paths, nhưng chưa có stage handoff với compatibility policy |

**Không có host ID, instance ID hoặc CPU model trong suite binding hiện tại.** Vì vậy đổi máy vật lý nhưng mọi trường runtime/source/data thực sự bằng nhau không tự làm lock sai. Một session mới cũng không tự là một cohort mới. Ngược lại cùng RTX 5090 nhưng khác driver/Python/VRAM được báo cáo có thể khác binding. Cần so diff thực tế; không dùng quy tắc “khác host bắt buộc làm lại mọi stage”.

Hash manifest bảo đảm bytes khớp bản đã kê, không tự bảo đảm xuất xứ đáng tin hoặc khoa học đúng. Acceptance là attestation của người review được người dùng giao, không phải chữ ký do runner tự cấp sau mỗi PASS.

## 3. Ba quyền phải tách riêng

1. **Lưu/reuse kết quả đã hoàn thành:** đọc metrics, raw scores, task checkpoints và tính lại thống kê trên CPU. GPU mới không được thay provenance của kết quả cũ. Không cần train lại chỉ để giữ kết quả này.
2. **Cho phép chạy fresh jobs ở stage sau:** cần gates trước đó còn hợp lệ về method/protocol, cùng cohort hoặc compatibility amendment được chấp thuận, runtime đích đạt kỹ thuật, budget mới và quyền chạy. Đây là mục tiêu chính.
3. **Resume một job đang học hoặc một cell đang tính dở:** cần replay/state/feature parity mạnh hơn. Không mở quyền này chỉ vì quyền 1/2 đã đạt.

Không truyền bank/learner của discarded preflight/pilot làm initialization cho main. Replicate/control mới luôn fresh theo factory và registered seed. Task states C có thể được đọc để kiểm transport hoặc scoring, không làm initialization cho D.

## 4. Identity mới đề xuất, giữ đầy đủ origin

Không thay suite digest cũ bằng suite digest mới trong các file cũ. Thêm lớp chỉ mục có kiểm chứng:

- **Origin execution:** giữ nguyên full source file map/digest, suite/job/cell locks, runtime, archive SHA, receipts và review nguyên bản. Old metrics tiếp tục thuộc origin đó.
- **Study semantic identity:** method/variant contracts, full-train IDs/order/file bytes, checkpoint, preprocessing/raw geometry, bank/proposer/scoring/Eq9, precision/backend policy, optimizer/clocks/RNG và evaluator/statistics definitions. Không chỉ dùng một chuỗi study name.
- **Target execution:** full source map và runtime mới, setup/environment receipts, segment/host ID để accounting, actual hourly rate và allocation.
- **Compatibility receipt:** allowlist rõ origin→target, compatibility class, hashes từng chứng cứ đã kiểm, gates được kế thừa, gates phải chạy mới, limitations và reviewer decision. Không là wildcard “mọi driver/GPU được chấp nhận”.

Source digest hiện chứa cả docs/tests. Chỉ bổ sung migration/docs cũng đổi full identity. Không tự loại docs/tests khỏi source hash để tránh việc đó. Nếu muốn kế thừa evidence source cũ, phải giữ nguyên source snapshot cũ và kiểm explicit file-by-file diff với target. Bản mapping ghi những file đổi, vì sao không đổi scientific path và tests/probes đã kiểm. Bất cứ thay đổi nào ở numerical/learner/proposer/feature/evaluator không chứng minh tương đương phải tách revision/cohort, không gắn nhãn “chỉ infrastructure” theo tên file.

Compatibility của từng origin/target không có tính bắc cầu mặc định: A→B và B→C không tự cấp A→C. Không tự gộp tất cả origins theo cùng method ID.

## 5. Ma trận quyết định tương thích

| Trường hợp | Giữ completed results? | Fresh stage tiếp theo? | In-flight learned-state resume? |
|---|---|---|---|
| Host/session mới, full identity bằng nhau | Có, bytes giữ nguyên | Có sau restore integrity/gates/budget; current loader semantics được giữ | Có khi full atomic state/event/feature checks đạt; cần test transport thật |
| Chỉ docs/launcher/migration source khác, scientific path đã chứng minh giữ nguyên | Có | Chỉ sau explicit source compatibility mapping và target technical checks | Không tự cho phép; phải có replay parity cho chính job/schema |
| Khác driver/GPU nhưng package/recipe/precision không đổi | Có với origin cũ | Chỉ trong runtime-compatibility amendment cụ thể, sau numerical bridge và reviewer; không bảo đảm được nhận | **Không hỗ trợ ở v1**; giữ checkpoint và khởi tạo nhánh continuation riêng chỉ sau spec mạnh hơn |
| Đổi packages/numerical backend/precision/feature transform | Có như historical results | Mặc định new numerical/scientific cohort; không silently kế thừa matched claims | Không |
| Đổi method/hyperparameters/data/order/K/checkpoint/test definition | Có như study cũ | New registered revision/study; old pilot không mở candidate mới | Không |

“Full identity bằng nhau” vẫn phải kiểm giá trị thật, không copy JSON runtime để làm nó bằng. CPU/host khác vẫn ảnh hưởng timing dù không nằm trong numerical identity; giữ host receipts để báo chi phí.

## 6. Gói bàn giao bất biến

Mỗi gói chỉ chốt sau run/export kết thúc ở atomic boundary và không có writer. Bao gồm:

1. Toàn bộ experiment `run/`, logs và mọi export session: `evidence.zip` + hash-bound `accounting.json`. Giữ pause/failure sessions, không chỉ ZIP đẹp nhất.
2. `workspace/records/` của segment nguồn và hai asset receipts; Git commit + full source inventory/file hashes; source snapshot/commit phải còn truy cập được. Commit SHA một mình chưa chứng minh dirty source đã chạy.
3. Các review/acceptance có thật gắn receipt SHA, không tự sinh một acceptance thay mặt người chưa review.
4. Outer `handoff_manifest.json` với relative paths, byte sizes, SHA256, schema/version, study/method/job identities, origin suite, source/runtime/data/backbone/test digests, completion coverage, permitted next stages và budget history references. Manifest không tự chứa hash của chính nó; outer delivery receipt giữ hash manifest/archive.
5. Trạng thái tổng hợp `handoff_ready_pending_import_review`, không đổi `candidate_adopted=false` hay `reportable` chỉ vì chuyển máy.

Artifact inventory phải kiểm **toàn bộ** payload, không chỉ selected score chunks. Refuse symlinks/path traversal/duplicate entries/unknown required-schema/missing files/hash mismatch/truncated payload. Chỉ giải nén vào staging trống; kiểm xong publish atomically. Không overwrite một live run khác. Rename physical directory không đổi logical relative-path identities trong manifest.

Dataset và pretrained tensors không bắt buộc upload lại nếu target provision đúng pinned bytes và verify receipt/file hashes. `.pixi/` không chuyển; target cài bằng `pixi install --locked`. Omit dữ liệu lớn không có nghĩa bỏ data identity hoặc chỉ tin filename. Receipts/logs không được chứa private keys/API tokens.

### 6.1. Nếu gói B đã hoàn thành

Phải có accepted preflight và accepted `pilot.json` thật; đủ chín completed arms ×256, events/states/counters/history/RNG/feature receipts để audit lại gates. Giữ cả nine arms và risk/probe identities, không rút còn vài summary numbers rồi gọi fully resumable. Target C fresh, không import pilot active states vào actor main.

### 6.2. Nếu gói C đã hoàn thành

Phải có cả ba jobs CADIC/R_EXACT/H32_EXACT seed0: train đủ3629,15 immutable task states,15 final cells và đầy đủ120 matrix cells/job cùng raw chunks/labels/masks/identities, metric summaries và completion locks. Giữ test manifest và hashes. Nhận chỉ `final_macro.json` không đủ kiểm statistics hoặc giữ matrix.

`development_screen.json` phải được reviewer kiểm lại từ **đúng ba completed C jobs trong gói**; record screen mới có thể dẫn hashes các origin jobs, nhưng không sửa origin screen/metrics. Thiếu matrix cell hoặc screen FAIL thì không mở D. Giữ negative hoặc incomplete C, không làm lại để thay kết quả bằng winner.

## 7. Giao thức import và numerical bridge

Các tên thao tác dưới đây mô tả interface cần implement; **không phải CLI hiện có**.

### 7.1. Offline trước khi thuê máy

`handoff inspect/pack` kiểm archive/accounting/source manifests, stage coverage, reviews và budget lineage; xuất plan chỉ rõ cần upload gì và job nào đã hoàn tất/chưa chạy. Không import tensor vào learner và không gọi GPU/API/SSH. Thiếu source snapshot, acceptance hoặc sidecar phải báo cụ thể, không cho phép auto-repair lịch sử.

Một plan đứng yên trước target probes định nghĩa allowed delta, bridge inputs/snapshots, acceptance mode và caps. Không chọn tolerance/IDs sau khi thấy anomaly metrics. Evidence cũ vẫn dùng format/schema cũ; adapter đọc nó chỉ tạo index mới.

### 7.2. Trên target, trước jobs mới

1. Verify actual source/data/checkpoint/runtime, clean environment, real GPU/driver/precision; so field-level origin→target diff và target allocation.
2. Import gói vào immutable origins store; target run/ledger/acceptance mới ở segment riêng. Chưa mở scientific job.
3. Chạy compatibility probe đã định nghĩa trên discarded clones. Không test/GT, không lấy thresholds theo AD scores.
4. Export target probe evidence và reviewer decision gắn cả origin package SHA/target receipt SHA. PASS kỹ thuật chưa tự ghi reviewer acceptance.
5. Chỉ khi allowed stage và remaining jobs hợp lệ mới mở C/D. Target runner resolve origin references thay vì yêu cầu file origin mang target runtime.

### 7.3. Phạm vi probe, chọn trước khi triển khai

Mức tối thiểu cho runtime không đổi: verify receipts + fixed input features + state load/scoring read-only + interrupted/replayed atomic boundary. Không cần nine fresh pilot arms chỉ vì máy có IP mới.

Khi GPU/driver đổi, bridge v1 phải bao phủ ít nhất:

- Re-extract fixed allowed-normal IDs và đối chiếu original FP32 tensor hashes; kiểm dtype/shape/finite không thay pixel recipe.
- Full-budget reference/optimized NN, victim/proposal/tie decisions và Eq9, với same frozen inputs/snapshots; so bank vector hashes, write sequence, gate actions và scores.
- Historical exact/FO forward/backward, actual fast/slow steps, history-query IDs/Algorithm-R/private RNG, pending/partial records, clocks và atomic replay trên **disposable cloned snapshots** đã có ≥32 labels/history active. Preflight28normals hiện tại riêng nó chưa làm đầy real learner history.
- Scoring read-only của frozen source checkpoint; all affected groups finite/connected; snapshots trước/sau không mutate. Không dùng true anomaly labels để chọn compatibility.

Inputs/snapshot indices cụ thể phải được khóa trong implementation contract. Cần tạo probe reference outputs trên source runtime khi source còn truy cập được; đã Destroy source thì chỉ dùng reference outputs/tensors thực sự có trong backup. Không thể bịa baseline gradient/intermediate tensor từ một summary PASS. Có thể chạy old source trên target để kiểm source-code mapping, nhưng đó không tái tạo phép tính GPU/driver nguồn.

**Default acceptance cho đường discrete/state parity là exact equality**, không chỉ `allclose`: chênh rất nhỏ ở distance có thể đổi ties, admission và bank trajectory. Tolerances khác chỉ được dùng sau amendment có lý do, trước run và với claim scope thu hẹp; không nới threshold để probe PASS. Probe hữu hạn không là theorem về mọi future input hoặc toàn training trajectory.

Bridge FAIL hoặc thiếu reference → giữ old results; không cho compatibility receipt tự mở mixed-cohort attribution. Reviewer chọn giữ đúng numerical runtime cũ nếu còn thuê được, dùng new cohort/thu hẹp report, hoặc phê duyệt thí nghiệm bổ sung hữu hạn. Không mặc định làm lại cả pilot/C, cũng không bảo đảm luôn tránh được mọi rerun.

## 8. Reuse pilot, C screen và fairness khi mở D

B→C: accepted pilot là evidence cơ chế/safety của đúng method/protocol và origin runtime. Same-runtime handoff kế thừa bằng identity. Changed-runtime bridge chỉ mở fresh C theo amendment cho phép; nếu không đủ bằng chứng về hành vi bank/learner, old pilot không trở thành target pilot bằng đổi nhãn. Không đòi retrain full AD để kiểm kỹ thuật; cũng không thay nine-arm safety gate bằng vài tensor finite checks.

C→D: ba completed C jobs giữ nguyên hashes, scores và origins, được dùng cho screen/seed0 aggregate. Target D chỉ schedule sáu controls seed0 và sáu replicates đã đăng ký, trừ jobs đã completed hợp lệ. Không chạy lại C để thay seed0 bằng kết quả đẹp hơn. Session split không mở thêm trial/candidate/order.

Fairness phải quyết định theo từng contrast:

- Trong mỗi seed, ưu tiên cùng execution runtime cho CADIC/R_EXACT/H32_EXACT. Seeds1/2 là fresh jobs, không resume seed0 hoặc copy weights giữa methods.
- C seed0 trên runtime A và FO/DIRECT controls seed0 trên runtime B tạo nguy cơ method×runtime confound. Một compatibility bridge PASS chỉ hỗ trợ equivalence trong scope đã đo; báo mixed-runtime origin và không gọi runtime effect đã cô lập toàn cục.
- Nếu numerical bridge không đủ cho matched attribution, vẫn được bảo toàn/report C nguyên trạng nhưng **không** dùng runtime khác để kết luận gain do exact meta-gradient/history. Có thể giới hạn kết luận descriptive, hoặc xin amendment/budget cho matched reference jobs trên target. Extra jobs nằm ngoài max15 hiện tại cần phê duyệt trước; không lén thêm hoặc thay các jobs cũ.
- Nếu target khác GPU model, không gọi cùng tài nguyên/hardware hoặc đưa timing vào matched overhead ratio. Báo wall/GPU/CPU/memory/cost theo segment và timing scope; không chia runtime theo DLPerf để “chuẩn hóa”.
- Aggregate phải đọc origin suite/job/cell identities và compatibility allowlist; validate từng origin riêng trước khi ghép. Không bỏ dòng `identity!=source` mà không thêm policy thay thế.
- Cùng tên job được transport nhiều lần chỉ deduplicate khi payload identity/hashes bằng nhau. Hai outputs khác nhau cho cùng logical job là conflict, không tự “latest wins”. Recomputed metrics từ cùng raw scores là analysis provenance mới, không sửa raw record.

Statistics vẫn paired whole-image/finite-three-seed đúng [study](study.md); disclose runtime blocks và development exposure. Không coi cells/seeds được tạo thêm bởi copy là independent replication. Missing jobs/cells không trở thành complete-study PASS.

## 9. Budget xuyên host và Stop/Destroy

Giữ budgets B12h/C24h và D allocation riêng; chúng là caps, không ETA. Dịch vụ được Start/Stop/Destroy bởi người dùng hoặc theo authorization vận hành cụ thể, không do import tool tự quyết.

Một **study-stage root ledger** append-only tham chiếu mọi origin/target segment và all attempts theo UUID. Segment có actual hourly rate và measured billed-time estimate riêng; tổng USD là tổng từng `seconds×rate/3600`, không đổi rate cũ thành rate máy mới. GPU wall cap cộng mọi segments của stage, bao gồm kỹ thuật/probes/migration/setup/idle khi GPU billed theo policy, không reset khi đổi folder/host. Disk/bandwidth sau Stop được báo riêng, không giả GPU ledger là hóa đơn Vast.

Allocation cũ giữ bytes; target allocation mới là child record của stage allocation. Cấp thêm tiền/giờ cần user amendment ghi cumulative cap mới, không xóa charged attempts hoặc tạo stage ID giả để né cap. Double-import cùng attempt UUID không charge lại; overlapping intervals/unknown setup time được flag. Failed/crash reservations giữ conservative charge cho đến có reconciliation được review, không tự refund để vừa budget.

Archive có open reservation; sidecar mới chốt export: giữ cả hai và verify archive SHA/attempt binding. Không sửa ZIP để ledger “đẹp”. Migration receipt giữ chain các lần reconciliation. Actual rate nhập sai cần correction record với stated effect, không edit old allocation in place.

Trước Destroy: confirmed completed/quiescent output → backup đầy đủ → verify outer manifest/ZIP/accounting → second durable copy → user quyết định. Stop không giữ GPU slot; chuyển ngắn giữa stages có thể đắt hơn nếu phải thuê host mới và probe lại. User cân đối chờ review với tiền idle; không tự mặc định Stop sau mọi stage bất kể nguy cơ mất chỗ.

## 10. Acceptance và negative tests trước khi gọi runnable

Implementation cần tests độc lập cho:

1. Same-runtime physical-host change và same-session path relocation; origin locks/checkpoints/events/cells nguyên bytes, next stage skip đúng work đã completed.
2. B→C fresh initialization, không import pilot bank/learner/Adam/history/pending/private RNG; C→D chỉ resolve completed C để screen/aggregate, new jobs fresh.
3. Missing/corrupt archive/chunk/task state/receipt/accounting/source; duplicate/path-traversal/symlink/unknown schema; interrupted import không publish partial acceptance.
4. Changed driver/GPU/precision/packages/source; compatible policy chỉ accept đúng ordered origin→target; unknown/failed/incomplete bridge bị reject, transitive bypass bị reject.
5. Real completed H32 partial/history states transport; resume parity qua mọi partial lengths1–7 cho same runtime, pending/censor/finalized/clock/RNG/event-chain/atomic crash. V1 vẫn reject changed-runtime in-flight learned-state resume.
6. C coverage đủ3×3629 và3×120cells, exact checkpoint/test bindings, screen recompute from frozen raw-linked origins; screen FAIL/partial/conflicting job outputs không mở D.
7. Aggregator giữ per-origin identity; allowed mixed runtime descriptive vs rejected unsupported matched claim; renamed seed/method không vượt guard; paired image/mask bindings và finite-three-seed rules còn nguyên.
8. Per-host rates, cumulative caps, imported reservation/sidecar reconciliation, repeated-import idempotence, no ledger reset/free budget; no accidental GPU/API/SSH/Stop from offline pack/import.
9. Source-diff receipt explicit, documented-only changes không sửa whole source digest; real scientific diff không lọt qua infrastructure label.

Local fixtures chứng minh implementation contract, không thay actual cross-host GPU bridge. Real acceptance tối thiểu cần một bounded same-runtime transport và một changed-runtime bridge với reference outputs thật; báo trường hợp chưa có. Không gọi fully portable trước kết quả đó.

## 11. Kế hoạch triển khai tối thiểu và điều kiện freeze

1. **Offline pack/inspect trước:** strict manifests, immutable origin index, coverage/gates, budget lineage. Không thay runner/core. Dùng old completed archives làm fixtures kiểm parser, không gọi là new GPU proof.
2. **Completed C→D, same recorded runtime:** explicit origin references trong gate resolver và aggregator, old loaders vẫn strict. Đây là ưu tiên giúp giữ 7.1/7.2; bảo toàn max15jobs và all old hashes.
3. **B→C same runtime:** rebind operational segment, new allocation, fresh-main invariants; no pilot rerun khi source/data/runtime thực sự bằng.
4. **Source mapping và changed-runtime bridge:** freeze exact fields/inputs/snapshots/reference outputs/equality/limitations trước GPU; amendment cohort comparability và statistics policy. Không biến thành wrapper `--ignore-runtime`.
5. **Real bounded transport acceptance:** copy+verify+probe+operator review trước khi tin đường mới cho paid main/extension. Changed-runtime in-flight training remains separate proposal.

Trước implementation phải chốt: schema names/version, supported origin deliveries, reference tensors/snapshots thực sự còn có, exact field policy, source mapping review scope, allowable claims khi mixed runtime, job budget/cap và tests/rollback. Không cần sửa learning equations để làm transport; nếu phát hiện phải đổi equations thì tách method revision.

**Kết luận đề xuất:** có thể thiết kế ba đợt độc lập mà không train lại toàn bộ phần đã xong. Hiện chưa có migration API đã nghiệm thu; chỉ đổi host với same identity đã không bị code chặn riêng theo machine ID. Các trường hợp numerical/source đổi cần giao thức trên, không hứa preflight mới tự động thay toàn bộ scientific pilot hoặc bảo đảm mọi host đều giữ nguyên trajectory.

## 12. Cơ sở bên ngoài và giới hạn áp dụng

[PyTorch 2.11: Reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html) không bảo đảm identical results xuyên releases/platforms, và lưu ý deterministic mode có chi phí tốc độ. Vì vậy đổi driver/GPU không được suy thành numerical parity chỉ từ seed hoặc tên model. Yêu cầu bridge ở trên là quyết định của dự án; tài liệu PyTorch không chứng nhận bridge hữu hạn đủ chứng minh mọi future trajectory.

[PyTorch: Saving & Loading a General Checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training) phân biệt inference weights với state cần để resume, gồm optimizer và tiến độ. Dự án còn phải giữ normal bank, fast/slow states, pending credit, history/RNG, events và locks theo contract riêng. Không thay full experiment bằng một file model weights hoặc giả rằng tải tensor trên device mới là đã xác nhận equivalence. Hai nguồn này hướng dẫn nguyên tắc lưu/truy vết; không được dùng làm bằng chứng thực nghiệm cho stage migration chưa implement.
