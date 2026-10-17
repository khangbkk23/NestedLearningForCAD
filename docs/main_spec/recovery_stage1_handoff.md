# Nested Learning for CAD — recovery stage 1: bàn giao Kaggle

Ngày: 2026-10-16. Delivery hiện hành: **lifecycle-r4**. Worktree triển khai: `D:/src/DATN/hope_cad_v5`, branch registry `hope_cad_v5`, base `4dd6f36fe093b2ae153d196a9333aeda1d196a06`. Tham chiếu read-only từ worktree `hope_cad_v2_2`. Không commit/push/merge trong lượt này.

## 1. Phạm vi

Đây là **gói điều tra trước method revision**, tiếp theo `research_recovery_report_v1.md`. Tên worktree v5 không có nghĩa đã có một thuật toán v5. Giữ nguyên HOPE-CAD v4, matched CADIC, W6 initialization/selection và kết quả M seed0 7.1–7.2.

Gói này triển khai:

- Matched `C1_seed0`, `A1_seed0`, `CADIC10000_seed0`: mỗi job stream đủ 3.599 normal images, lưu 15 task checkpoints và score **final row** đủ 1.725 test images. C1/A1 dùng initialization W6, không refit calibration.
- Diagnostics normal cho **M/C1/A1**, trên 8 online train identities/category được chọn bằng hash cố định, ở checkpoint ngay sau category và checkpoint cuối. Cùng samples giữa methods; không dùng validation/probe/test. Có 29 checkpoint–category pairs khác nhau × 8 = 232 image measurements/method; last-task diagonal/final được dùng chung, không tính lại.
- Actual stage norms, radial/angular residual decomposition, mean/p95/p99/max, prediction norm, max-score patch, discarded one-event own-gradient/update diagnostics và per-category diagonal→final normal-loss change.
- Observer lưu return losses/counters của `fit_task`, không thay update/state/RNG/score. CADIC ghi coreset counters. Trace theo từng effective transport call; CADIC giữ extraction batch8 và sequential per-image writes.
- Source/parent/runtime/data locks, strict JSON identity normalization fix đã có, failure export, resume và comparison mô tả seed0.

Không triển khai learned-memory proposal, slow online decoder mới, outer refit W6, full task matrices cho controls, full35-job W7.3, bootstrap CIs hoặc VisA. Các normal measurements là reused/in-sample forensic evidence; không phải independent held-out generalization và không được dùng để đổi selected recipe.

## 2. Tính toàn vẹn và newline checkout

Checkout mới từ Git tạo28 khác biệt LF/CRLF so với bytes của run cũ. Đã kiểm từng file: có thể đổi newline để khớp SHA256 cũ chính xác; không có content change. Đã khôi phục đúng bytes trong worktree này, bảo toàn119 frozen dependencies.

- Bản sao trước khôi phục: `results/recovery_stage1/checkout_pre_normalization.zip`.
- Biên nhận từng before/after SHA: `results/recovery_stage1/checkout_normalization.json`.
- Tham chiếu khóa: `notebook_support/w7_resume_fix/frozen_reference.json`.

Nếu clone Git mới rồi source guard fail, **không cập nhật frozen hashes cho qua**. Đối chiếu newline với bản đã khóa trước; bundle bàn giao ở đây đã giữ đúng bytes. Không thay `.gitattributes` để che mismatch. Helper `python -B -m notebook_support.recovery_stage1.restore_checkout` chỉ kiểm tra; thêm `--apply` mới backup và sửa đúng pure-newline variants. Helper từ chối mọi content difference, không cập nhật reference hashes.

## 3. Files cần dùng

1. `ultils/recovery_stage1/hope_cad_recovery_stage1_source_r4.zip` — source mới, Kaggle có thể giải nén thành directory; notebook nhận cả ZIP và extracted folder.
2. `notebooks/hope_cad_recovery_stage1_preflight_kaggle.ipynb`.
3. `notebooks/hope_cad_recovery_stage1_controls_kaggle.ipynb`.
4. `results/recovery_stage1/delivery.json` — bundle SHA/files và ops identity.

Đây là bundle/schema riêng, **không dùng source ZIP W7 cũ thay thế**. Frozen learner source vẫn giữ nguyên. Không cần input `mvtec_prefix_features_final.pt` hoặc chạy lại cache/calibration W6.

## 4. Lần đầu: chạy recovery preflight

Tạo/import notebook `hope_cad_recovery_stage1_preflight_kaggle.ipynb`. Chọn GPU T4 x2 (runner dùng một GPU). Giữ environment đã chạy W6; notebook kiểm exact runtime, không tự cài/upgrade thư viện. Internet không cần cho backbone vì dùng local checkpoint.

Attach **5 inputs**. R4 chạy fresh preflight: **không attach failed recovery output**; nó thuộc source identity cũ và chỉ giữ làm evidence. Không dùng source ZIP cũ.



| Input | Nội dung cần có |
|---|---|
| NEW recovery source | Source ZIP ở mục3, hoặc Kaggle-extracted contents có `bundle_manifest.json` schema `hope_recovery_stage1_source_v1` |
| Completed W6 v4 | `hope_cad_v4_normal_evidence.zip` từ `docs/w6_kaggle_v4.1`, hoặc extracted Output cùng manifest đã khóa |
| Completed W7.2 mới | `hope_cad_w7_evidence.zip` từ `docs/w7_kaggle_7_2_1`, hoặc đúng extracted Output; không phải preflight7.2/failed attempt/7.1 |
| MVTec AD | Archive hoặc extracted official dataset như các run trước |
| ViT-B/8 | `vit_base_patch8_224.augreg_in21k.safetensors` như trước |

W6 ZIP SHA `ae9d257ecf1ced738df651fa07887b838a96da5fe4cdda84e46c51533c9bcd71`.
W7.2 ZIP SHA `a4cba0f4c04c1c8e6677eb79e1f51b419efb8d88a02838fc91cd5f1086469563`.
ViT SHA `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`.

Bản sao extracted cùng pinned manifest được chấp nhận; hai **phiên recovery khác nhau** phải giữ latest hoặc đặt `RESUME_OVERRIDE` rõ. Notebook có overrides cho source/W6/W7.2/resume/MVTec/checkpoint, mặc định tự tìm đúng input. Không sửa các hàm bên dưới config cells.

Bấm **Save Version → Save & Run All**. Preflight:

1. Kiểm source files, import chọn lọc W6 weights/receipts và M states/metrics từ7.2; không copy toàn2.48GB reference vào output.
2. Kiểm runtime đúng W6, backbone/data/source identities. Byte integrity kiểm cả train manifest; learner/diagnostic chỉ forward legal normal samples theo event/checkpoint.
3. Chạy regression suite frozen và recovery tests.
4. Trên9 normal prefix images, so unobserved run với observed run ngắt sau5events/resume cho M/C1/A1. Kiểm exact state/RNG/score và diagnostics read-only.
5. Chạy CADIC10k cached probe trên16 normal prefix images qua cùng extractionB8/per-image update; kiểm full bank, score read-only, checkpoint roundtrip. Đầu tiên kiểm concrete registry bank chính là bank train/score/snapshot và fit-return counters khớp. Trên cloned full bank10k, so pair/decisions/state/Eq9 với reference dùng4actual normal patch candidates. Đây là operator fixture và proof/test-supported optimization, không giả nhận đã dual-run toàn16ảnh với reference chậm.
6. Export `hope_cad_recovery_stage1_evidence.zip` rồi mới báo failure nếu có.

**Được chuyển bước khi:** `handoff.json.status = recovery_preflight_passed` và `recovery_preflight.json.passed = true`, không chỉ vì notebook cells có output.

### Resource gate CADIC cần đọc kỹ

Đã đo local trên **features giả**, RTX3050Ti: một `_closest_pair` tại K10000,D768 mất khoảng1.75s. Đây không là T4 real throughput hoặc benchmark; actual replacement counts quyết định cost. Frozen reference quét pair distances lại mỗi replacement nên có nguy cơ runtime lớn. Run T4 đầu đã timeout600s ở route này; r2 bổ sung exact tile-min cache và reference parity gate, xem `recovery_stage1_cadic_timeout_review.md`.

Resource worker có watchdog **600giây**, hoặc ít hơn nếu hết session budget. Nếu chưa hoàn thành16normal images/full-bank route trong giới hạn, nó bị dừng và ghi `cadic_resource_probe.json.status = resource_timeout`; global preflight không PASS. Khi đó giữ ZIP/log để review engineering cost. **Không xóa gate, giảmK, đổi update unit, đổi chunk math hoặc thay source hash để tiếp tục như cùng baseline.** R2 có lỗi binding registry khiến cache chưa vào actual fit path; R3 sửa concrete owner và kiểm bank identity/counters/cache-used fail-fast. Xem `recovery_stage1_registry_fix_r3.md`. R3 đã đạt full16normal images trong122.40s và reference parity trênT4, nhưng thiếu close() nên fail sau compute; r4 sửa lifecycle. Xem `recovery_stage1_lifecycle_fix_r4.md`. Nếu resource timeout xuất hiện lại thì cần profile tiếp; không tăng timeout để che chi phí. Auxiliary cache được khai riêng, run identity CADIC có execution backend mới; science budget/update/scoring không đổi.

Vì vậy không hứa preflight vài phút hoặc baseline xong trong một đêm. Thời gian thực T4 và counts sẽ có trong profile. Gate này giúp phát hiện bottleneck trước long run, không được dùng để chọn method theo anomaly metrics.

## 5. Sau preflight PASS: chạy controls

Import `hope_cad_recovery_stage1_controls_kaggle.ipynb`. Dùng cùng source/W6/MVTec/ViT; **thay input W7.2 bằng Output/ZIP của recovery preflight vừa PASS**. Parent M đã nằm trong recovery output, không bắt buộc attach lại7.2. Giữ5inputs là đủ. Có thể giữ7.2 riêng nhưng không cần đưa nhiều phiên recovery cũ cùng lúc.

Giữ default:

```python
PHASE='controls'  # notebook đặt sẵn
JOBS=['C1_seed0','A1_seed0','CADIC10000_seed0']
SESSION_HOURS=10.5
```

Thứ tự C1→A1→CADIC là quyết định vận hành cố định để có predictor controls sớm, không phụ thuộc kết quả test. Có thể chọn một subset đã đăng ký để chia phiên; muốn hoàn tất giai đoạn cần đủ3jobs và M/C1/A1 diagnostics. Không thêm seeds/methods khác vào gói này.

Runner:

1. Kiểm/import parent và matching preflight, không refit W6.
2. Đọc15M checkpoints để diagnostics normal; không train/rescore M official test.
3. Với từng requested control, stream normal một lần, checkpoint/resume, final15cells và diagnostics nếu là C1/A1.
4. Lưu `comparison.json` chứa macro4metrics, deltas vs matched CADIC và A1−C1/M−A1 descriptive contrasts. Không chọn rates/steps hoặc đưa ra superiority claim tự động.
5. Hoàn tất hoặc pause thì export ZIP. Notebook có heartbeat30s khi child chưa xuất thêm log; không xem silence đơn thuần như error nếu heartbeat còn chạy.

## 6. Pause/resume và xuất artifacts

Nếu handoff là `paused_at_atomic_boundary`:

- Lưu version Output/ZIP mới nhất; lần sau dùng **latest recovery evidence** thay preflight hoặc phiên cũ.
- Giữ đúng source bundle, reviewed W6, MVTec và ViT. Không đổi method/rates/normal sample rule.
- Dùng controls notebook với originalJOBS; jobs đã train/final/diagnostic xong được kiểm và reuse, không train hoặc score lại những phần đã commit.
- Nếu chỉ chạy subset rồi status `requested_jobs_complete_remaining_pending`, chọn remaining jobs hoặc giữ đủJOBS để hoàn tất.

`SESSION_HOURS` giới hạn trong0.05..11, tính từ đầu notebook và trừ180s reserve. Cần chọn thấp hơn quota/session thực để đủ export. Atomic CADIC operation có thể dài; watchdog600s chỉ áp resource probe, không đổi scientific update. Không hứa chắc export nếu Kaggle hard-kill, hết disk/quota đột ngột hoặc mất kernel trước finally. Normal controlled pause/failure sẽ export trước khi raise.

ZIP nằm ở `/kaggle/working/hope_cad_recovery_stage1_evidence.zip`. SaveVersion output là bản cần bảo toàn; `FileLink` một mình không có nghĩa file đã về máy bạn. Tải ZIP và log, đặt vào worktree này, ví dụ `docs/recovery_stage1_kaggle/preflight/` và `docs/recovery_stage1_kaggle/controls/`. Không overwrite parent archives.

## 7. Điều kiện hoàn tất giai đoạn1

- Source/parent/runtime/data lock khớp; r4 preflight PASS trên T4 thật, registry binding/cache-used/reference parity và cleanup_complete đạt.
- C1/A1/CADIC10k đều3.599normal images,15checkpoints và đủ15final evaluation cells/1.725test images.
- M/C1/A1 normal diagnostics đủ15categories/8samples/history, không future-task feature reads hoặc main-state/RNG mutation.
- Observer traces lưu normal loss/counters, raw final score maps/labels/masks/identities có hash; failures và pauses được giữ. Mỗi control phải có `lifecycle_completion.json` sau cleanup thành công.
- Handoff `recovery_stage1_complete_pending_review`, `comparison.json.status=complete`, các diagnostics summaries complete; sau đó cần review artifacts.

Gói này **không hoàn tất full W7**. HOPE control state/run digests giữ frozen W7 construction; CADIC có execution optimization metadata nên run digest riêng. Muốn tái sử dụng cần verified identity bridge, nhưng recovery ZIP dùng schema riêng; muốn nhập về full W7 cần một verified bridge, không attach thẳng vào notebook W7 cũ rồi coi matrix đã có.

Kết quả tốt/xấu đều cần giữ. Dữ liệu MVTec đã development-exposed. Không dùng VisA để quyết định sửa kiến trúc. Sau review mới chọn một revision có contract đầy đủ, mechanism pilot và prospective VisA freeze.

## 8. Codespace và local verification

Code mới ở `notebook_support/recovery_stage1/` để không chèn file vào119frozen source map:

- `common.py`: pinned parent import, ops source lock, legal sampling và loss observer.
- `runner.py`: exact original W7 identities, final-only controls, M checkpoint diagnostics, comparison/status.
- `diagnostics.py`: real stage measurements, radial/angular decomposition, discarded gradient proposal và resumable per-image outputs.
- `preflight.py`: real normal RGB observer/resume parity và focused tests.
- `cadic_profile.py`: bounded full-budget resource probe và reference parity.
- `cadic_accel.py`: exact tile-min cache, cache invalidation, cloned reference audit, execution identity và memory accounting.
- `bootstrap.py`: input/schema/duplicate-copy/runtime checks, safe dataset extraction.
- `entry.py`: logs/heartbeat, export-before-error, explicit pause/failure status.
- `restore_checkout.py`: bảo toàn locked bytes qua fresh checkout, check-only mặc định.
- `build_delivery.py`: notebooks/source bundle; verifies119dependencies before packaging.
- `tests/recovery_stage1/`: new meaningful contracts and orchestration tests.

Local dùng Python của environment Pixi hiện có ở worktree cũ ở chế độ read-only runtime, vì worktree mới chưa có`.pixi`; mọi fixtures/outputs được ghi trong `hope_cad_v5`. Không chạy `pixi install` vào sibling hoặc sửa source sibling. Equivalent khi environment v5 đã có:

```text
pixi run python -B -m unittest discover -s tests/recovery_stage1 -v
pixi run python -B scripts/hope_cad_v2/verify_w7.py --output results/recovery_stage1/frozen_regressions.json
pixi run python -B -m notebook_support.recovery_stage1.build_delivery
```

Local tests/compile/packaging là engineering evidence. Chưa có T4 preflight hoặc research controls results từ gói recovery này tại thời điểm bàn giao; không nâng nhãn implemented/tested thành experimentally validated CAD gain.

## 9. Bằng chứng engineering local của lần bàn giao

- Frozen regression suite: **130/130 PASS**, không errors/failures/skips. Hai historical v3 notebook UI tests bị loại đúng danh sách cố định của frozen verifier, như các lần W7 trước; không loại test bất lợi mới.
- Recovery suite r4: **45 tests** về contracts, source ZIP/extracted bootstrap, exact observer/resume, normal stage arithmetic, resource watchdog, failure export và final-only reuse. Xem kết quả máy đọc được tại `results/recovery_stage1/recovery_verification.json`.
- Actual parent compatibility: strict load **7 W6 initializations** và **15 M checkpoints**, frozen outer weights bằng W6, CMS clocks/commits/pending đúng. `results/recovery_stage1/real_parent_compatibility.json`. Không restore T4 RNG trên GPU khác hoặc claim canonical ViT/T4 inference từ kiểm tra CPU này.
- New code được compile, notebooks/cellIDs và ZIP entry hashes được kiểm trước bàn giao. `results/recovery_stage1/delivery_validation.json` và `delivery.json`.
- Synthetic local CADIC pair-search profile ở `local_cadic_pair_profile.json`; không suy full-stream ETA hoặc AD từ profile đó.
- **Kaggle preflight/controls/normal diagnostics trên data thật còn pending.** Chỉ chuyển scientific status sau khi đọc artifacts.

## 10. Chạy preflight lifecycle-r4 sau review r3

1. Giữ các failed ZIP. Tại thời điểm bàn giao, file `hope_cad_recovery_stage1_evidence.zip` là r3, SHA `db16e48ab160b44811ecf602bfad098998783e42e8c61f69a6047d71bcaac9bb`,153entries; `_old_1.zip` là r2 SHA `2a241ed424d31de78eb9b2a1b0e1a8c88c9ff812732908723343569f7b728fa3`; `_old.zip` là r1 SHA `ff034113ab9f846054585f6b02eb938a1b14bdbd5e0781e624fc9e39962152f5`. Trước khi lưu lượt mới, rename r3 để tránh overwrite.
2. Import lại notebook preflight mới. Replace source input bằng `hope_cad_recovery_stage1_source_r4.zip` hoặc extracted contents của chính ZIP này. Không đổi tên ZIP; notebook yêu cầu revision `lifecycle-r4`.
3. Giữ4inputs original W6 v4, completed W7.2 `w7_kaggle_7_2_1`, MVTec, ViT. Tổng5inputs. Không attach failed recovery outputs; source identity khác sẽ bị từ chối trước restore/copy. GPU T4 x2, CUDA runtime đúng W6. Không chạy CPU hoặc tự pip install.
4. Save Version / Save & Run All. R4 chạy tests trọn worker subprocess và control lifecycle, rồi actual normal checks/resource probe như trước. Worker phải exit0 và ghi cleanup_complete=true mới được PASS. Không tăng watchdog600s.
5. Sau `recovery_preflight_passed`, giữ ZIP/log để review. Sau nghiệm thu dùng controls notebook mới, cùng source r4/W6/MVTec/ViT; replace W7.2 bằng đúng output preflight r4. Resume latest r4 output ở các phiên sau.

Không rerun W6/calibration hoặc các M official test metrics. Các failed archives giữ đúng status gốc; không chỉnh resource receipt r3 từ failed thành passed. R4 chỉ sửa lifecycle/acceptance/operational tests; thuật toán/backend pair-cache/scoring giữ nguyên. Source/notebooks/receipts cũ đã backup dưới `ultils/recovery_stage1/pre_timeout_delivery/`, `pre_r3_delivery/`, `pre_r4_delivery/`. Chi tiết và mức bằng chứng tại `recovery_stage1_lifecycle_fix_r4.md`.
