# Recovery r2 artifact audit và registry cache binding fix r3

Ngày2026-10-16, scope `D:/src/DATN/hope_cad_v5`. Không sửa sibling, frozen119files hoặc các resultsW6/M7.1/M7.2. Không commit/push. Machine review: `results/recovery_stage1/preflight_r2_review.json`.

## 1. Kết quả thực tế

Input r2 ZIP365033777bytes, SHA256 `2a241ed424d31de78eb9b2a1b0e1a8c88c9ff812732908723343569f7b728fa3`. Toàn151entry hashes và inventory đã verify. Source operations digest `a75f3f333fc37550bdf7812d62134224bfdfb2a732636fe9d78ec038e24932eb` đúng r2. RuntimeT4/CUDA12.8,130frozen+37recovery tests PASS. R2 vẫn resource_timeout600s, preflight chưa PASS, controls/AD comparison chưa chạy.

Mâu thuẫn quyết định: actual fit log ghi bank6272 và8effective image updates; `cadic_resource_probe.json` ghi `normal_train_images_completed=8` nhưng `coreset.count=0`, `seen_features=0`, `update_batches=0`. Đây không chỉ là T4 chậm: hai đường đọc khác bank. Source r2 quả thực có gọi accelerate; không phải user gắn nhầm source.

## 2. Root cause xác định

`models/benchmark_registry_v2.py:RegisteredAdapter` chuyển reads qua `__getattr__` đến concrete CADIC adapter. Nó không forward assignment. R2 gán `adapter.coreset=cached` trên RegisteredAdapter, tạo shadow attribute. `fit_task`/`score_batch` là bound methods của concrete inner adapter, vẫn dùng original coreset. Compact snapshot và resource report đọc shadow cached coreset rỗng.

Đây là bug tích hợp trong code hỗ trợ tôi bàn giao. Tests toán học của CachedPairCoreset đúng nhưng chưa kiểm production registry composition. SimpleNamespace fixture trước đó không đại diện cho forwarding wrapper;37tests PASS không phủ integration này. Nó giải thích cache r2 không tăng tốc actual fit. Chưa kết luận đây là bottleneck duy nhất sau sửa.

Gate timeout đã chặn trước controls; không có full matched benchmark trên bank rỗng. Không đổi nhãn failure này thành success hay diễn giải như negative research result. Nếu lỗi này không bị chặn, checkpoint/scoring consistency có thể hỏng; vì vậy r3 kiểm ownership và counters độc lập.

## 3. Sửa r3

`notebook_support/recovery_stage1/cadic_accel.py:CachedCADICAdapter` nhận diện RegisteredAdapter, gắn cache lên concrete owner (`adapter.adapter.coreset`), không tạo shadow trên registry. Kiểm cả registry read, concrete bank và cached bank là cùng object; fit/score/state/load bound methods cùng concrete owner. Fit-return counters phải bằng bank được snapshot. Thay binding ở runtime phải fail trước update/scoring.

Resource probe ghi binding receipt, cache usage/progress và coreset thật. Nếu hết16images nhưng `closest_pair_calls<=0`, fail. Supporting gate kiểm binding shared, actual registry wrapper, backend và cache-used count trước reference parity; không tin label optimization đơn lẻ. Lỗi worker giữ progress receipt để điều tra.

Pair-cache mathematical algorithm, distance primitive/tile shapes/ties/scoring không đổi so r2 intended cache. Execution backend tăng thành `cadic_exact_pair_tile_cache_v2`; delivery revision `registry-cache-r3`, source ZIP mới. HOPE dynamics/weights không thay. Các source/run identities mới khác r2; không import failed learned resume dưới source mới.

## 4. Regression và giới hạn

Thêm3tests qua actual `build_adapter -> RegisteredAdapter -> concrete CADIC adapter`, chỉ mock ViT bằng deterministic feature extractor. Cả3 FAIL trên r2 trước fix và PASS sau fix:

- Train/score/snapshot dùng cùng CachedPairCoreset; không shadow registry; actual closest-pair cache được gọi.
- Reference vs registered cached adapter cùng features/counters/Eq9; CompactCADIC state/load/continue giữ exact parity.
- Cố tình tạo shadow bank phải reject trước bất kỳ update.

Cùng37tests trước tạo40tests. Fixtures không là canonical ViT/T4 hoặc AD benchmark. Original119source hashes giữ nguyên;130frozen tests đã PASS trong r2 trênT4. Source r3 vẫn cần preflight T4 real RGB, full10k cache use và cloned real-normal reference parity. Không hứa hết mọi runtime bottleneck hoặc performance gain từ sửa integration này.

## 5. Bàn giao

Dùng notebook preflight đã cập nhật và `ultils/recovery_stage1/hope_cad_recovery_stage1_source_r3.zip`; giữ original W6/completed7.2/MVTec/ViT. Không attach failed outputs. GPU T4 x2, Save Version / Save & Run All. Sau PASS lưu ZIP/log để review trước controls. Không chạy lại W6. Handoff mục10 ghi từng bước. Giữ r1/r2 ZIP/source/receipts làm evidence, không ghi đè.
