# V7 — bảo toàn source, tối giản vận hành và chuyển runtime

Trạng thái: implementation engineering đang thực hiện theo yêu cầu22/10/2026. **V7 là tên workspace, method đối chiếu vẫn v5.1**. Không đổi frozen v5.1, không gán cleanup/GPU mới là performance gain.

## 1. Mốc bảo toàn

V6@84cccbae có149 untracked files,90Python và3 tracked modifications. Trước mọi cleanup, lưu toàn bộ269 working source files (gồm configs/specs/tests/papers cần có), manifest/hash, Git status và patches tại maintenance record chung. Không copy .pixi/caches/data. V7 có commit bảo toàn đầy đủ trước refactor; v6 không bị stage/commit/sửa.

V6 working bytes, Git blob normalized line endings và source bundles cũ là ba danh tính có thể khác nhau. Archive gốc giữ byte-exact; v7 phải giữ nguyên byte hash của scientific specs/kernels cần để design digest không đổi qua clean checkout Linux. Không gọi một SHA Git là identity của dirty v6.

## 2. Phạm vi cleanup

Giữ core models/CADIC/ViT, visual/CMS/oracle, v5 geometry dependency, data rights/preprocessing/QA, atomic engine, scoring/metrics/bootstrap và numerical/causal/resume tests. Không giảm K, anchors, replay/unroll, frequency, exact gradient, precision hoặc image count để chạy nhanh.

Thay notebook/build/bootstrap theo từng delivery bằng package `hope_cad`: CLI `preflight`, `train`, `evaluate`, `matrix`, `aggregate`, `export`. Common runtime/source/evidence/NN-backend dùng chung. Historical profile/bridge/continuation wrappers chỉ bỏ sau import review và replacement parity tests. Legacy wrapper-specific tests có thể chuyển khỏi active suite nếu entrypoint bị retire; inventory phải ghi rõ, core contracts không bị vô hiệu hóa để che failures.

Không tự resume migrated v6 owner bằng v7 execution identity. Existing7.1/7.2 archives/checkpoints được giữ nguyên; chạy fresh trên runtime mới, hoặc specification/bridge riêng trước khi hỗ trợ import. V7 không cần `.agents` hay sibling checkout để chạy.

## 3. CLI và nghiệm thu

- `preflight`: fail-fast source/runtime/GPU/architecture/checkpoint identity; discarded full28×28 exact/FO meta gate; same-device reference/optimized NN/counter/Eq9 parity; bounded24train normals cho NESTED/CADIC, strict state roundtrip/read-only scoring. CPU mode chỉ synthetic/local, không mở real benchmark.
- `train`: explicit run-dir/data-root/checkpoint/device/job, accepted preflight receipt đúng runtime/source/backbone/normal manifest. Fresh bank/init; source/operational/runtime/data/job identity. Pause/checkpoint giữ pending, partial episode, RNG/cursor. Không đọc official test trước train completion.
- `evaluate`: final row từ completed own training, không train. `matrix`: saved tasks, read-only120cells, reuse existing final15, không đổi checkpoints/finalrow. Locks từ chối source/runtime/data changes.
- `aggregate`: frozen study paired metrics/CI; thiếu seeds không suy superiority. Một execution/source/runtime cohort; không trộn với old T4 owners hoặc revisions mới.
- `export`: verified original payloads/manifest, output ngoài run tree, giữ failure/pause. Host kill có thể cản export; không cam kết server disk là backup máy người dùng.

Keep registered9methods×3seeds, excluded30normal split, unchanged ViT/raw geometry/Eq9. V5.1 current evidence MVTec seed0 khác oldv4 results. VisA chưa runnable theo contract này; cần freeze/protocol riêng.

## 4. Remote5090/SSH

Candidate Linux/Python3.13/PyTorch2.11.0+cu128/torchvision0.26.0 cùng primary package pins canonical T4; **không tái dùng T4 runtime lock**. Kiểm actual GPU/driver/CUDA, architecture hỗ trợ, deterministic FP32/TF32off và CPU thread setting; ghi runtime receipt mới. Không AMP/compile/DDP ngầm.

Setup scripts/runbook/logging/checkpoint/export tự đủ trong repo. Không credentials hoặc endpoint host giả. Chưa thuê, SSH, cài remote hay chạy5090 ở local implementation task. Measurement thật quyết định ETA/USD-job, không FLOPS ratio. Runtime transition giữ scientific recipe, chưa có cross-device numerical equivalence claim.

## 5. Gates và revision sau đó

G1: bảo toàn269files vàv6khôngđổi. G2: dependency cleanup, scientific filehash+initialization/event/score/state+interrupted/resume parity, clean-source tests. G3: remote preflight/numerics/normal throughput do người dùng chạy trên máy thuê; chưa PASS chỉ vì local tests. G4: một revision khoa học có architecture/contract/acceptance riêng, rồi mới implement/pilot.

IDEA006/HYP018 historical-query consolidation được giữ là proposal. Nó không nằm trong cleanup/port runtime và không tự đổi method ID vì worktree tênv7. Trước G4 cần khóa buffer/query split, old-label semantics, sampling/RNG, memory budget, support/query disjointness, counters và trial/stop rules; không lấy VisA để tune. RuntimeG3 cần evidence thật trước khi gọi v7 ready for full benchmark.
