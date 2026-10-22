# Nested Learning for CAD — addendum tối ưu ghi nearest-neighbor v5.1

Ngày 21/10/2026. Execution candidate `nearest_fixed_shape_writes_v1`. Đây là addendum engineering sau [G1 review](../../../.agents/evidence/reviews/support-v5-1-throughput-profile-session1-20261021/review.md), mở rộng nhóm candidate ở [spec throughput](throughput_v5_1_spec.md), không sửa scientific method/study. Contract máy đọc: [nn-write contract](throughput_nn_write_v5_1_contract.json).

## 1. Vì sao chọn đúng một patch

G1 T4 đo full proposal92,83% thời gian P1; nearest bên trong proposal76,54%, actual fast+slow learning2,10%. Vì vậy batch này chỉ thay cách ghi kết quả nearest, không tắt hoặc rút ngắn Nested Learning. CUDA synchronization do dynamic indexing là giả thuyết về chi phí, chưa là kết quả kernel timeline hoặc speedup đã đo.

File reference `models/cadic_paperfaithful/cadic_patch_coreset.py::CADICPatchCoresetV1._nearest` giữ bytes. Candidate ở module riêng, activation có scope và phục hồi method gốc kể cả exception. Giữ nguyên FP32distance primitive/GEMM, tile256, query/bank slices, min/reduction order, strict `<`, traversal và index dtype. Cùng backend áp dụng cho mọi caller trong scoped process, gồm proposal, anchors, risk và Eq9; không tối ưu riêng NESTED khi so CADIC sau này.

## 2. Phép biến đổi và lập luận tương đương

Reference sau mỗi tile tính `mask = values < best_part`, rồi `best_part[mask] = values[mask]` và `index_part[mask] = start + local[mask]`.

Candidate giữ đúng `mask` đó, dùng `best_part.copy_(torch.where(mask, values, best_part))` và `index_part.copy_(torch.where(mask, start + local, index_part))`. Với từng query j: mask true chọn đúng value/index mới; false giữ nguyên cặp cũ. Strict comparison false khi tie hoặc values NaN, nên tie chọn index nhỏ nhất như reference; inf giữ sentinel nếu không cải thiện. Không thêm phép toán FP vào distance/value, không regroup GEMM, không đổi epsilon/normalization hoặc precision. Index addition thực hiện cả vector thay vì subset, không overflow trong bank10k/int64.

Điều này cho lập luận lựa chọn tương đương từng tile bằng quy nạp; vẫn phải kiểm backend thực, bit patterns/NaN/zero/ties và full trajectory. Không coi proof này là universal CUDA certification. Candidate không cache distance, không sửa pair-cache invalidation, không thêm preprocessing hoặc giảm số replacements. Giữ fullimage proposal kể cả veto.

## 3. Bất biến học và checkpoint

Giữ toàn bộ frozen method: raw784×768/K10000/fullEq9b9, four-direction visual/five maps, CMS/persistent phi, causal next-image credit,32×8anchors, loss-gradient rồi chia denominator và derivative quotient trong exact replay,4support+4query mỗi8labels, Adam/clip, warmup32/p≥0, precision/runtime, rollback/pending/partial clocks, checkpoint mỗi8ảnh. Không dùng FIRST_ORDER thay NESTED. Không official test/GT/VisA hoặc normals chưa quan sát để chọn patch.

Original166benchmark payloads và183profiler file-map entries giữ nguyên. Source mới thêm manifest riêng, không sửa bundle_manifest/throughput_manifest cũ. Không nạp source candidate vào benchmark376 bằng resume cũ. G4/migration chưa triển khai trong batch này; kết quả từ validation clones bị discard.

## 4. G2 local và G3 T4 validation

G2: so exact outputs/indices/bit patterns trên random/zero/tie/partialtiles/emptyqueries/invalid geometry/NaN-distance fixtures; giữ original distance block call sequence; cache full/no-op/fork/load/in-place fixtures; Eq9; full synthetic session/pending/partial/rollback/scoring; actual and exact-meta gradients/Adamsteps. Existing core/benchmark/profiler regressions giữ nguyên.

G3 cố định bốn pass **R1,O1,O2,R2**,16arrivals179–194 mỗi pass, fresh process/owner từ task00cursor179. R=original, O=candidate; không component profiler.64attempts tối đa, soft90min cumulative,180s export reserve. Full manifest3599 chỉ dùng quyền/identity; không finalize ở195. Periodic checkpoints184/192 thực sự được ghi/fsync/hash trong measured pipeline. No silent retry hoặc nối partial timing qua restart; incomplete/failure giữ evidence và review.

Captures tại credit events181/189, ngay trước `NestedUtilityLearner.observe`: full learner serialized state (records7/block_initial/phi/theta/Adam) và incoming completed record. Cả R/O dùng cùng wrapper/capture schedule. Clock synchronize trước capture, dừng primary segment, copy CPU/capture, synchronize rồi mở segment mới. Report cả `pipeline_seconds_excluding_diagnostics`, capture time và total wall; không gọi segmented time là continuous production time. Không thêm arrival/replay16ảnh chỉ để lấy gradients.

Ngoài primary timing, dựng disposable learner từ mỗi capture, recompute raw actual fast gradients/loss/d/updates; support4increments/denominators; exact meanquery4loss/rawslowgradients trước clip; chạy frozen observe trên clone để lấy actual phi, theta/Adam sau update. Đây là diagnostic recomputation, không học thêm vào actor/benchmark; chi phí tính trong90min. Dùng private seeded owner/RNG preservation. So mọi tensor/scalar exact giữa4passes, bao toàn parameter groups. No approximate fallback.

Events/feature receipts so exact với reviewed parent. Final scientific state195 so exact across4owners và digest G1 đã review. Không lưu4bản bank60MB: R1state giữ riêng; O/R2 hash+runtimeassert againstR1. Raw gradients/steps và diagnostic capture digests giữ trongZIP để reviewer kiểm độc lập. No intermediate wholebank dump mỗi event trong timing.

Speed gate: aggregate Otime/Rtime ≤1; cả hai paired ratios phải<1 và aggregate gain lớn hơn max(abs(R1−R2)/meanR, abs(O1−O2)/meanO). Đây là conservative noise heuristic với2pairs, không statisticalCI. Nếu exact nhưng chưa rõ speedup, ghi `exact_parity_speed_inconclusive`, không tự thử thêm lần hoặc hạ gate. Memory gate: noOOM, report baseline/peaks; Opeakallocated không vượt max(Rpeaks)+64MiB (envelope engineering đã chốt trước run; không là modelcapacity cap). Vượt thì defer review, không bớt memory/learning để qua.

Complete fourpasses chỉ là `validation_complete_pending_review`; final acceptance cần artifacts review. Parity mismatch fail closed. Không sửa source/threshold từ output để giữ cùng frozencandidate. Source mới hoặc retry cần execution lineage/attempt ledger riêng.

## 5. Bàn giao và dừng

Notebook/source mới chỉ chạy G3 sau localchecks; bootstrap verify exact original parent/runtime/backbone trước việc nặng. Bốn inputs: newNNsource, reviewed376benchmark parent, MVTecraw, lockedViT. G1profile đã review được bind bằng manifest/source/receipt hashes trong contract; không cần attach profiler output hoặc W6/pilot learned states.

Output schema riêng; resume chỉ latest output của validation này, skip completed/sealed passes. Lưu `.agents/artifacts/runs/support_v5_1_nn_write_validation_01/session_01/`, tăngsession khi có hướng dẫn; giữ ZIP/log. Không ghi đè source/delivery cũ. Default budget cố định, không phải ETA.

Sau G3 user-run: review parity/cost/memory/gradients. Đạt mới bàn giao G4 compatibility loader và canonical bridge8arrivals376–383 theo spec gốc; chưa có quyền tự chạy GPU/VisA hoặc tự adopt candidate. Task implementation này dừng tại notebook/source để user chạy. Theo runtime đã đo, hai referencepasses riêng khoảng34phút trong window này; tổng phụ thuộc candidate, bootstrap/diagnostics và quota90min, không bảo đảm ETA hoặc speedup.
