# Nested Learning for CAD — bàn giao candidate NN-write v5.1

Ngày21/10/2026. [Engineering addendum/proof/acceptance](throughput_nn_write_v5_1.md), [contract](throughput_nn_write_v5_1_contract.json). Candidate `nearest_fixed_shape_writes_v1` chỉ thay masked gather/scatter bằng fixed-shape selection. Math v5.1, scientific method/study và frozen166benchmark/183profiler payloads giữ nguyên. Canonical candidate speedup/parity chưa có trước user run. Không tiếp tục benchmark bằng source mới trong notebook này.

## Mã nguồn và trạng thái

`training/support_nn_write_v5_1/candidate.py` chứa operator/scoped activation; distanceGEMM/min/tile order giữ nguyên. `diagnostics.py` có synchronized segmented clock và pre/post learner captures. `replay.py` giữ global179→195/full3599manifest/cadence184/192, pending/partial và nofinalize. `worker.py` thực hiện R/O, exactevents/receipts/state195 so reviewedG1, rồi actual/meta gradient recomputation và postobserve steps. `runner.py` quản lý R1/O1/O2/R2,64attempts/90min, failclosed/incomplete, paired speed/noise/memory gates. `contract.py` khóa source riêng và base183 bất biến.

`notebook_support/support_nn_write_v5_1/` gồm bootstrap/entry/build; reuse verified original16image assets, reject completedprofileroutput asresume, source/runtime/parent locks riêng và exportfailure. Notebook [NN-write validation](../../notebooks/hope_cad_v5_1_nn_write_validation_kaggle.ipynb) là interface user-run. `tests/support_nn_write_v5_1/` có NN-bit-pattern/ties/NaN/shapes/cache/Eq9/session/rollback/gradients/timing/budget/transport checks. Sốtests/hashes/logs thật trong delivery/review `.agents`; localtests không thay T4.

## Hướng dẫn chạy

1. Import notebook NN-write mới, kernel mới.
2. Attach source `.agents/artifacts/deliveries/hope_support_v5_1_nn_write_r1/hope_cad_support_v5_1_nn_write_source_r1.zip` (hoặc Datasetgiải nén). Tháo profiler source cũ.
3. Attach original reviewed376parent `.agents/artifacts/runs/support_v5_1_benchmark_NESTED_seed0/session_01/hope_cad_support_v5_1_benchmark_evidence.zip` (hoặc exactoutput đã giải nén). Tháo completedprofiler evidence; G1 được pin bằng receipt hash/endstate digest trong newcontract. Không lấy state195discardedclone làm benchmarkparent.
4. Giữ MVTecraw và locked `vit_base_patch8_224.augreg_in21k.safetensors`. Tổng4loại inputs như profiler, chỉ source/notebook mới.
5. GPU T4×2, PHASE='validate', overridesNone nếu auto discovery đủ rõ. SaveVersion→Save&RunAll. Không upgradepackages hoặc đổi precision/window/budget. MộtGPUcuda:0 thực hiện tính toán; haiGPU không poolVRAM.
6. Lưu ZIP `hope_cad_support_v5_1_nn_write_evidence.zip` và log vào `D:/src/DATN/.agents/artifacts/runs/support_v5_1_nn_write_validation_01/session_01/`, không ghiđè. Phiênsau tăngsession khi có hướngdẫn.

## Cách đọc trạng thái

`validation_complete_pending_review`: đủ4passes, parity/gradient checks đã chạy; xem `summary.json` cho speed/memory gates. Complete không tự là optimizationaccepted. `exact_parity_speed_or_memory_inconclusive` nghĩa giữ reference và review, không tự thử tiếp hoặc hạ tiêu chí. `validation_incomplete_budget`: thiếu pass/diagnostics, giữZIP/log; partialpass không tự retry/nối timing. Failure luôn export trước raise nếu platform còn cho phép.

Budget90min là cumulative actualpreexport +180s conservativeexportreserve. Actualexportseconds inKagglelog, không gọi reserve là thời gian thựcđo. Reserve16attempts/pass trước dispatch để hardkill không cấp lạiarrivals. Bootstrap/captures/audit/recomputation/export đều tínhquota; nofinalizecensor vì hết16ảnh. Pause/resume chỉ giữa completed/sealedpasses, không làm mất pending/partial benchmark vì mọi actor trongrun này bịdiscard.

Primarytiming ghép cácinterval giữa pre/post-learningcaptures ở events181/189; R/O cùng4excludedintervals. Capture CPUcopy, serialized snapshot, allrawgradient recomputations nằm ngoài primarytiming nhưng có ledger. Report bootstrappassGPUpeak, primary allocated/reserved và postmeasurement peak gồm diagnostics riêng; memorygate áp primarypeak. Không dùng CUDAspan/hosttime để suy CPU/GPUutilization hoặc ETAfullstream.

Gradientartifact giữ rawactualgrads/loss/d/increments,4support increments/denominators,4queryloss/mean,allslowrawgrads trướcclip, postobserve theta/phi/Adam và inputs đã capture. Reference/candidate exactcomparison bao mọigroup, không dùng tolerance để accept đổi quyếtđịnh. Captures không thêm arrivals hoặc đọc future normals. Các gradients không apply vào deployedbenchmark.

## Sau khi user chạy

Review rawartifacts và gates trước adoption. Nếu exact và tiếtkiệm rõ/withinmemorygate thì mới implement G4 loader/lineage/tests và bàngiao bridge8arrivals376–383. Parent376, states/events và source cũ giữ nguyên. Không tự resume oldbenchmark bằng candidateZIP, không rerunW6/pilot/full3599 để che migrationmismatch. NoADperformancegainclaim từ timing; mọi metric/selection và VisA giữ protocol đã khóa.
