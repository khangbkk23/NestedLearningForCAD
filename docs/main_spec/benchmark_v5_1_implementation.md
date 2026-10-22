# Nested Learning for CAD — bàn giao benchmark v5.1

Ngày 20/10/2026. Đây là implementation/handoff record mới của study v1 trong [benchmark spec](benchmark_v5_1.md), không sửa frozen method specs hoặc normal pilot. NESTED hiện hành quản lý raw normal-support bank bằng HOPE/CMS; prediction vẫn full CADIC Eq9 trên bank được learner thật sự thay đổi. V5.1 đã qua normal pilot; **anomaly benchmark và VisA chưa chạy/nghiệm thu** tại thời điểm bàn giao. Local test/delivery receipts nằm bên ngoài source ZIP để không tự thay file identity.

## 1. Codespace và entrypoints

| File/module | Vai trò |
|---|---|
| `conf/benchmarks/support_v5_1_study.json` | Registered27 jobs, task/data/scoring/metrics/phases/bootstrap/claims |
| `conf/benchmarks/support_v5_1_pilot_source.json` | Full144 file hashes map từ accepted pilot source; builder từ chối core drift |
| `training/support_benchmark_v5_1/contract.py` | Job parser, source/study locks, canonical JSON, fresh session factories và K24576 control |
| `data.py` | Hashed normal-only delivery, separate evaluator/GT rights, mask QA, no full raw-feature cache |
| `engine.py` | Image-atomic training, event hash chain, immutable task checkpoints; chunk evaluation/resume; final/matrix/F/BWT |
| `statistics.py` | Selected-evidence verifier/descriptive aggregate, paired whole-image CPU bootstrap; no model selection |
| `notebook_support/support_benchmark_v5_1/` | Input discovery/runtime/bootstrap/source schemas, filtered resume, single-job phase runner, heartbeat/export, immutable package/build verifier |
| `tests/support_benchmark_v5_1/` | Synthetic meaningful regression contracts; không real anomaly data |

Namespace core `models/hope_cad_v5_1`, primitive v2, raw CADIC và original144 pilot files giữ nguyên. Existing RGB adapter được giữ như interface đã localtested; runner mới gọi session raw-feature APIs để checkpoint từngảnh và tách learner khỏi evaluatorlabels/masks. Không thay updateunit hoặc score để tăng số.

## 2. Chạy local checks và package

Tại active checkout D:\\src\\DATN\\hope_cad_v6, dùng Pixi env:

```powershell
pixi run python -B -m unittest discover -s tests/support_benchmark_v5_1 -v
pixi run python -B -m notebook_support.support_v5_1.local_tests
pixi run python -B -m notebook_support.support_benchmark_v5_1.build_delivery
pixi run python -B -m notebook_support.support_benchmark_v5_1.verify_delivery
```

`build_delivery` tạo bốn notebooks và sourcebundle riêng. Default output ở shared `.agents/artifacts/deliveries/hope_support_v5_1_benchmark_r1/`. Khi triển khai, dùng `HOPE_BENCHMARK_DELIVERY_DIR` để stage+verify trước publish, không overwrite published r1. Verifier kiểm toàn source bytes/live map, notebook syntax, thực thi sourcebootstrap ZIP và Kaggle-extracted directory trong hai isolatedprocesses; chạy benchmark tests từ packagedsource. Wholecheckout regressions chạy riêng; các scopes/counts ghi trong receipts, không gọi localWindows là canonical T4 test. Đây không là benchmark result.

## 3. Bước đầu trên Kaggle: preflight

1. Import **`hope_cad_v5_1_benchmark_preflight_kaggle.ipynb`** từ `notebooks/`.
2. Tháo source/output của pilot cũ. Attach đúng **ba inputs**: source `hope_cad_support_v5_1_benchmark_source_r1.zip` (hoặc Dataset đã giải nén ZIP), MVTecAD archive, locked `vit_base_patch8_224.augreg_in21k.safetensors` đang dùng. Không attach W6/W7/recovery/pilot learned state.
3. Settings GPU T4×2; implementation dùng **cuda:0**, không DDP hoặc tự dùng cảhaiGPU. Giữ `PHASE='preflight'`, `JOB='NESTED_seed0'`. Có thể dùng SESSION_HOURS=9 mặc định hoặc giảm nếuquota; không ETA.
4. **Save Version → Save & Run All.** Runtime/source checks trước model learning; preflight không extractofficial test/GT. Tests/numerical/full-budget check và24 discarded normal clone có receipts. Preflight không là lặp full W6 và không trainpilot thêm.
5. `preflight_passed`: giữ ZIP/log/Saved Version. Rawdownload vào `.agents/artifacts/runs/support_v5_1_benchmark_preflight_01/session_01/`. Pause thì giữlatest output và resume; fail thì giữfailure ZIP/log để review, không bypassguards. Không cần giữlearned weights của preflight; clone bịdiscard.

## 4. 7.1 — first seed0 final view

1. Import **`hope_cad_v5_1_benchmark_7_1_kaggle.ipynb`**.
2. Cùng source/MVTec/ViT + **output/ZIP của benchmark preflight vừaPASS**. Chỉ một evidenceparent.
3. `PHASE='7.1'`, `JOB='NESTED_seed0'`. Không sửa learning-rate/warmup/threshold/score. Run Saved Version.
4. Fresh3599 online images/15 tasks, giữtask states rồi15 final cells. Không nạp242labelpilotstate. Sau mỗi8images cóprogress; nếu chưa đủ9h thì softpause/export. **Pause chưaxong**: detach parent cũ, attach latest job output, cùng source/assets/phase/job, chạytiếp. CPU/GPU RAM không là nơi giữstate duy nhất; ZIP chứa optimizer/privateRNG/pending/replay/bank.
5. `7.1_complete_pending_review`: giữ ZIP/log tại `.agents/artifacts/runs/support_v5_1_benchmark_NESTED_seed0/session_01/` (resume session_02,...). Macro/percategory tại `runs/NESTED_seed0/metrics/`. Review trước khi diễn giảiaccuracy.
6. Muốn matched quick comparison: chạy7.1 hai job **CADIC_seed0** và **FIRST_ORDER_seed0**, mỗi job một Saved Version/output riêng. Bắt đầujob mới attach preflight PASS hoặc completedoutputcũ chứacommonpreflight; bootstrap **không copy learned tensors** từkhác job. Giữ cùng protocol và compare bằng rawartifacts. NESTED final một mình chỉ descriptive, không phải matchedv5.1superiority.

## 5. 7.2 — thêm task matrix bằng checkpoint

Import **`hope_cad_v5_1_benchmark_7_2_kaggle.ipynb`**, attach latestcompleted7.1 **cùng JOB** +source/assets. Bỏ preflightevidence cũ nếuđãattach latest job output; tênZIP lặp là bình thường, khôngattachhai parentskhácnhau. `PHASE='7.2'`, `JOB='NESTED_seed0'` hoặc CADIC/FOseed0 nếujob đó đã xong7.1. Missingtrainstate/finalparent → stop, không tự train. Chỉ thêm 105 cells, final15cells/checkpoints/events giữbytes. Pause resumeđúngphase 7.2/latest job output. Matrix/F/BWT ở `runs/<JOB>/metrics/task_matrix.json`. Lưu cùngrunfolder, tăng session. Đây là thêm thôngtin retention/acquisition, không performancegain do học thêm.

## 6. 7.3 — coverage đầy đủ

Import **`hope_cad_v5_1_benchmark_7_3_kaggle.ipynb`**. Mỗi invocation một registered job. DefaultNESTED_seed1; các jobs theoCartesianproduct của9 methods/3 seeds trongstudyJSON. Với job mới, freshstate; với jobđãxong7.1/7.2, attach latest outputchínhjob đó vàreuse, không trainlại. Mọi7.3job yêu cầu120 cells. Không gọi xong3NESTEDseeds là xong27 jobs/ablations. Thiếu quota có thể báo partialcoverage, không cherry-pick/drop nhữngkếtquảxấu hoặc “full benchmark” thiếu controls.

Artifacts từng job/session giữriêng; currentSaved Version thườngkhông cần giữinputoutputcủa26jobs khác. Model/timescale/seed/taskorder không được chỉnh từAD metrics. Nếu phát hiệnbug, dừngvàreview, phânbiệt operationalpatch với scientificrevision; không lặnglẽ ghiđè study/source.

## 7. Tổng hợp và uncertainty, không chạyGPU

Chạy từ đúng packagedsource hoặc checkoutbyte-exact; chỉ supplylatest outputmỗi job, ZIP hoặc extracteddirectory:

```powershell
python -B -m training.support_benchmark_v5_1.statistics --evidence <NESTED_seed0.zip> <CADIC_seed0.zip> <FIRST_ORDER_seed0.zip> --output comparison_seed0.json
```

Khi đãđủNESTED/CADIC cả 3 seeds, thêm `--statistics` để paired2000whole-imagebootstrap. `--compare FIRST_ORDER` dùng3NESTED+3FO. Đây là CPU aggregation cóthểtốnđángkể doallpixel AP; không chạy mặcđịnh trongGPUnotebook, khôngclaimETA. Sampler/draw hash/CI rules khóa trongstudy. Full studyflag cần27jobmatrices; partial table không cófull studyclaim. Phảireviewraw endpoint/provenance/coverage trước scientificclaims. Script không mở VisA, sửa model hoặc tự claim superiority từpilot.

## 8. Các điểm cần user cung cấp sau bàn giao

User chạybenchmark preflight và gửiZIP/log. SaukhiPASS, chạyregistered jobs bằngSaved Versions, giữlatestparentvàrunledger. Nếu runtime Kagglekhác pin hoặctechnical failure, gửifailure ZIP/log thay vìupgrade pip/skip guard. ViT/MVTecassetSHA vàinputpathsautodiscovery cóoverridesrõ; không yêu cầuthêmbackbone/dataset mới. Agent cóthểreviewartifacts và tính metrics độc lập; khôngtựtruycậptàikhoản/quota Kaggle hoặc paid GPU. VisA cầnfreeze riêng theoexistingplan trướcconfirmation.
