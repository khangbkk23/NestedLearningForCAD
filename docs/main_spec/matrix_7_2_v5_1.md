# Nested Learning for CAD — 7.2 readonly task matrices, v1

Ngày 21/10/2026. Addendum vận hành cho `benchmark_v5_1.md`, không sửa phương pháp `hope_cad_v5_1_causal_scaled_utility`, study, score hoặc công thức metric. User giao chuẩn bị đồng thời NESTED/CADIC seed0 để chạy hai notebook Kaggle độc lập. Implementation/local tests và canonical acceptance được ghi riêng; tài liệu này không chứng minh kết quả GPU đã có.

## Phạm vi và nguồn gốc

Hai endpoint7.1 đã được review: NESTED I-AUROC86.9561%/P-AP47.4331%, matchedCADIC86.9983%/44.9012%, cùng3599train/1725test/rawfeature/runtime/ViT/protocol. Bản mới chỉ bổ sung105missingcells/method để đủ120triangularcells,15checkpoints. Không train, không observe, không resolvepending, không finalize, không lựa chọn/tune model theo matrix. 15finalcells và mọi payload7.1 trừ root handoff phải giữ nguyên bytes. Không claim statistical superiority hoặc VisA từ mộtseed.

Initial parents bị pin bằng archiveSHA, manifestSHA và toàn filemap đã review. Kaggle-extracted directory hợp lệ khi manifestbytes và tất cả payloadhashes khớp; duplicate ZIP/directory cùngmanifest là một identity. Chỉ nhận completed7.1 đúng job hoặc latest7.2 cùngjob/source/runtime. Hai method không dùng state của nhau; không nhận costclone/pilot/W6/partial7.1. Resume ưu tiên đúng7.2; nhiều resume versions khác nhau phải fail hoặc explicitoverride. Wrong-job explicitoverride luôn fail. Root `matrix_lock.json` giữ source/neworchestration/job/initialparent; operation/source7.1 giữ nguyên.

## Ownership và nguồn khóa

Source mới kế thừa toàn249entries của matchedCADICr1 byte-exact, gồm235continuationbaseentries. Extension riêng `notebook_support/matrix_v5_1/`, newmanifest và additive docs/notebooks/tests. Original model/engine/migrationresolver/scoring giữ nguyên. Metadata parents chứa manifest và reviewed references, không checkpoint tensors/raw data. NESTED dùng nguyên `Execution.evaluate(matrix=True)` và scoped task00 legacy resolver; tasks01–14 thuộc migratedowner. CADIC dùng original `engine.evaluate(matrix=True)`, cùng fixed-shapeNN activation đã review. Cả hai giữ taskcheckpoint/job/manifest locks cũ, không tạo owner mới cho scientificstate.

## Readonly execution và chứng từ

Bootstrap kiểm runtime canonical T4 trước model/datawork, backboneSHA và tất cả parentpayloads. Copy source evidence sang workdir mới, giữ originalmanifest/handoff dưới `matrix_origin/`; không ghi vào input. Không extract hoặc load train/good; chỉ stage officialtest/GT, hash và tái tạo byte-exact testmanifest đã khóa. Test/GT không đi vào learner.

Worker kiểm source/op/runtime, initialfrozenpayloads, completedtrain/final receipts, all15checkpoint hashes/owners/cursors và eventchain của active endpoint. Actor.load_state_dict đọc state; không cập nhật. Preflight tíchhợp kiểm actor.score trên testimage đầu của task00 vàlasttask, discardedowners: finite shape/read-only/roundtrip, không ghi lại finalcells và không optimizer. Sau đó originalengine thêm missingcells và tính task_matrix/forgetting/BWT. Cơ chế scoring/masks/chunk8/CPUFP32interpolation không đổi. Root status chỉ là vận hành; mọi checkpoint/events/finalcells của7.1 được hash kiểm trước/sau cảcompletion/pause/failure. Nếu frozenpayload đổi phải exportfailure, không tiếp tục.

Engine deadline áp ở chunkboundary,600s exportreserve, default9hsoftbudget, <=11h; không hardkill giữa atomicoperation. Stage/verify/export vẫn tốn thời gian, hardplatformkill có thể ngăn export. Resume7.2 giữ latestchunks có receipt, chỉ tính lại orphanpartial chunk đọc, không học lại hoặc làm lại completioncell. Worker luôn có heartbeat30s và `[BENCH CELL] i j complete`; kiểm progress count trong receipt. Output namespace/job và path khác nhau cho hai notebooks nên có thể chạy SavedVersions song song nếu Kaggle cho phép; quota cộng dồn, không assume2GPUs tự chia việc.

Handoff: `paused_at_atomic_boundary`, `7.2_complete_pending_review`, `failed_preserve_evidence`. Complete cần đúng120completioncells, matrix_complete.cell_count120/jobdigest/`train_repeated=false`, finalrow giữbytes và có i_auroc/p_ap matrices, forgetting/BWT finite. Failure xuấtZIP rồi raise; không nuốt lỗi thành notebookPASS. Không rerun7.1/pilot hoặc thay architecture.

## Test và nghiệm thu

Local: meaningful actualengine2task readonlymatrix + preservedfinalrow; interrupted/uninterrupted chunkresume; train/observe/learnerupdate forbidden; zerochange frozenpayloads; wrongjob/source/runtime/checkpoint/hash/multipleversions rejection; test-onlytar staging; sourceZIP/extracteddirectory bootstrap; actual15task parents CPU resolver smoke. Syntheticfixtures không thay realT4matrix acceptance. Sau user chạy, reviewallpayloadhashes,120cells/checkpointbindings/finalrowbyteparity/scoremetrics/F/BWT. Chưa có7.2GPUresult khi build.

Fullstudy27jobs/CI/seeds/orderstudies/VisA không thuộc launcher này. Matrix là thêm thông tin acquisition/retention, không performancegain. Không dùng matrix để sửa method rồi báo cùngfrozenstudy.
