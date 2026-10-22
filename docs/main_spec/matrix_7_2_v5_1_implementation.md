# 7.2 NESTED + matched CADIC — hướng dẫn vận hành

Đọc `matrix_7_2_v5_1.md` và `benchmark_v5_1.md`. Hai notebooks dùng cùngsource7.2, nhưng job/parent/output riêng. Không train lại. Trạng thái canonical run từ `.agents/evidence/status.json`; docs buildtime không là bằngchứng đã chạy.

## Chuẩn bị hai notebook Kaggle mới

| Notebook | JOB | Evidence7.1 duy nhất cần attach lần đầu |
|---|---|---|
| `hope_cad_v5_1_matrix_NESTED_seed0_kaggle.ipynb` | NESTED_seed0 | `hope_cad_support_v5_1_continuation_evidence.zip` — completed3599 đãreview |
| `hope_cad_v5_1_matrix_CADIC_seed0_kaggle.ipynb` | CADIC_seed0 | `hope_cad_matched_cadic_seed0_evidence.zip` — completed3599 đãreview |

Mỗi notebook đúng4inputs: **source mới `hope_cad_matrix_7_2_source_r1.zip`**, owncompleted7.1evidence (ZIP hoặc Kaggle giải nén), MVTec AD và checkpoint `vit_base_patch8_224.augreg_in21k.safetensors` cũ. Tháo source7.1, bridge, pilot/W6/cost và evidence methodkhác. Không đổi tên files bên trong ZIP; sourceDataset Kaggle tên khác không ảnhhưởng discovery nếu manifestduynhất.

Giữ `PHASE='7.2'`, đúngJOB đãđiền sẵn, `SESSION_HOURS=9.0`, cácoverrides=None nếu inputs rõ. GPU T4×2/canonicalenvironment như7.1. Save Version → Save & Run All cho mỗi notebook. Hai SavedVersions có thể được khởiđộng song song nếu quota/capacity/account cho phép; nếu Kaggle từchối concurrentGPU thì chạy lầnlượt, không ảnhhưởng phươngpháp. Không tự pipupgrade, chuyển GPU hoặc disable runtimeguard. Budget9h không phảiETA; đây là105task-evaluations/method, nhiều hơn15finalcells ở7.1.

Bootstrap import/verifyparent, stagetest/GT rồipreflightscore2images đọc; sau đó log `[BENCH CELL] i j complete`, heartbeat30s. Không có `[BENCH TRAIN]` mới. Mỗi job đủ120cells sẽxuấtstatus `7.2_complete_pending_review`. 15finalcells/checkpoints/events cũ giữbytes; finalmetrics không tăng do làm7.2.

## Pause/resume và output

Pause: SaveVersionOutput/downloadZIP/log. Phiênsau cùngnotebook/source/MVTec/ViT nhưng **thay own7.1input bằng latest own7.2output** (vẫn4inputs). Không giữ nhiều old7.2versions. `PARENT_OVERRIDE` chỉ khi discoveryambiguous, trỏ đúngroot evidence hoặcZIP, không folderchacha. Freshworkdir/kernel; khôngRunAlllạitrênworkdir cófiles. Failure: giữZIP/log đểreview trướcretry.

- NESTEDoutput: `hope_cad_matrix_NESTED_seed0_evidence.zip`, lưu `D:/src/DATN/.agents/artifacts/runs/support_v5_1_matrix_NESTED_seed0/session_01/`.
- CADICoutput: `hope_cad_matrix_CADIC_seed0_evidence.zip`, lưu `D:/src/DATN/.agents/artifacts/runs/support_v5_1_matrix_CADIC_seed0/session_01/`.
- Cácphiênresume tăng session_02,03…; khôngoverwrite. Mỗi folder giữZIP+Kagglelog.
- Metrics: `benchmark/runs/<JOB>/metrics/task_matrix.json`; receipts `matrix_complete.json`, `matrix_origin/`, `matrix_sessions/`. Giữ cảhai7.1rawarchives đãreview.

Source delivery tại `D:/src/DATN/.agents/artifacts/deliveries/hope_support_v5_1_matrix_7_2_r1/`. NhậnZIP/log xong mớireview120cells/F/BWT vàsohai methods. Không tựrẽthiếtkế hoặc mởVisA trong lúcbenchmark.
