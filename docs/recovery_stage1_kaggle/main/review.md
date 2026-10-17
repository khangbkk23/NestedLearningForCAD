# Review recovery stage1 — kết quả controls và chẩn đoán HOPE-CAD v4

**Kết luận:** nghiệm thu **gói recovery stage1 đã hoàn tất** ở phạm vi development seed0. C1, A1 và matched CADIC10000 đều đã train đủ 15 tasks và đánh giá final row; M được giữ nguyên từ W7.2 đã review. Bộ evidence hợp lệ, nhưng **M hiện còn kém matched CADIC rất xa**. Đây là kết quả âm có ích cho quyết định revision, không phải một bản sửa performance đã thành công, không phải full W7 và không phải validation trên VisA.

## 1. Nguồn, danh tính và phép kiểm

- Worktree được review: `D:\src\DATN\hope_cad_v5`, branch `hope_cad_v5`, HEAD `4dd6f36fe093b2ae153d196a9333aeda1d196a06`; checkout dirty từ trước. Không lấy commit đơn lẻ làm source identity.
- ZIP gốc: `hope_cad_recovery_stage1_evidence.zip` trong cùng folder; 2.014.930.291 bytes.
- SHA256 ZIP: `b3c5d204320cc7c66e1ecb908e0b3ab6d411df9f8298288906e92bb5e1a19dc1`.
- SHA256 manifest: `087f167eb0fbcb935e36148ceaf64d373aa196894e928ac9b17aad49b46db6cc`.
- Đã đọc/decompress và kiểm SHA256 **9.394/9.394 payload files**, không thiếu/thừa entry, không duplicate hoặc unsafe paths. ZIP gồm 9.395 entries tính cả manifest.
- Đã tính lại độc lập I-AUROC, I-AP, pixel AP/P-AUPR, P-AUROC của C1/A1/CADIC từ NPZ: **45 cells, 5.175 image evaluations, 259.660.800 pixel evaluations, 663 shards**. Dùng sklearn trực tiếp, float64 cho tính metric; không import metric helper của project. Sai khác float64 tối đa ghi trong machine receipt; tất cả dưới 1e-12. Không chạy lại neural model trên GPU.
- M: kiểm toàn bộ 47 entries reference subset với manifest của completed W7.2 và đối chiếu macro từ per-task metrics. Recovery không mang raw score shards của M; lượt này không tính lại M từ raw maps.
- 119 frozen source files và 40 operations files khớp live/archived hashes. Study digest `b732d8566d56362e87f6f2c6a4acafddd62f716df8c0f05585d7e269fd3e7cd7`; suite `17cd635b35e62484c5a000b4be4af503284e7d00da9aec79d64576c8d5055240`.
- Full source-identity digest là `ced707d247b1e84fc0fa9815e65c491615cae932df7af5922fb4ac3060b3ab47`. Digest của riêng map 40 extras là `11d530f8d236e0d071654b3f40c071e60f37be3b6e515da4c48f0020fa3274cd`; hai đối tượng khác nhau, không so lẫn.
- Runtime khớp W6: Python3.13.15, torch2.11.0+cu128, torchvision0.26.0+cu128, TeslaT4/CUDA12.8/cuDNN91900, TF32false. 130 frozen +45 recovery tests là receipts của accepted preflight được reuse nguyên bytes, không phải 175 tests được chạy lại trong phiên controls.

## 2. Hoàn tất đúng phạm vi nào?

| Hạng mục | Kết luận | Bằng chứng trong ZIP |
|---|---|---|
| Runner và export | Complete, exit0 | `handoff.json`, `notebook_execution.json`, log `[VERIFIED STATUS]` |
| C1/A1/CADIC online stream | 3.599 ảnh/method, 15 tasks, 15 task checkpoints | `runs/<job>/train_summary.json`, observations, states |
| Official final evaluation | 1.725 ảnh/method =467 normal+1.258 anomaly; đủ15 categories | test manifests, `scores/14_*/progress.json`, `result.json`, NPZ |
| Cleanup | Hoàn tất cả3 controls | `runs/<job>/lifecycle_completion.json` |
| W6 recipe và source | Giữ nguyên, không refit/reselect | run locks, assets receipt, suite recipes, operations archive |
| M reference | Không train/rescore official test trong recovery | `reference_w7/reference_import.json`, hashes của47 entries |
| Normal diagnostics | Complete M/C1/A1 | 232 records/method, 29 checkpoint-task cells; diagonal và final |
| Full W7.3/seeds/CI/VisA | Chưa hoàn tất | recovery completions có `matrix_complete:false`, `reportable:false` |

Handoff `recovery_stage1_complete_pending_review` được chấp nhận **bởi review này**, không sửa lại cờ trong archive. Runner elapsed14.367,55s≈3h59m28; log kết thúc14.517s≈4h01m57 tính cả đóng gói/nbconvert. Đây là phép đo phiên này, không ETA chung.

C1 có CMS clock0/commits(0,0) đúng vai trò static persistent control. A1 và M có clock3.599/commits(3.599,449), pending slow7; không flush ở cuối task. CADIC thấy2.821.616 patch vectors, update3.599 lần theo ảnh; bank10.000, accepted10.000/replaced95.410/rejected2.716.206. Transport batch8 không đổi update unit thành batch.

Cache `cadic_exact_pair_tile_cache_v2` hoạt động trên concrete bank, giữ reference kernel/tile/tie rule và có receipts parity/roundtrip/cleanup ở preflight. Parity trên cloned full bank với4 real candidates và các regression fixtures **không phải** exhaustive dual replay cả3.599 ảnh. Config vẫn giữ `exact_parity_claim:false` đối với paper/author pipeline; kết quả dưới đây là **matched local implementation**, không phải khẳng định tái lập trọn paper.

## 3. Kết quả benchmark development seed0

Đơn vị bảng là %. P-AUPR là pixel average precision theo implementation đã khóa, không đổi sang diện tích PR trapezoidal.

| Method | I-AUROC | P-AUPR | I-AP | P-AUROC |
|---|---:|---:|---:|---:|
| M, HOPE-CAD v4 | 66,8010 | 17,0163 | 83,2417 | 81,1590 |
| C1, persistent predictor tĩnh | 56,8491 | 12,3667 | 79,5140 | 75,5289 |
| A1, C1 initialization +dynamics của M | 64,3216 | 16,3055 | 82,5843 | 80,5109 |
| Matched CADIC10000 | **88,3571** | **46,0137** | **95,4480** | **90,5898** |

M−CADIC: **−21,5560pp I-AUROC /−28,9975pp P-AUPR**. Đây là gap matched có evidence, thay cho việc chỉ dựa số historical Khang87,72/46,71. Sự gần nhau của hai hàng CADIC không chứng minh hai protocol hoàn toàn giống nhau hoặc local baseline khớp paper.

A1−C1: +7,4725pp/+3,9388pp; M−A1: +2,4795pp/+0,7108pp. Hai contrasts cho thấy online CMS và recipe meta-learned có tín hiệu ích lợi trong cấu hình/seed này. Không suy rằng riêng tầng chậm hoặc Nested Learning nói chung đã được chứng minh vượt baseline. C1/A1 chưa có full task matrix và chưa có paired CI/multi-seed.

M thua CADIC I-AUROC ở14/15 categories, chỉ bottle nhỉnh hơn khoảng0,3175pp; pixel AP cũng thua14/15, bottle tốt hơn khoảng15,4983pp. Do đó không phải một category lỗi kéo toàn bộ macro xuống. Screw của M có I-AUROC6,8867% soCADIC30,2726%: cần giữ nguyên hướng score và phân tích failure, không flip theo category để đẹp số.

## 4. Chẩn đoán normal: các kết luận mới quan trọng

Samples: cố định theo salted identity hash,8 online-train normals/category. Có120 ảnh duy nhất/method; checkpoint diagonal và final trùng nhau ở task14 nên232 records/method. Đã kiểm identities, checkpoint hashes, normal-only/future-task permissions và tính lại summaries. Các flags state/RNG readonly và discarded proposals đều đúng trong records; source thực thi có guard tương ứng. **Đây là reused train normals, không phải held-out generalization.**

### 4.1 Acquisition yếu và retention cũng có vấn đề

| Method | Mean normal loss diagonal | Mean normal loss final | Số task cũ loss tăng |
|---|---:|---:|---:|
| M |0,604614|0,743455|14/14|
| C1 |2,127444|2,127444|0/14, static đúng kỳ vọng|
| A1 |0,517290|0,611220|14/14|

Mean là macro đều qua15 categories và8ảnh/category; loss là0,5×squared residual summed channels. M tăng0,138840 từ diagonal đến final; A1 tăng0,093930. Bottleneck không chỉ forgetting: loss đã cao ngay sau học category. Kết hợp W7.2 diagonal I-AUROC69,0439% và final66,8010%, không có cơ sở coi sửa retention là đủ để bắt kịp CADIC.

Một mốc toán học hữu ích: target norms quan sát nằm trong[0,999999821;1,000000119]. Predictor bằng0 có normal loss gần0,5, với bound[0,499999821;0,500000119]. M diagonal vượt0,5 ở9/15 categories; final14/15; cả hai mean0,6046/0,7435 đều lớn hơn mốc này. A1 tương ứng7/15 và10/15. Đây là dấu hiệu chất lượng khớp normal target còn yếu; **không phải** đề xuất zero predictor cho AD hoặc chứng minh metric của nó tốt hơn. Một constant-like score không giải được anomaly ranking.

### 4.2 Norm explosion không phải lời giải thích chính cho M hiện tại

Với prediction p,target u:

`||p−u||² = (||p||−||u||)² + 2(||p||||u||−pᵀu)`.

Đã aggregate từ decomposition thật, không thay mean của tích bằng tích của các means.

| Final normal diagnostics | M | C1 | A1 |
|---|---:|---:|---:|
| Mean prediction norm |0,982688|1,691461|0,854911|
| Mean cosine với teacher |0,271948|0,279271|0,314022|
| Radial /tổng residual, aggregate |1,7954%|34,8192%|2,8488%|
| Radial /score tại image-max patches, aggregate |18,6933%|64,1972%|10,6219%|

M có **98,2046%** residual tổng trên normal samples là angular component. Norm amplification nghiêm trọng hơn rõ rệt ở C1; CMS đã giúp hạn chế nó trong M/A1. Giả thuyết cũ “norm là nguyên nhân chính của M” phải hạ mức tin cậy. Tuy vậy max-patch endpoint vẫn có radial share18,69%, nên không nói norm không bao giờ ảnh hưởng image ranking.

Vì angular residual còn lớn và normal-support membership khác conditional masked predictability, chỉ normalize p hoặc đổi max sang aggregation khác chưa có cơ sở khép gap. Mọi score change là revision mới, không sửa score sau khi thấy test rồi coi cùng frozen benchmark.

### 4.3 Normal loss tốt hơn không đồng nghĩa AD tốt hơn

A1 có final normal loss0,6112 thấp hơnM0,7435, nhưng AD củaM vẫn tốt hơnA1. Vì vậy chọn method bằng một normal-loss scalar không bảo đảm ranking anomaly; đây là bằng chứng thực tế cho giới hạn surrogate. Không dùng quan sát này để tune beta/head theo MVTec test.

### 4.4 CMS có gradient nhưng đóng góp của nhiều timescale còn chưa được tách

Ở M final, mean relative fast proposed update≈0,000527 (0,0527% weight norm); slow≈0,0000839 (0,00839%). Một discarded next-view proposal giảm loss cùng ảnh trong120/120 measurements, mean delta≈−0,0290; A1≈−0,01165. Điều này loại bớt giả thuyết update hoàn toàn chết trong các measurements đó, **không chứng minh** tổng update tối ưu, beta cần tăng hoặc generalization tốt.

Tất cả final proposals ở cùng clock3.599; event giả định tiếp theo là3.600 chia hết8, nên slow commit120/120 là do vị trí clock. Không dùng nó để nói slow level cập nhật liên tục trong deployment. Cần controls C2/C3/C4 và interventions hợp lệ nếu muốn quy causal gain cho nhiều timescale.

## 5. Những điều đã xác định và chưa xác định

**Đã xác định:** phiên complete; evidence không hỏng; counts/identities/recipes và raw metrics khớp; matched CADIC mạnh hơn lớn; CMS tốt hơn static C1 trong contrast đã chạy; M nhỉnh hơnA1; M gặp angular/transfer/acquisition và retention issues trên normal measurements.

**Chưa xác định:** phần causal gap của frozen bottle-trained decoder, visual representation, conditional objective, capacity256, optimizer budget hoặc max aggregation. Không có theorem nói mọi predictor thất bại hoặc paper Nested Learning vô giá trị. Không phát hiện operational bug mới bắt buộc rerun gói này trong các checks đã làm; đây không là chứng minh toàn code không còn bug.

M có persistent continual state4.214.784bytes soCADIC30.720.140bytes, nhỏ hơn≈7,29lần. Nhưng total deployment356.038.144 so373.951.956bytes chỉ giảm≈4,79% vì chung backbone lớn. Không dùng memory-state ratio để tuyên bố toàn hệ thống nhỏ hơn7lần hoặc accuracy trade-off hiện tại là đủ tốt.

CADIC train wall7.914,30s≈2h11m54; C1≈17m28/A1≈18m24, chưa gồm final scoring/diagnostics. Recovery tổng≈4h02. Không so trực tiếp `feature_seconds` hoặc cộng feature+update+score giữa adapters: HOPE update chứa feature extraction, timer scope/synchronization khác CADIC; M completion còn chứa công việc matrix trước đó. W6 calibration cost phải tính riêng, không được coi initializations miễn phí.

## 6. Leakage, giới hạn claim và bước tiếp theo

Manifest online3.599 là normal train;30 normal-validation/probe của bottle bị loại cho cả HOPE và matched CADIC. Ba controls có cùng test identities/bytes/masks với M; không trùng paths train/test. Evaluator readonly guards đã được đối chiếu source và receipts. Diagnostics chỉ đọc category đã đến tại checkpoint tương ứng, không apply proposals. Trong scope này **không thấy direct train/test leakage**; không khẳng định đã forensic toàn bộ hoạt động ngoài artifacts.

MVTec final/matrix và giờ thêm3controls đã development-exposed. Mọi revision phát sinh sau review là adaptive development, phải có method ID/exposure/trial budget riêng. VisA không được dùng để chọn revision; xác minh lịch sử exposure cả nhóm, freeze model+transfer/calibration rule trước test. Giữ `reportable:false`; chưa có multi-seed/CI/full35streams.

**Việc tiếp theo được khuyến nghị:**

1. Giữ nguyên ZIP/log/checkpoints và kết quả v4; **không cần chạy lại recovery stage1 hoặc W6 chỉ để có lại các kết quả này**.
2. Chưa tiêu quota vào full7.3 v4 hoặc VisA chỉ vì controls complete. Nếu mục tiêu vẫn là ngang/hơn CADIC, cần chốt một revision nghiên cứu có cơ sở rõ.
3. Ưu tiên đánh giá đề xuất Nested Learning học quản lý normal-support memory, giữ raw ViT geometry thay vì thay anomaly score hoàn toàn bằng masked residual. Đây vẫn là **proposal**, chưa được user phê duyệt/freeze/implement; không hứa giữ baseline floor hoặc thắngCADIC nếu chưa có thiết kế/chứng minh phù hợp.
4. Trước implement, làm full revision spec: HOPE/CMS học cái gì, signal normal-only nào, tác động thật lên scoring/memory, retention/coverage objective, common baseline/memory/compute, ablations và bounded trial/stop rules. Không gắn module trang trí hay trộn nhiều mechanism cùng lúc.
5. Kiểm normal pilot/engineering trước, bounded MVTec development sau; báo mọi trials/failures. Chỉ freeze revision đã qua acceptance rồi đánh giá VisA theo rule khóa trước, không sửa tiếp theo VisA test.

Review này **chỉ ghi nhận và phân tích evidence**, không sửa algorithms, equations, frozen specs, notebooks, weights, raw evidence, Git branch hoặc environment. Các status/HYP trong `.agents` được cập nhật bằng review/change record mới; snapshots và register gốc giữ nguyên.

## 7. Truy nguyên code và receipts

- `notebook_support/recovery_stage1/runner.py::setup,make_adapter,run_control,reference_diagnostics,comparison,execute` — parent/source/recipe lock, final-row controls và lifecycle.
- `notebook_support/recovery_stage1/diagnostics.py::residual_components,measure_normal,run_diagnostics` — decomposition, same-image discarded proposals, normal sample permissions và summaries.
- `notebook_support/recovery_stage1/common.py::sampled_rows,assert_legal_normal,import_reference,verify_reference` — identities/scope và parent bytes.
- `training/hope_w7_engine.py::train,evaluate_cell,final` — atomic states, score shards, official counts, readonly state/RNG và metrics.
- `models/hope_cad_v2/benchmark_adapter_v4.py::fit_task,score_batch,memory_stats` — deployed scoring/update path và accounting.
- `notebook_support/recovery_stage1/cadic_accel.py` và `cadic_resource_probe.json` — backend/parity scope; không đổi paper-parity flag.
- Machine receipt: `results/recovery_stage1/main_review.json` tại worktree v5; bản sao trong shared context review pack. Raw metric/diagnostic paths được ghi trong receipt và archive.
