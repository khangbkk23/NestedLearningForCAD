# Kết quả hiện có và phạm vi bằng chứng

## Đã đo được gì?

Các số dưới đây kế thừa từ reviewed raw artifacts của phương pháp **v5.1** trên canonical Kaggle T4. Chúng không phải một lượt benchmark mới của package v7 hoặc H32, và không được tái tính trong lần tổ chức tài liệu này. Code v7 hợp nhất layout/import/CLI; tính tương đương có engineering evidence theo phạm vi, không thay một fresh benchmark trên máy khác.

| Metric | Matched CADIC | NESTED v5.1 | NESTED−CADIC |
|---|---:|---:|---:|
| Final image AUROC | 86,9983% | 86,9561% | −0,0422 điểm % |
| Final pixel AP | 44,9012% | 47,4331% | +2,5319 điểm % |
| Image forgetting | 6,8349 điểm % | 6,9384 điểm % | +0,1035 điểm % |
| Pixel forgetting | 8,9104 điểm % | 6,8252 điểm % | −2,0852 điểm % |
| Pixel BWT | −8,9047 điểm % | −6,4952 điểm % | +2,4096 điểm % |

Seed0, một task order,15categories, **3.599 train normals/1.725 test images**, K10000 raw vectors;120triangular cells/method.30Bottle normals070–099 bị loại ở cả hai comparator. Đây là matched internal comparison, không phải exact reproduction paper hoặc direct comparison với run riêng của nhóm. MVTec đã dùng để phát triển; chưa multi-seed superiority/noninferiority, chưa VisA.

Pixel AP cao hơn ở11/15categories; image retention chưa có lợi thế và có transient image-AUROC dip ở checkpoint metal_nut. Chi tiết per-category trong [báo cáo nhóm](../QUICK_UPDATE.md). Không chỉ chọn final row thuận lợi hoặc gọi pixel advantage là image advantage.

## Artifact identities để kiểm chứng độc lập

Raw ZIP/logs không đi cùng source và chưa có public download URL được xác nhận. Người nhận phải xin đúng archives từ người thực hiện; không cần một đường dẫn knowledge cụ thể. Nếu chỉ có branch thì có thể đọc code/spec và bảng số, **chưa thể độc lập tái tính raw metrics**.

| Archive | SHA256 |
|---|---|
| `hope_cad_matrix_NESTED_seed0_evidence.zip` | `668f9e348561d5ec55c4cf51762c5890b19328c2a950c7ba2fb09f037b1bea0d` |
| `hope_cad_matrix_CADIC_seed0_evidence.zip` | `035f88639f8faab8bf02fdfbcd5f502677620c3756af53dcf4815dd40e8cc889` |

Prior paired review kiểm22.943payload hashes và recompute240score cells (image tolerance<1e−12, pixel<1e−6). Đây là phạm vi của review trước; công bố bảng này không tương đương rerun/hash toàn raw data trong phiên hiện tại. Archives gồm `artifact_manifest.json`, `benchmark/manifest_train.json`, `benchmark/runs/{job}/...`; D0 chỉ đọc allowlist normal events trong [spec](research/attribution.md), không toàn bộ tensor payloads.

Raw feature hashes3599ảnh, manifests, runtime và evaluator của hai methods đã khớp trong accepted comparison. Checkpoint upstream SHA `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`; ImageNet mean/std + direct224bilinear, block8raw784×768, Eq9b9, CPUbilinear224map, nonzeroGTnearest. Native checkpoint metadata khác recipe này; ảnh hưởng riêng chưa đo.

## Pilot và chi phí

Normal-only pilot v5.1 cũ có CADIC/GEOMETRIC/FIRST_ORDER/NESTED, mỗi arm256ảnh; mechanism/safety gates đã được review đạt. Pool cũ179Bottle+77Cable và realized NESTED post-warm rejection145/211 là nguồn của fixed-rate RANDOM control được công bố, không phải rate nhìn trước từ full benchmark. Không dùng normal-risk gate làm anomaly-performance proof.

CADIC fresh3599 train≈7.918,867giây (2giờ12phút), final evaluation≈464,089giây. NESTED riêng continuation384→3599 train≈4giờ34phút44giây; đây không phải fresh full training và không cho exact overhead factor. Full matrix bổ sung→export≈62,78phút CADIC/73,63phút NESTED. Hardware/runtime, extraction/update/checkpoint/eval/export và cumulative sessions cần cùng scope khi báo chi phí;2T4 không được code tự ghép VRAM/chia training.

## Phần chưa có kết quả

- Full-train3629/H32 đã implementation/local verified và có technical preflight theo phạm vi bên dưới; chưa completed/accepted pilot hoặc AD benchmark, số liệu cũ không được relabel.
- R_EXACT/H32 và controls prospective chưa có attribution results; D0 analyzer đã viết/exercise normal trace, chưa AD intervention hoặc attribution causal proof.
- Controls/other seeds của full baseline study chưa hoàn tất; không nhận120cells là120independent replicates.
- VisA protocol/runner/freeze và exposure review còn riêng; ReplayCAD/ContCore integrations không tự là accepted baselines.
- Same K không equal total memory. Chưa có measured-total-memory-matched control hoặc full fresh same-runtime NESTED/CADIC timing ratio.

Xem [method hiện hành](method.md), [benchmark contract](benchmark.md), [thiết kế nghiên cứu](research/README.md) và [pipeline quyết định](pipeline_alignment_spec.md).

## Technical preflight trên máy thuê

Source đã publish `54d9049653178d8bc7ee29386444753db449542d`, remote source digest `4d7fe48a6d8ca487e8485fab9f022e7f261d524b65189eaa8b1ea46fe6ac409c` (169 files) có preflight RTX5090 instance57380117/driver580.173.02 đã review ngày23/10/2026: numerical, NN/Eq9 và resume checks đạt; 62 payload hashes,28 feature records và169 source blobs được đối chiếu. Invocation khoảng522giây, **không phải full train/pilot timing hoặc AD quality**. Source/runtime sau đó không tự kế thừa acceptance này.

Archive `hq32_5090_01/exports/session_01/evidence.zip` SHA256 `8603599c6823b18a40d16488b4b055d30b8b6fa45539d8ab28be033affbce7e3`, cần cấp riêng cùng accounting/log để tự kiểm. Host57385505 sau đó đã bị người chạy Destroy khi pilot còn dang dở; chưa có completed pilot được nhận/review trong repo. Không gộp thông tin tiến độ từ terminal thành accepted result.

## CADIC diagnostic trên RTX3090 — 24/10/2026

Run `cadic_check_01`, instance57535903, RTX3090/driver570.153.02, Ryzen Threadripper PRO3955WX, PyTorch2.11.0+cu128/Python3.13.17, source commit `bd78bda9bf2d5a18e87d2e1885f2fb4e5b290de9`; source digest Linux `1a3f729d651b5c4b3da16d5a08f42c4687a67189e4c698f99cfebf022db6b528` đối chiếu đúng178Gitblobs. Mỗi K xử lý đủ3629normals. Hai bank train xong; chưa có anomaly evaluation.

| Phạm vi timing | K2500 | K5000 |
|---|---:|---:|
| Đọc/hash/transform ảnh | 101,70s | 102,73s |
| ViT features | 48,71s | 48,99s |
| Cập nhật coreset | 441,82s | 1512,37s |
| Tổng pipeline, không checkpoint/QA/export | 592,23s | 1664,09s |
| Checkpoint và hash checkpoint | 1,64s | 3,13s |
| Số vector được thay | 28102 | 52335 |

Invocation gồm cả hai K và export2272,10s, khoảng37phút52giây. Update chiếm74,6%/90,9% pipeline; hai category đầu chiếm56,8%/55,7% tổng update. Không suy whole-stream ETA từ prefix hoặc quy bottleneck chi tiết cho NN/pair-cache/CPU-GPU khi chưa có profiler. Đây là một host/run, chưa là so sánh3090–4090 hoặc speedup so với paper vì phạm vi timer/implementation khác chưa được cô lập.

Export `cadic_check_01/exports/session_02/evidence.zip`,43420007bytes,SHA256 `7995561273f72f18b7cbb7ee97cafc457931a4b5313a77b4c0eb13a16ba818b7`; session01 là lượt lỗi thiếu dữ liệu, SHA256 `3985c325d6b84b9918bd50d7548413f69b2b7d2c1500486423b9e07ddb1ef04d`. Raw archives/timing/checkpoints cần cấp riêng. Bản tải local được đối chiếu CRC và18/18payload hashes, records/counters/order/timing và bank finite/unique shape2500×768/5000×768 bằng CPU; các QA GPU trong receipts ghiPASS nhưng chưa replay độc lập trênGPU.

Diagnostic không cấp research acceptance. User báo research preflight `hq32_3090_01` thất bại sau đó; chưa nhận raw handoff/error/export của research, nên chưa xác định nguyên nhân hoặc kết luận thiếu allocation. Bootstrap tại commit `bd78bda` có hai CPU transport fixtures lỗi khi CUDA hiện diện, dừng trước assets/allocation. Assets sau đó đã được user chuẩn bị riêng để chạy diagnostic thành công. Bản sửa sau đó cô lập GPU discovery của hai CPU fixtures, thêm regression CUDA-present/absent, fail-fast message và ready shell sau log-save; phạm vi local/mocked verification không tự thay nghiệm thu Linux GPU-visible suite hoặc research preflight mới. Host được user báo đã Stop và chưa thuê lại được cùng máy; availability chưa kiểm trực tiếp. Xem [runbook](operations.md) để kiểm setup trước fresh run.
