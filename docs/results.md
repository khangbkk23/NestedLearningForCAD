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

[CADIC K2500/K5000 diagnostic](cadic_diagnostic.md) mới được triển khai để đo trực tiếp trên cấu hình tiếp theo. CPU fixtures/local tests của nó không là kết quả chạy K2500/K5000 thật; chưa có GPU timing hoặc final AD metrics từ công cụ này.
