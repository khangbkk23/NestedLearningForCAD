# Protocol workload hiện hành

Đối chiếu source: `74d975b151f59ad0b5fbf23be392e2d891f350bf` — 11/11/2026.

[Đường đọc](../README.md#đường-đọc) · [B319 design/result](../experiments/B319/card.md)

## Nguồn định nghĩa phép chạy

| Nội dung | Nguồn |
| --- | --- |
| Seed, arms, primary, encoder identity, ramp, caps | [configs/b319.json](../configs/b319.json) |
| Training image identities/order/bytes | [train_manifest.json](../configs/train_manifest.json) |
| Task order/boundaries, full-test/GT mapping | [data.py](../src/nested_replay_cad/data.py) |
| Preprocessing, feature groups | [encoder.py](../src/nested_replay_cad/model/encoder.py) |
| Loss, optimizer, scorer | [decoder.py](../src/nested_replay_cad/model/decoder.py), [scoring.py](../src/nested_replay_cad/model/scoring.py) |
| Memory, contexts, clocks | [sampler.py](../src/nested_replay_cad/core/memory/sampler.py), [replay.py](../src/nested_replay_cad/replay.py) |
| Ranks và aggregation | [metrics.py](../src/nested_replay_cad/metrics.py), [summarize.py](../scripts/summarize.py) |
| Historical screen/results/disposition | [B319 card](../experiments/B319/card.md), [audit](../experiments/history/audit.json) |

Config chưa chứa mọi hằng số. Source + config + manifest + lock + actual bundle
cùng định nghĩa execution. Không áp protocol B319 ngược cho D0/D2 hoặc CADIC-NL.

## Dữ liệu và lịch evaluation

Training chỉ dùng MVTec `train/good`: một pass 3.629 arrivals. Không train bằng
`test/good`, anomaly images hoặc GT masks. CLI tái dựng training manifest từ
input tree rồi so với frozen manifest trước train.

Task order canonical ở `data.ORDER`: bottle → cable → capsule → carpet → grid →
hazelnut → leather → metal_nut → pill → screw → tile → toothbrush → transistor →
wood → zipper. Boundaries lấy từ manifest, không đoán theo số ảnh trung bình.

`data.prepare` yêu cầu đúng 1.725 test PNGs gồm good và mọi defect, GT mapping
không thừa/thiếu. Sau task `t`, evaluate tasks `1..t`: 120 category-boundary cells
mỗi arm. Observer không update learner và kiểm model/RNG invariance. Điều này
không đảo ngược việc người nghiên cứu đã xem test results để phát triển thiết kế.

Preprocessing: RGB, full-field bilinear resize theo encoder constants và ImageNet
normalization. Scorer tạo anomaly map, resize về evaluation resolution, Gaussian
smoothing; image score là mean top-ratio pixels. Masks resize nearest; good có
zero mask. Resolution/kernel/ratio nằm trong encoder/decoder/scoring source;
không tự đổi chúng chỉ để khớp paper.

## Metrics và forgetting

`exact_numpy` gộp tied scores thành nhóm. Pixel AP/AUROC tính trên các pixels
của từng category; final performance macro-average theo categories, khác pooled
AP trên tất cả categories. Image AP/AUROC dùng image scores/binary labels. AP là
stepwise precision-weighted recall increments, khác `PR_AUC_trapezoid`.

Summary giữ bốn ma trận `R[boundary][task]`: pixel/image × AP/AUROC; unseen cells
là `null`. `METRIC_VERSION` nằm trong summarize source.

- `Fmax`: max Pixel AP **trước final** trừ final Pixel AP.
- `Facq`: Pixel AP lúc acquisition trừ final Pixel AP.
- Cả hai có dấu, average trên 14 tasks có observation trước final. Task cuối
  không có history trước final và được loại khỏi denominator.
- Các trường `legacy_*` giữ định nghĩa final-inclusive/denominator khác để tra
  lịch sử, không tráo thành hai metrics chính.

`summarize.py` aggregate recorded scalars, không re-rank raw predictions. Raw-rank
audit cần tensors tương ứng. Primary/margins dẫn tới B319 card, không tạo bộ
threshold mới trong docs. Summary complete không tự quyết định scientific screen.

## So sánh và exposure

Ba arms dùng cùng stream/seed/fast commits/memory policy. Learned arms có thêm
context/query/backward work nên chưa matched compute. Một seed không là multi-seed
confirmation. So với Khang/papers còn cần match scorer, preprocessing, budgets,
aggregation và runtime; cùng test count không đủ gọi exact reproduction.

| Dataset | Exposure / vai trò |
| --- | --- |
| MVTec test | Đã dùng trong development; kết luận hiện tại exploratory |
| VisA | Ứng viên holdout, **chưa khóa**. Người dùng xác nhận ngày 11/11/2026 chưa dùng kết quả VisA để chọn/tune phương pháp hiện tại. Không nâng lời xác nhận này thành kiểm kê chứng minh mọi nhánh lịch sử chưa chạy VisA. Lượt history/docs không truy cập outcomes VisA. |

Trước outcome access cho phép kiểm độc lập: khóa candidate/source, split/exclusions,
preprocessing, seeds, metrics/decision rule; xác nhận exposure history và quyền
execution. Không dùng holdout để dò thiết kế rồi vẫn gọi nó confirmatory. Gói này
chưa có VisA workflow được hỗ trợ.
