# Provenance, artifacts và giới hạn xác minh

Đối chiếu source: `74d975b151f59ad0b5fbf23be392e2d891f350bf` — 11/11/2026.

[Đường đọc](../README.md#đường-đọc) · [History audit](../experiments/history/audit.json)

## Nguồn kế thừa

| Thành phần | Nguồn / giới hạn attribution |
| --- | --- |
| Nền repo và snapshot trao đổi với Khang | Giữ MIT copyright gốc; không coi worktree này là toàn NL-CAD workflow hoặc benchmark của Khang |
| Decoder, StableAdamW, loss/scorer | Dinomaly `1f252be03a918789b19848f0ca37166e7c28dada`; adapted ports, không claim detector mới của Tài |
| Frozen encoder | DINOv2 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; grouped forward/preprocessing port, pretrained checkpoint pinned |
| Memory/sampler/policy primitives | Kế thừa nhánh learned replay/spec engineering trước; attribution/source theo history, không claim mọi primitive mới |
| B319 integration | ER/tail/CTX, numerical route và workload integration của hướng Tài; scientific benefit chưa acceptance |

[LICENSE](../LICENSE) giữ `Copyright (c) 2026 Khang Bui Tran Duy`, Dinomaly/DINOv2
notices và Apache-2.0 text. Headers encoder/decoder/scoring dẫn root LICENSE và
upstream commits. [pyproject.toml](../pyproject.toml) khai báo `MIT AND Apache-2.0`,
`license-files = ["LICENSE"]`. Không có `licenses/` độc lập. Đây là kiểm files hiện
có, chưa thay audit đầy đủ mọi upstream NOTICE/weights/dataset terms trước phân phối.
Package license không tự cấp quyền dataset/weights không giới hạn.

[HISTORY](../experiments/HISTORY.md) giải thích D0 → D2 → Dinomaly. Không sao chép
locked specs cũ thành specs active; revisions/source và qualification phải đi cùng
phiên bản/run. Header lịch sử trong code không thay call graph thực tế.

## Source identity và evidence

Git commit giữ tracked tree; dirty/ignored payloads không tự được bảo toàn bằng tag.
[provenance.py](../src/nested_replay_cad/provenance.py) lưu exact bundle bytes,
`BUNDLE.json` member→SHA và `SOURCE.json` origin/design hashes.

Bundle selection hiện lấy `src`, `scripts`, `configs`, `tests`, `experiments`,
`notebooks` và selected root packaging/license files. **README và docs chưa được
bundle selection lấy vào**; Git checkout giữ chúng. Đây là scope hiện tại, không
đổi producer chỉ để hoàn thiện docs. Clone/hash không thay missing payload bytes.

Local `run.json` sống; `receipts/receipt.json` là terminal record. Provider status
chỉ xác nhận remote state. Download manifest ghi actual retrieved files/hashes;
hash khớp không chứng minh science. Giữ closed receipts; correction dùng analysis
ID/source riêng theo [experiment lifecycle](../experiments/README.md).

Catalog tách `source_fidelity` thành exact bytes, exact reconstruction, semantic
derivative, reference-only và unknown; declared SHA khác measured SHA. Từng record
có verification scope ở [catalog](../experiments/history/catalog.json) và
[audit](../experiments/history/audit.json). Không nâng read/hash thành raw-rank audit.

## Artifacts và .local

Canonical resolver: [paths.py](../src/nested_replay_cad/paths.py).
`NRCAD_LOCAL_ROOT` đổi dedicated root; default `.local/` của checkout này.

```text
.local/                         # ignored, không đồng nghĩa disposable
  inputs/
    mvtec/                      # train/good, test, ground_truth
    weights/                    # frozen encoder checkpoint
    assets.json                 # fetch metadata, chưa thay tree verification
  runs/b319/<run-id>/
    run.json                    # mutable run/package metadata
    package/                    # exact source ZIP, dataset/kernel payload
    logs/                       # local hoặc logs/v<version>/
    outputs/                    # arm outputs hoặc outputs/v<version>/
    receipts/                   # terminal/provider/download records
    audit/                      # retained analysis source/derivatives
  cache/                        # downloads, tests, library caches
  tmp/                          # staging, không là sole retained evidence
```

Tạo folder theo consumer, không cần dựng đủ trước. `kaggle_run.destination` map
provider layouts vào các nhóm trên. `.pixi/` ở ngoài `.local/`, do Pixi quản lý.
Catalog public dùng IDs/hashes thay absolute locators. Raw legacy, inventory và
mapping đầy đủ được maintainer giữ riêng; không có trong public clone. IDs không
là download URLs và không chứng minh payload hiện sẵn cho người đọc.
[Publication manifest](../experiments/history/publication.json) ghi phạm vi projection,
source identity và giới hạn validation. Public run records tên `publication.json`
là semantic derivatives với hash receipts gốc; originals không bị viết lại.

## Gaps và backup actions còn mở

Nguồn: [risk registry](../experiments/history/audit.json#/at_risk_artifacts).
Không coi một bản sao khác path trên cùng ổ đĩa là independent disaster backup.

| ID / phạm vi | Trạng thái | Phụ trách / hạn |
| --- | --- | --- |
| `G-legacy-archive`: archive lịch sử | Một container quan sát được; chưa fresh whole-file SHA/independent backup | Đề xuất Tài; owner/ngày chưa xác nhận, OPEN |
| `G-B317-parent`: B298 ER checkpoint | Chưa tìm payload verified trong scan; có metadata binding | Đề xuất Tài; ngày retrieve/preserve chưa chốt, OPEN |
| `G-B319-raw`: original scores/tensors | Một vùng lưu trữ local quan sát được; raw audit có giới hạn | Đề xuất Tài; ưu tiên trước raw re-audit/cleanup, ngày chưa chốt |
| `G-B148`: optimizer checkpoints | Bytes đã hash, backup độc lập chưa verified | Đề xuất Tài; ngày chưa chốt, OPEN |
| `G-standalone-source`: source ZIP | Bản local hash verified; remote reference chưa chứng minh recoverable backup | Đề xuất Tài; ngày chưa chốt, OPEN |

Đây là việc bảo toàn còn mở, không chỉ chú thích kết quả. Owner/deadline chưa được
người dùng chốt vẫn OPEN; docs không tự đặt ngày xong hoặc tự sao lưu. Trước xóa/
di chuyển payload quan trọng, người phụ trách cần xác nhận independent recoverable
copy và kiểm restore. Không xóa raw vì đã có report/catalog.

## Bảng nội dung → nguồn kiểm

| Nội dung | Nguồn | Mức kiểm khi viết |
| --- | --- | --- |
| Modules/entrypoints | Tracked tree + architecture source links | Source read; không suy support từ tên |
| Fast/slow/CTX | trainer/replay/fast_update/context/core policy | Callers/branches, không ML rerun |
| Data/metrics | config/manifest/data/model/metrics/summarize | Literals, selectors, aggregation |
| CLI/package/fetch/checkpoint | scripts/notebook/provenance/paths/storage | Static read + CLI help, không model |
| Scientific status/history | B319/B317 cards, catalog/audit, closed exports | Evidence đúng scope, không raw reranking |
| Attribution/license | LICENSE, pyproject, source headers | Files hiện tại, chưa full license audit |

Đây là bản đồ kiểm tài liệu, không là scientific acceptance ledger mới.
