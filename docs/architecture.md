# Kiến trúc workspace

```text
src/hope_cad/
  cli.py                # entrypoint thống nhất
  contract.py           # method identity và JSON contract
  paths.py              # package resources/source checkout, không sibling lookup
  models/               # learner, session, geometry, frozen ViT
    primitives/         # visual self-modification, CMS, scan và oracle
  baselines/cadic/      # bank, exact pair cache, adapters và compatible route
  data/                 # manifests, prefix rights, mask QA
  benchmark/            # study, train/evaluation/matrix/statistics; optional baseline harness
  runtime/              # precision, locks, preflight, atomic IO, shipping/export
  integrations/         # CADIC/ReplayCAD wrappers; không tự cài upstream hoặc chạy
  resources/            # method/study/runtime/protocol JSON/YAML và lineage
  testing/              # adapter giả cho harness tests, không phải method benchmark thật
tests/                  # chia theo chức năng; không theo version
scripts/remote/         # setup/run qua SSH
docs/                   # docs hiện hành, results, research specs/plan và papers
```

## Một đường chạy

`cli → runtime.execution → benchmark.engine → models.session → geometry/CADIC bank + utility learner`. Frozen ViT tạo raw patch features; utility learner không chạy trên anomaly test. `model.session` sở hữu clock, pending credit, replay, RNG và transaction; exact pair-cache/fixed-shape NN chỉ thay execution, không thay bank update/score.

Package cài bằng `pip install -e . --no-deps --no-build-isolation`. `src/` tránh import nhầm các thư mục cạnh repo. Wheel chứa model APIs và normative resource JSON/YAML; CLI `--help` và model imports dùng được ngoài repo. Các lệnh execution, source shipping và local tests cần **verified source checkout** vì identity bao gồm source/config/docs/tests; chúng từ chối nếu chỉ có wheel, thay vì tìm checkout lân cận. Remote workflow dùng editable install từ source ZIP.

Không giữ `models/hope_cad_v2`, `models/hope_cad_v5` hay `training/support_*` trong workspace này. Primitive còn dùng được chuyển vào tên chức năng; common geometry và validation được hợp nhất. Serialized class/config/schema names hoặc method ID có version được giữ khi thuộc contract lịch sử, không phải code version song song.

## Optional baseline integrations

Matched study chính dùng `hope-cad train --job CADIC_seed0`. Compatible CADIC và ReplayCAD wrappers được đặt trong `integrations`, không nằm trên hot path NESTED. ReplayCAD cần repo/dependencies/checkpoints riêng và chưa có matched result trong study này; không gọi adapter availability là benchmark acceptance. `testing.fake_adapter` chỉ phục vụ fixtures/explicit smoke; không biến nó thành scientific comparator.

## Ranh giới dữ liệu và artifact

Data/weights/runs ở ngoài source checkout hoặc các thư mục outputs bị ignore; không commit dữ liệu thật. Train chỉ dùng normals theo manifest; GT/masks chỉ vào evaluator; scoring không update state/RNG. Checkpoint atomic, lock identity, source hashes, feature receipts và task checkpoints được giữ.

Dataset, checkpoint và raw result ZIPs nằm ngoài source; identities/provenance và giới hạn truy cập được công bố trong [results](results.md). Docs/spec/plan cần cho hiểu và phát triển nằm trong repo. Không cần một kho context hoặc sibling checkout riêng.

## Research successor đã thêm

[Research specs](research/README.md) và [pipeline](pipeline_alignment_spec.md) định nghĩa successor full-train/H32, đã có CLI research và revision-aware factories/resources riêng; chưa remote acceptance. Code/resources baseline giữ strict. Xem [implementation](research/implementation.md). Không gộp identity candidate với v5.1 hoặc thay frozen JSON để aggregator nhận nhầm.

## Tổ chức tham khảo

- [Python Packaging: src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/): tách package cài đặt khỏi root repository.
- [Anomalib](https://github.com/open-edge-platform/anomalib): package, tests, docs và công cụ triển khai tách vai trò.
- [DINOv2](https://github.com/facebookresearch/dinov2): code chia theo chức năng nghiên cứu, entrypoints và configs rõ.

Đây là lựa chọn scaffold cho dự án này, không là chứng nhận SOTA hoặc bản sao cấu trúc của một repo khác.
