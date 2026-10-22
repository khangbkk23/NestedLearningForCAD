# Nested Learning for Continual Anomaly Detection

HOPE/CMS học utility để quản lý raw normal-support memory; frozen ViT và CADIC nearest-neighbor scoring giữ nguyên. Method: `hope_cad_v5_1_causal_scaled_utility`. **V7/package 0.7.0 là workspace hợp nhất, không phải một scientific revision mới.**

## Bắt đầu

```bash
python -m pip install -e . --no-deps --no-build-isolation
hope-cad --help
python -m hope_cad.local_tests
```

Cần cài dependencies trước: local Windows dùng `pixi install --locked` rồi `pixi run install`; Linux/RTX5090 dùng `bash scripts/remote/setup.sh /absolute/path/to/python3.13`. Không dùng local dependency solve thay remote runtime pins. Đọc [hướng dẫn máy thuê](docs/operations.md), chạy preflight GPU thật trước train. Chưa thuê/chạy 5090 trong lượt refactor.

## Đọc gì?

| Nội dung | Tài liệu |
|---|---|
| Toán học, causal utility, fast/slow learning, contracts | [Method](docs/method.md) |
| Package, ownership, luồng dữ liệu và optional integrations | [Architecture](docs/architecture.md) |
| Protocol, controls, metrics và acceptance | [Benchmark](docs/benchmark.md) |
| SSH/setup/preflight/train/resume/export | [Operations](docs/operations.md) |

```text
src/hope_cad/    package duy nhất, chia theo chức năng
tests/          numerical, causal, state/resume và benchmark regressions
scripts/        source inventory và remote setup/run
docs/           bốn tài liệu hiện hành và papers
requirements/   runtime GPU đã định danh
refs/           provenance upstream integrations
pyproject.toml  metadata, editable install, CLI và pytest
pixi.toml/lock  môi trường local có khóa
```

## Trạng thái bằng chứng

Phương pháp trước refactor đã có normal pilot và paired MVTec seed0 endpoints/matrices được review. Đó là development evidence, không phải multi-seed/VisA superiority. Refactor kiểm numerical/state/score/resume tương đương; không tự tạo benchmark hoặc performance gain. Các raw artifacts/reviews/source bundles cũ vẫn nằm ngoài repo. Runtime/source mới phải preflight rồi chạy fresh; không ghép cohort phần cứng khác để claim paired gain.

Không có dữ liệu, checkpoints, API keys hoặc lịch sử chat trong source distribution. `.agents` là context tùy chọn ngoài checkout, không là dependency của chương trình. Báo cáo học thuật đã nộp cũng được giữ ngoài repo. Scientific revision historical-query consolidation vẫn là proposal, chưa implement.
