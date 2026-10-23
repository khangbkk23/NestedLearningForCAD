# Nested Learning for Continual Anomaly Detection

HOPE/CMS học utility để chấp nhận hoặc từ chối proposal cập nhật normal-support bank của CADIC. Frozen ViT tạo raw patch features; bank thật được controller chọn quyết định anomaly map và image score bằng full CADIC Eq9.

**Code hiện tại:** `hope_cad_v5_1_causal_scaled_utility`, package0.7.0. Tên workspace **v7** là tổ chức mã nguồn, không phải một thuật toán mới. Runtime hiện hành dùng matched MVTec3.599normals. **Full-train3.629 và historical-query H32 đã implement/test local qua CLI `research`; chưa remote preflight/pilot hoặc AD benchmark.**

**Mốc source chính thức trước nâng cấp:** `89df9a21658b265731f98a7c2014afb1876f1924`, gồm code và bộ docs/spec/plan công khai. Dùng mốc này làm gốc phát triển/rollback; benchmark cũ vẫn giữ provenance của run thực tế.

## Kết quả đang có

| Matched MVTec seed0 | CADIC | NESTED v5.1 |
|---|---:|---:|
| Final image AUROC | 87,00% | 86,96% |
| Final pixel AP | 44,90% | 47,43% |
| Pixel forgetting, thấp hơn tốt hơn | 8,91điểm% | 6,83điểm% |

Một seed/order,15categories,120matrix cells/method;3.599train/1.725test. Đây là kết quả v5.1 trước refactor, không là fresh v7/5090/H32 benchmark hoặc proof superiority tổng quát. MVTec development-exposed; VisA chưa chạy. Đọc [kết quả và provenance](docs/results.md) trước khi so với paper hoặc run riêng.

## Bắt đầu với code hiện tại

```bash
pixi install --locked
pixi run --locked env-check
pixi run --locked cad --help
pixi run --locked test
pixi run --locked assets
```

Pixi là package/environment manager duy nhất; `pixi install --locked` cài Python, dependencies và project editable, không cần pip/uv/venv thủ công. Cùng `pixi.lock` khóa Linux (Python3.13/PyTorch2.11+cu128) và Windows local (Python3.14/PyTorch2.12+cu126). Linux lock đã solve, cài/kernel5090 cần preflight thật. `assets` là bước tải dữ liệu/ViT riêng, chỉ chạy khi được yêu cầu; không tự tải5GB khi install. [Runbook](docs/operations.md), [research handoff](docs/research/implementation.md).

Bàn giao máy thuê qua Git: người dùng review, commit và push branch `hope_cad_v7`, rồi clone branch và kiểm full commit SHA trước setup/preflight. Source ZIP cũ giữ provenance local verification, không là yêu cầu chuyển source. ZIP outputs của remote run vẫn cần giữ cùng accounting sidecar và log theo research handoff.

## Đọc và phát triển

- [Bản đồ tài liệu](docs/README.md): phân biệt code hiện tại, kết quả và candidate đã implement nhưng chưa nghiệm thu remote.
- [Architecture](docs/architecture.md), [method](docs/method.md), [benchmark](docs/benchmark.md): repo hiện hoạt động như thế nào.
- [Research spec/plan](docs/research/README.md): D0, H32 đầy đủ, controls, gates/budgets; [implementation và SSH handoff](docs/research/implementation.md); [pipeline](docs/pipeline_alignment_spec.md) chốt full-train và recipe.
- [QUICK_UPDATE](QUICK_UPDATE.md): Overview, milestones từ nền chung, protocol và chi tiết để trao đổi với nhóm.

```text
src/hope_cad/    package duy nhất, chia theo chức năng
tests/          gradient, causal, state/resume và benchmark regressions
scripts/        source inventory và remote setup/run
docs/           code hiện tại, kết quả, research specs/plan và papers
refs/           provenance upstream integrations
pyproject.toml  metadata, editable install, CLI và pytest
pixi.toml/lock  môi trường Linux/Windows và editable install có khóa
```

Repo Git tự chứa tài liệu/code, không phát hành dataset/checkpoint/raw outputs/API keys/chat history. Trên máy chạy, `workspace/` gom data, pretrained weights, records, checks và experiments; được ignore và loại khỏi source inventory. Mỗi experiment có run/logs và exports theo session bất biến. Muốn chạy phải cung cấp data/weights đúng identity; muốn tự tính lại kết quả lịch sử phải nhận các archives ghi trong results. Mọi spec/link cần đọc nằm trong repo hoặc nguồn khoa học công khai.
