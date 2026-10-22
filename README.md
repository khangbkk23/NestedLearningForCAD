# Nested Learning for Continual Anomaly Detection

HOPE/CMS học utility để chấp nhận hoặc từ chối proposal cập nhật normal-support bank của CADIC. Frozen ViT tạo raw patch features; bank thật được controller chọn quyết định anomaly map và image score bằng full CADIC Eq9.

**Code hiện tại:** `hope_cad_v5_1_causal_scaled_utility`, package0.7.0. Tên workspace **v7** là tổ chức mã nguồn, không phải một thuật toán mới. Runtime hiện hành dùng matched MVTec3.599normals. **Full-train3.629 và historical-query H32 đã có spec, chưa được triển khai.**

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
python -m pip install -e . --no-deps --no-build-isolation
hope-cad --help
python -m hope_cad.local_tests
```

Cài dependencies trước: Windows local dùng `pixi install --locked` rồi `pixi run install`; Linux/RTX5090 dùng `bash scripts/remote/setup.sh /absolute/path/to/python3.13`. Các pins chuẩn bị không thay actual remote preflight. [Runbook](docs/operations.md) mô tả inputs, lệnh, resume và export; source không tự thuê máy/chạy GPU.

## Đọc và phát triển

- [Bản đồ tài liệu](docs/README.md): phân biệt code hiện tại, kết quả và nâng cấp chưa triển khai.
- [Architecture](docs/architecture.md), [method](docs/method.md), [benchmark](docs/benchmark.md): repo hiện hoạt động như thế nào.
- [Research spec/plan](docs/research/README.md): D0, H32 đầy đủ, controls, gates/budgets và implementation plan; [pipeline](docs/pipeline_alignment_spec.md) chốt full-train và recipe.
- [QUICK_UPDATE](QUICK_UPDATE.md): Overview, milestones từ nền chung, protocol và chi tiết để trao đổi với nhóm.

```text
src/hope_cad/    package duy nhất, chia theo chức năng
tests/          gradient, causal, state/resume và benchmark regressions
scripts/        source inventory và remote setup/run
docs/           code hiện tại, kết quả, research specs/plan và papers
requirements/   runtime GPU đã định danh
refs/           provenance upstream integrations
pyproject.toml  metadata, editable install, CLI và pytest
pixi.toml/lock  môi trường local có khóa
```

Repo tự chứa các tài liệu cần để hiểu và phát triển phương pháp. Không chứa dataset/checkpoint/raw output archives/API keys/chat history. Muốn chạy phải cung cấp data/weights đúng identity; muốn tự tính lại kết quả lịch sử phải nhận các archives ghi trong results. Mọi spec/link cần đọc nằm trong repo hoặc nguồn khoa học công khai.
