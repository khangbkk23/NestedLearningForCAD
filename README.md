# Nested Learning for CAD

**Workspace v7, phương pháp v5.1.** HOPE/CMS học utility để quản lý normal-support bank, giữ raw ViT geometry và CADIC Eq9. V7 bảo toàn source v6 rồi gom vận hành để chạy từ checkout sạch trên Linux/GPU thuê; chưa phải revision khoa học mới.

Evidence v5.1 MVTec seed0: NESTED I-AUROC86,9561% /P-AP47,4331%; matched CADIC86,9983% /44,9012%. Hai120cell matrices có pixel-retention benefit quan sát được. Single-seed development chưa chứng minh superiority tổng quát; cleanup không sinh benchmark mới. Full controls/seeds và VisA chưa hoàn tất.

## Bắt đầu

```powershell
pixi install --locked
pixi run test
pixi run python -B -m hope_cad --help
```

Pixi cho local Windows/Python3.14; remote có Linux/Python3.13/cu128 candidate riêng. [Runbook5090](docs/remote_gpu.md). Local tests không nghiệm thu máy thuê. Không cần copy .pixi/.agents/old Kaggle outputs hoặc sibling checkout để chương trình chạy.

```text
python -m hope_cad preflight   # discarded gates; local-smoke không mở benchmark
python -m hope_cad train       # fresh/resume normal stream, không official test
python -m hope_cad evaluate    # final metrics, completed own training required
python -m hope_cad matrix      # saved task states, no training
python -m hope_cad aggregate   # registered metrics/CI khi đủ seeds
python -m hope_cad export      # verified ZIP ngoài payload tree
python -m hope_cad source      # portable source ZIP, không runtime/data/artifacts
```

Real runs cần explicit data-root/checkpoint/run-dir/job/device và passed remote preflight đúng source/runtime/data. Không có default program path trỏ vào shared .agents. Xem command examples, pause/stop/download rules trong runbook.

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| hope_cad | CLI/runtime/source locks, reviewed NN backend, discarded preflight, evidence/shipping |
| models/hope_cad_v5_1 | Current fast/slow learner và causal admission |
| models/hope_cad_v5 | Geometry dependency và predecessor đối chiếu |
| models/hope_cad_v2 | Visual/CMS, scan/state và independent oracle |
| models/cadic_paperfaithful, feature_extractors | CADIC bank/proposal/cache/Eq9, frozen ViT |
| training/support_benchmark_v5_1 | Frozen study, atomic stream engine, evaluation/statistics |
| training, dataset | Runtime/data rights/manifest/mask QA và baseline infrastructure |
| conf | Method/protocol/study, shipping inventory, runtime candidate, preserved scientific hashes |
| scripts/remote | Explicit setup/logging; không tự thuê/SSH |
| tests | Numerical/meta/causal/atomic/resume/scoring/metrics/portable contracts |
| docs | Papers, full specs/dependencies, workspace/runbook và research proposal |
| refs | External baseline provenance; không claim ReplayCAD đã reproduced |

K10k patch vectors/raw784×768, frozen ViT-B/8 ImageNet21k, FP32/noTF32 và inherited FP64 scalars. Proposal/anchors/alternative bank/delayed labels/exact4support4query vẫn nguyên; không cắt để tăng tốc.

## Specs và reproducibility

- [Architecture](docs/main_spec/architecture_decision_v5_1.md), [instruction](docs/main_spec/instruction_CAD_v5_1.md), [planning](docs/main_spec/planning_CAD_v5_1.md), [contract](docs/main_spec/revision_contract_v5_1.json), [acceptance](docs/main_spec/implementation_acceptance_v5_1.md).
- [Benchmark](docs/main_spec/benchmark_v5_1.md), [workspace gates](docs/main_spec/workspace_v7.md), [remote runbook](docs/remote_gpu.md).
- conf/provenance/workspace_v7.json giữ scientific hashes và lineage full269files. V6 có149untracked khi snapshot; HEAD một mình không đủ.
- [Historical-query proposal](docs/research/historical_query_consolidation.md) chưa approve/freeze/implement.

Full v6 source và old operational wrappers còn ở commit410ac6a của branch này; byte-exact archive/patches/run history ngoài repo được user quản lý. Chúng không cần mount cho fresh run. Existing T4 owner/migration không auto import vào portable cohort.

V5.1 giữ matched-local preprocessing; native-checkpoint normalization mismatch/author recipe chưa giải quyết causal impact. Không exact paper parity. MVTec development-exposed; freeze method/protocol trước VisA, study runner này chưa hỗ trợ VisA.
