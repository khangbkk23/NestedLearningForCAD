# Kiến trúc codebase hiện hành

Đối chiếu source: `74d975b151f59ad0b5fbf23be392e2d891f350bf` — 11/11/2026.

[Đường đọc](../README.md#đường-đọc) · [Method](method.md) · [Operations](operations.md)

## Luồng chạy

```text
config + train manifest + locked runtime
  → scripts/train.py: input check, source bundle và receipt
  → data.prepare: training selection, full-test/GT manifest
  → trainer.arm_trial: fresh initialization cho từng arm
      train/good → frozen encoder → grouped features
      pre-event memory → reservations → fast update (+ slow khi đến lịch)
      commit arrival → reservoir/FIFO admission
      task boundary → observer → scores → metrics → verified ZIP chunks
  → summarize.py → record_run.py: matrices/aggregates và receipt nhẹ
```

`--arm all` chạy ba child processes tuần tự. Mỗi arm khởi tạo fresh, không dùng
checkpoint arm trước. Notebook chỉ bootstrap/setup và launch, không chứa learner
thứ hai. Encoder frozen; decoder/fast optimizer thay đổi theo arrivals. Memory
giữ features; policy và slow Adam ở CPU, detector/features ở GPU trong loop.
Observer không gọi learner update và có model/RNG before-after checks.

## Package Python

Các đường dẫn dưới tính từ `src/nested_replay_cad/`. Files `__init__.py` đánh dấu
packages, không là một learner riêng.

| Module | Trách nhiệm |
| --- | --- |
| [trainer.py](../src/nested_replay_cad/trainer.py) | Stream loop, detector, admission, observer, guards/counters và final checkpoint |
| [replay.py](../src/nested_replay_cad/replay.py) | Contexts, replay weights, query/hypergradient và slow commit |
| [fast_update.py](../src/nested_replay_cad/fast_update.py) | Functional StableAdamW proposal và forward-bound first conditional VJP |
| [context.py](../src/nested_replay_cad/context.py) | CTX detached transform, không nhận test labels/category IDs |
| [data.py](../src/nested_replay_cad/data.py) | Task order/boundaries, training selection và full-test/GT identities |
| [metrics.py](../src/nested_replay_cad/metrics.py) | Tied-group ranks/AP/AUROC và memory/runtime measurements |
| [storage.py](../src/nested_replay_cad/storage.py) | Disk guard, ZIP chunks, hash/readback trước khi bỏ tensor trung gian |
| [paths.py](../src/nested_replay_cad/paths.py) | Artifact root, inputs/runs/cache/tmp và cache environment |
| [provenance.py](../src/nested_replay_cad/provenance.py) | Design check, exact source bundle, live/terminal receipts |
| [model/encoder.py](../src/nested_replay_cad/model/encoder.py) | Frozen DINOv2-reg, preprocessing và hai feature groups |
| [model/decoder.py](../src/nested_replay_cad/model/decoder.py) | Decoder/bottleneck, loss hooks, StableAdamW và recipe constants |
| [model/scoring.py](../src/nested_replay_cad/model/scoring.py) | Anomaly maps, resizing/smoothing và image readout |
| [core/contracts.py](../src/nested_replay_cad/core/contracts.py), [errors.py](../src/nested_replay_cad/core/errors.py), [identity.py](../src/nested_replay_cad/core/identity.py) | State dataclasses, errors và canonical identities |
| [core/data/manifest.py](../src/nested_replay_cad/core/data/manifest.py) | Dataset paths/bytes và training manifest construction |
| [core/memory/rng.py](../src/nested_replay_cad/core/memory/rng.py), [sampler.py](../src/nested_replay_cad/core/memory/sampler.py) | PCG streams, reservations và Algorithm R admission |
| [core/policy/layout.py](../src/nested_replay_cad/core/policy/layout.py), [policy.py](../src/nested_replay_cad/core/policy/policy.py), [numerical.py](../src/nested_replay_cad/core/policy/numerical.py) | Policy layout/forward và pairwise-Jacobian first backward |
| [core/learning/adam.py](../src/nested_replay_cad/core/learning/adam.py), [parameters.py](../src/nested_replay_cad/core/learning/parameters.py) | Slow functional Adam và gradient/state validation |
| [core/runtime/tensors.py](../src/nested_replay_cad/core/runtime/tensors.py) | Lazy reference-torch/runtime checks |

`model/` là detector; `core/policy/` là replay controller. Comments Spec3 hoặc
qualification lịch sử trong inherited source không tự làm toàn spec cũ thành
authority cho B319; xem [lineage](provenance.md).

## Files ngoài package

| Path | Vai trò |
| --- | --- |
| [configs/b319.json](../configs/b319.json) | Seed/arms/primary, encoder identity, hard-mining ramp và caps |
| [configs/train_manifest.json](../configs/train_manifest.json) | Frozen training identities/order/bytes |
| [pixi.toml](../pixi.toml), [pixi.lock](../pixi.lock), [pyproject.toml](../pyproject.toml) | Một root environment/lock pair và distribution metadata |
| [scripts/train.py](../scripts/train.py) | Input-only check, orchestration, receipt lifecycle |
| [scripts/preflight.py](../scripts/preflight.py) | GPU smoke có forward/backward/update nhỏ |
| [scripts/prepare_assets.py](../scripts/prepare_assets.py) | Download/hash/extract inputs |
| [scripts/kaggle_run.py](../scripts/kaggle_run.py) | Local `pack`, remote `fetch`; không có subcommand launch |
| [scripts/watch_kaggle.py](../scripts/watch_kaggle.py) | Live logs, reconnect, terminal status |
| [scripts/summarize.py](../scripts/summarize.py) | Aggregate recorded stage metrics, không raw reranking |
| [scripts/record_run.py](../scripts/record_run.py) | Export terminal receipts/analysis; giữ closed outputs |
| [activate_source.bat](../scripts/activate_source.bat), [activate_source.sh](../scripts/activate_source.sh) | Source-path activation trên Windows/Linux |
| [notebooks/b319.ipynb](../notebooks/b319.ipynb) | Pinned bundle loader, setup/assets/check/smoke/train |
| [tests/](../tests/) | Layout, provenance, scientific primitives và mocked integration |
| [experiments/](../experiments/README.md) | Experiment lifecycle, history/cards và reports nhẹ |

Tests gồm `test_local_layout.py`, `test_experiments.py`, `test_science.py`;
`context_oracle.py` và `numerical_oracle.py` là các đối chiếu nhỏ; `conftest.py`
thiết lập paths/cache. Chúng không thay usefulness/portability qualification.
Final checkpoint lưu audit state; chưa có CLI resume được qualification.
