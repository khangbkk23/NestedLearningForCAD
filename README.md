# Nested Learning for CAD

Active checkout: **hope_cad_v6**. This tree integrates Khang's `dikhang_space@b985944` with our `hope_cad_v5@9e86478`, then retains only the current method, baseline infrastructure, tests, notebook and essential specs/papers.

**Research status:** historical v5 completed its four-job normal T4 pilot and failed the registered risk gate (NESTED/CADIC=1.128397>1.10). No v5 anomaly benchmark is available. The successor **v5.1 causal scaled utility** has a separate implementation and local contract tests; **real-normal T4 preflight/pilot and AD validation are pending**. Frozen [architecture](docs/main_spec/architecture_decision_v5_1.md), [instruction](docs/main_spec/instruction_CAD_v5_1.md), [planning](docs/main_spec/planning_CAD_v5_1.md) and [contract](docs/main_spec/revision_contract_v5_1.json) retain their design-time status; the new [implementation record](docs/main_spec/v5_1_implementation.md) records what was actually built and tested.

## Setup

```powershell
pixi install --locked
pixi run test
```

Pixi owns the local Python environment and test feature; do not borrow `.pixi` from another checkout. Local PyTorch/CUDA and Kaggle T4 environments are separate identities. Pillow uses a self-contained wheel at the same version12.2.0 to avoid Windows Conda DLL collisions; pixel preprocessing parity is checked against the original environment. Canonical Kaggle versions remain pinned in `notebook_support/support_v5/runtime_lock.json`.

## Pilot evidence and next implementation

The completed v5 delivery and original evidence live outside Git in `../.agents/artifacts/`; the current result is recorded in `../.agents/evidence/status.json`. The [existing notebook](notebooks/hope_cad_support_v5_kaggle.ipynb) runs **v5**, not v5.1. Do not rerun it to evaluate the new design, resume its learned state into the successor, or overwrite the archived `hope_support_v5_v6_r4` delivery.

Read the [implementation record and Kaggle steps](docs/main_spec/v5_1_implementation.md) and [acceptance addendum](docs/main_spec/implementation_acceptance_v5_1.md). The new [v5.1 notebook](notebooks/hope_cad_support_v5_1_kaggle.ipynb) uses its own source inventory/bundle/evidence schemas. It integrates discarded preflight24 followed by four fresh256-normal pilot jobs. Do not attach old learned evidence. Scientific gate failure stops for review; local tests do not authorize an automatic anomaly benchmark. Historical W6/W7/recovery/v5 source and evidence remain unchanged.

## Code layout

- `models/hope_cad_v5_1`: current candidate learner, transactional causal session and RGB adapter.
- `models/hope_cad_v5`: preserved predecessor; its raw geometry helpers are immutable dependencies.
- `models/hope_cad_v2`: actively reused visual/CMS primitives and independent oracle; names retained for traceability.
- `models/cadic_paperfaithful`: reference coreset, exact tile cache and full Eq9 adapter.
- `dataset`: download, byte-bound manifests/protocols and mask QA.
- `training`: causal pilot, baseline engine/metrics, shared runtime.
- `notebook_support`: current bootstrap/evidence/heartbeat; no old recovery workflows.
- `scripts/benchmarks`: CADIC/ReplayCAD baseline orchestration and comparison.
- `conf`: protocols, baseline settings, delivery inventory and invariance provenance.
- `tests`: current numerical, state, baseline and shipping contracts.
- `docs`: papers and current full specs only.

## Baselines and data

Khang's latest MVTec downloader is kept at `dataset/download_mvtec.py` (use `--help`). Data stays ignored or is supplied through mount/env paths. Matched CADIC entry: `python -m scripts.hope_cad_v2.run_matched_cadic --help`; baseline scaffold and ReplayCAD commands are documented in [scripts](scripts/README.md).

ReplayCAD requires an explicit external checkout, separate detector/LDM Python environments and large assets. [Reference provenance](refs/replaycad.json) preserves Khang's gitlink and observed official HEAD without silently substituting one for the other. The native scaffold is retained; full continual-boundary reproduction is not claimed.

## Reproducibility and remaining work

The v5 equations/contract, frozen ViT producer and visual numerical kernels are retained; [invariance](conf/provenance/algorithm_invariance.json) tracks unchanged definitions/bytes. Cleanup does not improve or invalidate historical benchmark numbers.

After implementation, test and review of a fresh successor pilot: complete its matched benchmark orchestration, controls, seeds/task matrices/uncertainty and resource accounting, then freeze transfer before VisA. Those milestones are not completed by a clean repository, a mathematical design check or passing software tests. No CADIC superiority is claimed.

Shared context: `../.agents/AGENTS.md`. Historical source snapshots, run ZIPs/logs, migration mapping and backups are external to this repository. Do not delete the primary repository's `.git`: linked worktrees share it. No old checkout is removed as part of this integration.
