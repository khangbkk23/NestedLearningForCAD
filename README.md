# Nested Learning for CAD

Active checkout: **hope_cad_v6**. This tree integrates Khang's `dikhang_space@b985944` with our `hope_cad_v5@9e86478`, then retains only the current method, baseline infrastructure, tests, notebook and essential specs/papers.

**Research status:** v5 causal support controller is implemented; its real-normal T4 pilot and anomaly benchmark remain pending. Existing v4 negative results are preserved externally. V6 does not change the registered v5 mathematical method or claim CADIC parity.

## Setup

```powershell
pixi install --locked
pixi run test
```

Pixi owns the local Python environment and test feature; do not borrow `.pixi` from another checkout. Local PyTorch/CUDA and Kaggle T4 environments are separate identities. Pillow uses a self-contained wheel at the same version12.2.0 to avoid Windows Conda DLL collisions; pixel preprocessing parity is checked against the original environment. Canonical Kaggle versions remain pinned in `notebook_support/support_v5/runtime_lock.json`.

## Current pilot

```powershell
pixi run package
pixi run verify
```

The delivery is written outside Git to `../.agents/artifacts/deliveries/hope_support_v5_v6_r4/`. Set `HOPE_ARTIFACTS_ROOT` to override the shared artifact directory on another machine. No historical r4 source ZIP is needed to build this checkout.

Import [the notebook](notebooks/hope_cad_support_v5_kaggle.ipynb) into Kaggle. Attach exactly the newly built `hope_cad_support_v5_source_v6_r4.zip`, MVTec AD archive and locked ViT-B/8 ImageNet21k checkpoint. Keep `PHASE='pilot'`, GPU T4, Save Version → Save & Run All. It runs a discarded24-normal preflight then CADIC/GEOMETRIC/FIRST_ORDER/NESTED,256 normal images per fresh job, one trial. `SESSION_HOURS=3` is a soft pause budget, not an ETA. Preserve ZIP/log and review gates before anomaly evaluation.

The new source schema/layout cannot resume old source-v1 pilot states. Historical W6/W7/recovery evidence and their original source bundles remain unchanged in the shared store; they are not initialization inputs for this pilot. See [workspace and handoff](docs/main_spec/workspace_v6.md).

## Code layout

- `models/hope_cad_v5`: utility learner, causal bank owner and RGB adapter.
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

After accepted pilot: complete full matched v5 benchmark orchestration, controls, seeds/task matrices/uncertainty and resource accounting, then freeze transfer before VisA. Those milestones are not completed by a clean repository or passing tests.

Shared context: `../.agents/AGENTS.md`. Historical source snapshots, run ZIPs/logs, migration mapping and backups are external to this repository. Do not delete the primary repository's `.git`: linked worktrees share it. No old checkout is removed as part of this integration.
