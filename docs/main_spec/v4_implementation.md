# Nested Learning for CAD — corrective v4 implementation

Worktree duy nhất: `D:/src/DATN/hope_cad_v2_2`, branch `hope_cad_v2_2`, base
`d78faa69c066094deb9bb9decc59df84b5d470db`. Người dùng phê duyệt triển khai
2026-10-14. Không commit/push/merge; các changes v3 đã tồn tại trước lượt này được giữ.

Design ID `hope_cad_v4_warm_joint_cms_row_retention`; selection-rule ID
`prefix_warm_joint_stress_v4`. V3 specs và evidence giữ nguyên; full v4 specs là
`architecture_decision_v4.md`, `instruction_CAD_v4.md`, `planning_CAD_v4.md`.
`v4_inheritance_manifest.json` lưu parent hashes. `revision_contract_v4.json`
khóa thiết kế/recipe trước fits v4.

| Phần | Evidence/source | Trạng thái |
|---|---|---|
| Row retention rho=0.9/chunk | `visual_memory.py::scan_chunked`, `FourScanVisualMemory`; `reference.py::scan_reference` | Implemented; numerical tests local; chưa calibrated real v4 |
| Joint-final own partials | `cms.py::ContinuumMemory.event(objective='joint_final')` | Implemented; independent pre-event oracle, meta/deployment parity, exact16 finite difference |
| Warm outer + validation | `hope_outer_learning_v4.py::{episode_loss,short_validation_components,fit}` | Implemented; zero/16 components và data permissions tested |
| Finite search 400/gates | `hope_selection_v4.py`, `hope_cms_stress_v4.py`, `hope_search_journal_v4.py` | Implemented; gate constants v3 giữ nguyên; fresh real selection pending |
| Versioned initialization | `hope_initialization_v4.py` | Implemented; schemas/design/joint score/receipt/source/context/zero-state checks |
| RGB integration | `benchmark_adapter_v4.py`, `preflight_rgb_v4.py` | Implemented; fixture tests; new real RGB Kaggle preflight pending |
| Resume/runtime | `hope_runtime_v4.py`, `preflight_outer_v4.py`, `run_normal_v4.py` | Implemented; atomic optimizer/RNG and pipeline fixture tests; T4 v4 pending |
| Cache reuse | `import_frozen_cache_v4.py`, unchanged `hope_feature_contract_v3.py` | Only calibration tensor copied; parent/hash receipt; contextual + RGB verification remains mandatory |
| Mechanism | `hope_mechanism_v4.py`, `diagnostics_v4.py` | Implemented; continuous 0/16/32/pool histories; visual-off during adaptation+score; freeze levels; center/border diagnostics |
| Scientific W6 v4 | New saved Kaggle artifacts | NOT experimentally validated |
| W7/W8/W9 | Locked AD benchmark, VisA, thesis/repro | NOT completed |

## Numerical details found by the new tests

Joint-final updates cannot reuse a plain `autograd.grad(L_final, current_slow_weights)`
on a meta-linked state: faster current weights can themselves depend on unchanged
slow weights through past updates. That computes a total historical derivative
instead of the own-parameter local partial. `torch.func.grad` now differentiates
the current dictionaries as independent formal arguments, while returned gradients
retain their dependence on actual inputs, decoder, dictionaries and history for
the outer derivative. The oracle isolates dictionaries only to check numerical
local partials; production meta training never detaches their actual history.
This is a v4 implementation correction, not a discovered error in old v3 W6.

New near-floor Hessian fixtures exposed nonfinite FP32 derivatives in an inactive
spectral branch. V4 preserves the exact FP32 forward cap, floors, margins and ULP
rounding, with an FP64 continuous derivative bridge for that cap only. Recurrence,
CMS tensors, losses and accumulators remain FP32. Local FP64 oracle and derivative
parity, zero/saturated and near-floor regressions check the bridge. A squared
1e20 first-gradient norm itself overflows FP32; second derivative tests use fixed
representable cotangents, not an overflowing diagnostic norm. Default v3 numerical
path is unchanged. This bridge is disclosed; no claim of all Hessians being bounded.

## Local verification and evidence scope

Run with Pixi from the main environment, executing the absolute worktree entrypoint:

```powershell
pixi run --locked python -B D:/src/DATN/hope_cad_v2_2/scripts/hope_cad_v2/verify_v4.py --output D:/src/DATN/hope_cad_v2_2/results/hope_cad_v2/v4_verification_local.json
```

The final verification JSON/log and `v4_delivery_validation.json` record actual
counts, failures/skips, source hashes and runtime. Earlier debug failures are not
scientific fit failures and are not represented as passing runs. Local GPU is
RTX 3050 Ti 4 GiB; canonical scientific W6 requires >=12 GiB. Reduced smoke never
passes that resource gate. New T4 preflight measures exact16 warm graph, real RGB
parity and two real-prefix outer steps with optimizer/RNG resume.

Final local suite: **113/113 PASS, zero failures/errors/skips**, 92.849 seconds
at the recorded source snapshot. CUDA reduced exact16 warm-graph profile PASS;
full-grid FP32/FP64 forward/backward and independent first/second-derivative
fixtures PASS. Synthetic selection/export CLI smoke PASS (nonreportable, not a
real calibrated initialization). Delivery QA PASS: three v3 parent hashes unchanged,
all major sections inherited into v4, current source equals final-suite hashes,
both notebook code cells compile, seven historical frozen producers match.

No real v4 learning outcome, duration or CADIC win is established by these tests.
W6 v3/probe informed development; probe is exposed, not untouched confirmation.
No official test or future task is used by v4 fitting/selection. Historical official
test exposure remains disclosed in the full specs.

Kaggle files and next actions: [w6_v4_handoff.md](w6_v4_handoff.md).
