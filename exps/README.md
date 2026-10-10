# exps — experimental source layout

All flat files in `exps/` have been regrouped into task-code subdirectories
(`exps/<task_code>/`), each mirrored by a runner shim under `scripts/exps/<task_code>/`.

## Task-code groups

| Group | Files (implementation / test) | Runner shim |
|---|---|---|
| `ol01/` | `test_hope_outer_learning_stage0_v1.py` | `hope_outer_learning_stage0_v1.py` |
| `ol03/` | `hope_outer_learning_p1_v1.py` | — |
| `ol04/` | `hope_outer_learning_pilot_v1.py`, `test_hope_outer_learning_pilot_v1.py` | `hope_outer_learning_pilot_v1.py` |
| `ol04d/` | `hope_outer_learning_horizon_audit_v1.py`, `test_hope_outer_learning_horizon_audit_v1.py` | `hope_outer_learning_horizon_audit_v1.py` |
| `ol04e/` | `hope_outer_learning_h50_v1.py`, `test_hope_outer_learning_h50_v1.py` | `hope_outer_learning_h50_v1.py` |
| `task3r/` | `hope_gate3.py` | `hope_gate3.py` |
| `anomaly/` | `hope_anomaly_signal.py`, `hope_anomaly_score_ablation.py`, `hope_normality_audit.py` + tests | matching shims |

## Deliberately left flat (shared or single-experiment)

These are imported by several task groups, so nesting them would be misleading:

- `hope_image_synchronous_memory.py` — locked P1 oracle; its SHA256
  `f5e02238…bee668b` is a hard gate in `exps/ol04e/hope_outer_learning_h50_v1.py`,
  so the file must not be edited.
- `hope_retention_stabilization.py`, `hope_update_stabilization.py` — shared
  stabilization modules imported by `task3r/`, the P1 oracle and `anomaly/`.
- `hope_cad_ad01_*.py`, `test_hope_cad_ad01_normal_support.py` — AD-01 kept flat
  by explicit request.

## Conventions

- Every new Python file starts with `# <repo-relative-path>`.
- New runners under `scripts/exps/<group>/` must be thin shims that import the
  implementation from `exps.<group>.<module>`.
- Moving a module one level deeper changes `/scripts/exps/<group>/` repo-root
  resolution from `parents[2]` to `parents[3]` (`parents[1]` → `parents[2]` for
  `exps/<group>/`). Verify with `PYTHONPATH=. python -c "import <module>"`.
- Run tests from the repository root; stale `__pycache__` can mask a broken
  import path, so clear it after moving modules.
