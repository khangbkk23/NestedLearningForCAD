# Config Guide

Use one config per run. Do not merge these into one YAML; separate files make
Kaggle/server commands reproducible and keep each benchmark profile explicit.

## Main Choices

| Config | Use when | Notes |
| --- | --- | --- |
| `full_demo.yaml` | Default MVTec submission run | Conservative Phase 3, larger Kaggle/server batches, reportable if accepted |
| `experimental_nsp2_cbp.yaml` | Full-tech experimental run | Enables NSP2 + CBP reset; report as experimental unless acceptance passes |
| `visa_phase3.yaml` | Optional VisA Phase 3 validation | Conservative Phase 3 path, requires `data/visa` |
| `visa.yaml` | Legacy VisA smoke | Phase 1-2-only path, use only with `RUN_PHASE3=0` |

## Reference Choices

| Config | Use when | Notes |
| --- | --- | --- |
| `reference/phase1_baseline.yaml` | Local Phase 1-2 baseline or data verification | Small/default MVTec baseline |
| `reference/phase3_smoke.yaml` | Quick Phase 3 smoke/debug | Utility profile, not the submission benchmark |
| `reference/phase3_conservative_local.yaml` | Reproduce the earlier conservative local profile | Same conservative idea, smaller local batches than `full_demo.yaml` |

## Recommended Commands

```bash
# Submission / Kaggle
bash scripts/run_full_demo.sh

# Stronger MVTec pass
MAIN_MAX_TASKS=15 EXPERIMENTAL_MAX_TASKS=15 bash scripts/run_full_demo.sh

# Optional VisA Phase 3
bash scripts/run_server_visa.sh
```

Default demo path:

```text
full_demo.yaml -> mechanism smoke -> experimental_nsp2_cbp.yaml
```
