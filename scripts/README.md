# Scripts

Start here for submission:

```bash
bash scripts/run_full_demo.sh
```

## Layout

- `run_full_demo.sh`: primary 3-tier MVTec demo workflow.
- `run_server_phase3.sh`: conservative-only MVTec Phase 3 workflow wrapper.
- `run_server_visa.sh`: VisA conservative Phase 3 workflow wrapper.
- `pipeline/`: reproducible CLIs used by the full demo.
- `diagnostics/`: smoke tests, summaries, GPU checks, and metric helpers.
- `workflows/`: optional server workflows. Root-level `run_server_*.sh` files are thin compatibility wrappers.

## Pipeline CLIs

- `pipeline/run_phase3_consolidation.py`: Phase 3 N2B-NC consolidation.
- `pipeline/evaluate_checkpoint.py`: evaluate a saved checkpoint without retraining.
- `pipeline/phase3_acceptance.py`: before/after metric-gated acceptance report.
- `pipeline/compare_checkpoint_scores.py`: slow score-distribution diagnostics, not part of the default demo.

## Diagnostics

- `diagnostics/mechanism_smoke.py`: TITANS, CADIC, ACC, NSP2, CBP, and Subspace Recycling smoke test.
- `diagnostics/summarize_run.py`: markdown summary for a result directory.
- `diagnostics/compute_forgetting.py`: forgetting metric from an evaluation matrix.
- `diagnostics/check_gpu.py`: environment/GPU check.

## Workflows

- `workflows/run_server_phase3.sh`: conservative-only MVTec Phase 3 workflow.
- `workflows/run_server_visa.sh`: optional VisA Phase 3 workflow once `data/visa` is mounted.
