# Scripts Map

For submission, start here:

```bash
bash scripts/run_full_demo.sh
```

## Submission Entrypoint

- `run_full_demo.sh`: runs the full v1 demo tiers: reportable conservative
  Phase 3, mechanism smoke, and experimental NSP2/CBP benchmark.

## Internal Pipeline Steps

These are called by `run_full_demo.sh` and are kept as explicit CLIs for
reproducibility:

- `run_phase3_consolidation.py`: Phase 3 N2B-NC consolidation.
- `evaluate_checkpoint.py`: checkpoint evaluation without retraining.
- `phase3_acceptance.py`: before/after acceptance gate.
- `test_integration_2.py`: mechanism smoke/integration checks.

## Optional Utilities

- `run_server_phase3.sh`: conservative-only server workflow.
- `run_server_visa.sh`: VisA Phase 1-2 workflow once `data/visa` is mounted.
- `compare_checkpoint_scores.py`: slow diagnostic score comparison; not part of
  the default demo.
- `summarize_run.py`: markdown summary for a result directory.
- `compute_forgetting.py`: forgetting metric helper.
- `check_gpu.py`: environment check.
- `01_data_preparation.py`: dataset verification/preparation helper.
