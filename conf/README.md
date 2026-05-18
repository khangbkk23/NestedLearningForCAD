# Configuration Map

Use `conf/config.yaml` as the stable Phase 1-2 baseline. Other configs are
purpose-specific and should not replace the baseline unless the run explicitly
needs that mode.

| File | Purpose | Status |
| --- | --- | --- |
| `config.yaml` | DINOv2 Phase 1-2 MVTec baseline | Reportable baseline |
| `config_phase3.yaml` | Phase 3 anchor warmup and smoke settings with stored images | Utility config |
| `config_phase3_conservative.yaml` | Accepted conservative Phase 3.0 MVTec 8-task candidate | Reportable Phase 3.0 config |
| `config_phase3_kaggle_gpu.yaml` | Kaggle/server variant of the conservative Phase 3.0 config with larger batches and workers | Preferred Kaggle config |
| `config_phase3_experimental_nsp2_cbp.yaml` | NSP2/CBP reset smoke only | Experimental, not reportable |
| `config_visa.yaml` | VisA Phase 1-2 run once `data/visa` is mounted | Prepared, not yet benchmarked |

Kaggle entrypoint:

```text
notebooks/kaggle_full_phase3_workflow.ipynb
```

Server entrypoint:

```bash
bash scripts/run_server_phase3.sh
```
