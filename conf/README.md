# Configuration Map

For submission, the full demo uses only the reportable server config plus the
explicit experimental config. The other files are kept because they support
manual reproduction, smoke runs, or optional datasets.

## Submission Configs

| File | Purpose | Status |
| --- | --- | --- |
| `config_phase3_kaggle_gpu.yaml` | Default full-demo conservative Phase 3 config for Kaggle/server runs | Reportable if acceptance passes |
| `config_phase3_experimental_nsp2_cbp.yaml` | NSP2/CBP benchmark path used by the experimental tier | Experimental unless acceptance passes |

## Reference Configs

| File | Purpose | Status |
| --- | --- | --- |
| `config.yaml` | DINOv2 Phase 1-2 MVTec baseline | Reportable baseline |
| `config_phase3.yaml` | Phase 3 anchor warmup and smoke settings with stored images | Utility config |
| `config_phase3_conservative.yaml` | Accepted conservative Phase 3.0 MVTec 8-task candidate | Reportable Phase 3.0 config |
| `config_visa.yaml` | VisA Phase 1-2 run once `data/visa` is mounted | Prepared, not yet benchmarked |

Kaggle entrypoint:

```text
notebooks/kaggle_full_phase3_workflow.ipynb
```

Server entrypoint:

```bash
bash scripts/run_full_demo.sh
```

Conservative-only server entrypoint:

```bash
bash scripts/run_server_phase3.sh
```
