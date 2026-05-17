# Phase 3 Plan

Goal: add cloud/offline consolidation without disturbing the Phase 1-2 baseline.

## Assumptions

- Phase 1-2 remains the edge/light path.
- Phase 3 runs on Kaggle/Colab or another cloud GPU.
- Start with the current `facebook/dinov2-base` prototype by design; DINOv3 migration is a separate experiment after the DINOv2 path is locked.
- `store_images: true` is enabled only for Phase 3 runs that need raw anchor images.

## Success Criteria

- Standard forgetting matrix exists before and after consolidation.
- A single N2B-NC cycle runs on a small task subset without breaking frozen-backbone semantics after the cycle.
- Drift rollback works when cosine drift exceeds the configured threshold.
- Phase 1-2 baseline metrics remain reproducible.

## Implementation Order

1. **Forgetting matrix**
   - Use `evaluation.forgetting_matrix: true` on a small run.
   - Compute FM with `scripts/compute_forgetting.py`.

2. **Minimal N2B-NC consolidator, without NSP2 first**
   - Add `training/consolidation_engine.py`.
   - Select top-k coreset anchors by utility.
   - Balance anchors across stored task ids before utility tie-breaking.
   - Unfreeze only the final backbone blocks and norm.
   - Use CLS distillation, patch-token distillation, and LEJEPA surrogate.
   - Treat patch-token distillation as mandatory for pixel-localization safety.
   - Use at least a few optimizer steps when testing patch-token distillation;
     with a single step the patch loss starts at zero because teacher and
     student tokens are initially identical.
   - Clip gradients at `1.0`.
   - Do not apply NSP2 or CBP in the first smoke version.

3. **Drift rollback**
   - Roll back if drift is above `0.05`.
   - Keep the backbone frozen again after every consolidation attempt.
   - If rollback is not triggered, refresh/re-index coreset embeddings with
     the updated backbone before saving the Phase 3 checkpoint.

4. **Metric-gated acceptance**
   - Evaluate the source and candidate checkpoints on the same task subset.
   - Reject candidates that regress `final_cumulative_pixel_aupr` beyond the
     configured tolerance, even if image AUROC improves.

5. **NSP2 projector**
   - Add `models/null_space_proj.py`.
   - Fit SVD once per task/cycle.
   - Cache projector and log `null_dim`.
   - Use energy threshold `0.99`.
   - Apply NSP2 to eligible gradients only after minimal N2B-NC works.

6. **CBP monitor**
   - Add `models/cbp.py`.
   - Start with utility/dead-neuron logging only.
   - Enable reinit only after smoke tests pass.

7. **CBP reset**
   - Reset only neurons below the configured utility threshold.
   - Project reset weights through NSP2 where shapes are eligible.

8. **Subspace Recycling**
   - Add fallback thresholds `64 -> 32 -> 16`.
   - Skip the consolidation cycle if no safe null space remains.

9. **Notebook orchestration**
   - Add `notebooks/phase3_n2bnc_kaggle.ipynb`.
   - Keep the notebook thin: setup, load data/config, run module functions, save artifacts.

## Artifacts

- `forgetting_matrix_before.json`
- `forgetting_matrix_after.json`
- `phase3_summary.json`
- `consolidation_log.json`
- final light/full checkpoint, depending on the run mode

## Active Phase 3 Files

- `conf/config_phase3.yaml`: separate Phase 3 config; baseline config stays untouched.
- `conf/config_phase3_conservative.yaml`: accepted 8-task conservative candidate for metric-gated Phase 3.0.
- `conf/config_phase3_experimental_nsp2_cbp.yaml`: explicit experimental config for NSP2/CBP reset smoke only.
- `conf/config_visa.yaml`: VisA Phase 1-2 config; requires `data/visa`.
- `training/consolidation_engine.py`: N2B-NC consolidation and drift rollback.
- `models/null_space_proj.py`: NSP2 projector, disabled by default for the first smoke.
- `models/cbp.py`: CBP monitor/reset helper, monitor-only by default.
- `scripts/run_phase3_consolidation.py`: CLI entrypoint for local/Kaggle execution.
- `scripts/evaluate_checkpoint.py`: source/candidate checkpoint evaluation.
- `scripts/phase3_acceptance.py`: metric-gated source-vs-candidate acceptance report.
- `scripts/compare_checkpoint_scores.py`: targeted score distribution diagnostics.
- `scripts/run_server_phase3.sh`: reproducible Linux server workflow for anchor
  warmup, conservative Phase 3.0, evaluation, and acceptance.
- `scripts/run_server_visa.sh`: VisA Phase 1-2 server entrypoint.
- `notebooks/phase3_n2bnc_kaggle.ipynb`: minimal thin Kaggle orchestration.
- `notebooks/kaggle_full_phase3_workflow.ipynb`: full Kaggle orchestration
  notebook for branch `taitrn`, MVTec Phase 3.0, optional 15-task, optional VisA.

## First Smoke Commands

Create an anchor checkpoint with raw images:

```powershell
.\.pixi\envs\default\python.exe training\run_experiment.py --config conf\config_phase3.yaml --max_tasks 4 --disable_wandb --quiet --run_suffix phase3_anchor_warmup_4task
```

Run one consolidation cycle:

```powershell
.\.pixi\envs\default\python.exe scripts\run_phase3_consolidation.py --config conf\config_phase3.yaml --checkpoint results\<warmup_run>\last_checkpoint.pt --run_suffix phase3_smoke
```

Evaluate the consolidated checkpoint:

```powershell
.\.pixi\envs\default\python.exe scripts\evaluate_checkpoint.py --config conf\config_phase3.yaml --checkpoint results\<phase3_run>\last_checkpoint.pt --max_tasks 4 --quiet --run_suffix phase3_after_eval_4task
```

Apply metric-gated acceptance:

```powershell
.\.pixi\envs\default\python.exe scripts\phase3_acceptance.py --config conf\config_phase3.yaml --before results\<before_eval_run>\checkpoint_eval_summary.json --after results\<after_eval_run>\checkpoint_eval_summary.json --output results\<phase3_run>\acceptance_report.json
```
