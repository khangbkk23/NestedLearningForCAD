# Phase 3 Plan

Goal: add cloud/offline consolidation without disturbing the Phase 1-2 baseline.

## Assumptions

- Phase 1-2 remains the edge/light path.
- Phase 3 runs on Kaggle/Colab or another cloud GPU.
- Start with the current `facebook/dinov2-base` prototype; DINOv3 migration is a separate experiment.
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
   - Unfreeze only the final backbone blocks and norm.
   - Use distillation loss plus LEJEPA surrogate.
   - Clip gradients at `1.0`.
   - Do not apply NSP2 or CBP in the first smoke version.

3. **Drift rollback**
   - Roll back if drift is above `0.05`.
   - Keep the backbone frozen again after every consolidation attempt.

4. **NSP2 projector**
   - Add `models/null_space_proj.py`.
   - Fit SVD once per task/cycle.
   - Cache projector and log `null_dim`.
   - Use energy threshold `0.99`.
   - Apply NSP2 to eligible gradients only after minimal N2B-NC works.

5. **CBP monitor**
   - Add `models/cbp.py`.
   - Start with utility/dead-neuron logging only.
   - Enable reinit only after smoke tests pass.

6. **CBP reset**
   - Reset only neurons below the configured utility threshold.
   - Project reset weights through NSP2 where shapes are eligible.

7. **Subspace Recycling**
   - Add fallback thresholds `64 -> 32 -> 16`.
   - Skip the consolidation cycle if no safe null space remains.

8. **Notebook orchestration**
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
- `training/consolidation_engine.py`: N2B-NC consolidation and drift rollback.
- `models/null_space_proj.py`: NSP2 projector, disabled by default for the first smoke.
- `models/cbp.py`: CBP monitor/reset helper, monitor-only by default.
- `scripts/run_phase3_consolidation.py`: CLI entrypoint for local/Kaggle execution.
- `notebooks/phase3_n2bnc_kaggle.ipynb`: thin Kaggle orchestration.

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
