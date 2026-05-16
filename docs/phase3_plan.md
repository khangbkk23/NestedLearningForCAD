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

2. **NSP2 projector**
   - Add `models/null_space_proj.py`.
   - Fit SVD once per task/cycle.
   - Cache projector and log `null_dim`.
   - Use energy threshold `0.99`.

3. **Minimal N2B-NC consolidator**
   - Add `training/consolidation_engine.py`.
   - Select top-k coreset anchors by utility.
   - Unfreeze only the final backbone blocks and norm.
   - Use distillation loss plus LEJEPA surrogate.
   - Clip gradients at `1.0`.
   - Apply NSP2 to eligible gradients.
   - Roll back if drift is above `0.05`.

4. **CBP monitor**
   - Add `models/cbp.py`.
   - Start with utility/dead-neuron logging only.
   - Enable reinit only after smoke tests pass.

5. **Subspace Recycling**
   - Add fallback thresholds `64 -> 32 -> 16`.
   - Skip the consolidation cycle if no safe null space remains.

6. **Notebook orchestration**
   - Add `notebooks/phase3_n2bnc_kaggle.ipynb`.
   - Keep the notebook thin: setup, load data/config, run module functions, save artifacts.

## Artifacts

- `forgetting_matrix_before.json`
- `forgetting_matrix_after.json`
- `phase3_summary.json`
- `consolidation_log.json`
- final light/full checkpoint, depending on the run mode
