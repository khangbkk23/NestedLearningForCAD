# Legacy Code

This folder contains code from earlier prototypes that is not part of the active Meta-NATH CAD Phase 1-2 or Phase 3 path.

Kept for reference:

- `models/vit_cms.py`: old ViT-CMS baseline/decoder.
- `training/trainer.py`, `training/evaluator.py`, `training/cms_optim.py`: old supervised-style training stack.
- `training/run_sweep.py`: old sweep runner for ViT-CMS-style knobs.
- `scripts/train_mvtec.py`, `scripts/test_forward.py`, `scripts/test_integration.py`: old smoke/demo scripts.
- `scripts/get_mvtec_meta.py`: old replay metadata helper, not used by the active dataset loader.
- `notebooks/eval_checkpoint.ipynb`: old checkpoint notebook.

Use the active entrypoints documented in `README.md` and `docs/PIPELINE.md` for current experiments.
