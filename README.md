# Meta-NATH CAD

Phase 1-2 Light prototype for continual anomaly detection on MVTec AD.
DINOv2 is the intentional stable backbone for the current reportable code path;
DINOv3 is a later migration experiment after the DINOv2 pipeline is locked.

The active pipeline is:

1. Frozen backbone feature extraction (`facebook/dinov2-base` for the current prototype).
2. TITANS-style test-time memory update.
3. ACC gating.
4. CADIC unified coreset update with normal samples only.
5. Patch nearest-neighbor image and pixel anomaly scoring.

See `docs/instruction_CAD.md` for the full research target. Phase 3 now has
an accepted conservative 8-task N2B-NC consolidation path; NSP2 and CBP are
present behind config flags and remain experimental until accepted benchmark
runs are added to `docs/runs.md`. After a successful non-rollback
consolidation, coreset embeddings are refreshed with the updated backbone before
the Phase 3 checkpoint is saved.

## Setup

```powershell
.\.pixi\envs\default\python.exe --version
```

or install from `requirements.txt` in a separate environment.

## Verify

```powershell
.\.pixi\envs\default\python.exe scripts\diagnostics\mechanism_smoke.py
.\.pixi\envs\default\python.exe scripts\data\prepare_data.py --run_verify --config .\conf\reference\phase1_baseline.yaml
```

## Run

For submission, use the full v1 closure workflow. It is the single public
entrypoint for the current MVTec demo:

```bash
bash scripts/run_full_demo.sh
```

It runs three tiers:

- Main reportable demo: conservative Phase 3 before/after/acceptance.
- Mechanism demo: integration smoke for TITANS, CADIC, NSP2, CBP reset, and
  Subspace Recycling code paths.
- Experimental benchmark: NSP2/CBP config with the same before/after acceptance
  gate.

Useful submission knobs:

```bash
MAIN_MAX_TASKS=15 EXPERIMENTAL_MAX_TASKS=15 bash scripts/run_full_demo.sh
REQUIRE_EXPERIMENTAL_ACCEPTED=1 bash scripts/run_full_demo.sh
PYTHON_BIN=/path/to/python bash scripts/run_full_demo.sh
```

Kaggle workflow:

```text
notebooks/kaggle_full_phase3_workflow.ipynb
```

The notebook calls the same full-demo script, adds visible progress to artifact
inspection/packaging cells, and uses `conf/full_demo.yaml` by
default for larger Kaggle batches/workers.

Conservative-only and VisA server workflows are kept as optional utilities:

```bash
MAX_TASKS=8 RUN_TESTS=1 RUN_SCORE_COMPARE=0 bash scripts/run_server_phase3.sh
bash scripts/run_server_visa.sh
```

`RUN_SCORE_COMPARE=1` is an optional diagnostic pass and can take a long time
after acceptance has already passed; keep it off for normal Kaggle/server
demos.

## Inspect Results

```powershell
.\.pixi\envs\default\python.exe scripts\diagnostics\summarize_run.py results\<run_dir>
```

Baseline runs are tracked in `docs/runs.md`.

By default runs save `last_checkpoint.pt` only. Set
`logging.checkpoint_policy: "all"` if you need per-task checkpoints, or
`"best_and_last"` if you want best image/pixel checkpoints as well.

## Submission Surface

- `scripts/run_full_demo.sh`: full 3-tier v1 closure workflow.
- `notebooks/kaggle_full_phase3_workflow.ipynb`: one-notebook Kaggle workflow.
- `conf/full_demo.yaml`: default reportable server/Kaggle config.
- `conf/experimental_nsp2_cbp.yaml`: experimental NSP2/CBP benchmark config.
- `docs/runs.md`: verified run log.

## Project Layout

- `conf/`: run profiles; choose one YAML per run.
- `dataset/`: MVTec/VisA loading and synthetic anomaly generation.
- `models/`: Meta-NATH core modules and mechanisms.
- `training/`: importable training/evaluation/consolidation engines.
- `scripts/`: command-line entrypoints and diagnostics.
- `notebooks/`: Kaggle orchestration only; logic stays in scripts/modules.
- `utils/`: small shared runtime helpers.
- `docs/`: scope, pipeline notes, and verified run records.
- `legacy/`: old prototypes kept for reference only.

## Core Files

- `models/meta_nath_core.py`: frozen backbone, TITANS, ACC, CADIC orchestration.
- `models/cadic_coreset.py`: unified memory bank and patch-NN scoring.
- `training/meta_nath_engine.py`: normal-only streaming update and evaluation.
- `training/run_experiment.py`: main experiment entrypoint.
- `training/consolidation_engine.py`: Phase 3 N2B-NC consolidation.
- `dataset/load_dataset.py`: MVTec/VisA continual task stream.
- `scripts/pipeline/run_phase3_consolidation.py`: Phase 3 CLI entrypoint.
- `scripts/pipeline/evaluate_checkpoint.py`: evaluates a saved checkpoint without retraining.
- `scripts/pipeline/phase3_acceptance.py`: compares before/after checkpoint metrics and accepts or rejects a Phase 3 candidate.
- `scripts/pipeline/compare_checkpoint_scores.py`: optional score distribution diagnostics.
- `scripts/run_server_phase3.sh`: conservative-only Linux utility workflow.
- `scripts/run_server_visa.sh`: optional VisA conservative Phase 3 utility workflow.
- `scripts/diagnostics/summarize_run.py`: markdown run summaries.
- `scripts/diagnostics/compute_forgetting.py`: forgetting metric from an evaluation matrix.
- `utils/global_seed.py`: shared reproducibility helper.

## Legacy Code

Older ViT-CMS/trainer/evaluator scripts live under `legacy/`. They are kept for reference only and should not be used for Meta-NATH Phase 1-2 or Phase 3 benchmarks.
