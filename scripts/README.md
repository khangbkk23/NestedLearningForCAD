# Entrypoints

Run from the repository root with Pixi. Current HOPE pilot uses `pixi run package`, `pixi run verify` and the single Kaggle notebook. Full v5 anomaly orchestration remains a later milestone.

- Matched CADIC: `python -m scripts.hope_cad_v2.run_matched_cadic --help` (prepare/train/eval/fm, strict prefix protocol).
- Baseline harness: `python scripts/benchmarks/0_setup_benchmark.py --help`, then `1_run_benchmark.py --help`.
- ReplayCAD external native pipeline: `python -m scripts.benchmarks.replaycad.setup --help`, `python -m scripts.benchmarks.replaycad.run --help`.
- Normalized comparison: `python -m scripts.benchmarks.compare --help`.

Supply explicit run-dir/cache paths under `../.agents/artifacts/runs/`; data/checkpoint paths are external inputs. ReplayCAD requires separate environments/assets and does not implement full continual-boundary mode here. Do not infer a missing metric is zero or report structural smoke as a scientific run.
