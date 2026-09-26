# Benchmark harness v1

The three entry points are intentionally thin: `0_setup_benchmark.py` resolves a
protocol and method and writes the immutable train manifest, `1_run_benchmark.py`
trains and then evaluates a frozen final state (or runs one phase), and
`2_compare_benchmarks.py` produces a read-only table without ranking seeds.

Set `MVTEC_ROOT` before setup. Protocol files define the MVTec 1x15 task stream;
method files define CADIC-compatible or historical Meta-NATH behavior. The train
manifest contains only `train/good`; official test paths are constructed only in
the final evaluation and offline forgetting phases. Smoke runs are explicitly
`reportable: false`.

```bash
export MVTEC_ROOT=/datasets/mvtec
python scripts/benchmarks/0_setup_benchmark.py \
  --protocol conf/benchmarks/protocols/mvtec_1x15_v1.yaml \
  --method conf/benchmarks/methods/cadic_compatible_v1.yaml \
  --seed 0 --set memory.budget=2500
python scripts/benchmarks/1_run_benchmark.py --run-dir results/benchmarks/mvtec_1x15_v1/cadic_compatible_v1/<RUN>
python scripts/benchmarks/2_compare_benchmarks.py --runs <RUN_A> <RUN_B> --output comparison.md
```

The CADIC-compatible memory ladder is 2,500, 5,000, and 10,000 patch vectors;
these are not image counts. Checkpoints are in `states/`, metrics in `metrics/`,
and timing/memory records in `profile/` and `summary.json`. The current adapter
does not claim exact CADIC parity because the published checkpoint and several
protocol fields remain unresolved. HOPE is deliberately not part of this path.
