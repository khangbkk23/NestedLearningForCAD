#!/usr/bin/env bash
set -euo pipefail
BUDGET="${1:?budget (2500|5000|10000)}"; SEED="${2:-0}"
RUN_DIR="$(python scripts/benchmarks/0_setup_benchmark.py --protocol conf/benchmarks/protocols/mvtec_1x15_v1.yaml --method conf/benchmarks/methods/cadic_compatible_v1.yaml --seed "$SEED" --set memory.budget="$BUDGET")"
python scripts/benchmarks/1_run_benchmark.py --run-dir "$RUN_DIR"