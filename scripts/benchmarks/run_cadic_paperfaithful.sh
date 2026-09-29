#!/usr/bin/env bash
set -euo pipefail

BUDGET="${1:?Usage: $0 <budget> <physical_gpu> [run_tag]}"
GPU="${2:?Usage: $0 <budget> <physical_gpu> [run_tag]}"
TAG="${3:-cadic_paperfaithful_${BUDGET}}"

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="$PWD"
export MVTEC_ROOT="$PWD/data/mvtec"
export CADIC_CKPT="$PWD/checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

RUN=$(
python scripts/benchmarks/0_setup_benchmark.py \
  --protocol conf/benchmarks/protocols/mvtec_1x15_v1.yaml \
  --method conf/benchmarks/methods/cadic_compatible_v1.yaml \
  --seed 0 \
  --run-name "${TAG}_$(date -u +%Y%m%dT%H%M%SZ)" \
  --set extractor.checkpoint_path="$CADIC_CKPT" \
  --set extractor.checkpoint_identity="vit_b8_imagenet21k_local" \
  --set memory.budget="$BUDGET" \
  --set memory.update_unit=image \
  | tail -n 1
)

echo "RUN=$RUN"
echo "$RUN" > "/tmp/cadic_pf_${BUDGET}_gpu${GPU}.run"

run_phase () {
    PHASE="$1"

    python -u - "$RUN" "$PHASE" <<'PY'
import importlib
import runpy
import sys

run_dir = sys.argv[1]
phase = sys.argv[2]

# Force the benchmark runner to use the isolated paper-faithful adapter.
sys.modules["models.cadic_benchmark_adapter_v1"] = importlib.import_module(
    "models.cadic_paperfaithful.cadic_benchmark_adapter"
)

sys.argv = [
    "scripts/benchmarks/1_run_benchmark.py",
    "--run-dir", run_dir,
    "--phase", phase,
    "--device", "cuda",
]

runpy.run_path(
    "scripts/benchmarks/1_run_benchmark.py",
    run_name="__main__",
)
PY
}

echo "===== TRAIN START ====="
run_phase train > "$RUN/train_console.log" 2>&1

echo "===== EVAL START ====="
run_phase eval > "$RUN/eval_console.log" 2>&1

echo "===== DONE ====="
echo "RUN=$RUN"

if [ -f "$RUN/metrics/final_macro.json" ]; then
    cat "$RUN/metrics/final_macro.json"
fi
