#!/usr/bin/env bash
set -euo pipefail

# Server entrypoint for ViSA Phase 1-2 evaluation.
# Requires the VisA dataset at data/visa by default, with per-category folders
# matching conf/visa.yaml.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".pixi/envs/default/bin/python" ]]; then
    PYTHON_BIN=".pixi/envs/default/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

CONFIG="${CONFIG:-conf/visa.yaml}"
MAX_TASKS="${MAX_TASKS:-}"
RUN_SUFFIX="${RUN_SUFFIX:-visa_phase12}"
PROGRESS="${PROGRESS:-1}"

if [[ ! -d "data/visa" ]]; then
  echo "VisA dataset not found at data/visa." >&2
  echo "Set dataset.root_dir in $CONFIG or mount/copy VisA to data/visa before running." >&2
  exit 2
fi

cmd=("$PYTHON_BIN" "training/run_experiment.py" "--config" "$CONFIG" "--disable_wandb" "--run_suffix" "$RUN_SUFFIX")
if [[ "$PROGRESS" != "1" ]]; then
  cmd+=("--quiet")
fi
if [[ -n "$MAX_TASKS" ]]; then
  cmd+=("--max_tasks" "$MAX_TASKS")
fi

echo "==> Running VisA Phase 1-2"
echo "progress=$PROGRESS"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"
