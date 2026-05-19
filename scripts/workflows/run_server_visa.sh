#!/usr/bin/env bash
set -euo pipefail

# Server entrypoint for the VisA workflow.
# Default mode is the full conservative Phase 3 benchmark:
#   warmup -> before eval -> Phase 3 -> after eval -> acceptance.
# Set RUN_PHASE3=0 for a Phase 1-2-only smoke run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PIPELINE_DIR="scripts/pipeline"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".pixi/envs/default/bin/python" ]]; then
    PYTHON_BIN=".pixi/envs/default/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export METANATH_REQUIRE_HF_BACKBONE="${METANATH_REQUIRE_HF_BACKBONE:-1}"
export METANATH_LOCAL_FILES_ONLY="${METANATH_LOCAL_FILES_ONLY:-0}"

CONFIG="${CONFIG:-conf/visa_phase3.yaml}"
MAX_TASKS="${MAX_TASKS:-}"
RUN_PHASE3="${RUN_PHASE3:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-visa_phase12}"
REQUIRE_ACCEPTED="${REQUIRE_ACCEPTED:-0}"
PROGRESS="${PROGRESS:-1}"
STEP_TIMEOUT_SECONDS="${STEP_TIMEOUT_SECONDS:-7200}"
LOCAL_FILES_ONLY_AFTER_WARMUP="${LOCAL_FILES_ONLY_AFTER_WARMUP:-1}"

if [[ ! -d "data/visa" ]]; then
  echo "VisA dataset not found at data/visa." >&2
  echo "Set dataset.root_dir in $CONFIG or mount/copy VisA to data/visa before running." >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Required config not found: $CONFIG" >&2
  exit 2
fi

QUIET_ARGS=()
if [[ "$PROGRESS" != "1" ]]; then
  QUIET_ARGS=(--quiet)
fi

MAX_TASK_ARGS=()
TASK_LABEL="full"
if [[ -n "$MAX_TASKS" ]]; then
  MAX_TASK_ARGS=(--max_tasks "$MAX_TASKS")
  TASK_LABEL="${MAX_TASKS}task"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/server_visa_${STAMP}}"
mkdir -p "$LOG_DIR"

TOTAL_STEPS=3
if [[ "$RUN_PHASE3" == "1" ]]; then
  TOTAL_STEPS=7
fi
CURRENT_STEP=1

latest_dir() {
  local pattern="$1"
  local dir
  dir="$(ls -td $pattern 2>/dev/null | head -n 1 || true)"
  if [[ -z "$dir" ]]; then
    echo "Could not find result directory matching: $pattern" >&2
    exit 1
  fi
  echo "$dir"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

run_logged() {
  local name="$1"
  shift
  local start_ts end_ts duration status
  start_ts="$(date +%s)"
  echo
  echo "==> [$CURRENT_STEP/$TOTAL_STEPS] $name"
  echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "    $*" | tee "$LOG_DIR/${CURRENT_STEP}_${name}.cmd.txt"
  set +e
  if [[ "$STEP_TIMEOUT_SECONDS" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "$STEP_TIMEOUT_SECONDS" "$@" 2>&1 | tee "$LOG_DIR/${CURRENT_STEP}_${name}.log"
  else
    "$@" 2>&1 | tee "$LOG_DIR/${CURRENT_STEP}_${name}.log"
  fi
  status=${PIPESTATUS[0]}
  set -e
  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))
  echo "    end: $(date '+%Y-%m-%d %H:%M:%S') (${duration}s)" | tee -a "$LOG_DIR/${CURRENT_STEP}_${name}.log"
  if [[ "$status" -ne 0 ]]; then
    echo "Step failed: $name (exit $status)" >&2
    exit "$status"
  fi
  CURRENT_STEP=$((CURRENT_STEP + 1))
}

check_acceptance() {
  local report="$1"
  require_file "$report"
  local decision
  decision="$("$PYTHON_BIN" - "$report" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)
print("accepted" if report.get("accepted") else "rejected")
PY
)"
  echo "visa_phase3 decision=$decision report=$report"
  if [[ "$REQUIRE_ACCEPTED" == "1" && "$decision" != "accepted" ]]; then
    echo "VisA Phase 3 did not pass acceptance and REQUIRE_ACCEPTED=1." >&2
    exit 1
  fi
}

echo "Meta-NATH VisA workflow"
echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
if command -v git >/dev/null 2>&1; then
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
fi
echo "config=$CONFIG"
echo "max_tasks=${MAX_TASKS:-all}"
echo "run_phase3=$RUN_PHASE3"
echo "require_accepted=$REQUIRE_ACCEPTED"
echo "progress=$PROGRESS"
echo "step_timeout_seconds=$STEP_TIMEOUT_SECONDS"
echo "hf_hub_disable_xet=$HF_HUB_DISABLE_XET"
echo "metanath_require_hf_backbone=$METANATH_REQUIRE_HF_BACKBONE"
echo "metanath_local_files_only=$METANATH_LOCAL_FILES_ONLY"
echo "logs=$LOG_DIR"

run_logged py_compile "$PYTHON_BIN" -u -m py_compile \
  training/run_experiment.py \
  training/consolidation_engine.py \
  training/meta_nath_engine.py \
  "$PIPELINE_DIR/run_phase3_consolidation.py" \
  "$PIPELINE_DIR/evaluate_checkpoint.py" \
  "$PIPELINE_DIR/phase3_acceptance.py"

run_logged bash_syntax bash -n scripts/run_server_visa.sh scripts/workflows/run_server_visa.sh

if [[ "$RUN_PHASE3" != "1" ]]; then
  run_logged visa_phase12 "$PYTHON_BIN" -u training/run_experiment.py \
    --config "$CONFIG" \
    --disable_wandb \
    "${QUIET_ARGS[@]}" \
    "${MAX_TASK_ARGS[@]}" \
    --run_suffix "$RUN_SUFFIX"

  echo
  echo "VisA Phase 1-2 smoke workflow completed."
  exit 0
fi

ANCHOR_SUFFIX="visa_anchor_${TASK_LABEL}_${STAMP}"
BEFORE_SUFFIX="visa_before_phase3_${TASK_LABEL}_${STAMP}"
CANDIDATE_SUFFIX="visa_phase3_conservative_${TASK_LABEL}_${STAMP}"
AFTER_SUFFIX="visa_after_phase3_${TASK_LABEL}_${STAMP}"

run_logged visa_anchor_warmup "$PYTHON_BIN" -u training/run_experiment.py \
  --config "$CONFIG" \
  --disable_wandb \
  "${QUIET_ARGS[@]}" \
  "${MAX_TASK_ARGS[@]}" \
  --run_suffix "$ANCHOR_SUFFIX"

WARMUP_DIR="$(latest_dir "results/*_${ANCHOR_SUFFIX}")"
WARMUP_CKPT="$WARMUP_DIR/last_checkpoint.pt"
require_file "$WARMUP_CKPT"
export METANATH_LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY_AFTER_WARMUP"
echo "Using local HuggingFace cache after VisA warmup: METANATH_LOCAL_FILES_ONLY=$METANATH_LOCAL_FILES_ONLY"

run_logged visa_before_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
  --config "$CONFIG" \
  --checkpoint "$WARMUP_CKPT" \
  "${MAX_TASK_ARGS[@]}" \
  "${QUIET_ARGS[@]}" \
  --run_suffix "$BEFORE_SUFFIX"

BEFORE_DIR="$(latest_dir "results/MetaNATH_Eval_*_${BEFORE_SUFFIX}")"
require_file "$BEFORE_DIR/checkpoint_eval_summary.json"

run_logged visa_phase3_conservative "$PYTHON_BIN" -u "$PIPELINE_DIR/run_phase3_consolidation.py" \
  --config "$CONFIG" \
  --checkpoint "$WARMUP_CKPT" \
  --run_suffix "$CANDIDATE_SUFFIX"

PHASE3_DIR="$(latest_dir "results/*_${CANDIDATE_SUFFIX}")"
PHASE3_CKPT="$PHASE3_DIR/last_checkpoint.pt"
require_file "$PHASE3_CKPT"

run_logged visa_after_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
  --config "$CONFIG" \
  --checkpoint "$PHASE3_CKPT" \
  "${MAX_TASK_ARGS[@]}" \
  "${QUIET_ARGS[@]}" \
  --run_suffix "$AFTER_SUFFIX"

AFTER_DIR="$(latest_dir "results/MetaNATH_Eval_*_${AFTER_SUFFIX}")"
require_file "$AFTER_DIR/checkpoint_eval_summary.json"

ACCEPTANCE_REPORT="$PHASE3_DIR/acceptance_report.json"
run_logged visa_acceptance "$PYTHON_BIN" -u "$PIPELINE_DIR/phase3_acceptance.py" \
  --config "$CONFIG" \
  --before "$BEFORE_DIR/checkpoint_eval_summary.json" \
  --after "$AFTER_DIR/checkpoint_eval_summary.json" \
  --output "$ACCEPTANCE_REPORT"

check_acceptance "$ACCEPTANCE_REPORT"

echo
echo "VisA Phase 3 workflow completed."
echo "warmup_dir=$WARMUP_DIR"
echo "before_eval=$BEFORE_DIR/checkpoint_eval_summary.json"
echo "phase3_dir=$PHASE3_DIR"
echo "after_eval=$AFTER_DIR/checkpoint_eval_summary.json"
echo "acceptance_report=$ACCEPTANCE_REPORT"
