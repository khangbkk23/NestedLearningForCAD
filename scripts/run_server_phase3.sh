#!/usr/bin/env bash
set -euo pipefail

# Server entrypoint for the verified Phase 3.0 workflow.
# Defaults are intentionally conservative: 8 MVTec tasks, metric-gated acceptance,
# no NSP2/CBP reset/Subspace Recycling.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".pixi/envs/default/bin/python" ]]; then
    PYTHON_BIN=".pixi/envs/default/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

BASELINE_CONFIG="${BASELINE_CONFIG:-conf/config.yaml}"
PHASE3_CONFIG="${PHASE3_CONFIG:-conf/config_phase3.yaml}"
CONSERVATIVE_CONFIG="${CONSERVATIVE_CONFIG:-conf/config_phase3_conservative.yaml}"
MAX_TASKS="${MAX_TASKS:-8}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_PHASE12_FULL="${RUN_PHASE12_FULL:-0}"
RUN_SCORE_COMPARE="${RUN_SCORE_COMPARE:-0}"
PROGRESS="${PROGRESS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

QUIET_ARGS=()
if [[ "$PROGRESS" != "1" ]]; then
  QUIET_ARGS=(--quiet)
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/server_phase3_${STAMP}}"
mkdir -p "$LOG_DIR"

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

run_logged() {
  local name="$1"
  shift
  local start_ts end_ts duration status
  start_ts="$(date +%s)"
  echo
  echo "==> [$CURRENT_STEP/$TOTAL_STEPS] $name"
  echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "    $*" | tee "$LOG_DIR/${name}.cmd.txt"
  set +e
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  status=${PIPESTATUS[0]}
  set -e
  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))
  echo "    end: $(date '+%Y-%m-%d %H:%M:%S') (${duration}s)" | tee -a "$LOG_DIR/${name}.log"
  return "$status"
}

echo "Meta-NATH server Phase 3.0 workflow"
echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
if command -v git >/dev/null 2>&1; then
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
fi
echo "max_tasks=$MAX_TASKS"
echo "phase3_config=$PHASE3_CONFIG"
echo "conservative_config=$CONSERVATIVE_CONFIG"
echo "progress=$PROGRESS"
echo "logs=$LOG_DIR"

TOTAL_STEPS=7
if [[ "$RUN_TESTS" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "$RUN_PHASE12_FULL" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "$RUN_SCORE_COMPARE" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
CURRENT_STEP=1

run_logged py_compile "$PYTHON_BIN" -u -m py_compile \
  training/run_experiment.py \
  training/consolidation_engine.py \
  training/meta_nath_engine.py \
  scripts/run_phase3_consolidation.py \
  scripts/evaluate_checkpoint.py \
  scripts/phase3_acceptance.py \
  scripts/compare_checkpoint_scores.py
CURRENT_STEP=$((CURRENT_STEP + 1))

if [[ "$RUN_TESTS" == "1" ]]; then
  run_logged integration "$PYTHON_BIN" -u scripts/test_integration_2.py
  CURRENT_STEP=$((CURRENT_STEP + 1))
fi

if [[ "$RUN_PHASE12_FULL" == "1" ]]; then
  run_logged phase12_full "$PYTHON_BIN" -u training/run_experiment.py \
    --config "$BASELINE_CONFIG" \
    --disable_wandb \
    "${QUIET_ARGS[@]}" \
    --run_suffix "server_phase12_full15_${STAMP}"
  CURRENT_STEP=$((CURRENT_STEP + 1))
fi

ANCHOR_SUFFIX="server_phase3_anchor_${MAX_TASKS}task_${STAMP}"
BEFORE_SUFFIX="server_before_phase3_${MAX_TASKS}task_${STAMP}"
CANDIDATE_SUFFIX="server_phase3_conservative_${MAX_TASKS}task_${STAMP}"
AFTER_SUFFIX="server_after_phase3_conservative_${MAX_TASKS}task_${STAMP}"

run_logged anchor_warmup "$PYTHON_BIN" -u training/run_experiment.py \
  --config "$PHASE3_CONFIG" \
  --max_tasks "$MAX_TASKS" \
  --disable_wandb \
  "${QUIET_ARGS[@]}" \
  --run_suffix "$ANCHOR_SUFFIX"
CURRENT_STEP=$((CURRENT_STEP + 1))

WARMUP_DIR="$(latest_dir "results/MetaNATH_Phase3_*_${ANCHOR_SUFFIX}")"
WARMUP_CKPT="$WARMUP_DIR/last_checkpoint.pt"

run_logged before_eval "$PYTHON_BIN" -u scripts/evaluate_checkpoint.py \
  --config "$CONSERVATIVE_CONFIG" \
  --checkpoint "$WARMUP_CKPT" \
  --max_tasks "$MAX_TASKS" \
  "${QUIET_ARGS[@]}" \
  --run_suffix "$BEFORE_SUFFIX"
CURRENT_STEP=$((CURRENT_STEP + 1))

BEFORE_DIR="$(latest_dir "results/MetaNATH_Eval_*_${BEFORE_SUFFIX}")"

run_logged phase3_conservative "$PYTHON_BIN" -u scripts/run_phase3_consolidation.py \
  --config "$CONSERVATIVE_CONFIG" \
  --checkpoint "$WARMUP_CKPT" \
  --run_suffix "$CANDIDATE_SUFFIX"
CURRENT_STEP=$((CURRENT_STEP + 1))

PHASE3_DIR="$(latest_dir "results/MetaNATH_Phase3_*_${CANDIDATE_SUFFIX}")"
PHASE3_CKPT="$PHASE3_DIR/last_checkpoint.pt"

run_logged after_eval "$PYTHON_BIN" -u scripts/evaluate_checkpoint.py \
  --config "$CONSERVATIVE_CONFIG" \
  --checkpoint "$PHASE3_CKPT" \
  --max_tasks "$MAX_TASKS" \
  "${QUIET_ARGS[@]}" \
  --run_suffix "$AFTER_SUFFIX"
CURRENT_STEP=$((CURRENT_STEP + 1))

AFTER_DIR="$(latest_dir "results/MetaNATH_Eval_*_${AFTER_SUFFIX}")"
ACCEPTANCE_REPORT="$PHASE3_DIR/acceptance_report.json"

run_logged acceptance "$PYTHON_BIN" -u scripts/phase3_acceptance.py \
  --config "$CONSERVATIVE_CONFIG" \
  --before "$BEFORE_DIR/checkpoint_eval_summary.json" \
  --after "$AFTER_DIR/checkpoint_eval_summary.json" \
  --output "$ACCEPTANCE_REPORT"
CURRENT_STEP=$((CURRENT_STEP + 1))

if [[ "$RUN_SCORE_COMPARE" == "1" ]]; then
  run_logged score_compare "$PYTHON_BIN" -u scripts/compare_checkpoint_scores.py \
    --config "$CONSERVATIVE_CONFIG" \
    --before "$WARMUP_CKPT" \
    --after "$PHASE3_CKPT" \
    --max_tasks "$MAX_TASKS" \
    --task_ids "1,5,6" \
    --output "$PHASE3_DIR/score_compare_cable_hazelnut_leather.json"
  CURRENT_STEP=$((CURRENT_STEP + 1))
fi

echo
echo "Workflow completed."
echo "warmup_dir=$WARMUP_DIR"
echo "before_eval=$BEFORE_DIR/checkpoint_eval_summary.json"
echo "phase3_dir=$PHASE3_DIR"
echo "after_eval=$AFTER_DIR/checkpoint_eval_summary.json"
echo "acceptance_report=$ACCEPTANCE_REPORT"
