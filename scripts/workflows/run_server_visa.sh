#!/usr/bin/env bash
set -Eeuo pipefail

# Server entrypoint for the VisA workflow.
#
# Default mode mirrors the MVTec full-demo structure for VisA:
#   1. Conservative Phase 3 benchmark.
#   2. Mechanism smoke.
#   3. Experimental NSP2/CBP or max-power Phase 3 benchmark.
#
# Set RUN_PHASE3=0 RUN_EXPERIMENTAL=0 for a Phase 1-2-only smoke run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PIPELINE_DIR="scripts/pipeline"
DIAGNOSTICS_DIR="scripts/diagnostics"

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
EXPERIMENTAL_CONFIG="${EXPERIMENTAL_CONFIG:-conf/visa_experimental_nsp2_cbp.yaml}"
EXPERIMENTAL_PROFILE="${EXPERIMENTAL_PROFILE:-nsp2_cbp}"
if [[ "$EXPERIMENTAL_CONFIG" == *"visa_max_power"* && "${EXPERIMENTAL_PROFILE:-}" == "nsp2_cbp" ]]; then
  EXPERIMENTAL_PROFILE="max_power"
fi

MAX_TASKS="${MAX_TASKS:-}"
RUN_PHASE3="${RUN_PHASE3:-1}"
RUN_MECHANISM="${RUN_MECHANISM:-1}"
RUN_EXPERIMENTAL="${RUN_EXPERIMENTAL:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-visa_phase12}"
REQUIRE_ACCEPTED="${REQUIRE_ACCEPTED:-0}"
REQUIRE_EXPERIMENTAL_ACCEPTED="${REQUIRE_EXPERIMENTAL_ACCEPTED:-0}"
REUSE_CONSERVATIVE_ANCHOR_FOR_EXPERIMENTAL="${REUSE_CONSERVATIVE_ANCHOR_FOR_EXPERIMENTAL:-1}"
PROGRESS="${PROGRESS:-1}"
STEP_TIMEOUT_SECONDS="${STEP_TIMEOUT_SECONDS:-7200}"
LOCAL_FILES_ONLY_AFTER_WARMUP="${LOCAL_FILES_ONLY_AFTER_WARMUP:-1}"

if [[ ! -d "data/visa" ]]; then
  echo "VisA dataset not found at data/visa." >&2
  echo "Set dataset.root_dir in $CONFIG or mount/copy VisA to data/visa before running." >&2
  exit 2
fi

for required_config in "$CONFIG" "$EXPERIMENTAL_CONFIG"; do
  if [[ ! -f "$required_config" ]]; then
    echo "Required config not found: $required_config" >&2
    exit 2
  fi
done

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
VISA_DEMO_DIR="${VISA_DEMO_DIR:-results/MetaNATH_ViSA_Workflow_${STAMP}}"
VISA_DEMO_MANIFEST="$VISA_DEMO_DIR/manifest.json"
mkdir -p "$LOG_DIR" "$VISA_DEMO_DIR"

REUSE_EXPERIMENTAL_ANCHOR=0
if [[ "$RUN_PHASE3" == "1" && "$RUN_EXPERIMENTAL" == "1" && "$REUSE_CONSERVATIVE_ANCHOR_FOR_EXPERIMENTAL" == "1" ]]; then
  REUSE_EXPERIMENTAL_ANCHOR=1
fi

TOTAL_STEPS=2
if [[ "$RUN_PHASE3" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 5))
fi
if [[ "$RUN_MECHANISM" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "$RUN_EXPERIMENTAL" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 4))
  if [[ "$REUSE_EXPERIMENTAL_ANCHOR" != "1" ]]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
  fi
fi
if [[ "$RUN_PHASE3" != "1" && "$RUN_EXPERIMENTAL" != "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
CURRENT_STEP=1

WARMUP_DIR=""
WARMUP_CKPT=""
BEFORE_DIR=""
PHASE3_DIR=""
PHASE3_CKPT=""
AFTER_DIR=""
ACCEPTANCE_REPORT=""

EXP_SOURCE_DIR=""
EXP_SOURCE_CKPT=""
EXP_BEFORE_DIR=""
EXP_PHASE3_DIR=""
EXP_PHASE3_CKPT=""
EXP_AFTER_DIR=""
EXP_ACCEPTANCE_REPORT=""

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

progress_bar() {
  local current="$1"
  local total="$2"
  local name="$3"
  local width=24
  local filled=$((current * width / total))
  local empty=$((width - filled))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
  local pct=$((current * 100 / total))
  echo "[$bar] $current/$total ${pct}% :: $name"
}

run_logged() {
  local name="$1"
  shift
  local start_ts end_ts duration status
  start_ts="$(date +%s)"
  echo
  progress_bar "$CURRENT_STEP" "$TOTAL_STEPS" "$name"
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
  local label="$1"
  local report="$2"
  local require_accepted="$3"
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
  echo "$label decision=$decision report=$report"
  if [[ "$require_accepted" == "1" && "$decision" != "accepted" ]]; then
    echo "$label did not pass acceptance and REQUIRE_*_ACCEPTED=1." >&2
    exit 1
  fi
}

write_manifest() {
  export VISA_DEMO_MANIFEST STAMP LOG_DIR VISA_DEMO_DIR
  export RUN_PHASE3 RUN_MECHANISM RUN_EXPERIMENTAL
  export MAX_TASKS CONFIG EXPERIMENTAL_CONFIG EXPERIMENTAL_PROFILE
  export REQUIRE_ACCEPTED REQUIRE_EXPERIMENTAL_ACCEPTED
  export WARMUP_DIR WARMUP_CKPT BEFORE_DIR PHASE3_DIR PHASE3_CKPT AFTER_DIR ACCEPTANCE_REPORT
  export EXP_SOURCE_DIR EXP_SOURCE_CKPT EXP_BEFORE_DIR EXP_PHASE3_DIR EXP_PHASE3_CKPT EXP_AFTER_DIR EXP_ACCEPTANCE_REPORT

  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

def env(name):
    return os.environ.get(name, "")

manifest = {
    "workflow": "Meta-NATH VisA full demo",
    "stamp": env("STAMP"),
    "log_dir": env("LOG_DIR"),
    "visa_demo_dir": env("VISA_DEMO_DIR"),
    "tiers": {
        "main_reportable": {
            "enabled": env("RUN_PHASE3") == "1",
            "max_tasks": env("MAX_TASKS") or "all",
            "config": env("CONFIG"),
            "warmup_dir": env("WARMUP_DIR"),
            "warmup_checkpoint": env("WARMUP_CKPT"),
            "before_eval_dir": env("BEFORE_DIR"),
            "phase3_dir": env("PHASE3_DIR"),
            "phase3_checkpoint": env("PHASE3_CKPT"),
            "after_eval_dir": env("AFTER_DIR"),
            "acceptance_report": env("ACCEPTANCE_REPORT"),
            "require_accepted": env("REQUIRE_ACCEPTED") == "1",
        },
        "mechanism_smoke": {
            "enabled": env("RUN_MECHANISM") == "1",
            "script": "scripts/diagnostics/mechanism_smoke.py",
        },
        "experimental": {
            "enabled": env("RUN_EXPERIMENTAL") == "1",
            "max_tasks": env("MAX_TASKS") or "all",
            "profile": env("EXPERIMENTAL_PROFILE"),
            "experimental_config": env("EXPERIMENTAL_CONFIG"),
            "source_dir": env("EXP_SOURCE_DIR"),
            "source_checkpoint": env("EXP_SOURCE_CKPT"),
            "before_eval_dir": env("EXP_BEFORE_DIR"),
            "phase3_dir": env("EXP_PHASE3_DIR"),
            "phase3_checkpoint": env("EXP_PHASE3_CKPT"),
            "after_eval_dir": env("EXP_AFTER_DIR"),
            "acceptance_report": env("EXP_ACCEPTANCE_REPORT"),
            "require_accepted": env("REQUIRE_EXPERIMENTAL_ACCEPTED") == "1",
        },
    },
}

out = Path(env("VISA_DEMO_MANIFEST"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"manifest={out}")
PY
}

echo "Meta-NATH VisA workflow"
echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
if command -v git >/dev/null 2>&1; then
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
fi
echo "config=$CONFIG"
echo "experimental_config=$EXPERIMENTAL_CONFIG"
echo "experimental_profile=$EXPERIMENTAL_PROFILE"
echo "max_tasks=${MAX_TASKS:-all}"
echo "run_phase3=$RUN_PHASE3"
echo "run_mechanism=$RUN_MECHANISM"
echo "run_experimental=$RUN_EXPERIMENTAL"
echo "reuse_experimental_anchor=$REUSE_EXPERIMENTAL_ANCHOR"
echo "require_accepted=$REQUIRE_ACCEPTED"
echo "require_experimental_accepted=$REQUIRE_EXPERIMENTAL_ACCEPTED"
echo "progress=$PROGRESS"
echo "step_timeout_seconds=$STEP_TIMEOUT_SECONDS"
echo "hf_hub_disable_xet=$HF_HUB_DISABLE_XET"
echo "metanath_require_hf_backbone=$METANATH_REQUIRE_HF_BACKBONE"
echo "metanath_local_files_only=$METANATH_LOCAL_FILES_ONLY"
echo "logs=$LOG_DIR"
echo "visa_demo_dir=$VISA_DEMO_DIR"

run_logged py_compile "$PYTHON_BIN" -u -m py_compile \
  training/run_experiment.py \
  training/consolidation_engine.py \
  training/meta_nath_engine.py \
  "$PIPELINE_DIR/run_phase3_consolidation.py" \
  "$PIPELINE_DIR/evaluate_checkpoint.py" \
  "$PIPELINE_DIR/phase3_acceptance.py" \
  "$DIAGNOSTICS_DIR/mechanism_smoke.py"

run_logged bash_syntax bash -n scripts/run_server_visa.sh scripts/workflows/run_server_visa.sh

if [[ "$RUN_PHASE3" != "1" && "$RUN_EXPERIMENTAL" != "1" ]]; then
  run_logged visa_phase12 "$PYTHON_BIN" -u training/run_experiment.py \
    --config "$CONFIG" \
    --disable_wandb \
    "${QUIET_ARGS[@]}" \
    "${MAX_TASK_ARGS[@]}" \
    --run_suffix "$RUN_SUFFIX"

  write_manifest
  echo
  echo "VisA Phase 1-2 smoke workflow completed."
  echo "manifest=$VISA_DEMO_MANIFEST"
  exit 0
fi

if [[ "$RUN_PHASE3" == "1" ]]; then
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

  check_acceptance "visa_conservative" "$ACCEPTANCE_REPORT" "$REQUIRE_ACCEPTED"
fi

if [[ "$RUN_MECHANISM" == "1" ]]; then
  run_logged mechanism_smoke "$PYTHON_BIN" -u "$DIAGNOSTICS_DIR/mechanism_smoke.py"
fi

if [[ "$RUN_EXPERIMENTAL" == "1" ]]; then
  EXP_BEFORE_SUFFIX="visa_exp_before_${EXPERIMENTAL_PROFILE}_${TASK_LABEL}_${STAMP}"
  EXP_CANDIDATE_SUFFIX="visa_exp_phase3_${EXPERIMENTAL_PROFILE}_${TASK_LABEL}_${STAMP}"
  EXP_AFTER_SUFFIX="visa_exp_after_${EXPERIMENTAL_PROFILE}_${TASK_LABEL}_${STAMP}"

  if [[ "$REUSE_EXPERIMENTAL_ANCHOR" == "1" ]]; then
    EXP_SOURCE_DIR="$WARMUP_DIR"
    EXP_SOURCE_CKPT="$WARMUP_CKPT"
    echo
    echo "Reusing conservative VisA warmup checkpoint for experimental benchmark:"
    echo "  $EXP_SOURCE_CKPT"
  else
    EXP_ANCHOR_SUFFIX="visa_exp_anchor_${EXPERIMENTAL_PROFILE}_${TASK_LABEL}_${STAMP}"
    export METANATH_LOCAL_FILES_ONLY="${METANATH_LOCAL_FILES_ONLY_BEFORE_EXP_ANCHOR:-0}"
    run_logged visa_exp_anchor_warmup "$PYTHON_BIN" -u training/run_experiment.py \
      --config "$EXPERIMENTAL_CONFIG" \
      --disable_wandb \
      "${QUIET_ARGS[@]}" \
      "${MAX_TASK_ARGS[@]}" \
      --run_suffix "$EXP_ANCHOR_SUFFIX"
    EXP_SOURCE_DIR="$(latest_dir "results/*_${EXP_ANCHOR_SUFFIX}")"
    EXP_SOURCE_CKPT="$EXP_SOURCE_DIR/last_checkpoint.pt"
    export METANATH_LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY_AFTER_WARMUP"
    echo "Using local HuggingFace cache after experimental VisA warmup: METANATH_LOCAL_FILES_ONLY=$METANATH_LOCAL_FILES_ONLY"
  fi
  require_file "$EXP_SOURCE_CKPT"

  run_logged visa_exp_before_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --checkpoint "$EXP_SOURCE_CKPT" \
    "${MAX_TASK_ARGS[@]}" \
    "${QUIET_ARGS[@]}" \
    --run_suffix "$EXP_BEFORE_SUFFIX"

  EXP_BEFORE_DIR="$(latest_dir "results/MetaNATH_Eval_*_${EXP_BEFORE_SUFFIX}")"
  require_file "$EXP_BEFORE_DIR/checkpoint_eval_summary.json"

  run_logged "visa_exp_phase3_${EXPERIMENTAL_PROFILE}" "$PYTHON_BIN" -u "$PIPELINE_DIR/run_phase3_consolidation.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --checkpoint "$EXP_SOURCE_CKPT" \
    --run_suffix "$EXP_CANDIDATE_SUFFIX"

  EXP_PHASE3_DIR="$(latest_dir "results/*_${EXP_CANDIDATE_SUFFIX}")"
  EXP_PHASE3_CKPT="$EXP_PHASE3_DIR/last_checkpoint.pt"
  require_file "$EXP_PHASE3_CKPT"

  run_logged visa_exp_after_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --checkpoint "$EXP_PHASE3_CKPT" \
    "${MAX_TASK_ARGS[@]}" \
    "${QUIET_ARGS[@]}" \
    --run_suffix "$EXP_AFTER_SUFFIX"

  EXP_AFTER_DIR="$(latest_dir "results/MetaNATH_Eval_*_${EXP_AFTER_SUFFIX}")"
  require_file "$EXP_AFTER_DIR/checkpoint_eval_summary.json"

  EXP_ACCEPTANCE_REPORT="$EXP_PHASE3_DIR/acceptance_report.json"
  run_logged visa_exp_acceptance "$PYTHON_BIN" -u "$PIPELINE_DIR/phase3_acceptance.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --before "$EXP_BEFORE_DIR/checkpoint_eval_summary.json" \
    --after "$EXP_AFTER_DIR/checkpoint_eval_summary.json" \
    --output "$EXP_ACCEPTANCE_REPORT"

  check_acceptance "visa_experimental_${EXPERIMENTAL_PROFILE}" "$EXP_ACCEPTANCE_REPORT" "$REQUIRE_EXPERIMENTAL_ACCEPTED"
fi

write_manifest

echo
echo "VisA workflow completed."
echo "manifest=$VISA_DEMO_MANIFEST"
if [[ -n "$ACCEPTANCE_REPORT" ]]; then
  echo "acceptance_report=$ACCEPTANCE_REPORT"
fi
if [[ -n "$EXP_ACCEPTANCE_REPORT" ]]; then
  echo "experimental_acceptance_report=$EXP_ACCEPTANCE_REPORT"
fi
