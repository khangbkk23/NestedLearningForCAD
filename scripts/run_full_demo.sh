#!/usr/bin/env bash
set -Eeuo pipefail

# Full Meta-NATH CAD v1 demo workflow.
#
# Tiers:
#   1. Main reportable Phase 3 conservative benchmark.
#   2. Mechanism smoke for TITANS, CADIC, NSP2, CBP, and Subspace Recycling.
#   3. Experimental NSP2/CBP benchmark with the same before/after acceptance gate.
#
# DINOv3 is intentionally out of scope for this v1 demo.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

STEP_TIMEOUT_SECONDS="${STEP_TIMEOUT_SECONDS:-7200}"
LOCAL_FILES_ONLY_AFTER_WARMUP="${LOCAL_FILES_ONLY_AFTER_WARMUP:-1}"

RUN_MAIN="${RUN_MAIN:-1}"
RUN_MECHANISM="${RUN_MECHANISM:-1}"
RUN_EXPERIMENTAL="${RUN_EXPERIMENTAL:-1}"

MAIN_MAX_TASKS="${MAIN_MAX_TASKS:-${MAX_TASKS:-8}}"
EXPERIMENTAL_MAX_TASKS="${EXPERIMENTAL_MAX_TASKS:-$MAIN_MAX_TASKS}"

MAIN_CONFIG="${MAIN_CONFIG:-conf/full_demo.yaml}"
CONSERVATIVE_CONFIG="${CONSERVATIVE_CONFIG:-$MAIN_CONFIG}"
EXPERIMENTAL_CONFIG="${EXPERIMENTAL_CONFIG:-conf/experimental_nsp2_cbp.yaml}"
EXPERIMENTAL_PROFILE="${EXPERIMENTAL_PROFILE:-nsp2_cbp}"
if [[ "$EXPERIMENTAL_CONFIG" == *"mvtec_max_power"* && "${EXPERIMENTAL_PROFILE:-}" == "nsp2_cbp" ]]; then
  EXPERIMENTAL_PROFILE="max_power"
fi

REUSE_MAIN_ANCHOR_FOR_EXPERIMENTAL="${REUSE_MAIN_ANCHOR_FOR_EXPERIMENTAL:-1}"
REQUIRE_MAIN_ACCEPTED="${REQUIRE_MAIN_ACCEPTED:-1}"
REQUIRE_EXPERIMENTAL_ACCEPTED="${REQUIRE_EXPERIMENTAL_ACCEPTED:-0}"
PROGRESS="${PROGRESS:-1}"

QUIET_ARGS=()
if [[ "$PROGRESS" != "1" ]]; then
  QUIET_ARGS=(--quiet)
fi

for required_config in "$MAIN_CONFIG" "$CONSERVATIVE_CONFIG" "$EXPERIMENTAL_CONFIG"; do
  if [[ ! -f "$required_config" ]]; then
    echo "Required config not found: $required_config" >&2
    echo "This checkout expects the cleaned config names: conf/full_demo.yaml and conf/experimental_nsp2_cbp.yaml." >&2
    exit 2
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/full_demo_${STAMP}}"
FULL_DEMO_DIR="${FULL_DEMO_DIR:-results/MetaNATH_FullDemo_${STAMP}}"
FULL_DEMO_MANIFEST="$FULL_DEMO_DIR/manifest.json"
mkdir -p "$LOG_DIR" "$FULL_DEMO_DIR"

REUSE_EXPERIMENTAL_ANCHOR=0
if [[ "$RUN_MAIN" == "1" && "$RUN_EXPERIMENTAL" == "1" && "$REUSE_MAIN_ANCHOR_FOR_EXPERIMENTAL" == "1" && "$MAIN_MAX_TASKS" == "$EXPERIMENTAL_MAX_TASKS" ]]; then
  REUSE_EXPERIMENTAL_ANCHOR=1
fi

TOTAL_STEPS=2
if [[ "$RUN_MAIN" == "1" ]]; then
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
CURRENT_STEP=1

MAIN_WARMUP_DIR=""
MAIN_WARMUP_CKPT=""
MAIN_BEFORE_DIR=""
MAIN_PHASE3_DIR=""
MAIN_PHASE3_CKPT=""
MAIN_AFTER_DIR=""
MAIN_ACCEPTANCE_REPORT=""

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

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
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
  export FULL_DEMO_MANIFEST STAMP LOG_DIR FULL_DEMO_DIR
  export RUN_MAIN RUN_MECHANISM RUN_EXPERIMENTAL
  export MAIN_MAX_TASKS EXPERIMENTAL_MAX_TASKS
  export MAIN_CONFIG CONSERVATIVE_CONFIG EXPERIMENTAL_CONFIG EXPERIMENTAL_PROFILE
  export REQUIRE_MAIN_ACCEPTED REQUIRE_EXPERIMENTAL_ACCEPTED
  export MAIN_WARMUP_DIR MAIN_WARMUP_CKPT MAIN_BEFORE_DIR MAIN_PHASE3_DIR MAIN_PHASE3_CKPT MAIN_AFTER_DIR MAIN_ACCEPTANCE_REPORT
  export EXP_SOURCE_DIR EXP_SOURCE_CKPT EXP_BEFORE_DIR EXP_PHASE3_DIR EXP_PHASE3_CKPT EXP_AFTER_DIR EXP_ACCEPTANCE_REPORT

  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

def env(name):
    return os.environ.get(name, "")

manifest = {
    "workflow": "Meta-NATH CAD full demo v1",
    "stamp": env("STAMP"),
    "log_dir": env("LOG_DIR"),
    "full_demo_dir": env("FULL_DEMO_DIR"),
    "tiers": {
        "main_reportable": {
            "enabled": env("RUN_MAIN") == "1",
            "max_tasks": int(env("MAIN_MAX_TASKS") or 0),
            "main_config": env("MAIN_CONFIG"),
            "conservative_config": env("CONSERVATIVE_CONFIG"),
            "warmup_dir": env("MAIN_WARMUP_DIR"),
            "warmup_checkpoint": env("MAIN_WARMUP_CKPT"),
            "before_eval_dir": env("MAIN_BEFORE_DIR"),
            "phase3_dir": env("MAIN_PHASE3_DIR"),
            "phase3_checkpoint": env("MAIN_PHASE3_CKPT"),
            "after_eval_dir": env("MAIN_AFTER_DIR"),
            "acceptance_report": env("MAIN_ACCEPTANCE_REPORT"),
            "require_accepted": env("REQUIRE_MAIN_ACCEPTED") == "1",
        },
        "mechanism_smoke": {
            "enabled": env("RUN_MECHANISM") == "1",
            "script": "scripts/diagnostics/mechanism_smoke.py",
        },
        "experimental_nsp2_cbp": {
            "enabled": env("RUN_EXPERIMENTAL") == "1",
            "max_tasks": int(env("EXPERIMENTAL_MAX_TASKS") or 0),
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

out = Path(env("FULL_DEMO_MANIFEST"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"manifest={out}")
PY
}

echo "Meta-NATH CAD full demo v1"
echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
if command -v git >/dev/null 2>&1; then
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
fi
echo "run_main=$RUN_MAIN"
echo "run_mechanism=$RUN_MECHANISM"
echo "run_experimental=$RUN_EXPERIMENTAL"
echo "main_max_tasks=$MAIN_MAX_TASKS"
echo "experimental_max_tasks=$EXPERIMENTAL_MAX_TASKS"
echo "main_config=$MAIN_CONFIG"
echo "conservative_config=$CONSERVATIVE_CONFIG"
echo "experimental_config=$EXPERIMENTAL_CONFIG"
echo "experimental_profile=$EXPERIMENTAL_PROFILE"
echo "reuse_experimental_anchor=$REUSE_EXPERIMENTAL_ANCHOR"
echo "require_main_accepted=$REQUIRE_MAIN_ACCEPTED"
echo "require_experimental_accepted=$REQUIRE_EXPERIMENTAL_ACCEPTED"
echo "step_timeout_seconds=$STEP_TIMEOUT_SECONDS"
echo "hf_hub_disable_xet=$HF_HUB_DISABLE_XET"
echo "metanath_require_hf_backbone=$METANATH_REQUIRE_HF_BACKBONE"
echo "metanath_local_files_only=$METANATH_LOCAL_FILES_ONLY"
echo "logs=$LOG_DIR"
echo "full_demo_dir=$FULL_DEMO_DIR"

run_logged py_compile "$PYTHON_BIN" -u -m py_compile \
  training/run_experiment.py \
  training/consolidation_engine.py \
  training/meta_nath_engine.py \
  "$PIPELINE_DIR/run_phase3_consolidation.py" \
  "$PIPELINE_DIR/evaluate_checkpoint.py" \
  "$PIPELINE_DIR/phase3_acceptance.py" \
  "$PIPELINE_DIR/compare_checkpoint_scores.py" \
  "$DIAGNOSTICS_DIR/mechanism_smoke.py"

run_logged bash_syntax bash -n \
  scripts/run_full_demo.sh \
  scripts/run_server_phase3.sh \
  scripts/workflows/run_server_phase3.sh \
  scripts/run_server_visa.sh \
  scripts/workflows/run_server_visa.sh

if [[ "$RUN_MAIN" == "1" ]]; then
  MAIN_ANCHOR_SUFFIX="full_main_anchor_${MAIN_MAX_TASKS}task_${STAMP}"
  MAIN_BEFORE_SUFFIX="full_main_before_${MAIN_MAX_TASKS}task_${STAMP}"
  MAIN_CANDIDATE_SUFFIX="full_main_phase3_conservative_${MAIN_MAX_TASKS}task_${STAMP}"
  MAIN_AFTER_SUFFIX="full_main_after_${MAIN_MAX_TASKS}task_${STAMP}"

  run_logged main_anchor_warmup "$PYTHON_BIN" -u training/run_experiment.py \
    --config "$MAIN_CONFIG" \
    --max_tasks "$MAIN_MAX_TASKS" \
    --disable_wandb \
    "${QUIET_ARGS[@]}" \
    --run_suffix "$MAIN_ANCHOR_SUFFIX"

  MAIN_WARMUP_DIR="$(latest_dir "results/MetaNATH_Phase3_*_${MAIN_ANCHOR_SUFFIX}")"
  MAIN_WARMUP_CKPT="$MAIN_WARMUP_DIR/last_checkpoint.pt"
  require_file "$MAIN_WARMUP_CKPT"
  export METANATH_LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY_AFTER_WARMUP"
  echo "Using local HuggingFace cache after main warmup: METANATH_LOCAL_FILES_ONLY=$METANATH_LOCAL_FILES_ONLY"

  run_logged main_before_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
    --config "$CONSERVATIVE_CONFIG" \
    --checkpoint "$MAIN_WARMUP_CKPT" \
    --max_tasks "$MAIN_MAX_TASKS" \
    "${QUIET_ARGS[@]}" \
    --run_suffix "$MAIN_BEFORE_SUFFIX"

  MAIN_BEFORE_DIR="$(latest_dir "results/MetaNATH_Eval_*_${MAIN_BEFORE_SUFFIX}")"
  require_file "$MAIN_BEFORE_DIR/checkpoint_eval_summary.json"

  run_logged main_phase3_conservative "$PYTHON_BIN" -u "$PIPELINE_DIR/run_phase3_consolidation.py" \
    --config "$CONSERVATIVE_CONFIG" \
    --checkpoint "$MAIN_WARMUP_CKPT" \
    --run_suffix "$MAIN_CANDIDATE_SUFFIX"

  MAIN_PHASE3_DIR="$(latest_dir "results/MetaNATH_Phase3_*_${MAIN_CANDIDATE_SUFFIX}")"
  MAIN_PHASE3_CKPT="$MAIN_PHASE3_DIR/last_checkpoint.pt"
  require_file "$MAIN_PHASE3_CKPT"

  run_logged main_after_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
    --config "$CONSERVATIVE_CONFIG" \
    --checkpoint "$MAIN_PHASE3_CKPT" \
    --max_tasks "$MAIN_MAX_TASKS" \
    "${QUIET_ARGS[@]}" \
    --run_suffix "$MAIN_AFTER_SUFFIX"

  MAIN_AFTER_DIR="$(latest_dir "results/MetaNATH_Eval_*_${MAIN_AFTER_SUFFIX}")"
  require_file "$MAIN_AFTER_DIR/checkpoint_eval_summary.json"

  MAIN_ACCEPTANCE_REPORT="$MAIN_PHASE3_DIR/acceptance_report.json"
  run_logged main_acceptance "$PYTHON_BIN" -u "$PIPELINE_DIR/phase3_acceptance.py" \
    --config "$CONSERVATIVE_CONFIG" \
    --before "$MAIN_BEFORE_DIR/checkpoint_eval_summary.json" \
    --after "$MAIN_AFTER_DIR/checkpoint_eval_summary.json" \
    --output "$MAIN_ACCEPTANCE_REPORT"

  check_acceptance "main_reportable" "$MAIN_ACCEPTANCE_REPORT" "$REQUIRE_MAIN_ACCEPTED"
fi

if [[ "$RUN_MECHANISM" == "1" ]]; then
  run_logged mechanism_smoke "$PYTHON_BIN" -u "$DIAGNOSTICS_DIR/mechanism_smoke.py"
fi

if [[ "$RUN_EXPERIMENTAL" == "1" ]]; then
  EXP_BEFORE_SUFFIX="full_exp_before_${EXPERIMENTAL_PROFILE}_${EXPERIMENTAL_MAX_TASKS}task_${STAMP}"
  EXP_CANDIDATE_SUFFIX="full_exp_phase3_${EXPERIMENTAL_PROFILE}_${EXPERIMENTAL_MAX_TASKS}task_${STAMP}"
  EXP_AFTER_SUFFIX="full_exp_after_${EXPERIMENTAL_PROFILE}_${EXPERIMENTAL_MAX_TASKS}task_${STAMP}"

  if [[ "$REUSE_EXPERIMENTAL_ANCHOR" == "1" ]]; then
    EXP_SOURCE_DIR="$MAIN_WARMUP_DIR"
    EXP_SOURCE_CKPT="$MAIN_WARMUP_CKPT"
    echo
    echo "Reusing main warmup checkpoint for experimental benchmark:"
    echo "  $EXP_SOURCE_CKPT"
  else
    EXP_ANCHOR_SUFFIX="full_exp_anchor_${EXPERIMENTAL_MAX_TASKS}task_${STAMP}"
    export METANATH_LOCAL_FILES_ONLY="${METANATH_LOCAL_FILES_ONLY_BEFORE_EXP_ANCHOR:-0}"
    run_logged exp_anchor_warmup "$PYTHON_BIN" -u training/run_experiment.py \
      --config "$EXPERIMENTAL_CONFIG" \
      --max_tasks "$EXPERIMENTAL_MAX_TASKS" \
      --disable_wandb \
      "${QUIET_ARGS[@]}" \
      --run_suffix "$EXP_ANCHOR_SUFFIX"
    EXP_SOURCE_DIR="$(latest_dir "results/MetaNATH_Phase3_*_${EXP_ANCHOR_SUFFIX}")"
    EXP_SOURCE_CKPT="$EXP_SOURCE_DIR/last_checkpoint.pt"
    export METANATH_LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY_AFTER_WARMUP"
    echo "Using local HuggingFace cache after experimental warmup: METANATH_LOCAL_FILES_ONLY=$METANATH_LOCAL_FILES_ONLY"
  fi
  require_file "$EXP_SOURCE_CKPT"

  run_logged exp_before_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --checkpoint "$EXP_SOURCE_CKPT" \
    --max_tasks "$EXPERIMENTAL_MAX_TASKS" \
    "${QUIET_ARGS[@]}" \
    --run_suffix "$EXP_BEFORE_SUFFIX"

  EXP_BEFORE_DIR="$(latest_dir "results/MetaNATH_Eval_*_${EXP_BEFORE_SUFFIX}")"
  require_file "$EXP_BEFORE_DIR/checkpoint_eval_summary.json"

  run_logged "exp_phase3_${EXPERIMENTAL_PROFILE}" "$PYTHON_BIN" -u "$PIPELINE_DIR/run_phase3_consolidation.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --checkpoint "$EXP_SOURCE_CKPT" \
    --run_suffix "$EXP_CANDIDATE_SUFFIX"

  EXP_PHASE3_DIR="$(latest_dir "results/MetaNATH_Phase3_*_${EXP_CANDIDATE_SUFFIX}")"
  EXP_PHASE3_CKPT="$EXP_PHASE3_DIR/last_checkpoint.pt"
  require_file "$EXP_PHASE3_CKPT"

  run_logged exp_after_eval "$PYTHON_BIN" -u "$PIPELINE_DIR/evaluate_checkpoint.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --checkpoint "$EXP_PHASE3_CKPT" \
    --max_tasks "$EXPERIMENTAL_MAX_TASKS" \
    "${QUIET_ARGS[@]}" \
    --run_suffix "$EXP_AFTER_SUFFIX"

  EXP_AFTER_DIR="$(latest_dir "results/MetaNATH_Eval_*_${EXP_AFTER_SUFFIX}")"
  require_file "$EXP_AFTER_DIR/checkpoint_eval_summary.json"

  EXP_ACCEPTANCE_REPORT="$EXP_PHASE3_DIR/acceptance_report.json"
  run_logged exp_acceptance "$PYTHON_BIN" -u "$PIPELINE_DIR/phase3_acceptance.py" \
    --config "$EXPERIMENTAL_CONFIG" \
    --before "$EXP_BEFORE_DIR/checkpoint_eval_summary.json" \
    --after "$EXP_AFTER_DIR/checkpoint_eval_summary.json" \
    --output "$EXP_ACCEPTANCE_REPORT"

  check_acceptance "experimental_nsp2_cbp" "$EXP_ACCEPTANCE_REPORT" "$REQUIRE_EXPERIMENTAL_ACCEPTED"
fi

write_manifest

echo
echo "Full demo workflow completed."
echo "manifest=$FULL_DEMO_MANIFEST"
if [[ -n "$MAIN_ACCEPTANCE_REPORT" ]]; then
  echo "main_acceptance_report=$MAIN_ACCEPTANCE_REPORT"
fi
if [[ -n "$EXP_ACCEPTANCE_REPORT" ]]; then
  echo "experimental_acceptance_report=$EXP_ACCEPTANCE_REPORT"
fi
