#!/usr/bin/env bash
# VITB8-Q2 full run: trainer with automatic resume from the latest completed stage boundary (max 5 retries),
# plus the stage evaluator following checkpoints in order. Run from the repository root.
set -u
PY=${PY:-/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python}
LOG=results/hope_cad/vitb8_q2/logs
mkdir -p "$LOG"
export PYTHONPATH=.

( for attempt in 1 2 3 4 5; do
    PYTHONPATH=. $PY -m exps.vitb8_q2.evaluate follow >> "$LOG/eval.log" 2>&1 && break
    echo "evaluator exited (attempt $attempt), retrying in 60s" >> "$LOG/eval.log"; sleep 60
  done ) &
EVAL_PID=$!

for attempt in 1 2 3 4 5; do
  if $PY -m exps.vitb8_q2.train train >> "$LOG/train.log" 2>&1; then
    echo "train finished" >> "$LOG/train.log"; break
  fi
  echo "trainer exited (attempt $attempt), resuming in 30s" >> "$LOG/train.log"; sleep 30
done
if ! grep -q "train finished" "$LOG/train.log"; then
  echo "trainer failed 5 times: BLOCKED" >> "$LOG/train.log"; pkill -P $EVAL_PID; kill $EVAL_PID; exit 1
fi

wait $EVAL_PID
$PY -m exps.vitb8_q2.evaluate summary >> "$LOG/eval.log" 2>&1
$PY -m exps.vitb8_q2.evaluate latency >> "$LOG/eval.log" 2>&1
echo "chain done" >> "$LOG/eval.log"
