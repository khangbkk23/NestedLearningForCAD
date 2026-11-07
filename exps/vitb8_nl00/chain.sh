#!/usr/bin/env bash
# VITB8-NL-00 Q2-OFF: trainer (auto-resume, 5 attempts) + follower evaluator; summary only after both finish.
set -u
PY=${PY:-/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python}
LOG=results/hope_cad/vitb8_nl00/logs
export PYTHONPATH=.
( for a in 1 2 3 4 5; do $PY -m exps.vitb8_nl00.run follow >> $LOG/eval.log 2>&1 && break; echo "evaluator exited ($a)" >> $LOG/eval.log; sleep 60; done ) &
EP=$!
ok=0
for a in 1 2 3 4 5; do
  if $PY -m exps.vitb8_nl00.run train >> $LOG/train.log 2>&1; then ok=1; echo "train finished" >> $LOG/train.log; break; fi
  echo "trainer exited ($a), resuming" >> $LOG/train.log; sleep 30
done
[ $ok = 1 ] || { echo "BLOCKED" >> $LOG/train.log; pkill -P $EP; kill $EP; exit 1; }
wait $EP
$PY -m exps.vitb8_nl00.run summary >> $LOG/eval.log 2>&1
echo "chain done" >> $LOG/eval.log
