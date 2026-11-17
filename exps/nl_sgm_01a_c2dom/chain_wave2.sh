#!/bin/bash
# Wave 2 (after B8_DECISION.json): B8 streams (PQ on certified GPU FPS, FP32 canonical CPU) | scorer check + evaluation
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=6
L=results/hope_cad/nl_sgm_01a_c2_dom_sweep/logs
$PY -m exps.nl_sgm_01a_c2dom.stream run DOM_PQ_B8 --gpu > $L/pq_B8.stdout 2>&1 &
( $PY -m exps.nl_sgm_01a_c2dom.stream run SEL_FP32_B8 > $L/fp32_B8.stdout 2>&1
  $PY -m exps.nl_sgm_01a_c2dom.evaluate arm SEL_FP32_B8 > $L/eval_SEL_FP32_B8.stdout 2>&1 ) &
( $PY -m exps.nl_sgm_01a_c2dom.evaluate check > $L/eval_check.stdout 2>&1 || exit 1
  for a in DOM_PQ_B1 SEL_FP32_B1 DOM_PQ_B2 SEL_FP32_B2 DOM_PQ_B4 SEL_FP32_B4; do
    $PY -m exps.nl_sgm_01a_c2dom.evaluate arm $a > $L/eval_$a.stdout 2>&1
  done ) &
wait
