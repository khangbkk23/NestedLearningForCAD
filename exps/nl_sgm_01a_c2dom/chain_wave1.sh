#!/bin/bash
# Wave 1: codebook fits (CPU) | FP32 controls (CPU) | PQ streams B4, B2 (certified GPU FPS), B1 (CPU) — PQ streams
# wait for each category's codebook marker (causal pipeline).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=6
L=results/hope_cad/nl_sgm_01a_c2_dom_sweep/logs; mkdir -p $L
$PY -m exps.nl_sgm_01a_c2dom.stream fit > $L/fit.stdout 2>&1 &
( for b in B1 B2 B4; do $PY -m exps.nl_sgm_01a_c2dom.stream run SEL_FP32_$b || exit 1; done ) > $L/fp32_chain.stdout 2>&1 &
$PY -m exps.nl_sgm_01a_c2dom.stream run DOM_PQ_B4 --gpu > $L/pq_B4.stdout 2>&1 &
$PY -m exps.nl_sgm_01a_c2dom.stream run DOM_PQ_B2 --gpu > $L/pq_B2.stdout 2>&1 &
wait
