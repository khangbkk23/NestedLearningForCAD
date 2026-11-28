#!/bin/bash
# After RIGEL-ENG-01 check: full test suite, then RIGEL-BUDGET-00 sweep, then the 1.1.1 speed benchmark.
# One GPU job at a time, strictly sequential. Nothing existing is overwritten (new output directories).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
echo "$(date +%FT%T) START pytest"
$PY -B -m pytest -q rigel_cad/tests exps/rigel_q8/test_q8.py > results/rigel_cad/eng_01/pytest_111.out 2>&1
echo "$(date +%FT%T) END pytest rc=$?"
for DS in mpdd mvtec visa; do
  echo "$(date +%FT%T) START sweep $DS"
  PYTHONPATH=. $PY -B -m exps.rigel_budget_00.sweep $DS 2>&1 | grep -v Warn > results/rigel_cad/budget_sweep/sweep_$DS.log
  echo "$(date +%FT%T) END sweep $DS rc=${PIPESTATUS[0]}"
done
PYTHONPATH=. $PY -B -m exps.rigel_budget_00.sweep summary > results/rigel_cad/budget_sweep/summary.out 2>&1
for ST in fp16 int4_triangular; do
  echo "$(date +%FT%T) START bench mvtec $ST"
  $PY -B -m rigel_cad bench --dataset mvtec --data-root data/mvtec --storage $ST --out results/rigel_cad/bench_v111 \
      > results/rigel_cad/bench_v111/bench_mvtec_$ST.log 2>&1
  echo "$(date +%FT%T) END bench mvtec $ST rc=$?"
done
