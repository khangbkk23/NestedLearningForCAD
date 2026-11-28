#!/bin/bash
# RIGEL-ENG-01: bitwise check of the rigel_cad 1.1.1 engineering changes (one GPU job at a time, sequential).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
for ds in mvtec visa; do
  PYTHONPATH=. $PY -B -m exps.rigel_eng_01.check $ds 2>&1 | grep -v Warn > results/rigel_cad/eng_01/check_$ds.log
done
PYTHONPATH=. $PY -B -m exps.rigel_eng_01.check summary > results/rigel_cad/eng_01/check_summary.out 2>&1
