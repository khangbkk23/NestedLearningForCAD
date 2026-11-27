#!/bin/bash
# RIGEL-Q4T-INT verification (one GPU job, datasets sequential).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
for ds in mpdd mvtec visa; do
  $PY -B -m exps.rigel_q4t_int.verify stream $ds || exit 1
done
$PY -B -m exps.rigel_q4t_int.verify summary
