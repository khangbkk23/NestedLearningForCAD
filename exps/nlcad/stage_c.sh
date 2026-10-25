#!/bin/bash
# NL-CAD Stage C (exploratory, one seed): synthetic bank for seed 2711, then Variant N from the RH-01b R15 stage-6 state.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NLCAD_CAP_GIB=2.3
L=results/hope_cad/nlcad/logs; mkdir -p $L
st() { echo "$(date -u +%FT%TZ) $1" >> $L/stage_c.status; }
if [ ! -f results/hope_cad/nlcad/BANK_seed2711.json ]; then st "START bank"; $PY -m exps.nlcad.bank build --seed 2711 > $L/bank_2711.log 2>&1 || { st "FAIL bank"; exit 1; }; st "DONE bank"; fi
st "START N_per_task_seed2711_from6"
$PY -m exps.nlcad.run --arm N --seed 2711 --start 6 --base per_task > $L/N_per_task_seed2711_from6.log 2>&1 || { st "FAIL N"; exit 1; }
st "DONE N_per_task_seed2711_from6"
st "EXIT"
