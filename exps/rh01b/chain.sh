#!/bin/bash
# RH-01b execution chain (detached). Stops at the first failure; the futility rule decides whether seeds 2712/2713 run.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
O=results/hope_cad/rh01b; L=$O/logs; mkdir -p $L
step() { echo "$(date -u +%FT%TZ) START $2" >> $L/chain.status
         $PY -m $1 > $L/$2.log 2>&1 || { echo "$(date -u +%FT%TZ) FAIL $2" >> $L/chain.status; exit 1; }
         echo "$(date -u +%FT%TZ) DONE $2" >> $L/chain.status; }
[ -f $O/CACHE_CHECK.json ] || step "exps.rh01b.run cache" cache
[ -f $O/PROBE.json ] || step "exps.rh01b.run probe" probe
for arm in R15 J15; do [ -f $O/runs/${arm}_seed2711/TRAIN.json ] || step "exps.rh01b.run train --arm $arm --seed 2711" train_${arm}_2711; done
[ -f $O/scores/SCORES_2711.json ] || step "exps.rh01b.run score --seeds 2711" score_2711
[ -f $O/metrics/METRICS_2711.json ] || step "exps.rh01b.analysis metrics --seeds 2711" metrics_2711
[ -f $O/FUTILITY.json ] || step "exps.rh01b.analysis futility" futility
if grep -q '"futile": true' $O/FUTILITY.json; then step "exps.rh01b.analysis decide" decide; echo "$(date -u +%FT%TZ) EXIT futility" >> $L/chain.status; exit 0; fi
for s in 2712 2713; do for arm in R15 J15; do
  [ -f $O/runs/${arm}_seed$s/TRAIN.json ] || step "exps.rh01b.run train --arm $arm --seed $s" train_${arm}_$s; done; done
[ -f $O/scores/SCORES_2712_2713.json ] || step "exps.rh01b.run score --seeds 2712 2713" score_2712_2713
[ -f $O/metrics/METRICS_2712_2713.json ] || step "exps.rh01b.analysis metrics --seeds 2712 2713" metrics_2712_2713
step "exps.rh01b.analysis decide" decide
echo "$(date -u +%FT%TZ) EXIT complete" >> $L/chain.status
