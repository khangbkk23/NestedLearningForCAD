#!/bin/bash
# NLCAD-06C-ANCHOR chain: ID evaluations (seeds 2712, 2713), the five new runs, then the predeclared analysis.
# Launches only through the gate (preregistered sources, integrity-passing preflight within the budget cap).
# Resumable (finished steps are skipped); stops at the first failure (integrity, numerical or resource).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NLCAD_CAP_GIB=2.6
O=results/hope_cad/nlcad; L=$O/logs; S=$L/nlcad06c.status
mkdir -p $L
st() { echo "$(date -u +%FT%TZ) $1" >> $S; }
step() { st "START $2"; $PY -m $1 > $L/06c_$2.log 2>&1 || { st "FAIL $2"; exit 1; }; st "DONE $2"; }

[ -f $O/PREFLIGHT_06C.json ] || { st "HOLD no PREFLIGHT_06C (no launch)"; exit 0; }
$PY -m exps.nlcad.anchor_06c gate >> $S 2>&1 || { st "HOLD gate closed (no launch)"; exit 0; }
if pgrep -f "exps[./](rh01b|nlcad\.(run|run_02|run_04|run_05b|lhrec|retention_credit|closed_loop_06b))" > /dev/null; then
  st "HOLD GPU busy (no launch)"; exit 0
fi
st "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for SD in 2712 2713; do
  [ -f $O/runs/CL06C_ID_seed$SD/TRAIN.json ] || step "exps.nlcad.anchor_06c eval-id --seed $SD" eval_ID_$SD
done
for R in ANCHOR:2711 REP-05:2712 ANCHOR:2712 REP-05:2713 ANCHOR:2713; do
  A=${R%%:*}; SD=${R##*:}
  [ -f $O/runs/CL06C_${A}_seed${SD}/TRAIN.json ] || step "exps.nlcad.anchor_06c train --arm $A --seed $SD" train_${A}_$SD
done
[ -f $O/DECISION_06C.json ] || step "exps.nlcad.anchor_06c analyze" analyze
st "EXIT"
