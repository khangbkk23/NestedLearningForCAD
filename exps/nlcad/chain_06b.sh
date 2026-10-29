#!/bin/bash
# NLCAD-06B-MECH chain: the seven fixed-policy arms, sequentially, then the predeclared analysis. Runs only when
# PREFLIGHT_06B.json has status PASS (projected <= 2 GPU-h, loop parity, operator checks); `train` enforces it again.
# ID first (bitwise R15 parity at stages 7-9), then the primary comparison set, then the secondary arms.
# Resumable (completed arms are skipped); stops at the first failure (integrity, numerical or resource).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NLCAD_CAP_GIB=2.6
O=results/hope_cad/nlcad; L=$O/logs; S=$L/nlcad06b.status
mkdir -p $L
st() { echo "$(date -u +%FT%TZ) $1" >> $S; }
step() { st "START $2"; $PY -m $1 > $L/06b_$2.log 2>&1 || { st "FAIL $2"; exit 1; }; st "DONE $2"; }

[ -f $O/PREFLIGHT_06B.json ] || { st "HOLD no PREFLIGHT_06B (no launch)"; exit 0; }
$PY -m exps.nlcad.closed_loop_06b gate >> $S 2>&1 || { st "HOLD gate closed (no launch)"; exit 0; }
if pgrep -f "exps[./](rh01b|nlcad\.(run|run_02|run_04|run_05b|lhrec|retention_credit))" > /dev/null; then st "HOLD GPU busy (no launch)"; exit 0; fi
st "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for A in ID REP-05 NORM-05 RAND-05 REP-10 ACQ-05 ORACLE-05; do
  [ -f $O/runs/CL06B_${A}_seed2711/TRAIN.json ] || step "exps.nlcad.closed_loop_06b train --arm $A" train_$A
done
[ -f $O/DECISION_06B.json ] || step "exps.nlcad.closed_loop_06b analyze" analyze
st "EXIT"
