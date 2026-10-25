#!/bin/bash
# NLCAD-03 Phase 1 (LH-REC) chain (detached). Starts only after the NLCAD-02 chain has exited cleanly (which itself
# waited for RH-01b) and no RH-01b / NLCAD-02 process is alive; holds (no launch) if NLCAD-02 failed or held.
# Record-only: no learning, no Stage D. Phase 2 is NOT launched from here.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LHREC_CAP_GIB=4.0
O=results/hope_cad/nlcad; L=$O/logs; S=$L/nlcad03.status; N2=$L/nlcad02.status
mkdir -p $L
st() { echo "$(date -u +%FT%TZ) $1" >> $S; }
step() { st "START $2"; $PY -m $1 > $L/03_$2.log 2>&1 || { st "FAIL $2"; exit 1; }; st "DONE $2"; }

st "WAIT nlcad02"
until grep -q "EXIT\|FAIL\|HOLD" $N2 2>/dev/null; do sleep 300; done
if grep -q "FAIL\|HOLD" $N2; then st "HOLD nlcad02_not_clean (no launch)"; exit 0; fi
while pgrep -f "exps[./](rh01b|nlcad\.(run|run_02|stage_d|analyze_02))" > /dev/null; do sleep 60; done
st "GPU free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
[ -f $L/03_preflight.ok ] || { step "pytest -q exps/nlcad/test_lhrec.py" preflight; touch $L/03_preflight.ok; }
[ -f $O/runs/LHREC_seed2711_from6/STAGE15.json ] || step "exps.nlcad.lhrec train" lhrec
[ -f $O/LHREC_ANALYSIS.json ] || step "exps.nlcad.lhrec_analyze" analyze
st "EXIT"
