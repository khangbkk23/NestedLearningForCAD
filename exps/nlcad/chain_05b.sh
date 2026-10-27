#!/bin/bash
# NLCAD-05B Stage C/D chain for the single RET1 learned arm. NOT started automatically: launch only after the user's
# explicit approval, with nothing else on the GPU. Order: preflight (CPU suites) -> train RET1 -> lock (before any
# score) -> Stage D score/metrics/compare -> dynamics -> historical comparison. Resumable; stops at the first failure.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NLCAD_CAP_GIB=2.6
O=results/hope_cad/nlcad; L=$O/logs; S=$L/nlcad05b.status; T=RET1_per_task_seed2711_from6
mkdir -p $L
st() { echo "$(date -u +%FT%TZ) $1" >> $S; }
step() { st "START $2"; $PY -m $1 > $L/05b_$2.log 2>&1 || { st "FAIL $2"; exit 1; }; st "DONE $2"; }

[ -f $O/PREREG_05B.json ] || { st "HOLD no PREREG_05B (no launch)"; exit 0; }
if pgrep -f "exps[./](rh01b|nlcad\.(run|run_02|run_04|lhrec|retention_credit|run_05b train))" > /dev/null; then st "HOLD GPU busy (no launch)"; exit 0; fi
st "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
[ -f $L/05b_preflight.ok ] || { CUDA_VISIBLE_DEVICES= step "pytest -q exps/nlcad/test_nlcad.py exps/nlcad/test_nlcad_02.py exps/nlcad/test_nlcad_04.py exps/nlcad/test_retention_credit.py exps/nlcad/test_nlcad_05b.py" preflight; touch $L/05b_preflight.ok; }
[ -f $O/runs/$T/STAGE15.json ] || step "exps.nlcad.run_05b train" train_RET1
[ -f $O/LOCK_05B.json ] || step "exps.nlcad.run_05b lock" lock
[ -f $O/stage_d/$T/SCORES_nl.json ] || step "exps.nlcad.stage_d score --run $T --identity-too" score_RET1
[ -f $O/stage_d/$T/METRICS_nl.json ] || step "exps.nlcad.stage_d metrics --run $T" metrics_RET1
[ -f $O/stage_d/$T/COMPARE.json ] || step "exps.nlcad.stage_d compare --run $T" compare_RET1
[ -f $O/ANALYSIS_05B.json ] || step "exps.nlcad.run_05b dynamics" dynamics
[ -f $O/COMPARE_05B.json ] || step "exps.nlcad.run_05b compare" compare_05b
st "EXIT"
