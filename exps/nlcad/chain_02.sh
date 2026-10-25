#!/bin/bash
# NLCAD-02 execution chain (detached). Touches the GPU only after the RH-01b chain has exited and no RH-01b process is
# alive; holds (no launch) if RH-01b failed. Exactly three training arms, then the combined lock, then Stage D, then the
# diagnostics. Resumable: every step is skipped when its output exists. Stops at the first failure.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NLCAD_CAP_GIB=2.3
O=results/hope_cad/nlcad; L=$O/logs; S=$L/nlcad02.status; RB=results/hope_cad/rh01b/logs/chain.status
mkdir -p $L
st() { echo "$(date -u +%FT%TZ) $1" >> $S; }
step() { st "START $2"; $PY -m $1 > $L/02_$2.log 2>&1 || { st "FAIL $2"; exit 1; }; st "DONE $2"; }
tag() { echo "$1_per_task_seed2711_from6"; }

st "WAIT rh01b"
until grep -q "EXIT\|FAIL" $RB; do sleep 300; done
if grep -q "FAIL" $RB; then st "HOLD rh01b_failed (no launch)"; exit 0; fi
while pgrep -f "exps[./]rh01b" > /dev/null; do sleep 60; done
st "GPU released by rh01b: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

[ -f $L/02_preflight.ok ] || { step "pytest -q exps/nlcad/test_nlcad.py exps/nlcad/test_nlcad_02.py" preflight; touch $L/02_preflight.ok; }
[ -f $O/runs/$(tag N1)/STAGE15.json ] || step "exps.nlcad.run_02 train --arm N1" train_N1
[ -f $O/F1_PROFILE.json ] || step "exps.nlcad.run_02 profiles" profiles
for arm in F1 RND; do [ -f $O/runs/$(tag $arm)/STAGE15.json ] || step "exps.nlcad.run_02 train --arm $arm" train_$arm; done
[ -f $O/LOCK_02.json ] || step "exps.nlcad.run_02 lock" lock
for arm in N1 F1 RND; do
  T=$(tag $arm)
  [ -f $O/stage_d/$T/SCORES_nl.json ] || step "exps.nlcad.stage_d score --run $T --identity-too" score_$arm
  [ -f $O/stage_d/$T/METRICS_nl.json ] || step "exps.nlcad.stage_d metrics --run $T" metrics_$arm
  [ -f $O/stage_d/$T/COMPARE.json ] || step "exps.nlcad.stage_d compare --run $T" compare_$arm
done
[ -f $O/ANALYSIS_02.json ] || step "exps.nlcad.analyze_02 dynamics" dynamics
[ -f $O/SYN_EVAL_02.json ] || step "exps.nlcad.analyze_02 gpu" diag_gpu
[ -f $O/COMPARE_02.json ] || step "exps.nlcad.analyze_02 compare" compare_02
st "EXIT"
