#!/bin/bash
# NLCAD-04 Stage C/D chain for the single S1 pilot arm. NOT started automatically: launch only after the user's explicit
# approval, with nothing else on the GPU. Order: preflight (CPU suites + GPU parity test) -> train S1 -> lock (before
# any score) -> Stage D score/metrics/compare -> dynamics -> historical comparison. Resumable; stops at the first failure.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD || exit 1
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NLCAD_CAP_GIB=2.3
O=results/hope_cad/nlcad; L=$O/logs; S=$L/nlcad04.status; T=S1_per_task_seed2711_from6
mkdir -p $L
st() { echo "$(date -u +%FT%TZ) $1" >> $S; }
step() { st "START $2"; $PY -m $1 > $L/04_$2.log 2>&1 || { st "FAIL $2"; exit 1; }; st "DONE $2"; }

[ -f $O/PREREG_04.json ] || { st "HOLD no PREREG_04 (no launch)"; exit 0; }
if pgrep -f "exps[./](rh01b|nlcad\.(run|run_02|lhrec|run_04 train))" > /dev/null; then st "HOLD GPU busy (no launch)"; exit 0; fi
st "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
[ -f $L/04_preflight.ok ] || { step "pytest -q exps/nlcad/test_nlcad.py exps/nlcad/test_nlcad_02.py exps/nlcad/test_nlcad_04.py" preflight; touch $L/04_preflight.ok; }
[ -f $O/runs/$T/STAGE15.json ] || step "exps.nlcad.run_04 train" train_S1
[ -f $O/LOCK_04.json ] || step "exps.nlcad.run_04 lock" lock
[ -f $O/stage_d/$T/SCORES_nl.json ] || step "exps.nlcad.stage_d score --run $T --identity-too" score_S1
[ -f $O/stage_d/$T/METRICS_nl.json ] || step "exps.nlcad.stage_d metrics --run $T" metrics_S1
[ -f $O/stage_d/$T/COMPARE.json ] || step "exps.nlcad.stage_d compare --run $T" compare_S1
[ -f $O/ANALYSIS_04.json ] || step "exps.nlcad.run_04 dynamics" dynamics
[ -f $O/COMPARE_04.json ] || step "exps.nlcad.run_04 compare" compare_04
st "EXIT"
