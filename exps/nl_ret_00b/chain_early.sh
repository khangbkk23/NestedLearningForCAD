#!/usr/bin/env bash
# NL-RET-00B addendum: lock -> GPU predict (stages 1-6, label-free) -> seal (+acq parity) -> CPU trajectory.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
R=results/hope_cad/nl_ret_00b
x() { echo "== $(date -u +%FT%TZ) $*"; $PY -m exps.nl_ret_00b.early "$@"; }
[ -f $R/EARLY_LOCK.json ] || x lock
x predict
[ -f $R/EARLY_SEAL.json ] || CUDA_VISIBLE_DEVICES= x seal
CUDA_VISIBLE_DEVICES= x traj
echo "== $(date -u +%FT%TZ) early chain done"
