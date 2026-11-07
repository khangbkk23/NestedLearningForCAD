#!/usr/bin/env bash
# NL-RET-00B: lock -> parity (stop on failure) -> run -> transitions. CPU only; never reruns a consumed step.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
R=results/hope_cad/nl_ret_00b
x() { echo "== $(date -u +%FT%TZ) $*"; $PY -m exps.nl_ret_00b.decomp "$@"; }
[ -f $R/AUDIT_LOCK.json ] || x lock
[ -f $R/PARITY.json ] || x parity
x run
[ -f $R/TRANSITIONS.jsonl ] || x transitions
echo "== $(date -u +%FT%TZ) chain done"
