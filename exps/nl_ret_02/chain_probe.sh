#!/usr/bin/env bash
# NL-RET-02 G1: preregistration lock, then no-update probes for every boundary/seed/arm (one GPU process at a time).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
R=results/hope_cad/nl_ret_02
[ -f $R/AUDIT_LOCK.json ] || { echo "== $(date -u +%FT%TZ) prereg"; CUDA_VISIBLE_DEVICES= $PY -m exps.nl_ret_02.prereg; }
for s in 2711 2712 2713; do
  echo "== $(date -u +%FT%TZ) probe seed $s"; $PY -m exps.nl_ret_02.probe run --seed $s
done
echo "== $(date -u +%FT%TZ) probes done"
