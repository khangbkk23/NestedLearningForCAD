#!/usr/bin/env bash
# NL-RET-01A Phase 2: GPU inference only (no training). One process at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
for s in 2711 2712 2713; do
  echo "== $(date -u +%FT%TZ) extract $s"; $PY -m exps.nl_ret_01a.extract extract --seed $s
  echo "== $(date -u +%FT%TZ) grads $s"; [ -f results/hope_cad/nl_ret_01a/extract/seed$s/grads.json ] || $PY -m exps.nl_ret_01a.extract grads --seed $s
done
echo "== $(date -u +%FT%TZ) extraction done"
