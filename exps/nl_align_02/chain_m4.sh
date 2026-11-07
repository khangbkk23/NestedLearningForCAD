#!/usr/bin/env bash
# NL-ALIGN-02-MVTec M4: prereg (final rule) -> predictions stages 10-15 -> seal (+ID parity vs M1) -> evaluate -> decide.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTHONPATH=$PWD
R=results/hope_cad/nl_align_02_mvtec
x() { echo "== $(date -u +%FT%TZ) $*"; $PY -m exps.nl_align_02.m4 "$@"; }
[ -f $R/M4_PREREG.json ] || x prereg
for s in 2711 2712 2713; do x predict --seed $s; done
[ -f $R/M4_PRED_SEAL.json ] || x seal
[ -f $R/M4_METRICS.json ] || CUDA_VISIBLE_DEVICES= x evaluate
[ -f $R/FINAL_VERDICT.json ] || x decide
echo "== $(date -u +%FT%TZ) M4 done"
