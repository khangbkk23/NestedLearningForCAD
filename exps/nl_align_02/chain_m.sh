#!/usr/bin/env bash
# NL-ALIGN-02-MVTec M0 -> M1 -> M2 chain (stops at the first failure; never retries a scientific step).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTHONPATH=$PWD
R=results/hope_cad/nl_align_02_mvtec
m() { echo "== $(date -u +%FT%TZ) $*"; $PY -m exps.nl_align_02.m "$@"; }
[ -f $R/G0_AUDIT.json ] || { echo "G0 missing"; exit 1; }
[ -f $R/M_PREREG.json ] || m prereg
[ -f $R/M0_PARITY.json ] || m parity
for s in 2711 2712 2713; do m predict --seed $s; done
[ -f $R/PRED_SEAL.json ] || m seal
[ -f $R/M1_METRICS.json ] || CUDA_VISIBLE_DEVICES= m evaluate
[ -f $R/M2_DECISION.json ] || m decide
echo "== $(date -u +%FT%TZ) chain done"
