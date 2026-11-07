#!/usr/bin/env bash
# NL-ALIGN-01 G1 chain (sequential; one GPU process at a time). Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTHONPATH=$PWD
while [ ! -f results/hope_cad/nl_align_01/BANK_seed2713.json ] || pgrep -f "exps.nl_align_01.build_bank" >/dev/null; do sleep 30; done
g() { echo "== $(date -u +%FT%TZ) $*"; $PY -m exps.nl_align_01.g1 "$@"; }
[ -f results/hope_cad/nl_align_01/g1/G1_PREREG.json ] || g prereg
[ -f results/hope_cad/nl_align_01/g1/PARITY.json ] || g parity
for s in 2711 2712 2713; do for a in FS_A FS_B CN; do g train --arm $a --seed $s; done; done
for s in 2712 2713; do g train --arm FP --seed $s; done
for s in 2711 2712 2713; do for a in ID FS_A FS_B CN FP; do g diag --arm $a --seed $s; done; done
[ -f results/hope_cad/nl_align_01/g1/G1_SEAL.json ] || g seal
[ -f results/hope_cad/nl_align_01/g1/G1_DECISION.json ] || g decide
[ -f results/hope_cad/nl_align_01/g1/G1_REAL.json ] || g real
echo "== $(date -u +%FT%TZ) G1 chain done"
