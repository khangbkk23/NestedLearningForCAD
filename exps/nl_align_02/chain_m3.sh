#!/usr/bin/env bash
# NL-ALIGN-02-MVTec M3: bank extension -> lock -> runner-equivalence check -> FS_B/CN stages 10-15 (stops on failure).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTHONPATH=$PWD
R=results/hope_cad/nl_align_02_mvtec/m3
x() { echo "== $(date -u +%FT%TZ) $*"; $PY -m exps.nl_align_02.m3 "$@"; }
for s in 2711 2712 2713; do [ -f $R/BANK_EXT_seed$s.json ] || x bank --seed $s; done
[ -f $R/M3_LOCK.json ] || x lock
[ -f $R/RERUN_CHECK.json ] || x rerun
for s in 2711 2712 2713; do for a in FS_B CN; do x train --arm $a --seed $s; done; done
echo "== $(date -u +%FT%TZ) M3 done"
