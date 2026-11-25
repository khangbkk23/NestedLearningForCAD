#!/usr/bin/env bash
# NANM-SYS-FINAL P6 driver (registration §19). Runs only after P5_TIER_A_PASS. Physical GPU 2 only.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
PY=/media/RAID5Array/backup_home/tindd4/anaconda3/envs/nlcad/bin/python
L=results/hope_cad/nanm_sys/final/logs; mkdir -p "$L"
grep -q '"verdict": "P5_TIER_A_PASS"' results/hope_cad/nanm_sys/final/p5/P5_RESULT.json
run() { echo "$(date -u +%FT%TZ) START $*"; "$PY" -B -m exps.nanm_sys.final "$@" >> "$L/p6.log" 2>&1; echo "$(date -u +%FT%TZ) DONE $*"; }
run encoder
grep -q '"all_bitwise": true' results/hope_cad/nanm_sys/final/p6/ENCODER.json
for ds in mpdd visa; do for s in 2711 2712 2713; do run oscore $ds $s; run nscore $ds $s; done; done
python3 - <<'PY'
import json, glob
bad = [f for f in glob.glob('results/hope_cad/nanm_sys/final/p6/NSCORE_*.json') if json.load(open(f))['failures']]
assert not bad, bad
PY
for ds in mpdd visa; do for src in oracle nanm; do run metrics $ds $src; done; done
run mcmp
