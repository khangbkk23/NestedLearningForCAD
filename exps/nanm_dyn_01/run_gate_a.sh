#!/usr/bin/env bash
# NANM-DYN-01 Gate A driver (registration). Physical GPU 2 only.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
PY=/media/RAID5Array/backup_home/tindd4/anaconda3/envs/nlcad/bin/python
L=results/hope_cad/nanm_dyn_01/logs; mkdir -p "$L"
for r in "MIX 32" "MIX_REV 32" "VISA 32" "REC 32" "MIX 64" "MIX_REV 64"; do
  set -- $r; echo "$(date -u +%FT%TZ) START A $1 $2"
  "$PY" -B -m exps.nanm_dyn_01.policies run A "$1" "$2" 2711 > "$L/A_$1_$2.log" 2>&1
  echo "$(date -u +%FT%TZ) DONE A $1 $2"
done
"$PY" -B -m exps.nanm_dyn_01.policies analyze_a > "$L/analyze_a.log" 2>&1
