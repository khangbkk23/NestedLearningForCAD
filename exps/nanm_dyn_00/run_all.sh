#!/usr/bin/env bash
# NANM-DYN-00 driver (registration §5). Physical GPU 2 only.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
PY=/media/RAID5Array/backup_home/tindd4/anaconda3/envs/nlcad/bin/python
L=results/hope_cad/nanm_dyn_00/logs; mkdir -p "$L"
for r in "MIX 32" "MIX_REV 32" "MIX 64" "VISA 32" "VISA 64" "VISA_REV 32" "VISA_REV 64" "MIX_REV 64" "REC 32" "REC 64"; do
  set -- $r; echo "$(date -u +%FT%TZ) START $1 $2"
  "$PY" -B -m exps.nanm_dyn_00.stress run "$1" "$2" > "$L/run_$1_$2.log" 2>&1
  echo "$(date -u +%FT%TZ) DONE $1 $2"
done
"$PY" -B -m exps.nanm_dyn_00.stress analyze > "$L/analyze.log" 2>&1
