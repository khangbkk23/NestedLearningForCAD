#!/usr/bin/env bash
# NANM-DYN-01 Gate B driver (runs only because GATE_A.json decision = RUN_GATE_B). Physical GPU 2; GPU 0 forbidden.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
PY=/media/RAID5Array/backup_home/tindd4/anaconda3/envs/nlcad/bin/python
L=results/hope_cad/nanm_dyn_01/logs; mkdir -p "$L"
step() { echo "$(date -u +%FT%TZ) START $*"; "$PY" -B -m exps.nanm_dyn_01.policies "$@" > "$L/$(echo "$*" | tr ' ' '_').log" 2>&1; echo "$(date -u +%FT%TZ) DONE $*"; }
step run B MIX 32 2711
step run B MIX_REV 32 2711
step prep_seed 2712
step prep_seed 2713
for sd in 2712 2713; do step run B MIX 32 $sd; step run B MIX_REV 32 $sd; done
step run B MIX 64 2711
step run B MIX_REV 64 2711
step analyze_b
