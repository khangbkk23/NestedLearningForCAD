#!/usr/bin/env bash
# NANM-SYS-FINAL P5 driver (registration §19). Physical GPU 2 only. Stops at the first error.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
PY=/media/RAID5Array/backup_home/tindd4/anaconda3/envs/nlcad/bin/python
L=results/hope_cad/nanm_sys/final/logs; mkdir -p "$L"
run() { echo "$(date -u +%FT%TZ) START $*"; "$PY" -B -m exps.nanm_sys.final "$@" >> "$L/p5.log" 2>&1; echo "$(date -u +%FT%TZ) DONE $*"; }
for s in 2712 2713; do run p5a mpdd G3 $s; done
for a in G3 G3@C; do for s in 2711 2712 2713; do run p5a visa $a $s; done; done
for s in 2711 2712 2713; do run p5b mpdd G3 $s 0 2; run p5b mpdd G3 $s 2 4; run p5b mpdd G3 $s 4 6; done
for c in "G3 2711" "G3 2712" "G3@C 2711"; do set -- $c; run p5b visa $1 $2 0 2; run p5b visa $1 $2 2 10; run p5b visa $1 $2 10 12; done
run p5cmp
