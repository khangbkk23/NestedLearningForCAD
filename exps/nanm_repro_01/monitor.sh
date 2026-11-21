#!/usr/bin/env bash
# Storage watchdog: every 120 s log D:/K: free + VHDX size; kill the run if D: free < 30 GiB. Exits when pipeline ends.
cd "$(dirname "$0")/../.."
LOG=results/hope_cad/nanm_repro_01/logs/storage_monitor.log; mkdir -p "$(dirname "$LOG")"
RES=$((30*1024*1024*1024))
while pgrep -f "exps/nanm_repro_01/pipeline.sh" >/dev/null; do
  d=$(df -B1 --output=avail /mnt/d | tail -1 | tr -d ' '); k=$(df -B1 --output=avail /mnt/k | tail -1 | tr -d ' ')
  v=$(stat -c %s /mnt/d/wsl/Ubuntu_new/ext4.vhdx 2>/dev/null); r=$(df -B1 --output=used / | tail -1 | tr -d ' ')
  kc=$(du -sb /mnt/k/NLCAD/cache/nanm_repro_01 2>/dev/null | cut -f1)
  echo "$(date -Is) D_free=$d K_free=$k VHDX=$v root_used=$r cache=$kc" >> "$LOG"
  if [[ -n "$d" && "$d" -lt "$RES" ]]; then echo "$(date -Is) D RESERVE BREACH - killing run" >> "$LOG"; pkill -f "exps/nanm_repro_01/pipeline.sh"; pkill -f "exps.nanm_repro_01.run"; exit 2; fi
  sleep 120
done
