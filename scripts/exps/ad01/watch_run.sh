#!/usr/bin/env bash
# scripts/exps/ad01/watch_run.sh
#
# Lightweight, independent progress monitor for an AD-01 sweep.
#
# It never touches the benchmark: it only reads process status, GPU utilisation,
# file sizes and checkpoint files, and appends one status block to a persistent
# log every interval. This exists because Python block-buffers stdout when a
# worker's output is redirected to a file, so an unbuffered worker log can stay
# empty until the run finishes.
#
# Usage:
#   scripts/exps/ad01/watch_run.sh <namespace_dir> [interval_seconds] [iterations]
#
# Example:
#   scripts/exps/ad01/watch_run.sh results/hope_cad/ad01_phase0/arms_gpu_valid_v2 30 200

set -u

NS="${1:?usage: watch_run.sh <namespace_dir> [interval_seconds] [iterations]}"
INTERVAL="${2:-30}"
ITERATIONS="${3:-1000}"
STATUS="$NS/status.log"
mkdir -p "$NS"

prev_total=0
started=$(date +%s)

for i in $(seq 1 "$ITERATIONS"); do
  ts=$(date +%H:%M:%S)
  now=$(date +%s)
  elapsed=$((now - started))

  mapfile -t pids < <(pgrep -f "hope_cad_ad01_run.py --worker" 2>/dev/null || true)
  nworkers=${#pids[@]}

  total_ticks=0
  worker_detail=""
  for pid in "${pids[@]}"; do
    if [ -r "/proc/$pid/stat" ]; then
      ticks=$(awk '{print $14+$15}' "/proc/$pid/stat" 2>/dev/null || echo 0)
      total_ticks=$((total_ticks + ticks))
      worker_detail="$worker_detail pid=$pid ticks=$ticks"
    fi
  done
  delta=$((total_ticks - prev_total))
  prev_total=$total_ticks

  done_arms=$(find "$NS" -name "seed*_*.json" 2>/dev/null | wc -l)
  done_seeds=$(find "$NS" -name "worker_seeds_*.json" 2>/dev/null | wc -l)
  logs=$(find "$NS" -name "worker.log" -printf '%s ' 2>/dev/null)

  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')

  {
    echo "[$ts] elapsed=${elapsed}s workers=$nworkers arms_done=$done_arms/12 seed_workers_done=$done_seeds"
    echo "        cpu_delta_ticks=${delta}${worker_detail}"
    echo "        worker_log_bytes: ${logs:-none}"
    echo "        gpu: ${gpu:-unavailable}"
    if [ "$nworkers" -eq 0 ]; then
      echo "        state: NO WORKER PROCESSES RUNNING"
    elif [ "$delta" -le 0 ]; then
      echo "        state: WARNING workers alive but consuming no CPU"
    else
      echo "        state: running"
    fi
  } >> "$STATUS"

  if [ "$nworkers" -eq 0 ]; then
    {
      echo "[$ts] all workers finished; arms_done=$done_arms/12"
      if [ -f "$NS/arms_results.json" ]; then
        echo "[$ts] arms_results.json present -> sweep complete"
      else
        echo "[$ts] arms_results.json MISSING -> parent may have failed"
      fi
    } >> "$STATUS"
    break
  fi

  sleep "$INTERVAL"
done
