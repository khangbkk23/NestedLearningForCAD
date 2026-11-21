#!/usr/bin/env bash
# NANM-REPRO-01 gate pipeline: one process per phase, stop on first failure. Usage: pipeline.sh [first_phase]
set -uo pipefail
cd "$(dirname "$0")/../.."
source docs/recovery/recovery_env.sh || exit 1
unset CUDA_VISIBLE_DEVICES                       # recovery_env is CPU-only; this run is authorized GPU
export NANM_REPRO_CACHE=/mnt/k/NLCAD/cache/nanm_repro_01
PY=~/miniconda3/envs/nested_learning/bin/python
LOG=results/hope_cad/nanm_repro_01/logs; mkdir -p "$LOG"
PHASES=(gate1 custody extract_mpdd gate2 extract_visa vlock normal_mpdd normal_visa relock extract_test score gate4 metrics_mpdd metrics_visa compare)
start=${1:-gate1}; go=0
for ph in "${PHASES[@]}"; do
  [[ $ph == "$start" ]] && go=1; [[ $go == 0 ]] && continue
  echo "$(date -Is) START $ph" | tee -a "$LOG/pipeline.log"
  $PY -B -m exps.nanm_repro_01.run "$ph" > "$LOG/$ph.log" 2>&1; rc=$?
  echo "$(date -Is) END $ph rc=$rc" | tee -a "$LOG/pipeline.log"
  if [[ $rc != 0 ]]; then tail -5 "$LOG/$ph.log" | tee -a "$LOG/pipeline.log"; exit $rc; fi
done
echo "$(date -Is) PIPELINE COMPLETE" | tee -a "$LOG/pipeline.log"
