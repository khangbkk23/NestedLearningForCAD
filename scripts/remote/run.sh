#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [[ ! -x .venv/bin/python ]]; then echo 'Run remote/setup.sh with Python3.13 first.' >&2; exit 1; fi
if [[ $# -lt 1 ]]; then echo 'Usage: bash scripts/remote/run.sh <CLI command and explicit arguments>' >&2; exit 1; fi
mkdir -p logs
LOG_FILE="logs/$(date -u +%Y%m%dT%H%M%SZ)-$1-$$.log"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -B -m hope_cad "$@" 2>&1 | tee "$LOG_FILE"
