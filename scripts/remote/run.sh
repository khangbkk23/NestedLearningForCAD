#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if ! command -v pixi >/dev/null; then echo 'Install Pixi and run pixi install --locked first.' >&2; exit 1; fi
if [[ $# -lt 1 ]]; then echo 'Usage: bash scripts/remote/run.sh <CLI command and explicit arguments>' >&2; exit 1; fi
for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    exec pixi run --locked cad "$@"
  fi
done
# Parse through the same CLI and output guard before creating the log directory.
LOG_DIR="$(pixi run --locked python -B -c '
import sys
from hope_cad.cli import parser
from hope_cad.paths import CHECKS
from hope_cad.runtime.identity import output_directory
a=parser().parse_args(sys.argv[1:])
w=output_directory(a.run_dir,data_root=getattr(a,"data_root",None),checkpoint=getattr(a,"checkpoint",None)) if hasattr(a,"run_dir") else None
print(w.parent/"logs" if w is not None and w.name=="run" else CHECKS/"invocations"/"logs")
' "$@")"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$1-$$.log"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
pixi run --locked cad "$@" 2>&1 | tee "$LOG_FILE"
