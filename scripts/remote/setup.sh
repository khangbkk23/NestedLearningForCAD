#!/usr/bin/env bash
set -euo pipefail
# Run explicitly after SSH. No rental/account/driver changes.
PYTHON_BIN="${1:-python3.13}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,13), "Supply Python3.13 explicitly"'
if [[ -e .venv ]]; then
  echo '.venv already exists; inspect/reuse explicitly instead of overwriting it.' >&2
  exit 1
fi
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements/remote-cu128.txt
.venv/bin/python -m pip install 'setuptools>=68,<82' wheel
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv/bin/python -m pip check
mkdir -p logs
.venv/bin/python -m pip freeze --all > logs/installed-packages.txt
.venv/bin/python -m torch.utils.collect_env > logs/torch-environment.txt
echo 'Setup finished. Read docs/operations.md, run tests, then explicit preflight on the actual GPU.'
