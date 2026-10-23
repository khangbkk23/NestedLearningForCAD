#!/usr/bin/env bash
set -euo pipefail
# Convenience wrapper: Pixi owns Python, packages and editable installation.
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if ! command -v pixi >/dev/null; then
  echo 'Install Pixi first: curl -fsSL https://pixi.sh/install.sh | bash' >&2
  exit 1
fi
if [[ $# -ne 0 ]]; then echo 'No Python path needed; Pixi uses pixi.lock.' >&2; exit 1; fi
pixi install --locked
pixi run --locked env-check
