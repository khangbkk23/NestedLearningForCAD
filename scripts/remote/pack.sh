#!/usr/bin/env bash
# Run after a phase/invocation AND its review if acceptance must travel with it.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
exec pixi run --locked cad handoff-pack "$@"
