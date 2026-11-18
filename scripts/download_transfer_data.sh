#!/usr/bin/env bash
set -euo pipefail

# Transfer helper: use the existing dataset preparer without its HEAD-size check.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARCHIVE_PATH="${1:-}"
DOWNLOAD_URL="https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/c67a6eab4c08d65c1990db9a273127039af7f9cc/mvtec_anomaly_detection.tar.xz"
EXPECTED_SHA256="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"

if [[ "${ARCHIVE_PATH}" == "--help" || "${ARCHIVE_PATH}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/download_transfer_data.sh [LOCAL_ARCHIVE]

Without an argument: download the pinned MVTec AD mirror (5.26 GB), resume an
interrupted transfer, verify SHA-256, and prepare data/mvtec using the existing
dataset/download_mvtec.py. Requires Python 3 and curl; allow roughly 12 GB free.
With LOCAL_ARCHIVE: prepare that archive without downloading it.
This helper prepares MVTec only. It does not train or evaluate any model.
EOF
  exit 0
fi
if [[ "$#" -gt 1 ]]; then
  echo "Usage: bash scripts/download_transfer_data.sh [LOCAL_ARCHIVE]" >&2
  exit 2
fi

if [[ -z "${ARCHIVE_PATH}" ]]; then
  command -v curl >/dev/null
  mkdir -p "${REPO_ROOT}/data"
  ARCHIVE_PATH="${REPO_ROOT}/data/mvtec_anomaly_detection.tar.xz"
  if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Downloading MVTec AD (resume enabled)..."
    curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 30 \
      --continue-at - --output "${ARCHIVE_PATH}.part" "${DOWNLOAD_URL}"
    CANDIDATE_PATH="${ARCHIVE_PATH}.part"
  else
    CANDIDATE_PATH="${ARCHIVE_PATH}"
  fi
  "${PYTHON_BIN}" - "${CANDIDATE_PATH}" "${EXPECTED_SHA256}" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != sys.argv[2]:
    raise SystemExit(f"SHA-256 mismatch; preserved for inspection: {path}")
print(f"SHA-256 verified: {path.name}")
PY
  if [[ "${CANDIDATE_PATH}" != "${ARCHIVE_PATH}" ]]; then
    mv "${CANDIDATE_PATH}" "${ARCHIVE_PATH}"
  fi
fi

"${PYTHON_BIN}" "${REPO_ROOT}/dataset/download_mvtec.py" "${ARCHIVE_PATH}"
