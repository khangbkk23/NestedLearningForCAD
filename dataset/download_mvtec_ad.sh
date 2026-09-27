#!/usr/bin/env bash
set -euo pipefail

# This script intentionally does not download anything.  Place the archive in
# ./data or pass its path as the only argument; the prepared dataset always
# ends up in <repository>/data/mvtec.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
TARGET_DIR="${DATA_DIR}/mvtec"
ARCHIVE_PATH="${1:-}"

CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut
  pill screw tile toothbrush transistor wood zipper
)

usage() {
  cat <<EOF
Usage:
  bash dataset/download_mvtec_ad.sh [ARCHIVE]

ARCHIVE may be a .tar.xz, .tar.gz, .tar, or .zip file. If omitted, exactly one
MVTec archive is selected from ${DATA_DIR}. The archive is never deleted.
Prepared output: ${TARGET_DIR}
EOF
}

if [[ "${ARCHIVE_PATH}" == "-h" || "${ARCHIVE_PATH}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${DATA_DIR}"

if [[ -z "${ARCHIVE_PATH}" ]]; then
  mapfile -t candidates < <(find "${DATA_DIR}" -maxdepth 1 -type f \( \
    -iname '*.tar.xz' -o -iname '*.txz' -o -iname '*.tar.gz' -o \
    -iname '*.tgz' -o -iname '*.tar' -o -iname '*.zip' \
  \) -print | sort)
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "[ERROR] Expected exactly one downloaded archive in ${DATA_DIR}; found ${#candidates[@]}." >&2
    usage >&2
    exit 2
  fi
  ARCHIVE_PATH="${candidates[0]}"
fi

if [[ ! -f "${ARCHIVE_PATH}" || ! -s "${ARCHIVE_PATH}" ]]; then
  echo "[ERROR] Archive does not exist or is empty: ${ARCHIVE_PATH}" >&2
  exit 2
fi
ARCHIVE_PATH="$(cd "$(dirname "${ARCHIVE_PATH}")" && pwd)/$(basename "${ARCHIVE_PATH}")"

validate_tree() {
  local root="$1"
  local category
  for category in "${CATEGORIES[@]}"; do
    [[ -d "${root}/${category}/train/good" ]] || return 1
    [[ -d "${root}/${category}/test" ]] || return 1
    [[ -d "${root}/${category}/ground_truth" ]] || return 1
  done
}

count_images() {
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print | wc -l | tr -d ' '
}

if [[ -d "${TARGET_DIR}" ]] && validate_tree "${TARGET_DIR}"; then
  echo "[OK] MVTec is already prepared at ${TARGET_DIR}; archive was not modified."
  exit 0
fi

# Keep an incomplete existing destination intact rather than silently mixing
# files from two archives.
if [[ -e "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[ERROR] Destination exists but is incomplete: ${TARGET_DIR}" >&2
  echo "        Move it aside before preparing a new archive." >&2
  exit 3
fi

STAGE_DIR="$(mktemp -d "${DATA_DIR}/.mvtec_extract.XXXXXX")"
cleanup() { rm -rf "${STAGE_DIR}"; }
trap cleanup EXIT

export ARCHIVE_PATH STAGE_DIR
python3 - <<'PY'
import os
import stat
import tarfile
import zipfile
from pathlib import Path

archive = Path(os.environ["ARCHIVE_PATH"])
stage = Path(os.environ["STAGE_DIR"]).resolve()

def safe_target(name: str) -> Path:
    path = (stage / name).resolve()
    if os.path.commonpath((str(stage), str(path))) != str(stage):
        raise RuntimeError(f"unsafe archive path: {name}")
    return path

if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            safe_target(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise RuntimeError(f"symlink in archive is not allowed: {info.filename}")
        zf.extractall(stage)
elif tarfile.is_tarfile(archive):
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            safe_target(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"link/device in archive is not allowed: {member.name}")
        tf.extractall(stage, members=members)
else:
    raise RuntimeError(f"unsupported archive format: {archive}")
PY

SOURCE_ROOT=""
for candidate in "${STAGE_DIR}"/*; do
  [[ -d "${candidate}" ]] || continue
  if validate_tree "${candidate}"; then
    SOURCE_ROOT="${candidate}"
    break
  fi
done
if [[ -z "${SOURCE_ROOT}" ]] && validate_tree "${STAGE_DIR}"; then
  SOURCE_ROOT="${STAGE_DIR}"
fi

if [[ -z "${SOURCE_ROOT}" ]]; then
  echo "[ERROR] Archive extracted, but no canonical 15-category MVTec tree was found." >&2
  echo "        Expected <root>/<category>/{train/good,test,ground_truth}." >&2
  exit 4
fi

mkdir -p "${TARGET_DIR}"
for category in "${CATEGORIES[@]}"; do
  mv "${SOURCE_ROOT}/${category}" "${TARGET_DIR}/${category}"
done

if ! validate_tree "${TARGET_DIR}"; then
  echo "[ERROR] Prepared tree failed post-move validation: ${TARGET_DIR}" >&2
  exit 5
fi

failed=0
total_train=0
total_test=0
printf '%-14s %10s %10s %12s\n' category train_good test_imgs gt_masks
printf '%-14s %10s %10s %12s\n' '--------------' '----------' '----------' '------------'
for category in "${CATEGORIES[@]}"; do
  root="${TARGET_DIR}/${category}"
  train_count="$(count_images "${root}/train/good")"
  test_count="$(count_images "${root}/test")"
  gt_count="$(count_images "${root}/ground_truth")"
  printf '%-14s %10d %10d %12d\n' "${category}" "${train_count}" "${test_count}" "${gt_count}"
  total_train=$((total_train + train_count))
  total_test=$((total_test + test_count))
  [[ "${train_count}" -gt 0 && "${test_count}" -gt 0 ]] || failed=1
done

echo
echo "Total train/good images: ${total_train}"
echo "Total test images      : ${total_test}"
if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] One or more categories contain no train or test images." >&2
  exit 6
fi

echo "[OK] MVTec AD benchmark tree is ready at ${TARGET_DIR}"
echo "[INFO] Archive retained at ${ARCHIVE_PATH}"
echo "[INFO] Benchmark root: ${TARGET_DIR}"
