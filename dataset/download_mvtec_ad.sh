#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-${HOME}/datasets/mvtec_ad}"
ARCHIVE_NAME="mvtec_anomaly_detection.tar.xz"
ARCHIVE_PATH="${TARGET_DIR}/${ARCHIVE_NAME}"

DEFAULT_URL="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
MVTEC_AD_URL="${MVTEC_AD_URL:-${DEFAULT_URL}}"

CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut
  pill screw tile toothbrush transistor wood zipper
)

echo "============================================================"
echo "MVTec AD downloader"
echo "Target : ${TARGET_DIR}"
echo "URL    : ${MVTEC_AD_URL}"
echo "============================================================"
echo
echo "License reminder: MVTec AD is CC BY-NC-SA 4.0 (non-commercial)."
echo "Official page: https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
echo

mkdir -p "${TARGET_DIR}"

if [[ ! -s "${ARCHIVE_PATH}" ]]; then
  echo "[INFO] Downloading archive..."
  if command -v wget >/dev/null 2>&1; then
    wget --continue --show-progress -O "${ARCHIVE_PATH}" "${MVTEC_AD_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl --fail --location --continue-at - -o "${ARCHIVE_PATH}" "${MVTEC_AD_URL}"
  else
    echo "[ERROR] Install wget or curl first."
    exit 1
  fi
else
  echo "[INFO] Archive already exists: ${ARCHIVE_PATH}"
fi

echo "[INFO] Validating archive..."
if ! tar -tJf "${ARCHIVE_PATH}" >/dev/null 2>&1; then
  echo "[ERROR] Invalid tar.xz archive."
  echo "MVTec may have changed its download endpoint."
  echo "Obtain the current link from the official page and retry with:"
  echo "  MVTEC_AD_URL='<CURRENT_URL>' bash $0 '${TARGET_DIR}'"
  exit 2
fi

missing=0
for c in "${CATEGORIES[@]}"; do
  [[ -d "${TARGET_DIR}/${c}" ]] || { missing=1; break; }
done

if [[ "${missing}" -eq 1 ]]; then
  echo "[INFO] Extracting..."
  tar -xJf "${ARCHIVE_PATH}" -C "${TARGET_DIR}"
fi

for outer in \
  "${TARGET_DIR}/mvtec_anomaly_detection" \
  "${TARGET_DIR}/mvtec_ad" \
  "${TARGET_DIR}/MVTec_AD"; do
  if [[ -d "${outer}/bottle" ]]; then
    echo "[INFO] Normalizing nested archive layout..."
    for c in "${CATEGORIES[@]}"; do
      if [[ -d "${outer}/${c}" && ! -e "${TARGET_DIR}/${c}" ]]; then
        mv "${outer}/${c}" "${TARGET_DIR}/${c}"
      fi
    done
    rmdir "${outer}" 2>/dev/null || true
    break
  fi
done

echo
echo "[INFO] Validating canonical 15-category structure..."
failed=0
total_train=0
total_test=0

printf "%-14s %10s %10s %12s\n" "category" "train_good" "test_imgs" "gt_masks"
printf "%-14s %10s %10s %12s\n" "--------------" "----------" "----------" "------------"

for c in "${CATEGORIES[@]}"; do
  root="${TARGET_DIR}/${c}"
  train_dir="${root}/train/good"
  test_dir="${root}/test"
  gt_dir="${root}/ground_truth"

  if [[ ! -d "${train_dir}" || ! -d "${test_dir}" || ! -d "${gt_dir}" ]]; then
    echo "[ERROR] Missing required structure for ${c}"
    failed=1
    continue
  fi

  n_train=$(find "${train_dir}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
  n_test=$(find "${test_dir}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
  n_gt=$(find "${gt_dir}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)

  total_train=$((total_train + n_train))
  total_test=$((total_test + n_test))

  printf "%-14s %10d %10d %12d\n" "${c}" "${n_train}" "${n_test}" "${n_gt}"

  [[ "${n_train}" -gt 0 && "${n_test}" -gt 0 ]] || failed=1
done

echo
echo "Total train/good images: ${total_train}"
echo "Total test images      : ${total_test}"

[[ "${total_train}" -eq 3629 ]] || echo "[WARN] Canonical publication total is 3629 train/validation images."
[[ "${total_test}" -eq 1725 ]] || echo "[WARN] Canonical publication total is 1725 test images."

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] Dataset validation failed."
  exit 3
fi

echo "[OK] MVTec AD structure is valid."

if [[ "${KEEP_ARCHIVE:-0}" != "1" ]]; then
  rm -f "${ARCHIVE_PATH}"
fi

echo
echo "[DONE]"
echo "Run:"
echo "  export MVTEC_ROOT=\"${TARGET_DIR}\""
echo
echo "Then:"
echo "  python scripts/benchmarks/0_setup_benchmark.py \\"
echo "    --protocol conf/benchmarks/protocols/mvtec_smoke_v1.yaml \\"
echo "    --method conf/benchmarks/methods/cadic_compatible_v1.yaml \\"
echo "    --seed 0 --smoke --set memory.budget=256"
