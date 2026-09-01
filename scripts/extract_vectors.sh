#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <adapter> <model-path> <calibration-json> <output-dir> <device-id>" >&2
  exit 2
fi

ADAPTER="$1"
MODEL_PATH="$2"
DATA_PATH="$3"
OUTPUT_DIR="$4"
DEVICE_ID="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for MODE in behavior conditional_c1 conditional_c2; do
  python -m conflictguard.vectors.extract \
    --adapter "${ADAPTER}" \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --mode "${MODE}" \
    --n_samples 300 \
    --pca_method pca_pairwise \
    --seed 42 \
    --device_id "${DEVICE_ID}" \
    --print_variance_table
done
