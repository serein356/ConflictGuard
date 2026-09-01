#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <vanilla|prompt|conflictguard> <model-key> <model-path> <test-json> <output-json> <device-id> [extra args...]" >&2
  exit 2
fi

METHOD="$1"
MODEL_KEY="$2"
MODEL_PATH="$3"
TEST_SET="$4"
OUTPUT="$5"
DEVICE_ID="$6"
shift 6

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/configs/configs.json"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${METHOD}" == "vanilla" || "${METHOD}" == "prompt" ]]; then
  exec python -m conflictguard.eval.run_eval_steered \
    --model_adapter "$(python -c "import json; print(json.load(open('${CONFIG}'))['${MODEL_KEY}']['adapter'])")" \
    --model_path "${MODEL_PATH}" \
    --test_set "${TEST_SET}" \
    --steer_mode none \
    --device_id "${DEVICE_ID}" \
    --output "${OUTPUT}" \
    "$@"
fi

if [[ "${METHOD}" != "conflictguard" ]]; then
  echo "Unknown method: ${METHOD}" >&2
  exit 2
fi

readarray -t PAPER_CONFIG < <(
  python - "${CONFIG}" "${MODEL_KEY}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2] not in config:
    raise SystemExit(f"Unknown model key: {sys.argv[2]}")
row = config[sys.argv[2]]
for key in (
    "adapter", "vector_dir", "behavior_vector", "condition_c1_layer",
    "condition_c2_layer", "condition_threshold", "behavior_strength",
    "behavior_layers",
):
    print(row[key])
PY
)

ADAPTER="${PAPER_CONFIG[0]}"
VECTOR_DIR="${ROOT}/vectors/${PAPER_CONFIG[1]}"
BEHAVIOR_VECTOR="${PAPER_CONFIG[2]}"
C1_LAYER="${PAPER_CONFIG[3]}"
C2_LAYER="${PAPER_CONFIG[4]}"
THRESHOLD="${PAPER_CONFIG[5]}"
STRENGTH="${PAPER_CONFIG[6]}"
BEHAVIOR_LAYERS="${PAPER_CONFIG[7]}"

exec python -m conflictguard.eval.run_eval_steered \
  --model_adapter "${ADAPTER}" \
  --model_path "${MODEL_PATH}" \
  --test_set "${TEST_SET}" \
  --behavior_vector "${VECTOR_DIR}/${BEHAVIOR_VECTOR}" \
  --condition_c1 "${VECTOR_DIR}/conditional_c1.pt" \
  --condition_c2 "${VECTOR_DIR}/conditional_c2.pt" \
  --steer_mode multi \
  --behavior_strength "${STRENGTH}" \
  --behavior_layers "${BEHAVIOR_LAYERS}" \
  --condition_c1_layers "${C1_LAYER}" \
  --condition_c2_layers "${C2_LAYER}" \
  --condition_threshold "${THRESHOLD}" \
  --condition_comparison_mode last \
  --device_id "${DEVICE_ID}" \
  --output "${OUTPUT}" \
  "$@"
