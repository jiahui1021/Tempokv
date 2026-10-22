#!/usr/bin/env bash
# Suite B only. Suite A remains under scripts/spectral/.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin=python3
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${MODEL_ROOT:=${WAN_MODEL_DIR:-}}"
: "${MODEL_ROOT:?MODEL_ROOT or WAN_MODEL_DIR is required}"
: "${PROMPTS:?PROMPTS is required}"
experiment="${EXPERIMENT:-B_E1_SPECTRUM}"
config="${SPECTRAL_KV_CONFIG:-$root/scripts/paper_experiments/spectral/configs/$experiment.yaml}"
command=("$python_bin" "$root/scripts/paper_experiments/spectral/run.py"
  --config "$config" --runtime "${RUNTIME:-self_forcing}" --checkpoint "$CHECKPOINT"
  --model-root "$MODEL_ROOT" --prompts "$PROMPTS" --output "${SPECTRAL_KV_OUTPUT_DIR:-$root/reports/spectral_kv}"
  --frames "${FRAMES:-189}" --window "${LOCAL_ATTN_SIZE:-21}" --sink "${SINK_SIZE:-0}"
  --seed "${SEED:-0}" --end-index "${END_INDEX:-2}" --start-index "${START_INDEX:-0}"
  --gpus "${NUM_GPUS:-1}" --samples "${NUM_SAMPLES:-1}")
[[ -n "${OUTPUT_DIR:-}" ]] && command+=(--video-output "$OUTPUT_DIR")
[[ "${DRY_RUN:-0}" == "1" ]] && command+=(--dry-run)
exec "${command[@]}"
