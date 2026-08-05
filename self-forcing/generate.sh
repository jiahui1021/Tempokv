#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"
mode="${MODE:-tempokv}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf 'Error: %s must be set.\n' "$name" >&2
    exit 2
  fi
}

# Build a short quantization label for the results subdirectory, e.g.
# kh3-cross-int2 for history_length=3, CrossKV predictor, INT2 residuals.
_quant_config_slug() {
  if [[ "$mode" == "bf16" ]]; then
    printf 'bf16'
    return 0
  fi
  local k_bits v_bits history
  k_bits=$("$python_bin" - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("k_bits", 0))
PY
  )
  v_bits=$("$python_bin" - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("v_bits", 0))
PY
  )
  history=$("$python_bin" - "$param_dir/k_temporal.npz" <<'PY'
import numpy as np, sys
k = np.load(sys.argv[1])
print(int(np.asarray(k["history_length"]).item()))
PY
  )
  printf 'kh%s-cross-int%s' "$history" "$k_bits"
}

if [[ "$mode" != "tempokv" && "$mode" != "bf16" ]]; then
  printf 'Error: MODE must be tempokv or bf16.\n' >&2
  exit 2
fi

for name in MODEL_REPO CHECKPOINT WAN_MODEL_DIR PROMPTS; do
  require_env "$name"
done
if [[ "$mode" == "tempokv" ]]; then
  for name in CONFIG; do
    require_env "$name"
  done
fi

for flag_name in DRY_RUN ALLOW_TRITON_FALLBACK SKIP_EXISTING; do
  flag_value="${!flag_name:-0}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    printf 'Error: %s must be 0 or 1.\n' "$flag_name" >&2
    exit 2
  fi
done
model_type="${MODEL_TYPE:-self_forcing}"
case "$model_type" in
  */) model_type="${model_type%/}" ;;
esac
param_dir="${PARAM_DIR:-$project_root/parameters}"
case "$param_dir" in
  /*) ;;
  *) param_dir="$(realpath -m -- "$param_dir")" ;;
esac
output_dir="${OUTPUT_DIR:-$project_root/results/$model_type/$(_quant_config_slug)}"
case "$output_dir" in
  /*) ;;
  *) output_dir="$(realpath -m -- "$output_dir")" ;;
esac
if [[ "$output_dir" == "/" ]]; then
  printf 'Error: OUTPUT_DIR must not be the filesystem root.\n' >&2
  exit 2
fi

command=(
  "$python_bin" "$script_dir/generate.py"
  --mode "$mode"
  --model-repo "$MODEL_REPO"
  --checkpoint "$CHECKPOINT"
  --prompts "$PROMPTS"
  --output-dir "$output_dir"
  --frames "${FRAMES:-27}"
  --samples "${NUM_SAMPLES:-1}"
  --seed "${SEED:-0}"
  --start-index "${START_INDEX:-0}"
  --end-index "${END_INDEX:--1}"
  --num-gpus "${NUM_GPUS:-1}"
)

if [[ "$mode" == "tempokv" ]]; then
  command+=(
    --config "$CONFIG"
    --k-params "$param_dir/k_temporal.npz"
    --v-params "$param_dir/v_cross_kv.npz"
  )
fi

if [[ -n "${MODEL_CONFIG:-}" ]]; then
  command+=(--model-config "$MODEL_CONFIG")
fi
if [[ -n "${LOCAL_ATTN_SIZE:-}" ]]; then
  command+=(--local-attn-size "$LOCAL_ATTN_SIZE")
fi
if [[ -n "${TRITON_CACHE:-}" ]]; then
  command+=(--triton-cache "$TRITON_CACHE")
fi
if [[ "${ALLOW_TRITON_FALLBACK:-0}" == "1" ]]; then
  command+=(--allow-triton-fallback)
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  command+=(--skip-existing)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  command+=(--dry-run)
fi

export FORCING_WAN_MODEL_DIR="$WAN_MODEL_DIR"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "${command[@]}"
  exit 0
fi
mkdir -p -- "$output_dir"
"${command[@]}" 2>&1 | tee "$output_dir/generate.log"
