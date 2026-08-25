#!/usr/bin/env bash
# Generate one layer_shared+TF32 video and compare it with a saved baseline.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf '缺少必需环境变量：%s\n' "$name" >&2
    exit 2
  fi
}

require_env CALIBRATION_TRACE
require_env BASELINE_REPORT
require_env CHECKPOINT
require_env WAN_MODEL_DIR

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  python_bin="$project_root/.venv/bin/python"
else
  python_bin="python3"
fi

prompts="${PROMPTS:-$project_root/assets/moviegen128.txt}"
output_dir="${OUTPUT_DIR:-$project_root/results/paper/layer_shared_tf32_one_video}"
base_config="${BASE_CONFIG:-$project_root/configs/legacy.json}"

command=(
  "$python_bin" "$script_dir/run_tf32_one_video.py"
  --base-config "$base_config"
  --calibration-trace "$CALIBRATION_TRACE"
  --baseline-report "$BASELINE_REPORT"
  --checkpoint "$CHECKPOINT"
  --prompts "$prompts"
  --output "$output_dir"
  --frames "${FRAMES:-717}"
  --seed "${SEED:-0}"
  --prompt-index "${PROMPT_INDEX:-0}"
  --num-gpus "${NUM_GPUS:-1}"
  --fit-device "${FIT_DEVICE:-auto}"
)
[[ -n "${MODEL_CONFIG:-}" ]] && command+=(--model-config "$MODEL_CONFIG")
[[ -n "${LOCAL_ATTN_SIZE:-}" ]] && command+=(--local-attn-size "$LOCAL_ATTN_SIZE")
[[ -n "${SPAN_FRAMES:-}" ]] && command+=(--span-frames "$SPAN_FRAMES")
[[ "${DRY_RUN:-0}" == "1" ]] && command+=(--dry-run)

export WAN_MODEL_DIR
exec "${command[@]}"
