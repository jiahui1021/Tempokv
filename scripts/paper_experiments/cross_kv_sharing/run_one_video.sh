#!/usr/bin/env bash
# Run the formal five-video CrossKV sharing comparison.

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
output_dir="${OUTPUT_DIR:-$project_root/results/paper/cross_kv_sharing_one_video}"
base_config="${BASE_CONFIG:-$project_root/configs/legacy.json}"

command=(
  "$python_bin" "$script_dir/run_study.py"
  --base-config "$base_config"
  --calibration-trace "$CALIBRATION_TRACE"
  --checkpoint "$CHECKPOINT"
  --prompts "$prompts"
  --output "$output_dir"
  --frames "${FRAMES:-717}"
  --samples 1
  --seed "${SEED:-0}"
  --start-index 0
  --end-index 1
  --num-gpus "${NUM_GPUS:-1}"
  --fit-device "${FIT_DEVICE:-auto}"
  --reconstruction-device "${RECONSTRUCTION_DEVICE:-cpu}"
  --quality-device "${QUALITY_DEVICE:-auto}"
  --fid-batch-size "${FID_BATCH_SIZE:-32}"
  --execute
)

[[ -n "${HELD_OUT_TRACE:-}" ]] && command+=(--held-out-trace "$HELD_OUT_TRACE")
[[ -n "${MODEL_CONFIG:-}" ]] && command+=(--model-config "$MODEL_CONFIG")
[[ -n "${LOCAL_ATTN_SIZE:-}" ]] && command+=(--local-attn-size "$LOCAL_ATTN_SIZE")
[[ -n "${SPAN_FRAMES:-}" ]] && command+=(--span-frames "$SPAN_FRAMES")
[[ -n "${MAX_QUALITY_FRAMES:-}" ]] && command+=(--max-quality-frames "$MAX_QUALITY_FRAMES")

if [[ "${VBENCH_LONG:-0}" == "1" ]]; then
  require_env VBENCH_ROOT
  require_env VBENCH_CACHE_DIR
  require_env VBENCH_DEVICES
  command+=(
    --vbench-long
    --vbench-root "$VBENCH_ROOT"
    --vbench-cache-dir "$VBENCH_CACHE_DIR"
    --vbench-devices "$VBENCH_DEVICES"
  )
fi

export WAN_MODEL_DIR
exec "${command[@]}"
