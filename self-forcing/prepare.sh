#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"
dry_run="${DRY_RUN:-0}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf 'Error: %s must be set.\n' "$name" >&2
    exit 2
  fi
}

print_command() {
  printf 'DRY_RUN:'
  printf ' %q' "$@"
  printf '\n'
}

for name in TRACE_RUNNER CHECKPOINT WAN_MODEL_DIR PROMPTS WORK_DIR CONFIG; do
  require_env "$name"
done

if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  printf 'Error: DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi

case "$WORK_DIR" in
  /*) ;;
  *) WORK_DIR="$(realpath -m -- "$WORK_DIR")" ;;
esac
raw_trace_dir="$WORK_DIR/raw_trace/traces"
trace_output_dir="$WORK_DIR/raw_trace/output"
param_dir="${PARAM_DIR:-$project_root/parameters}"
case "$param_dir" in
  /*) ;;
  *) param_dir="$(realpath -m -- "$param_dir")" ;;
esac
trace_file="$param_dir/self_forcing_trace.npz"
start_index="${START_INDEX:-0}"
end_index="${END_INDEX:-8}"
trace_frames="${TRACE_FRAMES:-51}"
trace_tokens_per_frame="${TRACE_TOKENS_PER_FRAME:-8}"
num_gpus="${NUM_GPUS:-1}"
base_seed="${BASE_SEED:-0}"
skip_existing="${SKIP_EXISTING:-0}"
min_free_vram_gb="${MIN_FREE_VRAM_GB:-70}"
history_length="${HISTORY_LENGTH:-3}"
iterations="${ITERATIONS:-2}"
temporal_ridge="${TEMPORAL_RIDGE:-0.1}"
cross_ridge="${CROSS_RIDGE:-0.1}"
max_gain="${MAX_GAIN:-1.0}"
trace_runner="$TRACE_RUNNER"

for value_name in START_INDEX END_INDEX TRACE_FRAMES TRACE_TOKENS_PER_FRAME NUM_GPUS BASE_SEED HISTORY_LENGTH ITERATIONS; do
  value="${!value_name:-}"
  case "$value_name" in
    START_INDEX) value="$start_index" ;;
    END_INDEX) value="$end_index" ;;
    TRACE_FRAMES) value="$trace_frames" ;;
    TRACE_TOKENS_PER_FRAME) value="$trace_tokens_per_frame" ;;
    NUM_GPUS) value="$num_gpus" ;;
    BASE_SEED) value="$base_seed" ;;
    HISTORY_LENGTH) value="$history_length" ;;
    ITERATIONS) value="$iterations" ;;
  esac
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf 'Error: %s must be a non-negative integer.\n' "$value_name" >&2
    exit 2
  fi
done
if (( trace_frames < 27 )); then
  printf 'Error: TRACE_FRAMES must be at least 27.\n' >&2
  exit 2
fi
if (( trace_tokens_per_frame < 1 || num_gpus < 1 || history_length < 1 || iterations < 1 )); then
  printf 'Error: trace tokens, GPU count, history length, and iterations must be positive.\n' >&2
  exit 2
fi
for flag_name in OVERWRITE SKIP_EXISTING; do
  flag_value="${!flag_name:-0}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    printf 'Error: %s must be 0 or 1.\n' "$flag_name" >&2
    exit 2
  fi
done
if [[ "$WORK_DIR" == "/" || "$param_dir" == "/" ]]; then
  printf 'Error: output paths must not be the filesystem root.\n' >&2
  exit 2
fi

trace_command=(
  env
  "SELF_FORCING_CHECKPOINT_PATH=$CHECKPOINT"
  "FORCING_WAN_MODEL_DIR=$WAN_MODEL_DIR"
  "CROSS_KV_TRACE_DIR=$raw_trace_dir"
  "OUTPUT_FOLDER=$trace_output_dir"
  "PROMPTS_PATH=$PROMPTS"
  "START_INDEX=$start_index"
  "END_INDEX=$end_index"
  "NUM_OUTPUT_FRAMES=$trace_frames"
  "TRACE_LAYERS=all"
  "TRACE_HEADS=all"
  "TRACE_TOKENS_PER_FRAME=$trace_tokens_per_frame"
  "TRACE_CAPTURE_HIDDEN=0"
  "TRACE_CAPTURE_ATTENTION=0"
  "TRACE_QUANT_TYPE=none"
  "NUM_GPUS=$num_gpus"
  "PYTHON_BIN=$python_bin"
  "BASE_SEED=$base_seed"
  "SKIP_EXISTING=$skip_existing"
  "MIN_FREE_VRAM_GB=$min_free_vram_gb"
  bash "$trace_runner"
)
merge_command=(
  "$python_bin" "$script_dir/prepare_trace.py"
  "$raw_trace_dir" "$trace_file"
)
fit_command=(
  "$python_bin" "$project_root/scripts/fit_params.py"
  "$trace_file"
  --config "$CONFIG"
  --output-dir "$param_dir"
  --history-length "$history_length"
  --iterations "$iterations"
  --temporal-ridge "$temporal_ridge"
  --cross-ridge "$cross_ridge"
  --max-gain "$max_gain"
  --stride 1560
  --coordinate-system pre_rope_normalized
)

if [[ "$dry_run" == "1" ]]; then
  print_command "${trace_command[@]}"
  print_command "${merge_command[@]}"
  print_command "${fit_command[@]}"
  exit 0
fi

if [[ ! -f "$trace_runner" ]]; then
  printf 'Error: trace runner does not exist: %s\n' "$trace_runner" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'Error: CONFIG does not exist: %s\n' "$CONFIG" >&2
  exit 2
fi

if [[ "${OVERWRITE:-0}" != "1" ]]; then
  existing_artifact=""
  for artifact in \
    "$trace_file" "$param_dir/k_temporal.npz" "$param_dir/v_cross_kv.npz"; do
    if [[ -e "$artifact" ]]; then
      existing_artifact="$artifact"
      break
    fi
  done
  if [[ -z "$existing_artifact" && -d "$raw_trace_dir" ]]; then
    existing_artifact="$(find "$raw_trace_dir" -type f -name 'layer_*.npz' -print -quit)"
  fi
  if [[ -n "$existing_artifact" ]]; then
    printf 'Error: existing trace or parameters found: %s\n' "$existing_artifact" >&2
    printf 'Set OVERWRITE=1 to replace files in place, or choose a new WORK_DIR.\n' >&2
    exit 2
  fi
fi

mkdir -p -- "$param_dir" "$raw_trace_dir" "$trace_output_dir" "$(dirname -- "$trace_file")"
"${trace_command[@]}"
"${merge_command[@]}"
"${fit_command[@]}"

for artifact in "$param_dir/k_temporal.npz" "$param_dir/v_cross_kv.npz"; do
  if [[ ! -s "$artifact" ]]; then
    printf 'Error: expected parameter file was not created: %s\n' "$artifact" >&2
    exit 1
  fi
done

printf 'Prepared TempoKV parameters:\n  %s\n  %s\n' \
  "$param_dir/k_temporal.npz" "$param_dir/v_cross_kv.npz"
