#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
python_bin="${PYTHON_BIN:-python3}"
mode="${MODE:-tempokv}"

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      printf 'Error: missing required variable %s\n' "$name" >&2
      exit 2
    fi
  done
}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$project_root" "${1#./}" ;;
  esac
}

require_env LONGCAT_REPO CHECKPOINT_DIR INIT_VIDEO_PATH OUTPUT_DIR
if [[ "$mode" == "tempokv" ]]; then
  require_env CONFIG K_PARAMS V_PARAMS
fi

args=(
  --mode "$mode"
  --longcat-repo "$(resolve_path "$LONGCAT_REPO")"
  --checkpoint_dir "$(resolve_path "$CHECKPOINT_DIR")"
  --workload "${WORKLOAD:-480p_long_gen}"
  --init_video_path "$(resolve_path "$INIT_VIDEO_PATH")"
  --output_dir "$(resolve_path "$OUTPUT_DIR")"
  --num_segments "${NUM_SEGMENTS:-1}"
  --num_frames "${NUM_FRAMES:-93}"
  --num_cond_frames "${NUM_COND_FRAMES:-53}"
  --prompt_source "${PROMPT_SOURCE:-direct_prompt}"
  --prompt "${PROMPT:-a cinematic video}"
  --prompt_idx "${PROMPT_IDX:-0}"
  --seed "${SEED:-0}"
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE:-1}"
  --attn_sink_frames "${ATTN_SINK_FRAMES:-0}"
)
if [[ "${OFFLOAD_KV_CACHE:-0}" == "1" ]]; then
  args+=(--offload_kv_cache)
else
  args+=(--no_offload_kv_cache)
fi
[[ -n "${CACHE_STRIDE:-}" ]] && args+=(--stride "$CACHE_STRIDE")
[[ -n "${TRITON_CACHE:-}" ]] && args+=(--triton-cache "$(resolve_path "$TRITON_CACHE")")
if [[ "$mode" == "tempokv" ]]; then
  args+=(
    --config "$(resolve_path "$CONFIG")"
    --k-params "$(resolve_path "$K_PARAMS")"
    --v-params "$(resolve_path "$V_PARAMS")"
  )
fi
[[ -n "${TRACE_DIR:-}" ]] && args+=(--trace-dir "$(resolve_path "$TRACE_DIR")")

if [[ "${PREFLIGHT:-1}" != "0" ]]; then
  preflight_args=(
    --mode "$mode"
    --longcat-repo "$(resolve_path "$LONGCAT_REPO")"
    --checkpoint-dir "$(resolve_path "$CHECKPOINT_DIR")"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
  )
  [[ -n "${CACHE_STRIDE:-}" ]] && preflight_args+=(--stride "$CACHE_STRIDE")
  if [[ "$mode" == "tempokv" ]]; then
    preflight_args+=(
      --config "$(resolve_path "$CONFIG")"
      --k-parameters "$(resolve_path "$K_PARAMS")"
      --v-parameters "$(resolve_path "$V_PARAMS")"
    )
  fi
  "$python_bin" "$script_dir/preflight.py" "${preflight_args[@]}"
fi

exec "$torchrun_bin" \
  --nproc_per_node="${NUM_GPUS:-${CONTEXT_PARALLEL_SIZE:-1}}" \
  --standalone \
  "$script_dir/generate.py" "${args[@]}" "$@"
