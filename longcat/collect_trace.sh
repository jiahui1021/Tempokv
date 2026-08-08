#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
python_bin="${PYTHON_BIN:-python3}"

for name in LONGCAT_REPO CHECKPOINT_DIR INIT_VIDEO_PATH TRACE_DIR; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Error: missing required variable %s\n' "$name" >&2
    exit 2
  fi
done

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$project_root" "${1#./}" ;;
  esac
}

args=(
  --mode trace
  --longcat-repo "$(resolve_path "$LONGCAT_REPO")"
  --trace-dir "$(resolve_path "$TRACE_DIR")"
  --checkpoint_dir "$(resolve_path "$CHECKPOINT_DIR")"
  --init_video_path "$(resolve_path "$INIT_VIDEO_PATH")"
  --output_dir "$(resolve_path "${OUTPUT_DIR:-$project_root/results/longcat_trace}")"
  --workload "${WORKLOAD:-480p_long_gen}"
  --num_segments "${NUM_SEGMENTS:-1}"
  --num_frames "${NUM_FRAMES:-93}"
  --num_cond_frames "${NUM_COND_FRAMES:-53}"
  --prompt_source "${PROMPT_SOURCE:-direct_prompt}"
  --prompt "${PROMPT:-a cinematic video}"
  --prompt_idx "${PROMPT_IDX:-0}"
  --seed "${SEED:-0}"
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE:-1}"
  --attn_sink_frames "${ATTN_SINK_FRAMES:-0}"
  --max-trace-spatial-positions "${MAX_TRACE_SPATIAL_POSITIONS:-32}"
)
if [[ "${OFFLOAD_KV_CACHE:-0}" == "1" ]]; then
  args+=(--offload_kv_cache)
else
  args+=(--no_offload_kv_cache)
fi
[[ -n "${CACHE_STRIDE:-}" ]] && args+=(--stride "$CACHE_STRIDE")

if [[ "${PREFLIGHT:-1}" != "0" ]]; then
  preflight_args=(
    --mode trace
    --longcat-repo "$(resolve_path "$LONGCAT_REPO")"
    --checkpoint-dir "$(resolve_path "$CHECKPOINT_DIR")"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
  )
  [[ -n "${CACHE_STRIDE:-}" ]] && preflight_args+=(--stride "$CACHE_STRIDE")
  "$python_bin" "$script_dir/preflight.py" "${preflight_args[@]}"
fi

exec "$torchrun_bin" \
  --nproc_per_node="${NUM_GPUS:-${CONTEXT_PARALLEL_SIZE:-1}}" \
  --standalone \
  "$script_dir/generate.py" "${args[@]}" \
  "$@"
