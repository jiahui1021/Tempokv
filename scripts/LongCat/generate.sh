#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/LongCat/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
mode="${MODE:-tempokv}"

case "$mode" in
  tempokv|bf16|trace) ;;
  *)
    die "MODE 必须是 tempokv、bf16 或 trace（当前值：$mode）"
    ;;
esac

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
[[ -n "${MAX_TRACE_SPATIAL_POSITIONS:-}" ]] && \
  args+=(--max-trace-spatial-positions "$MAX_TRACE_SPATIAL_POSITIONS")

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
  "$python_bin" "$adapter_root/preflight.py" "${preflight_args[@]}"
fi

exec "$torchrun_bin" \
  --nproc_per_node="${NUM_GPUS:-${CONTEXT_PARALLEL_SIZE:-1}}" \
  --standalone \
  "$adapter_root/generate.py" "${args[@]}" "$@"
