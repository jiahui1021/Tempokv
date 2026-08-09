#!/usr/bin/env bash
# Collect and merge clean LongCat K/V traces with the bundled runtime.
set -euo pipefail

# shellcheck source=scripts/LongCat/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

require_env CHECKPOINT_DIR

trace_root="$(resolve_path "${TRACE_DIR:-data/longcat/trace}")"
raw_trace_dir="$trace_root/raw"
trace_file="$(resolve_path "${TRACE_FILE:-$trace_root/clean_trace.npz}")"
output_dir="$(resolve_path "${OUTPUT_DIR:-$trace_root/output}")"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-$CONTEXT_PARALLEL_SIZE}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT="${PREFLIGHT:-1}"

check_int CONTEXT_PARALLEL_SIZE 1
check_int NUM_GPUS 1
check_flag DRY_RUN
check_flag PREFLIGHT

args=(
  --mode trace
  --trace-dir "$raw_trace_dir"
  --checkpoint_dir "$(resolve_path "$CHECKPOINT_DIR")"
  --output_dir "$output_dir"
  --workload 480p_long_gen
  --num_segments "${NUM_SEGMENTS:-1}"
  --num_frames "${NUM_FRAMES:-93}"
  --num_cond_frames "${NUM_COND_FRAMES:-53}"
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}"
  --guidance_scale "${GUIDANCE_SCALE:-4.0}"
  --prompt_source direct_prompt
  --prompt "${PROMPT:-a cinematic video}"
  --prompt_idx "${PROMPT_IDX:-0}"
  --seed "${SEED:-0}"
  --context_parallel_size "$CONTEXT_PARALLEL_SIZE"
  --attn_sink_frames "${ATTN_SINK_FRAMES:-0}"
  --max-trace-spatial-positions "${MAX_TRACE_SPATIAL_POSITIONS:-32}"
  --no_offload_kv_cache
)
[[ -n "${INIT_VIDEO_PATH:-}" ]] && \
  args+=(--init_video_path "$(resolve_path "$INIT_VIDEO_PATH")")
[[ -n "${CACHE_STRIDE:-}" ]] && args+=(--stride "$CACHE_STRIDE")

collect_command=(
  "$torchrun_bin"
  "--nproc_per_node=$NUM_GPUS"
  --standalone
  "$adapter_root/generate.py"
  "${args[@]}"
  "$@"
)
merge_command=(
  "$python_bin"
  "$adapter_root/prepare_trace.py"
  "$raw_trace_dir"
  "$trace_file"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[LongCat trace 1/2] '
  printf '%q ' "${collect_command[@]}"
  printf '\n[LongCat trace 2/2] '
  printf '%q ' "${merge_command[@]}"
  printf '\n'
  exit 0
fi

[[ -f "$project_root/experiments/LongCat/inference.py" ]] || \
  die "仓库内置 LongCat 运行代码不完整"
[[ -d "$(resolve_path "$CHECKPOINT_DIR")" ]] || \
  die "CHECKPOINT_DIR 不存在：$(resolve_path "$CHECKPOINT_DIR")"
if [[ -n "${INIT_VIDEO_PATH:-}" && ! -f "$(resolve_path "$INIT_VIDEO_PATH")" ]]; then
  die "INIT_VIDEO_PATH 不存在：$(resolve_path "$INIT_VIDEO_PATH")"
fi

if [[ "$PREFLIGHT" == "1" ]]; then
  preflight_args=(
    --mode trace
    --checkpoint-dir "$(resolve_path "$CHECKPOINT_DIR")"
    --context-parallel-size "$CONTEXT_PARALLEL_SIZE"
  )
  [[ -n "${CACHE_STRIDE:-}" ]] && preflight_args+=(--stride "$CACHE_STRIDE")
  "$python_bin" "$adapter_root/preflight.py" "${preflight_args[@]}"
fi

mkdir -p -- "$raw_trace_dir" "$output_dir"
"${collect_command[@]}"
"${merge_command[@]}"
printf 'LongCat clean trace：%s\n' "$trace_file"
