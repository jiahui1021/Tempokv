#!/usr/bin/env bash
# LongCat BF16 / TempoKV generation using the bundled runtime.
set -euo pipefail

# shellcheck source=scripts/LongCat/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"

mode="${MODE:-tempokv}"
case "$mode" in
  tempokv|bf16) ;;
  *) die "MODE 必须是 tempokv 或 bf16（当前值：${mode}）" ;;
esac

require_env CHECKPOINT_DIR
if [[ "$mode" == "tempokv" ]]; then
  require_env CONFIG K_PARAMS V_PARAMS
fi

CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-$CONTEXT_PARALLEL_SIZE}"
NUM_SEGMENTS="${NUM_SEGMENTS:-1}"
NUM_FRAMES="${NUM_FRAMES:-93}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-53}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT="${PREFLIGHT:-1}"
OFFLOAD_KV_CACHE="${OFFLOAD_KV_CACHE:-0}"
ENABLE_COMPILE="${ENABLE_COMPILE:-0}"

check_int CONTEXT_PARALLEL_SIZE 1
check_int NUM_GPUS 1
check_int NUM_SEGMENTS 1
check_int NUM_FRAMES 1
check_int NUM_COND_FRAMES 1
check_flag DRY_RUN
check_flag PREFLIGHT
check_flag OFFLOAD_KV_CACHE
check_flag ENABLE_COMPILE

if (( (NUM_FRAMES - 1) % 4 != 0 )); then
  die "NUM_FRAMES 必须满足 (NUM_FRAMES - 1) 能被 4 整除"
fi
if (( NUM_COND_FRAMES > NUM_FRAMES || (NUM_COND_FRAMES - 1) % 4 != 0 )); then
  die "NUM_COND_FRAMES 必须不大于 NUM_FRAMES，且 (NUM_COND_FRAMES - 1) 能被 4 整除"
fi

if [[ "$mode" == "tempokv" ]]; then
  default_output="results/longcat/tempokv"
else
  default_output="results/longcat/bf16"
fi
output_dir="$(resolve_path "${OUTPUT_DIR:-$default_output}")"

args=(
  --mode "$mode"
  --checkpoint_dir "$(resolve_path "$CHECKPOINT_DIR")"
  --output_dir "$output_dir"
  --workload 480p_long_gen
  --num_segments "$NUM_SEGMENTS"
  --num_frames "$NUM_FRAMES"
  --num_cond_frames "$NUM_COND_FRAMES"
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}"
  --guidance_scale "${GUIDANCE_SCALE:-4.0}"
  --prompt_source direct_prompt
  --prompt "${PROMPT:-a cinematic video}"
  --prompt_idx "${PROMPT_IDX:-0}"
  --seed "${SEED:-0}"
  --context_parallel_size "$CONTEXT_PARALLEL_SIZE"
  --attn_sink_frames "${ATTN_SINK_FRAMES:-0}"
)
[[ -n "${INIT_VIDEO_PATH:-}" ]] && \
  args+=(--init_video_path "$(resolve_path "$INIT_VIDEO_PATH")")
[[ "$OFFLOAD_KV_CACHE" == "1" ]] && args+=(--offload_kv_cache) || args+=(--no_offload_kv_cache)
[[ "$ENABLE_COMPILE" == "1" ]] && args+=(--enable_compile)
[[ -n "${CACHE_STRIDE:-}" ]] && args+=(--stride "$CACHE_STRIDE")
[[ -n "${K_BITS:-}" ]] && args+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && args+=(--v-bits "$V_BITS")
[[ -n "${V_PREDICTOR:-}" ]] && args+=(--v-predictor-mode "$V_PREDICTOR")
[[ -n "${TRITON_CACHE:-}" ]] && args+=(--triton-cache "$(resolve_path "$TRITON_CACHE")")
if [[ "$mode" == "tempokv" ]]; then
  args+=(
    --config "$(resolve_path "$CONFIG")"
    --k-params "$(resolve_path "$K_PARAMS")"
    --v-params "$(resolve_path "$V_PARAMS")"
  )
fi
[[ -n "${TRACE_DIR:-}" ]] && args+=(--trace-dir "$(resolve_path "$TRACE_DIR")")

preflight_args=(
  --mode "$mode"
  --checkpoint-dir "$(resolve_path "$CHECKPOINT_DIR")"
  --context-parallel-size "$CONTEXT_PARALLEL_SIZE"
)
[[ -n "${CACHE_STRIDE:-}" ]] && preflight_args+=(--stride "$CACHE_STRIDE")
if [[ "$mode" == "tempokv" ]]; then
  preflight_args+=(
    --config "$(resolve_path "$CONFIG")"
    --k-parameters "$(resolve_path "$K_PARAMS")"
    --v-parameters "$(resolve_path "$V_PARAMS")"
  )
fi

command=(
  "$torchrun_bin"
  "--nproc_per_node=$NUM_GPUS"
  --standalone
  "$adapter_root/generate.py"
  "${args[@]}"
  "$@"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[LongCat generate] 干跑模式：\n  '
  printf '%q ' "${command[@]}"
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
  "$python_bin" "$adapter_root/preflight.py" "${preflight_args[@]}"
fi

mkdir -p -- "$output_dir"
exec "${command[@]}"
