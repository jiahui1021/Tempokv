#!/usr/bin/env bash
# LongCat BF16 / TempoKV generation using the bundled runtime.
#
# 环境变量：
#   TEMPOKV_BF16_LAG_FRAMES  最新多少个条件 latent 保持精确 BF16、不压缩（默认 0）。
#                   LongCat 每段一次性生成整段条件 cache、段末丢弃，所以没有
#                   增量调度，也没有结束后要 flush 的尾巴；只保留 lag 这一项。
#                   有 attention sink 时只从最近的 span 里扣，sink span 不会被切
set -euo pipefail

# shellcheck source=scripts/LongCat/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_HISTORY_LENGTH="$(resolve_history K_HISTORY_LENGTH KH)"
V_HISTORY_LENGTH="$(resolve_history V_HISTORY_LENGTH VH)"
K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"
CONFIG="${CONFIG:-$default_codec_config}"

mode="${MODE:-tempokv}"
case "$mode" in
  tempokv|bf16) ;;
  *) die "MODE 必须是 tempokv 或 bf16（当前值：${mode}）" ;;
esac

require_env CHECKPOINT_DIR

if [[ "$mode" == "tempokv" ]]; then
  config_path="$(resolve_path "$CONFIG")"
  FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
  parameter_name="${PARAM_NAME:-$(
    codec_slug "$config_path" \
      "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" \
      "${K_BITS:-}" "${V_BITS:-}" "${CACHE_STRIDE:-}" \
      "${V_PREDICTOR:-}" "$FIT_HISTORY_SOURCE" "${ANCHOR_BITS:-}"
  )}"
  param_dir="$(resolve_path "${PARAM_DIR:-data/longcat/params/$parameter_name}")"
  read -r _ v_predictor_mode _ <<<"$(codec_modes "$config_path" "${V_PREDICTOR:-}")"
  if [[ -n "${K_PARAMS:-}" || -n "${V_PARAMS:-}" ]]; then
    [[ -n "${K_PARAMS:-}" && -n "${V_PARAMS:-}" ]] || \
      die "K_PARAMS 与 V_PARAMS 必须同时设置"
  fi
  k_params="$(resolve_path "${K_PARAMS:-$param_dir/k_temporal.npz}")"
  v_params="$(resolve_path "${V_PARAMS:-$param_dir/$(v_artifact_name "$v_predictor_mode")}")"
  default_units_per_launch "$v_predictor_mode"
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
SAVE_SEGMENT_SNAPSHOTS="${SAVE_SEGMENT_SNAPSHOTS:-0}"

check_int CONTEXT_PARALLEL_SIZE 1
check_int NUM_GPUS 1
check_int NUM_SEGMENTS 1
check_int NUM_FRAMES 1
check_int NUM_COND_FRAMES 1
check_flag DRY_RUN
check_flag PREFLIGHT
check_flag OFFLOAD_KV_CACHE
check_flag ENABLE_COMPILE
check_flag SAVE_SEGMENT_SNAPSHOTS

if (( (NUM_FRAMES - 1) % 4 != 0 )); then
  die "NUM_FRAMES 必须满足 (NUM_FRAMES - 1) 能被 4 整除"
fi
if (( NUM_COND_FRAMES > NUM_FRAMES || (NUM_COND_FRAMES - 1) % 4 != 0 )); then
  die "NUM_COND_FRAMES 必须不大于 NUM_FRAMES，且 (NUM_COND_FRAMES - 1) 能被 4 整除"
fi

if [[ "$mode" == "tempokv" ]]; then
  # Keep each resolved codec variant in its own result directory.  The same
  # signature already selects the matching fitted parameters above, so mixed
  # K/V bit widths cannot silently overwrite another generation run.
  default_output="results/longcat/tempokv/$parameter_name"
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
[[ "${CUMULATIVE_COND:-0}" == "1" ]] && args+=(--cumulative_cond)
[[ "$ENABLE_COMPILE" == "1" ]] && args+=(--enable_compile)
[[ "$SAVE_SEGMENT_SNAPSHOTS" == "1" ]] && args+=(--save_segment_snapshots)
[[ -n "${CACHE_STRIDE:-}" ]] && args+=(--stride "$CACHE_STRIDE")
[[ -n "${K_HISTORY_LENGTH:-}" ]] && args+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && args+=(--v-history-length "$V_HISTORY_LENGTH")
[[ -n "${K_BITS:-}" ]] && args+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && args+=(--v-bits "$V_BITS")
[[ -n "${ANCHOR_BITS:-}" ]] && args+=(--anchor-bits "$ANCHOR_BITS")
[[ -n "${V_PREDICTOR:-}" ]] && args+=(--v-predictor-mode "$V_PREDICTOR")
[[ -n "${TRITON_CACHE:-}" ]] && args+=(--triton-cache "$(resolve_path "$TRITON_CACHE")")
if [[ "$mode" == "tempokv" ]]; then
  args+=(
    --config "$config_path"
    --k-params "$k_params"
    --v-params "$v_params"
  )
fi
[[ -n "${TRACE_DIR:-}" ]] && args+=(--trace-dir "$(resolve_path "$TRACE_DIR")")

preflight_args=(
  --mode "$mode"
  --checkpoint-dir "$(resolve_path "$CHECKPOINT_DIR")"
  --context-parallel-size "$CONTEXT_PARALLEL_SIZE"
)
[[ -n "${CACHE_STRIDE:-}" ]] && preflight_args+=(--stride "$CACHE_STRIDE")
[[ -n "${K_BITS:-}" ]] && preflight_args+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && preflight_args+=(--v-bits "$V_BITS")
[[ -n "${ANCHOR_BITS:-}" ]] && preflight_args+=(--anchor-bits "$ANCHOR_BITS")
[[ -n "${V_PREDICTOR:-}" ]] && preflight_args+=(--v-predictor-mode "$V_PREDICTOR")
if [[ "$mode" == "tempokv" ]]; then
  preflight_args+=(
    --config "$config_path"
    --k-parameters "$k_params"
    --v-parameters "$v_params"
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
