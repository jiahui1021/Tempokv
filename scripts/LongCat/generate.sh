#!/usr/bin/env bash
# LongCat BF16 / TempoKV generation using the bundled runtime.
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
  [[ -f "$config_path" ]] || die "CONFIG 不存在：$config_path"
  if [[ -n "${V_PREDICTOR:-}" ]]; then
    case "$V_PREDICTOR" in
      cross_kv|temporal) ;;
      *) die "V_PREDICTOR 必须是 cross_kv 或 temporal（当前值：${V_PREDICTOR}）" ;;
    esac
  fi
  FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
  parameter_name="${PARAM_NAME:-$(
    codec_slug "$config_path" \
      "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" \
      "${K_BITS:-}" "${V_BITS:-}" "${CACHE_STRIDE:-}" \
      "${V_PREDICTOR:-}" "$FIT_HISTORY_SOURCE"
  )}"
  param_dir="$(resolve_path "${PARAM_DIR:-data/longcat/params/$parameter_name}")"
  v_predictor_mode="$("$python_bin" - "$config_path" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config.get("codec", config).get("v_predictor_mode", "cross_kv"))
PY
)"
  v_predictor_mode="${V_PREDICTOR:-$v_predictor_mode}"
  if [[ "$v_predictor_mode" == "temporal" ]]; then
    v_artifact_name="v_temporal.npz"
  else
    v_artifact_name="v_cross_kv.npz"
  fi
  if [[ -n "${K_PARAMS:-}" || -n "${V_PARAMS:-}" ]]; then
    [[ -n "${K_PARAMS:-}" && -n "${V_PARAMS:-}" ]] || \
      die "K_PARAMS 与 V_PARAMS 必须同时设置"
  fi
  k_params="$(resolve_path "${K_PARAMS:-$param_dir/k_temporal.npz}")"
  v_params="$(resolve_path "${V_PARAMS:-$param_dir/$v_artifact_name}")"

  # Use the validated Cross production launch size by default.  Temporal
  # keeps the shared runtime default, and explicit overrides still win.
  if [[ "$v_predictor_mode" == "cross_kv" ]]; then
    export TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH="${TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH:-8}"
  fi
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
[[ "$SAVE_SEGMENT_SNAPSHOTS" == "1" ]] && args+=(--save_segment_snapshots)
[[ -n "${CACHE_STRIDE:-}" ]] && args+=(--stride "$CACHE_STRIDE")
[[ -n "${K_HISTORY_LENGTH:-}" ]] && args+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && args+=(--v-history-length "$V_HISTORY_LENGTH")
[[ -n "${K_BITS:-}" ]] && args+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && args+=(--v-bits "$V_BITS")
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
