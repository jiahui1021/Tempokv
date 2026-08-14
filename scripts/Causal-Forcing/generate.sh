#!/usr/bin/env bash
#
# generate.sh — 读取已拟合参数，生成 Causal Forcing 视频
#
# 不采集 trace、不重新拟合，只使用 PARAM_DIR 下的参数。更换 PROMPTS /
# OUTPUT_DIR 后可以重复运行。
#
# 两种模式：
#   MODE=tempokv  使用 TempoKV 量化参数（默认），需要 CONFIG + PARAM_DIR
#   MODE=bf16     不量化的对照基线，不需要 CONFIG / PARAM_DIR
#
# 用法（所有参数都用环境变量传入）：
#   CHECKPOINT=/models/causal_forcing/chunkwise/causal_ode.pt \
#   MODEL_ROOT=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/generation_prompts.txt \
#   CONFIG=configs/native.json \
#   FRAMES=81 \
#   bash scripts/Causal-Forcing/generate.sh
#
# 必需变量：
#   CHECKPOINT   Causal Forcing 模型权重
#   MODEL_ROOT   Wan2.1-T2V-1.3B 目录
#   PROMPTS      prompt 列表
#   CONFIG       codec 配置 JSON（仅 MODE=tempokv）
#
# 常用可选变量：
#   GENERATION_MODE  chunkwise（默认）或 framewise，决定 num_frame_per_block
#   PARAM_DIR        参数目录（默认 data/causal-forcing/params/<模式>/<参数方案>）
#   OUTPUT_DIR       结果输出目录（默认 results/causal_forcing/<模式>/<量化标签>）
#   FRAMES           最终视频帧数（默认 81）
#   SPAN_FRAMES      每个 TempoKV span 的 latent 帧数（默认 12）
#                    需满足 latent_frames >= SPAN_FRAMES + num_frame_per_block，
#                    否则 span 直到最后一个 block 才打包，没有任何一次注意力读过它
#   NUM_SAMPLES      每条 prompt 采样数（默认 1）
#   SEED             随机种子（默认 0）
#   START_INDEX / END_INDEX  prompt 序号范围（默认 0 .. -1 全部）
#   NUM_GPUS         torchrun 使用的 GPU 数（默认 1）
#   LOCAL_ATTN_SIZE  局部注意力窗口（默认 -1，即全序列）
#   USE_EMA          是否读取 checkpoint 的 generator_ema（默认 0，官方权重只有 generator）
#   BITS / K_BITS / V_BITS  位宽 override（2/4/8；默认读取 CONFIG）
#   K_HISTORY_LENGTH / V_HISTORY_LENGTH  历史帧数（默认读取 CONFIG）
#
set -euo pipefail

# shellcheck source=scripts/Causal-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_HISTORY_LENGTH="$(resolve_history K_HISTORY_LENGTH KH)"
V_HISTORY_LENGTH="$(resolve_history V_HISTORY_LENGTH VH)"
K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"
mode="${MODE:-tempokv}"

case "$mode" in
  tempokv|bf16) ;;
  *) die "MODE 必须是 tempokv 或 bf16（当前值：${mode}）" ;;
esac

require_env CHECKPOINT MODEL_ROOT PROMPTS
if [[ "$mode" == "tempokv" ]]; then
  require_env CONFIG
fi

generation_mode="${GENERATION_MODE:-chunkwise}"
case "$generation_mode" in
  chunkwise) num_frame_per_block=3 ;;
  framewise) num_frame_per_block=1 ;;
  *) die "GENERATION_MODE 必须是 chunkwise 或 framewise（当前值：${generation_mode}）" ;;
esac

runtime_root="$project_root/experiments/Causal-Forcing"
if [[ -n "${MODEL_CONFIG:-}" ]]; then
  model_config="$(resolve_path "$MODEL_CONFIG")"
else
  model_config="$runtime_root/configs/causal_forcing_dmd_$generation_mode.yaml"
fi

if [[ "$mode" == "tempokv" ]]; then
  config_path="$(resolve_path "$CONFIG")"
  [[ -f "$config_path" ]] || die "CONFIG 不存在：$config_path"
  read -r fit_mode v_predictor_mode <<<"$(resolve_codec_modes "$config_path" "${V_PREDICTOR:-}")"
  # This only selects the matching fitted parameter directory; runtime always
  # consumes reconstructed cache state regardless of the fitting ablation.
  FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
  parameter_name="${PARAM_NAME:-$(codec_slug "$config_path" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" "${K_BITS:-}" "${V_BITS:-}" "" "$v_predictor_mode" "$FIT_HISTORY_SOURCE")}"
  param_dir="$(resolve_path "${PARAM_DIR:-data/causal-forcing/params/$generation_mode/$parameter_name}")"
  result_label="$parameter_name"
  if [[ "$v_predictor_mode" == "temporal" ]]; then
    v_artifact_name="v_temporal.npz"
  else
    v_artifact_name="v_cross_kv.npz"
  fi
else
  param_dir=""
  result_label="bf16"
fi

output_dir="$(resolve_path "${OUTPUT_DIR:-results/causal_forcing/$generation_mode/$result_label}")"
if [[ "$output_dir" == "/" ]]; then
  die "OUTPUT_DIR 不能是文件系统根目录"
fi

FRAMES="${FRAMES:-81}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-0}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
NUM_GPUS="${NUM_GPUS:-1}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:--1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_TRITON_FALLBACK="${ALLOW_TRITON_FALLBACK:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_EMA="${USE_EMA:-0}"

check_flag DRY_RUN
check_flag ALLOW_TRITON_FALLBACK
check_flag SKIP_EXISTING
check_flag USE_EMA
check_int FRAMES 1
check_int NUM_SAMPLES 1
check_int SEED 0
check_int START_INDEX 0
check_int END_INDEX -1
check_int NUM_GPUS 1
check_int LOCAL_ATTN_SIZE -1

period=$(( 4 * num_frame_per_block ))
if (( (FRAMES + 3) % period != 0 )); then
  die "FRAMES 必须满足 (FRAMES + 3) % $period == 0，保证视频帧精确对应完整的 ${num_frame_per_block}-frame latent block（当前值：${FRAMES}）"
fi
if [[ -n "${K_HISTORY_LENGTH:-}" ]]; then
  check_int K_HISTORY_LENGTH 1
  (( K_HISTORY_LENGTH <= 8 )) || die "K_HISTORY_LENGTH 必须 <= 8（当前值：${K_HISTORY_LENGTH}）"
fi
if [[ -n "${V_HISTORY_LENGTH:-}" ]]; then
  check_int V_HISTORY_LENGTH 0
  (( V_HISTORY_LENGTH <= 5 )) || die "V_HISTORY_LENGTH 必须 <= 5（当前值：${V_HISTORY_LENGTH}）"
fi
if [[ -n "${SPAN_FRAMES:-}" ]]; then
  check_int SPAN_FRAMES 2
fi

generate_command=(
  "$python_bin" "$runtime_root/causal_forcing/tempokv_adapter/generate.py"
  --mode "$mode"
  --checkpoint "$(resolve_path "$CHECKPOINT")"
  --model-root "$(resolve_path "$MODEL_ROOT")"
  --model-config "$model_config"
  --prompts "$(resolve_path "$PROMPTS")"
  --output-dir "$output_dir"
  --frames "$FRAMES"
  --num-frame-per-block "$num_frame_per_block"
  --samples "$NUM_SAMPLES"
  --seed "$SEED"
  --start-index "$START_INDEX"
  --end-index "$END_INDEX"
  --num-gpus "$NUM_GPUS"
  --local-attn-size "$LOCAL_ATTN_SIZE"
)
if [[ "$mode" == "tempokv" ]]; then
  generate_command+=(
    --config "$(resolve_path "$CONFIG")"
    --k-params "$param_dir/k_temporal.npz"
    --v-params "$param_dir/$v_artifact_name"
  )
  [[ -n "${K_HISTORY_LENGTH:-}" ]] && generate_command+=(--k-history-length "$K_HISTORY_LENGTH")
  [[ -n "${V_HISTORY_LENGTH:-}" ]] && generate_command+=(--v-history-length "$V_HISTORY_LENGTH")
  [[ -n "${K_BITS:-}" ]] && generate_command+=(--k-bits "$K_BITS")
  [[ -n "${V_BITS:-}" ]] && generate_command+=(--v-bits "$V_BITS")
  generate_command+=(--v-predictor-mode "$v_predictor_mode")
  [[ -n "${SPAN_FRAMES:-}" ]] && generate_command+=(--span-frames "$SPAN_FRAMES")
fi

[[ -n "${TRITON_CACHE:-}" ]] && generate_command+=(--triton-cache "$(resolve_path "$TRITON_CACHE")")
[[ "$ALLOW_TRITON_FALLBACK" == "1" ]] && generate_command+=(--allow-triton-fallback)
[[ "$SKIP_EXISTING" == "1" ]] && generate_command+=(--skip-existing)
[[ "$USE_EMA" == "1" ]] && generate_command+=(--use-ema)
[[ "$DRY_RUN" == "1" ]] && generate_command+=(--dry-run)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[generate] 干跑模式：打印最终命令，不执行\n'
  printf '  $'
  printf ' %q' "${generate_command[@]}"
  printf '\n'
  "${generate_command[@]}"
  exit 0
fi

mkdir -p -- "$output_dir"
printf '[generate] 输出目录：%s\n' "$output_dir"
"${generate_command[@]}" 2>&1 | tee "$output_dir/generate.log"
