#!/usr/bin/env bash
#
# generate.sh — 读取已拟合参数，生成 Self-Forcing 视频
#
# 不采集 trace、不重新拟合，只使用 PARAM_DIR 下的参数。更换 PROMPTS /
# OUTPUT_DIR 后可以重复运行。
#
# 两种模式：
#   MODE=tempokv  使用 TempoKV 量化参数（默认），需要 CONFIG + PARAM_DIR
#   MODE=bf16     不量化的对照基线，不需要 CONFIG / PARAM_DIR
#
# 用法（所有参数都用环境变量传入）：
#   CHECKPOINT=/models/self_forcing_dmd.pt \
#   WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/generation_prompts.txt \
#   CONFIG=configs/int2.json \
#   K_HISTORY_LENGTH=2 \
#   V_HISTORY_LENGTH=2 \
#   FRAMES=717 \
#   bash scripts/Self-Forcing/generate.sh
#
# 必需变量：
#   CHECKPOINT    Self-Forcing 模型权重
#   WAN_MODEL_DIR Wan 模型目录
#   PROMPTS       prompt 列表
#   CONFIG        codec 配置 JSON（仅 MODE=tempokv）
#
# 常用可选变量：
#   PARAM_DIR       参数目录（默认 data/self-forcing/params/<参数方案>）
#   OUTPUT_DIR      结果输出目录（默认 results/$MODEL_TYPE/<量化标签>）
#   FRAMES          最终视频帧数（两种模式默认均为 717）
#   NUM_SAMPLES     每条 prompt 采样数（默认 1）
#   SEED            随机种子（默认 0）
#   START_INDEX / END_INDEX  prompt 序号范围（默认 0 .. -1 全部）
#   NUM_GPUS        torchrun 使用的 GPU 数（默认 1）
#   K_HISTORY_LENGTH  K 使用的历史帧数（1 到 8；默认读取 CONFIG）
#   V_HISTORY_LENGTH  V 使用的历史帧数（0 到 2；默认读取 CONFIG）
#   BENCHMARK       要求本次完整生成且不允许跳过已有视频，0 或 1（默认 0）
#   TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH  单次持续解码帧数（默认 16；范围 1 到 16）
#
set -euo pipefail

# shellcheck source=scripts/Self-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
script_dir="$self_forcing_dir"
mode="${MODE:-tempokv}"

# ---------- 模式与必需变量 ----------

case "$mode" in
  tempokv) ;;
  bf16) ;;
  *) die "MODE 必须是 tempokv 或 bf16（当前值：$mode）" ;;
esac

require_env CHECKPOINT WAN_MODEL_DIR PROMPTS
if [[ "$mode" == "tempokv" ]]; then
  require_env CONFIG
fi

# ---------- 目录 ----------

if [[ "$mode" == "tempokv" ]]; then
  parameter_name="${PARAM_NAME:-$(codec_slug "$(resolve_path "$CONFIG")" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}")}"
  param_dir="$(resolve_path "${PARAM_DIR:-data/self-forcing/params/$parameter_name}")"
  result_label="$parameter_name"
else
  param_dir=""
  result_label="bf16"
fi

model_type="${MODEL_TYPE:-self_forcing}"
model_type="${model_type%/}"
output_dir="$(resolve_path "${OUTPUT_DIR:-$project_root/results/$model_type/$result_label}")"
if [[ "$output_dir" == "/" ]]; then
  die "OUTPUT_DIR 不能是文件系统根目录"
fi

# ---------- 生成参数 ----------

FRAMES="${FRAMES:-717}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-0}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_TRITON_FALLBACK="${ALLOW_TRITON_FALLBACK:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BENCHMARK="${BENCHMARK:-0}"

check_flag DRY_RUN
check_flag ALLOW_TRITON_FALLBACK
check_flag SKIP_EXISTING
check_flag BENCHMARK
check_int FRAMES 1
check_int NUM_SAMPLES 1
check_int SEED 0
check_int START_INDEX 0
check_int END_INDEX -1
check_int NUM_GPUS 1
if [[ "$BENCHMARK" == "1" && "$SKIP_EXISTING" == "1" ]]; then
  die "BENCHMARK=1 不允许 SKIP_EXISTING=1"
fi
if (( (FRAMES + 3) % 12 != 0 )); then
  die "FRAMES 必须满足 (FRAMES + 3) % 12 == 0，保证视频帧精确对应完整的 3-frame latent block（当前值：${FRAMES}）"
fi
if [[ -n "${K_HISTORY_LENGTH:-}" ]]; then
  check_int K_HISTORY_LENGTH 1
  (( K_HISTORY_LENGTH <= 8 )) || die "K_HISTORY_LENGTH 必须 <= 8（当前值：$K_HISTORY_LENGTH）"
fi
if [[ -n "${V_HISTORY_LENGTH:-}" ]]; then
  check_int V_HISTORY_LENGTH 0
  (( V_HISTORY_LENGTH <= 2 )) || die "V_HISTORY_LENGTH 必须 <= 2（当前值：$V_HISTORY_LENGTH）"
fi

latent_frames=$(( (FRAMES + 3) / 4 ))
if [[ "$mode" == "tempokv" && "$latent_frames" -lt 27 ]]; then
  die "tempokv 模式至少需要 27 个 latent 帧（约 105 个视频帧）才能触发量化，当前 FRAMES=$FRAMES"
fi

# ---------- 组装命令 ----------

generate_command=(
  "$python_bin" "$project_root/experiments/Self-Forcing/tempokv_adapter/generate.py"
  --mode "$mode"
  --checkpoint "$(resolve_path "$CHECKPOINT")"
  --prompts "$(resolve_path "$PROMPTS")"
  --output-dir "$output_dir"
  --frames "$FRAMES"
  --samples "$NUM_SAMPLES"
  --seed "$SEED"
  --start-index "$START_INDEX"
  --end-index "$END_INDEX"
  --num-gpus "$NUM_GPUS"
)
if [[ "$mode" == "tempokv" ]]; then
  generate_command+=(
    --config "$(resolve_path "$CONFIG")"
    --k-params "$param_dir/k_temporal.npz"
    --v-params "$param_dir/v_cross_kv.npz"
  )
  [[ -n "${K_HISTORY_LENGTH:-}" ]] && generate_command+=(--k-history-length "$K_HISTORY_LENGTH")
  [[ -n "${V_HISTORY_LENGTH:-}" ]] && generate_command+=(--v-history-length "$V_HISTORY_LENGTH")
fi

[[ -n "${MODEL_CONFIG:-}" ]] && generate_command+=(--model-config "$(resolve_path "$MODEL_CONFIG")")
[[ -n "${LOCAL_ATTN_SIZE:-}" ]] && generate_command+=(--local-attn-size "$LOCAL_ATTN_SIZE")
[[ -n "${TRITON_CACHE:-}" ]] && generate_command+=(--triton-cache "$(resolve_path "$TRITON_CACHE")")
[[ "$ALLOW_TRITON_FALLBACK" == "1" ]] && generate_command+=(--allow-triton-fallback)
[[ "$SKIP_EXISTING" == "1" ]] && generate_command+=(--skip-existing)
[[ "$BENCHMARK" == "1" ]] && generate_command+=(--benchmark)
[[ "$DRY_RUN" == "1" ]] && generate_command+=(--dry-run)

export FORCING_WAN_MODEL_DIR="$(resolve_path "$WAN_MODEL_DIR")"
if [[ "$DRY_RUN" != "1" ]]; then
  [[ -d "$FORCING_WAN_MODEL_DIR" ]] || \
    die "WAN_MODEL_DIR 不存在：$FORCING_WAN_MODEL_DIR"
fi

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
