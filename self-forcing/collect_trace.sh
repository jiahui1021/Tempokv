#!/usr/bin/env bash
#
# collect_trace.sh — 采集 clean K/V trace 并合并成拟合输入
#
# 两步完成：
#   1. 调用外部 TRACE_RUNNER 采集未量化的逐层 trace
#      <WORK_DIR>/raw_trace/traces/   （layer_*.npz，逐层 trace）
#      <WORK_DIR>/raw_trace/output/   （采集器中间产物）
#   2. 调用 prepare_trace.py 把逐层 trace 合并成 <TRACE_FILE>
#      （默认 parameters/self_forcing_trace.npz，供 fit_params.sh 使用）
#
# 用法（所有参数都用环境变量传入）：
#   TRACE_RUNNER=/path/to/runner.sh \
#   CHECKPOINT=/models/self_forcing_dmd.pt \
#   WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/fit_prompts.txt \
#   WORK_DIR=runs/self_forcing_prepare \
#   bash self-forcing/collect_trace.sh
#
# 必需变量：
#   TRACE_RUNNER   外部 Self-Forcing 采集入口（SH 脚本）
#   CHECKPOINT     Self-Forcing 模型权重
#   WAN_MODEL_DIR  Wan 模型目录
#   PROMPTS        采集用 prompt 列表
#   WORK_DIR       中间产物目录（会生成 raw_trace/ 子目录）
#
# 常用可选变量：
#   TRACE_FILE     合并后的拟合输入（默认 parameters/self_forcing_trace.npz）
#   NUM_GPUS       采集用 GPU 数量（默认 1）
#   TRACE_FRAMES   每段生成的帧数（默认 51，至少 27）
#   TRACE_TOKENS_PER_FRAME  每帧抽取的 token 数（默认 8）
#   START_INDEX / END_INDEX  prompt 序号范围（默认 0 .. 8）
#   BASE_SEED      随机种子（默认 0）
#   SKIP_EXISTING  采集时跳过已存在的 trace（0/1，默认 0）
#   MIN_FREE_VRAM_GB  采集前要求的最少空闲显存 GB（默认 70）
#   OVERWRITE      已存在 trace 或 TRACE_FILE 时覆盖（0/1，默认 0）
#   DRY_RUN        只打印命令不执行（0/1，默认 0）
#   PYTHON_BIN     Python 解释器（默认 python3）
#
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      die "缺少必需变量 $name（请用环境变量传入）"
    fi
  done
}

resolve_path() {
  local value="$1"
  case "$value" in
    /*) printf '%s\n' "$value" ;;
    *)  printf '%s/%s\n' "${project_root%/}" "${value#./}" ;;
  esac
}

check_flag() {
  local name="$1" value="${!1:-0}"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name 必须是 0 或 1（当前值：$value）"
}

check_int() {
  local name="$1" min="$2" value="${!1:-}"
  if [[ "$value" =~ ^-?[0-9]+$ ]] && (( value >= min )); then
    return 0
  fi
  die "$name 必须是整数且 >= $min（当前值：$value）"
}

require_env TRACE_RUNNER CHECKPOINT WAN_MODEL_DIR PROMPTS WORK_DIR

trace_runner="$(resolve_path "$TRACE_RUNNER")"
trace_file="$(resolve_path "${TRACE_FILE:-parameters/self_forcing_trace.npz}")"

work_dir="$(resolve_path "$WORK_DIR")"
raw_trace_dir="$work_dir/raw_trace/traces"
trace_output_dir="$work_dir/raw_trace/output"

if [[ "$work_dir" == "/" ]]; then
  die "WORK_DIR 不能是文件系统根目录"
fi

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-8}"
TRACE_FRAMES="${TRACE_FRAMES:-51}"
TRACE_TOKENS_PER_FRAME="${TRACE_TOKENS_PER_FRAME:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_SEED="${BASE_SEED:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MIN_FREE_VRAM_GB="${MIN_FREE_VRAM_GB:-70}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

check_flag OVERWRITE
check_flag SKIP_EXISTING
check_flag DRY_RUN
check_int START_INDEX 0
check_int END_INDEX 0
check_int TRACE_FRAMES 27
check_int TRACE_TOKENS_PER_FRAME 1
check_int NUM_GPUS 1
check_int BASE_SEED 0

trace_command=(
  env
  "SELF_FORCING_CHECKPOINT_PATH=$(resolve_path "$CHECKPOINT")"
  "FORCING_WAN_MODEL_DIR=$(resolve_path "$WAN_MODEL_DIR")"
  "CROSS_KV_TRACE_DIR=$raw_trace_dir"
  "OUTPUT_FOLDER=$trace_output_dir"
  "PROMPTS_PATH=$(resolve_path "$PROMPTS")"
  "START_INDEX=$START_INDEX"
  "END_INDEX=$END_INDEX"
  "NUM_OUTPUT_FRAMES=$TRACE_FRAMES"
  "TRACE_LAYERS=all"
  "TRACE_HEADS=all"
  "TRACE_TOKENS_PER_FRAME=$TRACE_TOKENS_PER_FRAME"
  "TRACE_CAPTURE_HIDDEN=0"
  "TRACE_CAPTURE_ATTENTION=0"
  "TRACE_QUANT_TYPE=none"
  "NUM_GPUS=$NUM_GPUS"
  "PYTHON_BIN=$python_bin"
  "BASE_SEED=$BASE_SEED"
  "SKIP_EXISTING=$SKIP_EXISTING"
  "MIN_FREE_VRAM_GB=$MIN_FREE_VRAM_GB"
  bash "$trace_runner"
)

merge_command=(
  "$python_bin" "$script_dir/prepare_trace.py"
  "$raw_trace_dir" "$trace_file"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[collect_trace] 干跑模式：打印将要执行的命令，不执行\n'
  printf '  [1/2] '
  printf '%q ' "${trace_command[@]}"
  printf '\n'
  printf '  [2/2] '
  printf '%q ' "${merge_command[@]}"
  printf '\n'
  exit 0
fi

[[ -f "$trace_runner" ]] || die "TRACE_RUNNER 不存在：$trace_runner"

if [[ "$OVERWRITE" != "1" ]]; then
  if [[ -e "$trace_file" ]]; then
    die "已存在拟合输入：$trace_file。设置 OVERWRITE=1 覆盖，或换一个 TRACE_FILE。"
  fi
  if [[ -d "$raw_trace_dir" ]]; then
    existing="$(find "$raw_trace_dir" -type f -name 'layer_*.npz' -print -quit)"
    if [[ -n "$existing" ]]; then
      die "已存在逐层 trace：$existing。设置 OVERWRITE=1 覆盖，或换一个 WORK_DIR。"
    fi
  fi
fi

mkdir -p -- "$raw_trace_dir" "$trace_output_dir"

printf '\n[1/2] 采集 clean trace ...\n'
"${trace_command[@]}"

printf '\n[2/2] 合并逐层 trace -> %s ...\n' "$trace_file"
"${merge_command[@]}"
printf '\n完成。拟合输入已写入：\n  %s\n' "$trace_file"
