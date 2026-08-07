#!/usr/bin/env bash
#
# prepare.sh — 一次性采集 clean K/V trace 并拟合 TempoKV 参数
#
# 执行流程（只需运行一次）：
#   1. 运行外部 TRACE_RUNNER，采集未量化的 clean trace
#   2. prepare_trace.py 把逐层 trace 合并成 self_forcing_trace.npz
#   3. fit_params.py 拟合出 k_temporal.npz 与 v_cross_kv.npz
#
# 用法（所有参数都用环境变量传入）：
#   TRACE_RUNNER=/path/to/runner.sh \
#   CHECKPOINT=/models/self_forcing_dmd.pt \
#   WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/fit_prompts.txt \
#   WORK_DIR=runs/self_forcing_prepare \
#   CONFIG=configs/int2.json \
#   bash self-forcing/prepare.sh
#
# 必需变量：
#   TRACE_RUNNER   外部 Self-Forcing 采集入口（SH 脚本）
#   CHECKPOINT     Self-Forcing 模型权重
#   WAN_MODEL_DIR  Wan 模型目录
#   PROMPTS        采集用 prompt 列表
#   WORK_DIR       中间产物目录（会生成 raw_trace/ 子目录）
#   CONFIG         codec 配置 JSON（如 configs/int2.json）
#
# 常用可选变量：
#   PARAM_DIR      参数输出目录（默认 parameters/，相对仓库根目录）
#   NUM_GPUS       采集用 GPU 数量（默认 1）
#   TRACE_FRAMES   每段生成的帧数（默认 51，至少 27）
#   TRACE_TOKENS_PER_FRAME  每帧抽取的 token 数（默认 8）
#   START_INDEX / END_INDEX  prompt 序号范围（默认 0 .. 8）
#   BASE_SEED      随机种子（默认 0）
#   ITERATIONS     拟合迭代次数（默认 2）
#   TEMPORAL_RIDGE / CROSS_RIDGE / MAX_GAIN  拟合正则与增益（默认 0.1/0.1/1.0）
#   OVERWRITE      已存在产物时覆盖（0/1，默认 0）
#   SKIP_EXISTING  采集时跳过已存在的 trace（0/1，默认 0）
#   DRY_RUN        只打印命令不执行（0/1，默认 0）
#   PYTHON_BIN     Python 解释器（默认 python3）
#
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"

# ---------- 工具函数 ----------

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

# 相对路径统一按“仓库根目录”解析，保证从任何目录运行行为一致
resolve_path() {
  local value="$1"
  case "$value" in
    /*) printf '%s\n' "$value" ;;
    *)  realpath -m -- "$project_root/$value" ;;
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

# ---------- 必需变量 ----------

require_env TRACE_RUNNER CHECKPOINT WAN_MODEL_DIR PROMPTS WORK_DIR CONFIG

trace_runner="$(resolve_path "$TRACE_RUNNER")"

# ---------- 目录 ----------

work_dir="$(resolve_path "$WORK_DIR")"
raw_trace_dir="$work_dir/raw_trace/traces"     # 采集器写入逐层 trace
trace_output_dir="$work_dir/raw_trace/output"  # 采集器写入视频等产物
param_dir="$(resolve_path "${PARAM_DIR:-parameters}")"
trace_file="$param_dir/self_forcing_trace.npz" # 合并后的拟合输入

if [[ "$work_dir" == "/" || "$param_dir" == "/" ]]; then
  die "WORK_DIR 和 PARAM_DIR 不能是文件系统根目录"
fi

# ---------- 采集与拟合参数 ----------

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-8}"
TRACE_FRAMES="${TRACE_FRAMES:-51}"
TRACE_TOKENS_PER_FRAME="${TRACE_TOKENS_PER_FRAME:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_SEED="${BASE_SEED:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MIN_FREE_VRAM_GB="${MIN_FREE_VRAM_GB:-70}"
ITERATIONS="${ITERATIONS:-2}"
TEMPORAL_RIDGE="${TEMPORAL_RIDGE:-0.1}"
CROSS_RIDGE="${CROSS_RIDGE:-0.1}"
MAX_GAIN="${MAX_GAIN:-1.0}"
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
check_int ITERATIONS 1

# ---------- 组装三条命令 ----------

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

fit_command=(
  "$python_bin" "$project_root/scripts/fit_params.py"
  "$trace_file"
  --config "$(resolve_path "$CONFIG")"
  --output-dir "$param_dir"
  --iterations "$ITERATIONS"
  --temporal-ridge "$TEMPORAL_RIDGE"
  --cross-ridge "$CROSS_RIDGE"
  --max-gain "$MAX_GAIN"
  --stride 1560
  --coordinate-system pre_rope_normalized
)

print_command() {
  printf '  $'
  printf ' %q' "$@"
  printf '\n'
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[prepare] 干跑模式：打印将要执行的命令，不执行\n'
  print_command "${trace_command[@]}"
  print_command "${merge_command[@]}"
  print_command "${fit_command[@]}"
  exit 0
fi

# 干跑模式不检查文件存在，便于先预览命令；真实执行前再校验
[[ -f "$trace_runner" ]] || die "TRACE_RUNNER 不存在：$trace_runner"
[[ -f "$CONFIG" ]] || die "CONFIG 不存在：$CONFIG"

# 防止意外覆盖已有参数
existing=""
if [[ "$OVERWRITE" != "1" ]]; then
  for artifact in "$trace_file" "$param_dir/k_temporal.npz" "$param_dir/v_cross_kv.npz"; do
    if [[ -e "$artifact" ]]; then
      existing="$artifact"
      break
    fi
  done
  if [[ -z "$existing" && -d "$raw_trace_dir" ]]; then
    existing="$(find "$raw_trace_dir" -type f -name 'layer_*.npz' -print -quit)"
  fi
  if [[ -n "$existing" ]]; then
    die "已存在 trace 或参数：$existing。设置 OVERWRITE=1 覆盖，或换一个 WORK_DIR/PARAM_DIR。"
  fi
fi

mkdir -p -- "$param_dir" "$raw_trace_dir" "$trace_output_dir"

printf '\n[1/3] 采集 clean trace ...\n'
"${trace_command[@]}"

printf '\n[2/3] 合并逐层 trace -> %s\n' "$trace_file"
"${merge_command[@]}"

printf '\n[3/3] 拟合 TempoKV 参数 ...\n'
"${fit_command[@]}"

for artifact in "$param_dir/k_temporal.npz" "$param_dir/v_cross_kv.npz"; do
  [[ -s "$artifact" ]] || die "拟合后未生成参数文件：$artifact"
done

printf '\n完成。参数已写入：\n  %s\n  %s\n' \
  "$param_dir/k_temporal.npz" "$param_dir/v_cross_kv.npz"
