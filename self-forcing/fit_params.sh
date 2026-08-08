#!/usr/bin/env bash
#
# fit_params.sh — 只拟合 TempoKV 参数（读 trace，不采集、不合并）
#
# 读取 <TRACE_FILE>（collect_trace.sh 采集并合并出的 self_forcing_trace.npz），
# 调用 scripts/fit_params.py 拟合出 k_temporal.npz 与 v_cross_kv.npz。每个
# k_history_length / v_history_length 配置拟合一套专属参数，用
# --k-history-length / --v-history-length 显式指定，用不同 --output-dir 区分。
#
# 用法（所有参数都用环境变量传入）：
#   TRACE_FILE=parameters/self_forcing_trace.npz \
#   CONFIG=configs/int2.json \
#   K_HISTORY_LENGTH=2 \
#   V_HISTORY_LENGTH=2 \
#   OUTPUT_DIR=parameters/self_forcing/kh2-vh2-int2 \
#   bash self-forcing/fit_params.sh
#
# 必需变量：
#   CONFIG      codec 配置 JSON（如 configs/int2.json）
#   OUTPUT_DIR  参数输出目录（如 parameters/self_forcing/kh2-vh2-int2）
#
# 常用可选变量：
#   TRACE_FILE      拟合输入 trace（默认 parameters/self_forcing_trace.npz）
#   K_HISTORY_LENGTH  K 历史长度（默认读取 CONFIG）
#   V_HISTORY_LENGTH  V 历史长度（默认读取 CONFIG）
#   ITERATIONS     拟合迭代次数（默认 2）
#   TEMPORAL_RIDGE / CROSS_RIDGE / MAX_GAIN  拟合正则与增益（默认 0.1/0.1/1.0）
#   FIT_DEVICE     拟合设备（默认 auto，有 CUDA 时自动使用 GPU）
#   OVERWRITE      已存在参数时覆盖（0/1，默认 0）
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

require_env CONFIG OUTPUT_DIR

trace_file="$(resolve_path "${TRACE_FILE:-parameters/self_forcing_trace.npz}")"
config_path="$(resolve_path "$CONFIG")"
output_dir="$(resolve_path "$OUTPUT_DIR")"

if [[ "$output_dir" == "/" ]]; then
  die "OUTPUT_DIR 不能是文件系统根目录"
fi

ITERATIONS="${ITERATIONS:-2}"
TEMPORAL_RIDGE="${TEMPORAL_RIDGE:-0.1}"
CROSS_RIDGE="${CROSS_RIDGE:-0.1}"
MAX_GAIN="${MAX_GAIN:-1.0}"
FIT_DEVICE="${FIT_DEVICE:-auto}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

check_flag OVERWRITE
check_flag DRY_RUN
check_int ITERATIONS 1

fit_command=(
  "$python_bin" "$project_root/scripts/fit_params.py"
  "$trace_file"
  --config "$config_path"
  --output-dir "$output_dir"
  --iterations "$ITERATIONS"
  --temporal-ridge "$TEMPORAL_RIDGE"
  --cross-ridge "$CROSS_RIDGE"
  --max-gain "$MAX_GAIN"
  --device "$FIT_DEVICE"
  --stride 1560
  --coordinate-system pre_rope_normalized
)
[[ -n "${K_HISTORY_LENGTH:-}" ]] && fit_command+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && fit_command+=(--v-history-length "$V_HISTORY_LENGTH")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[fit_params] 干跑模式：打印将要执行的命令，不执行\n'
  printf '  $'
  printf ' %q' "${fit_command[@]}"
  printf '\n'
  exit 0
fi

[[ -f "$config_path" ]] || die "CONFIG 不存在：$config_path"
[[ -f "$trace_file" ]] || die "TRACE_FILE 不存在：$trace_file"

if [[ "$OVERWRITE" != "1" ]]; then
  for artifact in "$output_dir/k_temporal.npz" "$output_dir/v_cross_kv.npz"; do
    if [[ -e "$artifact" ]]; then
      die "已存在参数：$artifact。设置 OVERWRITE=1 覆盖，或换一个 OUTPUT_DIR。"
    fi
  done
fi

mkdir -p -- "$output_dir"

printf '\n[2/2] 拟合 TempoKV 参数 ...\n'
"${fit_command[@]}"

for artifact in "$output_dir/k_temporal.npz" "$output_dir/v_cross_kv.npz"; do
  [[ -s "$artifact" ]] || die "拟合后未生成参数文件：$artifact"
done

printf '\n完成。参数已写入：\n  %s\n  %s\n' \
  "$output_dir/k_temporal.npz" "$output_dir/v_cross_kv.npz"
