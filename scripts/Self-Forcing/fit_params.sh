#!/usr/bin/env bash
#
# fit_params.sh — 只拟合 TempoKV 参数（读 trace，不采集、不合并）
#
# 读取 <TRACE_FILE>（collect_trace.sh 采集并合并出的 self_forcing_trace.npz），
# 调用 scripts/common/fit_params.py 拟合 k_temporal.npz 与 v_cross_kv.npz。每个
# k_history_length / v_history_length 配置拟合一套专属参数，用
# --k-history-length / --v-history-length 显式指定，用不同 --output-dir 区分。
#
# 用法（所有参数都用环境变量传入）：
#   CONFIG=configs/int2.json \
#   KH=2 \
#   VH=1 \
#   bash scripts/Self-Forcing/fit_params.sh
#
# 必需变量：
#   CONFIG      codec 配置 JSON（如 configs/int2.json）
#
# 常用可选变量：
#   TRACE_FILE      拟合输入（默认 data/self-forcing/trace/clean_trace.npz）
#   OUTPUT_DIR      输出目录（默认 data/self-forcing/params/<参数方案>）
#   K_HISTORY_LENGTH  K 历史长度（默认读取 CONFIG）
#   V_HISTORY_LENGTH  V 历史长度（默认读取 CONFIG）
#   ITERATIONS     拟合迭代次数（默认 2）
#   TEMPORAL_RIDGE / CROSS_RIDGE / V_HISTORY_RIDGE / MAX_GAIN
#                  拟合正则与增益（legacy 模式默认 1.0/0.1/0.0/1.0）
#   FIT_DEVICE     拟合设备（默认 auto，有 CUDA 时自动使用 GPU）
#   OVERWRITE      已存在参数时覆盖（0/1，默认 0）
#   DRY_RUN        只打印命令不执行（0/1，默认 0）
#   PYTHON_BIN     Python 解释器（默认 python3）
#
set -euo pipefail

# shellcheck source=scripts/Self-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_HISTORY_LENGTH="$(resolve_history K_HISTORY_LENGTH KH)"
V_HISTORY_LENGTH="$(resolve_history V_HISTORY_LENGTH VH)"

require_env CONFIG

trace_file="$(resolve_path "${TRACE_FILE:-data/self-forcing/trace/clean_trace.npz}")"
config_path="$(resolve_path "$CONFIG")"
parameter_name="${PARAM_NAME:-$(codec_slug "$config_path" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}")}"
output_dir="$(resolve_path "${OUTPUT_DIR:-data/self-forcing/params/$parameter_name}")"

if [[ "$output_dir" == "/" ]]; then
  die "OUTPUT_DIR 不能是文件系统根目录"
fi

ITERATIONS="${ITERATIONS:-2}"
fit_mode="$("$python_bin" - "$config_path" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config.get("codec", config).get("fit_mode", "native"))
PY
)"
if [[ "$fit_mode" == "legacy" ]]; then
  default_temporal_ridge=1.0
else
  default_temporal_ridge=0.1
fi
TEMPORAL_RIDGE="${TEMPORAL_RIDGE:-$default_temporal_ridge}"
CROSS_RIDGE="${CROSS_RIDGE:-0.1}"
V_HISTORY_RIDGE="${V_HISTORY_RIDGE:-0.0}"
MAX_GAIN="${MAX_GAIN:-1.0}"
FIT_DEVICE="${FIT_DEVICE:-auto}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

check_flag OVERWRITE
check_flag DRY_RUN
check_int ITERATIONS 1

fit_command=(
  "$python_bin" "$project_root/scripts/common/fit_params.py"
  "$trace_file"
  --config "$config_path"
  --output-dir "$output_dir"
  --iterations "$ITERATIONS"
  --temporal-ridge "$TEMPORAL_RIDGE"
  --cross-ridge "$CROSS_RIDGE"
  --v-history-ridge "$V_HISTORY_RIDGE"
  --v-history-mode diagonal
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
