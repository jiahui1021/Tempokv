#!/usr/bin/env bash
#
# fit_params.sh — 只拟合 TempoKV 参数（读 trace，不采集、不合并）
#
# 读取 collect_trace.sh 合并出的 clean_trace.npz，调用通用的
# scripts/common/fit_params.py 拟合 k_temporal.npz 与 v_cross_kv.npz。
#
# 输出目录默认从 CONFIG 推导（与 Self-Forcing、Causal Forcing 一致），因此
# INT4 与 INT2 不会互相覆盖。
#
# 用法：
#   CONFIG=configs/longcat_int4.json \
#   bash scripts/LongCat/fit_params.sh
#
# 常用可选变量：
#   TRACE_FILE / TRACE  拟合输入（默认 data/longcat/trace/clean_trace.npz）
#   OUTPUT_DIR          输出目录（默认 data/longcat/params/<参数方案>）
#   K_HISTORY_LENGTH / V_HISTORY_LENGTH  历史长度（默认读取 CONFIG），可简写 KH / VH
#   PARAM_NAME          直接指定参数方案目录名
#   FIT_DEVICE          拟合设备（默认 auto：有 CUDA 就用 GPU，否则回退 CPU 并告警）
#   DRY_RUN             只打印命令不执行（0/1，默认 0）
#   ITERATIONS / TEMPORAL_RIDGE / CROSS_RIDGE / MAX_GAIN  拟合控制
#
# 其余参数原样透传给 scripts/common/fit_params.py。
#
set -euo pipefail

# shellcheck source=scripts/LongCat/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_HISTORY_LENGTH="$(resolve_history K_HISTORY_LENGTH KH)"
V_HISTORY_LENGTH="$(resolve_history V_HISTORY_LENGTH VH)"

# collect_trace.sh 写出的是 TRACE_FILE；TRACE 是本脚本的旧名字，继续支持。
trace="$(resolve_path "${TRACE_FILE:-${TRACE:-data/longcat/trace/clean_trace.npz}}")"
config="$(resolve_path "${CONFIG:-configs/longcat_int4.json}")"
parameter_name="${PARAM_NAME:-$(codec_slug "$config" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}")}"
output_dir="$(resolve_path "${OUTPUT_DIR:-data/longcat/params/$parameter_name}")"

if [[ "$output_dir" == "/" ]]; then
  die "OUTPUT_DIR 不能是文件系统根目录"
fi
[[ -f "$config" ]] || die "CONFIG 不存在：$config"

FIT_DEVICE="${FIT_DEVICE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
check_flag DRY_RUN
ITERATIONS="${ITERATIONS:-2}"
TEMPORAL_RIDGE="${TEMPORAL_RIDGE:-0.1}"
CROSS_RIDGE="${CROSS_RIDGE:-0.1}"
MAX_GAIN="${MAX_GAIN:-1.0}"
check_int ITERATIONS 1

fit_command=(
  "$python_bin" "$project_root/scripts/common/fit_params.py"
  "$trace"
  --config "$config"
  --output-dir "$output_dir"
  --coordinate-system pre_rope_normalized
  --iterations "$ITERATIONS"
  --temporal-ridge "$TEMPORAL_RIDGE"
  --cross-ridge "$CROSS_RIDGE"
  --max-gain "$MAX_GAIN"
  --device "$FIT_DEVICE"
)
[[ -n "${K_HISTORY_LENGTH:-}" ]] && fit_command+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && fit_command+=(--v-history-length "$V_HISTORY_LENGTH")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[fit_params] 干跑模式：打印将要执行的命令，不执行\n'
  printf '  $'
  printf ' %q' "${fit_command[@]}" "$@"
  printf '\n'
  exit 0
fi

[[ -f "$trace" ]] || die "TRACE_FILE 不存在：$trace（先跑 collect_trace.sh）"

exec "${fit_command[@]}" "$@"
