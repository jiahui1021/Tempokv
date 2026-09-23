#!/usr/bin/env bash
#
# fit_params.sh — 只拟合 TempoKV 参数（读 trace，不采集、不合并）
#
# 读取 collect_trace.sh 合并出的 clean_trace.npz，调用通用的
# scripts/common/fit_params.py 拟合 k_temporal.npz 与 V 参数
#（v_cross_kv.npz、v_innovation.npz 或 v_temporal.npz，取决于 v_predictor_mode）。
#
# 输出目录默认从 CONFIG 推导（与 Self-Forcing、Causal Forcing 一致），因此
# INT4 与 INT2 不会互相覆盖。
#
# 用法：
#   CONFIG=configs/legacy.json \
#   bash scripts/LongCat/fit_params.sh
#
# 常用可选变量：
#   CONFIG          codec 配置 JSON（默认 configs/legacy.json）
#   TRACE_FILE / TRACE  拟合输入（默认 data/longcat/trace/clean_trace.npz）
#   OUTPUT_DIR          输出目录（默认 data/longcat/params/<参数方案>）
#   BITS / K_BITS / V_BITS  K/V residual 位宽 override（1 到 8；默认读取 CONFIG）
#   ANCHOR_BITS     anchor 位宽（1 到 15 量化，16 表示不量化直接存 bf16；默认 INT4）
#   K_HISTORY_LENGTH / V_HISTORY_LENGTH  历史长度（默认读取 CONFIG），可简写 KH / VH
#   PARAM_NAME          直接指定参数方案目录名
#   FIT_DEVICE          拟合设备（默认 auto：有 CUDA 就用 GPU，否则回退 CPU 并告警）
#   FIT_HISTORY_SOURCE  reconstructed（默认）或 clean；仅用于拟合消融
#   DRY_RUN             只打印命令不执行（0/1，默认 0）
#   OVERWRITE           已存在参数时覆盖（0/1，默认 0）
#   ITERATIONS / TEMPORAL_RIDGE / CROSS_RIDGE / MAX_GAIN  拟合控制
#
# 其余参数原样透传给 scripts/common/fit_params.py。
#
set -euo pipefail

# shellcheck source=scripts/LongCat/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_HISTORY_LENGTH="$(resolve_history K_HISTORY_LENGTH KH)"
V_HISTORY_LENGTH="$(resolve_history V_HISTORY_LENGTH VH)"
K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"

# collect_trace.sh 写出的是 TRACE_FILE；TRACE 是本脚本的旧名字，继续支持。
trace="$(resolve_path "${TRACE_FILE:-${TRACE:-data/longcat/trace/clean_trace.npz}}")"
config="$(resolve_path "${CONFIG:-$default_codec_config}")"
FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
parameter_name="${PARAM_NAME:-$(codec_slug "$config" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" "${K_BITS:-}" "${V_BITS:-}" "${CACHE_STRIDE:-}" "${V_PREDICTOR:-}" "$FIT_HISTORY_SOURCE" "${ANCHOR_BITS:-}")}"
output_dir="$(resolve_path "${OUTPUT_DIR:-data/longcat/params/$parameter_name}")"
read -r fit_mode v_predictor_mode _ <<<"$(codec_modes "$config" "${V_PREDICTOR:-}")"
v_artifact_name="$(v_artifact_name "$v_predictor_mode")"
# Legacy fitting reproduces the reference implementation and pins its ridge;
# fit_params.py rejects any other value, so the default follows fit_mode.
if [[ "$fit_mode" == "legacy" ]]; then
  default_temporal_ridge=1.0
else
  default_temporal_ridge=0.1
fi

DRY_RUN="${DRY_RUN:-0}"
ITERATIONS="${ITERATIONS:-2}"
check_flag DRY_RUN
check_flag OVERWRITE
check_int ITERATIONS 1

fit_command=(
  "$python_bin" "$project_root/scripts/common/fit_params.py"
  "$trace"
  --config "$config"
  --output-dir "$output_dir"
  --coordinate-system pre_rope_normalized
  --iterations "$ITERATIONS"
  --fit-history-source "$FIT_HISTORY_SOURCE"
  --temporal-ridge "${TEMPORAL_RIDGE:-$default_temporal_ridge}"
  --cross-ridge "${CROSS_RIDGE:-0.1}"
  --max-gain "${MAX_GAIN:-1.0}"
  --device "${FIT_DEVICE:-auto}"
)
[[ -n "${K_HISTORY_LENGTH:-}" ]] && fit_command+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && fit_command+=(--v-history-length "$V_HISTORY_LENGTH")
[[ -n "${K_BITS:-}" ]] && fit_command+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && fit_command+=(--v-bits "$V_BITS")
[[ -n "${ANCHOR_BITS:-}" ]] && fit_command+=(--anchor-bits "$ANCHOR_BITS")
[[ -n "${CACHE_STRIDE:-}" ]] && fit_command+=(--stride "$CACHE_STRIDE")
[[ -n "${V_PREDICTOR:-}" ]] && fit_command+=(--v-predictor-mode "$V_PREDICTOR")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[fit_params] 干跑模式：打印将要执行的命令，不执行\n'
  print_command "${fit_command[@]}" "$@"
  exit 0
fi

[[ -f "$trace" ]] || die "TRACE_FILE 不存在：${trace}（先跑 collect_trace.sh）"
refuse_existing_params "$output_dir/k_temporal.npz" "$output_dir/$v_artifact_name"
mkdir -p -- "$output_dir"
"${fit_command[@]}" "$@"
require_params_written "$output_dir/k_temporal.npz" "$output_dir/$v_artifact_name"
