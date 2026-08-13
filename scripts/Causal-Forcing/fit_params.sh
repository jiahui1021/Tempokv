#!/usr/bin/env bash
#
# fit_params.sh — 只拟合 TempoKV 参数（读 trace，不采集、不合并）
#
# 读取 collect_trace.sh 合并出的 clean_trace.npz，调用通用的
# scripts/common/fit_params.py 拟合 k_temporal.npz 与 v_cross_kv.npz。
# Causal Forcing 不需要专用的拟合器。
#
# Self-Forcing 的参数不能直接复用：即使 30 层 / 12 头 / D=128 / stride=1560
# 完全相同，checkpoint、训练方式和 cache 分布都不同，必须重新采 trace 重新拟合。
#
# 用法：
#   CONFIG=configs/causal_forcing.json \
#   bash scripts/Causal-Forcing/fit_params.sh
#
# 必需变量：
#   CONFIG      codec 配置 JSON
#
# 常用可选变量：
#   GENERATION_MODE  chunkwise（默认）或 framewise
#   TRACE_FILE       拟合输入（默认 data/causal-forcing/trace/<模式>/clean_trace.npz）
#   OUTPUT_DIR       输出目录（默认 data/causal-forcing/params/<模式>/<参数方案>）
#   BITS / K_BITS / V_BITS  位宽 override（2/4/8；默认读取 CONFIG）
#   K_HISTORY_LENGTH / V_HISTORY_LENGTH  历史长度（默认读取 CONFIG）
#   ITERATIONS / TEMPORAL_RIDGE / CROSS_RIDGE / V_HISTORY_RIDGE / MAX_GAIN
#   FIT_DEVICE
#   OVERWRITE / DRY_RUN / PYTHON_BIN
#
set -euo pipefail

# shellcheck source=scripts/Causal-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_HISTORY_LENGTH="$(resolve_history K_HISTORY_LENGTH KH)"
V_HISTORY_LENGTH="$(resolve_history V_HISTORY_LENGTH VH)"
K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"

require_env CONFIG

generation_mode="${GENERATION_MODE:-chunkwise}"
case "$generation_mode" in
  chunkwise|framewise) ;;
  *) die "GENERATION_MODE 必须是 chunkwise 或 framewise（当前值：${generation_mode}）" ;;
esac

trace_file="$(resolve_path "${TRACE_FILE:-data/causal-forcing/trace/$generation_mode/clean_trace.npz}")"
config_path="$(resolve_path "$CONFIG")"
parameter_name="${PARAM_NAME:-$(codec_slug "$config_path" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" "${K_BITS:-}" "${V_BITS:-}" "" "${V_PREDICTOR:-}")}"
output_dir="$(resolve_path "${OUTPUT_DIR:-data/causal-forcing/params/$generation_mode/$parameter_name}")"
# One interpreter launch for both switches the rest of this script needs.
read -r fit_mode v_predictor_mode <<<"$("$python_bin" - "$config_path" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
config = config.get("codec", config)
print(config.get("fit_mode", "native"), config.get("v_predictor_mode", "cross_kv"))
PY
)"
# V_PREDICTOR overrides the config so one file covers both predictors.
if [[ -n "${V_PREDICTOR:-}" ]]; then
  case "$V_PREDICTOR" in
    cross_kv|temporal) ;;
    *) die "V_PREDICTOR 必须是 cross_kv 或 temporal（当前值：${V_PREDICTOR}）" ;;
  esac
  v_predictor_mode="$V_PREDICTOR"
fi
if [[ "$v_predictor_mode" == "temporal" ]]; then
  v_artifact_name="v_temporal.npz"
else
  v_artifact_name="v_cross_kv.npz"
fi

if [[ "$output_dir" == "/" ]]; then
  die "OUTPUT_DIR 不能是文件系统根目录"
fi

ITERATIONS="${ITERATIONS:-2}"
# Legacy fitting reproduces the reference implementation and pins its controls;
# fit_params.py rejects any other value, so the default has to follow fit_mode.
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
  --max-gain "$MAX_GAIN"
  --device "$FIT_DEVICE"
  --stride 1560
  --coordinate-system pre_rope_normalized
  # The runtime CrossKV decoder only accelerates a diagonal V history; a dense
  # per-lag matrix is refused by the bridge, so it must never be fitted here.
  --v-history-mode diagonal
)
[[ -n "${K_HISTORY_LENGTH:-}" ]] && fit_command+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && fit_command+=(--v-history-length "$V_HISTORY_LENGTH")
[[ -n "${K_BITS:-}" ]] && fit_command+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && fit_command+=(--v-bits "$V_BITS")
[[ -n "${V_PREDICTOR:-}" ]] && fit_command+=(--v-predictor-mode "$V_PREDICTOR")

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
  for artifact in "$output_dir/k_temporal.npz" "$output_dir/$v_artifact_name"; do
    if [[ -e "$artifact" ]]; then
      die "已存在参数：${artifact}。设置 OVERWRITE=1 覆盖，或换一个 OUTPUT_DIR。"
    fi
  done
fi

mkdir -p -- "$output_dir"

printf '\n拟合 TempoKV 参数 ...\n'
"${fit_command[@]}"

for artifact in "$output_dir/k_temporal.npz" "$output_dir/$v_artifact_name"; do
  [[ -s "$artifact" ]] || die "拟合后未生成参数文件：$artifact"
done

printf '\n完成。参数已写入：\n  %s\n  %s\n' \
  "$output_dir/k_temporal.npz" "$output_dir/$v_artifact_name"
