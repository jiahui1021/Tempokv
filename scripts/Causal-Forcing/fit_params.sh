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
#   CONFIG=configs/legacy.json \
#   bash scripts/Causal-Forcing/fit_params.sh
#
# 常用可选变量：
#   CONFIG          codec 配置 JSON（默认 configs/legacy.json）
#   GENERATION_MODE  chunkwise（默认）或 framewise
#   TRACE_FILE       拟合输入（默认使用 legacy trace 目录）
#   OUTPUT_DIR       输出目录（默认 data/causal-forcing/params/<模式>/<参数方案>）
#   BITS / K_BITS / V_BITS  K/V residual 位宽 override（1 到 8；默认读取 CONFIG）
#   ANCHOR_BITS     anchor 位宽 override（1 到 16；默认读取 CONFIG，即 INT4）
#   K_HISTORY_LENGTH / V_HISTORY_LENGTH  历史长度（默认读取 CONFIG）
#   ITERATIONS / TEMPORAL_RIDGE / CROSS_RIDGE / V_HISTORY_RIDGE / MAX_GAIN
#   FIT_HISTORY_SOURCE  reconstructed（默认）或 clean；仅用于拟合消融
#   ATTENTION_STEPS  每个 K/V phase 的最大 Adam 步数（留空则不精调）
#   ATTENTION_LR / ATTENTION_PATIENCE / ATTENTION_MIN_RELATIVE_IMPROVEMENT
#                    Adam 学习率、早停 patience 与最小相对改善
#   ATTENTION_OPTIMIZER  adam（默认）或 sign_gradient
#   ATTENTION_RECONSTRUCTION_WEIGHT / ATTENTION_OUTPUT_WEIGHT  objective 权重
#   ATTENTION_REFINE_TENSORS  key、value 或 key_value（CrossKV 默认 key；temporal V 默认 key_value）
#   ATTENTION_SIGN_STEP_SIZES  sign-gradient 候选步长，空格分隔
#   ATTENTION_MAX_TOKENS  refinement 使用的 sampled token-track 上限
#   ATTENTION_LAYER_CHUNK  refinement 每批层数（默认 4）
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
CONFIG="${CONFIG:-$default_codec_config}"

generation_mode="${GENERATION_MODE:-chunkwise}"
case "$generation_mode" in
  chunkwise|framewise) ;;
  *) die "GENERATION_MODE 必须是 chunkwise 或 framewise（当前值：${generation_mode}）" ;;
esac

config_path="$(resolve_path "$CONFIG")"
[[ -f "$config_path" ]] || die "CONFIG 不存在：$config_path"
read -r fit_mode v_predictor_mode <<<"$(resolve_codec_modes "$config_path" "${V_PREDICTOR:-}")"
# The legacy protocol collects into its own directory, so the default trace
# file follows the config's fit mode.
if [[ "$fit_mode" == "legacy" ]]; then
  default_trace_file="data/causal-forcing/trace/${generation_mode}_legacy/clean_trace.npz"
else
  default_trace_file="data/causal-forcing/trace/$generation_mode/clean_trace.npz"
fi
trace_file="$(resolve_path "${TRACE_FILE:-$default_trace_file}")"
FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
parameter_name="${PARAM_NAME:-$(codec_slug "$config_path" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" "${K_BITS:-}" "${V_BITS:-}" "" "$v_predictor_mode" "$FIT_HISTORY_SOURCE" "${ANCHOR_BITS:-}")}"
output_dir="$(resolve_path "${OUTPUT_DIR:-data/causal-forcing/params/$generation_mode/$parameter_name}")"
case "$v_predictor_mode" in
  temporal|identity|taylor|taylor2|direct) v_artifact_name="v_temporal.npz" ;;
  cross_kv_identity|cross_kv_taylor|cross_kv_taylor2) v_artifact_name="v_cross_kv.npz" ;;
  innovation) v_artifact_name="v_innovation.npz" ;;
  cross_kv) v_artifact_name="v_cross_kv.npz" ;;
  *) v_artifact_name="v_cross_kv.npz" ;;
esac

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
ATTENTION_LAYER_CHUNK="${ATTENTION_LAYER_CHUNK:-4}"
ATTENTION_LR="${ATTENTION_LR:-0.001}"
ATTENTION_PATIENCE="${ATTENTION_PATIENCE:-10}"
ATTENTION_MIN_RELATIVE_IMPROVEMENT="${ATTENTION_MIN_RELATIVE_IMPROVEMENT:-0.0001}"
ATTENTION_OPTIMIZER="${ATTENTION_OPTIMIZER:-adam}"
ATTENTION_RECONSTRUCTION_WEIGHT="${ATTENTION_RECONSTRUCTION_WEIGHT:-0.5}"
ATTENTION_OUTPUT_WEIGHT="${ATTENTION_OUTPUT_WEIGHT:-0.5}"
if [[ -z "${ATTENTION_REFINE_TENSORS:-}" ]]; then
  if [[ "$v_predictor_mode" == "cross_kv" ]]; then
    ATTENTION_REFINE_TENSORS="key"
  else
    ATTENTION_REFINE_TENSORS="key_value"
  fi
fi
read -r -a attention_sign_step_sizes <<<"${ATTENTION_SIGN_STEP_SIZES:-0.01}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

check_flag OVERWRITE
check_flag DRY_RUN
check_int ITERATIONS 1
check_int ATTENTION_LAYER_CHUNK 1
if [[ -n "${ATTENTION_STEPS:-}" ]]; then
  check_int ATTENTION_STEPS 1
  check_int ATTENTION_PATIENCE 1
  [[ -z "${ATTENTION_MAX_TOKENS:-}" ]] || check_int ATTENTION_MAX_TOKENS 1
  case "$ATTENTION_OPTIMIZER" in adam|sign_gradient) ;; *) die "ATTENTION_OPTIMIZER 必须是 adam 或 sign_gradient" ;; esac
  case "$ATTENTION_REFINE_TENSORS" in key|value|key_value) ;; *) die "ATTENTION_REFINE_TENSORS 必须是 key、value 或 key_value" ;; esac
fi

fit_command=(
  "$python_bin" "$project_root/scripts/common/fit_params.py"
  "$trace_file"
  --config "$config_path"
  --output-dir "$output_dir"
  --iterations "$ITERATIONS"
  --fit-history-source "$FIT_HISTORY_SOURCE"
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
  --v-predictor-mode "$v_predictor_mode"
)
if [[ -n "${ATTENTION_STEPS:-}" ]]; then
  fit_command+=(
    --attention-steps "$ATTENTION_STEPS"
    --attention-lr "$ATTENTION_LR"
    --attention-patience "$ATTENTION_PATIENCE"
    --attention-min-relative-improvement "$ATTENTION_MIN_RELATIVE_IMPROVEMENT"
    --attention-layer-chunk "$ATTENTION_LAYER_CHUNK"
    --attention-optimizer "$ATTENTION_OPTIMIZER"
    --attention-reconstruction-weight "$ATTENTION_RECONSTRUCTION_WEIGHT"
    --attention-output-weight "$ATTENTION_OUTPUT_WEIGHT"
    --attention-refine-tensors "$ATTENTION_REFINE_TENSORS"
    --attention-sign-step-sizes "${attention_sign_step_sizes[@]}"
  )
  [[ -n "${ATTENTION_MAX_TOKENS:-}" ]] && \
    fit_command+=(--attention-max-tokens "$ATTENTION_MAX_TOKENS")
fi
[[ -n "${K_HISTORY_LENGTH:-}" ]] && fit_command+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && fit_command+=(--v-history-length "$V_HISTORY_LENGTH")
[[ -n "${K_BITS:-}" ]] && fit_command+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && fit_command+=(--v-bits "$V_BITS")
[[ -n "${ANCHOR_BITS:-}" ]] && fit_command+=(--anchor-bits "$ANCHOR_BITS")

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
