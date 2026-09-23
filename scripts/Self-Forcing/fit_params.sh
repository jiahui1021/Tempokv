#!/usr/bin/env bash
#
# fit_params.sh — 只拟合 TempoKV 参数（读 trace，不采集、不合并）
#
# 读取 <TRACE_FILE>（collect_trace.sh 采集并合并出的 self_forcing_trace.npz），
# 调用 scripts/common/fit_params.py 拟合 temporal 参数与对应的投影参数文件（文件名
# 取决于 V_PREDICTOR 与 DIRECTION，见下方 k_artifact_name / v_artifact_name）。每个
# k_history_length / v_history_length 配置拟合一套专属参数，用
# --k-history-length / --v-history-length 显式指定，用不同 --output-dir 区分。
#
# 用法（所有参数都用环境变量传入）：
#   CONFIG=configs/legacy.json \
#   KH=2 \
#   VH=1 \
#   bash scripts/Self-Forcing/fit_params.sh
#
# 常用可选变量：
#   CONFIG          codec 配置 JSON（默认 configs/legacy.json）
#   TRACE_FILE      拟合输入（默认 data/self-forcing/trace/clean_trace.npz）
#   OUTPUT_DIR      输出目录（默认 data/self-forcing/params/<参数方案>）
#   BITS / K_BITS / V_BITS  K/V residual 位宽 override（1 到 8；默认读取 CONFIG）
#   ANCHOR_BITS     anchor 位宽（1 到 15 量化，16 表示不量化直接存 bf16；默认 INT4）
#   K_HISTORY_LENGTH  K 历史长度（默认读取 CONFIG）
#   V_HISTORY_LENGTH  V 历史长度（默认读取 CONFIG）
#   ITERATIONS     拟合迭代次数（默认 2）
#   FIT_HISTORY_SOURCE  reconstructed（默认）或 clean；仅用于拟合消融
#   TEMPORAL_RIDGE / CROSS_RIDGE / V_HISTORY_RIDGE / MAX_GAIN
#                  拟合正则与增益（legacy 模式默认 1.0/0.1/0.0/1.0）
#   ATTENTION_STEPS  每个 K/V phase 的最大 Adam 步数（留空则不精调）
#   ATTENTION_LR / ATTENTION_PATIENCE / ATTENTION_MIN_RELATIVE_IMPROVEMENT
#                    Adam 学习率、早停 patience 与最小相对改善
#   ATTENTION_OPTIMIZER  adam（默认）或 sign_gradient
#   ATTENTION_RECONSTRUCTION_WEIGHT / ATTENTION_OUTPUT_WEIGHT  objective 权重
#   ATTENTION_REFINE_TENSORS  key、value 或 key_value（CrossKV 默认 key；temporal V 默认 key_value）
#   ATTENTION_SIGN_STEP_SIZES  sign-gradient 候选步长，空格分隔
#   ATTENTION_MAX_TOKENS  refinement 使用的 sampled token-track 上限
#   ATTENTION_LAYER_CHUNK  attention refinement 每批层数（默认 4）
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
K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"
CONFIG="${CONFIG:-$default_codec_config}"

trace_file="$(resolve_path "${TRACE_FILE:-data/self-forcing/trace/clean_trace.npz}")"
config_path="$(resolve_path "$CONFIG")"
FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
parameter_name="${PARAM_NAME:-$(codec_slug "$config_path" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" "${K_BITS:-}" "${V_BITS:-}" "" "${V_PREDICTOR:-}" "$FIT_HISTORY_SOURCE" "${ANCHOR_BITS:-}")}"
output_dir="$(resolve_path "${OUTPUT_DIR:-data/self-forcing/params/$parameter_name}")"
read -r fit_mode v_predictor_mode _ <<<"$(codec_modes "$config_path" "${V_PREDICTOR:-}")"
# DIRECTION mirrors the K/V roles: v_to_k predicts K from the reconstructed V.
# The artifacts are named for the role each parameter set plays, so the two
# directions cannot be confused on disk.
predictor_direction="${DIRECTION:-k_to_v}"
if [[ "$predictor_direction" == "v_to_k" ]]; then
  k_artifact_name="v_temporal_primary.npz"
  v_artifact_name="k_cross_from_v.npz"
else
  k_artifact_name="k_temporal.npz"
  v_artifact_name="$(v_artifact_name "$v_predictor_mode")"
fi

ITERATIONS="${ITERATIONS:-2}"
# Legacy fitting reproduces the reference implementation and pins its ridge;
# fit_params.py rejects any other value, so the default follows fit_mode.
if [[ "$fit_mode" == "legacy" ]]; then
  default_temporal_ridge=1.0
else
  default_temporal_ridge=0.1
fi
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
  --fit-history-source "$FIT_HISTORY_SOURCE"
  --temporal-ridge "${TEMPORAL_RIDGE:-$default_temporal_ridge}"
  --cross-ridge "${CROSS_RIDGE:-0.1}"
  --v-history-ridge "${V_HISTORY_RIDGE:-0.0}"
  --v-history-mode diagonal
  --max-gain "${MAX_GAIN:-1.0}"
  --device "${FIT_DEVICE:-auto}"
  --stride 1560
  --coordinate-system pre_rope_normalized
)
append_attention_refinement "$v_predictor_mode"
[[ -n "${K_HISTORY_LENGTH:-}" ]] && fit_command+=(--k-history-length "$K_HISTORY_LENGTH")
[[ -n "${V_HISTORY_LENGTH:-}" ]] && fit_command+=(--v-history-length "$V_HISTORY_LENGTH")
[[ -n "${K_BITS:-}" ]] && fit_command+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && fit_command+=(--v-bits "$V_BITS")
[[ -n "${ANCHOR_BITS:-}" ]] && fit_command+=(--anchor-bits "$ANCHOR_BITS")
[[ -n "${V_PREDICTOR:-}" ]] && fit_command+=(--v-predictor-mode "$V_PREDICTOR")
[[ -n "${DIRECTION:-}" ]] && fit_command+=(--predictor-direction "$predictor_direction")
[[ -n "${SPAN_FRAMES:-}" ]] && fit_command+=(--span-frames "$SPAN_FRAMES")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[fit_params] 干跑模式：打印将要执行的命令，不执行\n'
  print_command "${fit_command[@]}"
  exit 0
fi

[[ -f "$trace_file" ]] || die "TRACE_FILE 不存在：$trace_file"
refuse_existing_params "$output_dir/$k_artifact_name" "$output_dir/$v_artifact_name"
mkdir -p -- "$output_dir"

printf '\n[2/2] 拟合 TempoKV 参数 ...\n'
"${fit_command[@]}"
require_params_written "$output_dir/$k_artifact_name" "$output_dir/$v_artifact_name"

printf '\n完成。参数已写入：\n  %s\n  %s\n' \
  "$output_dir/$k_artifact_name" "$output_dir/$v_artifact_name"
