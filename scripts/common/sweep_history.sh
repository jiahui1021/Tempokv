#!/usr/bin/env bash
#
# sweep_history.sh — 历史帧数敏感性实验（三个模型共用实现）
#
# 不要直接调用，用各模型的入口：
#   bash scripts/Self-Forcing/sweep_history.sh
#   bash scripts/Causal-Forcing/sweep_history.sh
#   bash scripts/LongCat/sweep_history.sh
#
# 扫 K 和 V 的历史帧数，在每个取值上报告三项：
#
#   重建误差        ||rebuilt - clean||^2 / ||clean||^2，闭环重建后的 cache 误差
#   Attention 输出   同一份重建经过 causal softmax attention 后的相对输出误差
#   预测项占比      1 - ||clean - prediction||^2 / ||clean||^2，
#                   其补集就是量化器要花比特去编的残差
#
# K 扫 k_history_length（1 到 8），V 在 CrossKV 项之上再扫 v_history_length
# （0 到 5），并额外拆出 V 预测里 CrossKV 项与 history 项各自的能量占比。
#
# 每个取值都会**重新拟合一次**，确保每个历史长度都使用自己的 legacy/shared 参数。
#
# 常用可选变量：
#   TRACE_FILE   拟合输入（默认取该模型 collect_trace.sh 的输出）
#   CONFIG       位宽等固定设置（默认取该模型的 INT4 配置）
#   K_HISTORY    K 扫描点（默认 0,1,2,3,4,5；0=直接量化）
#   V_HISTORY    V 扫描点（默认 0,1,2,4,5）
#   BITS / K_BITS / V_BITS  位宽 override（1 到 8）
#   V_PREDICTOR  cross_kv、temporal、identity、taylor、taylor2、cross_kv_identity、cross_kv_taylor、cross_kv_taylor2、innovation 或 direct
#   BASELINE_K_HISTORY / BASELINE_V_HISTORY  扫另一侧时固定的取值
#   MAX_SAMPLES  下采样空间位置数，先跑小规模时用
#   MAX_FRAMES   截断时间轴
#   LAYERS       只统计部分层，例如 0,15,29
#   ITERATIONS / TEMPORAL_RIDGE / CROSS_RIDGE / MAX_GAIN  拟合控制
#   FIT_DEVICE   auto / cpu / cuda（默认 auto）
#   OUTPUT       JSON 报告路径
#   DRY_RUN      只打印命令不执行（0/1，默认 0）
#
# Causal Forcing 另外支持 GENERATION_MODE=chunkwise|framewise，两种模式的
# trace、配置和报告分开存放。
#
set -euo pipefail

sweep_common_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
sweep_project_root="$(cd -- "$sweep_common_dir/../.." && pwd -P)"

case "${MODEL:-}" in
  self-forcing)
    model_dir="Self-Forcing"
    default_trace="data/self-forcing/trace/clean_trace.npz"
    default_config="configs/legacy.json"
    default_output="results/self_forcing/history_sweep.json"
    ;;
  causal-forcing)
    model_dir="Causal-Forcing"
    generation_mode="${GENERATION_MODE:-chunkwise}"
    case "$generation_mode" in
      chunkwise|framewise) ;;
      *)
        printf 'Error: GENERATION_MODE 必须是 chunkwise 或 framewise（当前值：%s）\n' \
          "$generation_mode" >&2
        exit 2
        ;;
    esac
    default_trace="data/causal-forcing/trace/${generation_mode}_legacy/clean_trace.npz"
    default_config="configs/legacy.json"
    default_output="results/causal_forcing/$generation_mode/history_sweep.json"
    ;;
  longcat)
    model_dir="LongCat"
    default_trace="data/longcat/trace/clean_trace.npz"
    default_config="configs/legacy.json"
    default_output="results/longcat/history_sweep.json"
    ;;
  *)
    printf 'Error: MODEL 必须是 self-forcing、causal-forcing 或 longcat\n' >&2
    exit 2
    ;;
esac

# shellcheck source=scripts/Self-Forcing/_common.sh
source "$sweep_project_root/scripts/$model_dir/_common.sh"

trace_file="$(resolve_path "${TRACE_FILE:-$default_trace}")"
config_path="$(resolve_path "${CONFIG:-$default_config}")"
output="$(resolve_path "${OUTPUT:-$default_output}")"

K_HISTORY="${K_HISTORY:-0,1,2,3,4,5}"
V_HISTORY="${V_HISTORY:-0,1,2,4,5}"
K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"
ITERATIONS="${ITERATIONS:-2}"
TEMPORAL_RIDGE="${TEMPORAL_RIDGE:-1.0}"
CROSS_RIDGE="${CROSS_RIDGE:-0.1}"
MAX_GAIN="${MAX_GAIN:-1.0}"
FIT_DEVICE="${FIT_DEVICE:-auto}"
DRY_RUN="${DRY_RUN:-0}"

check_flag DRY_RUN
check_int ITERATIONS 1

sweep_command=(
  "$python_bin" "$project_root/scripts/tools/sweep_history_length.py"
  --trace "$trace_file"
  --config "$config_path"
  --k-history "$K_HISTORY"
  --v-history "$V_HISTORY"
  --iterations "$ITERATIONS"
  --temporal-ridge "$TEMPORAL_RIDGE"
  --cross-ridge "$CROSS_RIDGE"
  --max-gain "$MAX_GAIN"
  --device "$FIT_DEVICE"
  --output "$output"
)
[[ -n "${K_BITS:-}" ]] && sweep_command+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && sweep_command+=(--v-bits "$V_BITS")
[[ -n "${V_PREDICTOR:-}" ]] && sweep_command+=(--v-predictor-mode "$V_PREDICTOR")
[[ -n "${BASELINE_K_HISTORY:-}" ]] && sweep_command+=(--baseline-k-history "$BASELINE_K_HISTORY")
[[ -n "${BASELINE_V_HISTORY:-}" ]] && sweep_command+=(--baseline-v-history "$BASELINE_V_HISTORY")
[[ -n "${MAX_SAMPLES:-}" ]] && sweep_command+=(--max-samples "$MAX_SAMPLES")
[[ -n "${MAX_FRAMES:-}" ]] && sweep_command+=(--max-frames "$MAX_FRAMES")
[[ -n "${LAYERS:-}" ]] && sweep_command+=(--layers "$LAYERS")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[sweep_history] 干跑模式：打印将要执行的命令，不执行\n'
  printf '  $'
  printf ' %q' "${sweep_command[@]}"
  printf '\n'
  exit 0
fi

[[ -f "$config_path" ]] || die "CONFIG 不存在：$config_path"
[[ -f "$trace_file" ]] || die "TRACE_FILE 不存在：${trace_file}（先跑 collect_trace.sh）"

mkdir -p -- "$(dirname -- "$output")"
"${sweep_command[@]}"
printf '\n完成。报告已写入：\n  %s\n' "$output"
