#!/usr/bin/env bash
#
# run_study.sh — 跑完整的 §3.2 observation 实验（采集 → 校验 → 阶梯 → 图表）
#
# 用法：
#   CHECKPOINT=/models/self_forcing_dmd.pt \
#   WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
#   CAL_PROMPTS=assets/calibration/cal_prompts.txt \
#   EVAL_PROMPTS=assets/calibration/heldout_prompts.txt \
#   bash scripts/paper_experiments/observation/run_study.sh
#
# 两组 prompt 必须来自已有的 prompt-disjoint 划分。trace 已存在时跳过采集，
# 此时 CHECKPOINT / WAN_MODEL_DIR / *_PROMPTS 都不需要，可以在无 GPU 的机器上
# 只跑分析。
#
# 可选变量：
#   OUTPUT_ROOT   输出目录（默认 results/observation）
#   TRACE_ROOT    trace 目录（默认 data/self-forcing/trace）
#   CAL_TRACE / EVAL_TRACE   两份 .npz（默认 <TRACE_ROOT>/observation_{cal,eval}.npz）
#   FRAMES        每条 prompt 的 latent 帧数（默认 180）
#   TOKENS        每帧抽取的 spatial token 数（默认 64）
#   LAYERS        分析用层（默认 0,7,15,22,29）
#   K_ORDER / V_ORDER   预测器阶数（默认 2 / 1）
#   CONFIG        闭环位宽来源（默认 configs/legacy.json）
#   DEVICE        分析设备（默认 cpu）
#   PYTHON_BIN    Python 解释器
#
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"
cd "$project_root"

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
step() { printf '\n=== %s ===\n' "$*"; }

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  python_bin="$project_root/.venv/bin/python"
else
  python_bin="python3"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/observation}"
TRACE_ROOT="${TRACE_ROOT:-data/self-forcing/trace}"
CAL_TRACE="${CAL_TRACE:-$TRACE_ROOT/observation_cal.npz}"
EVAL_TRACE="${EVAL_TRACE:-$TRACE_ROOT/observation_eval.npz}"
FRAMES="${FRAMES:-180}"
TOKENS="${TOKENS:-64}"
LAYERS="${LAYERS:-0,7,15,22,29}"
K_ORDER="${K_ORDER:-2}"
V_ORDER="${V_ORDER:-1}"
CONFIG="${CONFIG:-configs/legacy.json}"
DEVICE="${DEVICE:-cpu}"

# 1. 采集。LEGACY_PROTOCOL=0 是硬要求：legacy 归档保留原始 B/P 轴，阶梯读的是
# 扁平 tempokv_flat_samples_v1 布局，loader 会拒绝而不是重新解释轴。held-out
# 一组标成 diagnostic，fitter 便无法误当成 calibration 使用。
collect() {
  local prompts="$1" trace="$2" role="$3"
  [[ -f "$trace" ]] && { printf 'trace 已存在，跳过采集：%s\n' "$trace"; return; }
  [[ -n "$prompts" ]] || die "缺少 trace $trace，且未提供对应的 prompt 列表"
  [[ -n "${CHECKPOINT:-}" && -n "${WAN_MODEL_DIR:-}" ]] || \
    die "采集需要 CHECKPOINT 与 WAN_MODEL_DIR"
  CHECKPOINT="$CHECKPOINT" WAN_MODEL_DIR="$WAN_MODEL_DIR" PROMPTS="$prompts" \
  LEGACY_PROTOCOL=0 TRACE_ROLE="$role" \
  TRACE_FRAMES="$FRAMES" TRACE_TOKENS_PER_FRAME="$TOKENS" \
  TRACE_DIR="${trace%.npz}" TRACE_FILE="$trace" \
    bash scripts/Self-Forcing/collect_trace.sh
}

step "1/4 采集 trace"
collect "${CAL_PROMPTS:-}" "$CAL_TRACE" calibration
collect "${EVAL_PROMPTS:-}" "$EVAL_TRACE" diagnostic

# 2. 两个必查项：帧数不能是 latent 帧数的整数倍（denoising 漏进来了），
# K 不能像旋转过的张量那样沿帧翻转符号（拿到的是 post-RoPE）。
step "2/4 校验 trace"
for trace in "$CAL_TRACE" "$EVAL_TRACE"; do
  "$python_bin" scripts/paper_experiments/observation/check_trace.py \
    "$trace" --expected-frames "$FRAMES" \
    --output "$OUTPUT_ROOT/check_$(basename "${trace%.npz}").json" >/dev/null \
    || die "trace 校验未通过：$trace（详见 $OUTPUT_ROOT/check_*.json）"
  printf 'ok  %s\n' "$trace"
done

# 3. 阶梯。clean 与闭环两套一次跑完，正文只能用 clean 讲信号，任何设计选择
# 都必须引闭环那一列。
step "3/4 分解阶梯"
"$python_bin" scripts/paper_experiments/observation/run_decompose.py \
  --calibration-trace "$CAL_TRACE" \
  --held-out-trace "$EVAL_TRACE" \
  --layers "$LAYERS" --k-order "$K_ORDER" --v-order "$V_ORDER" \
  --config "$CONFIG" --device "$DEVICE" \
  --output "$OUTPUT_ROOT/decompose.json"

step "4/4 表 1 与 Fig 3(a)"
"$python_bin" scripts/paper_experiments/observation/plot_fig3a.py \
  --report "$OUTPUT_ROOT/decompose.json" \
  --output-dir "$OUTPUT_ROOT" --loop closed_loop >/dev/null
"$python_bin" scripts/paper_experiments/observation/plot_fig3a.py \
  --report "$OUTPUT_ROOT/decompose.json" --output-dir "$OUTPUT_ROOT" --loop clean

printf '\n输出：%s/{decompose.json,table1_tiers.csv,fig3a_segments.csv,fig3a_*.pdf}\n' \
  "$OUTPUT_ROOT"
