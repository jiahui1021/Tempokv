#!/usr/bin/env bash
#
# collect_trace.sh — 采集长时程（默认 1401 视频帧）的原始 K/V trace
#
# 1401 视频帧对应 351 个 latent 帧（换算见 generate.py 的 latent_frames_for：
# shifted = FRAMES + 3，要求能被 12 整除，再除以 4）。legacy 采集协议把
# TRACE_FRAMES 钉死在 51，装不下这个长度，所以本脚本强制 LEGACY_PROTOCOL=0；
# 拟合侧的 legacy 方程与形状无关，照常接受这种 [L,S,T,H,D] 布局。
#
# 必需变量：
#   CHECKPOINT     Self-Forcing 权重
#   WAN_MODEL_DIR  Wan 模型目录
#   PROMPTS        prompt 列表（长时程采集很贵，默认只用第一条）
#
# 常用可选变量：
#   HORIZON_FRAMES 视频帧数（默认 1401，必须满足 (FRAMES+3)%12==0）
#   TRACE_DIR      trace 目录（默认 data/long-horizon/<HORIZON_FRAMES>f）
#   TRACE_FILE     合并后的 .npz（默认 <TRACE_DIR>/clean_trace.npz）
#   TOKENS_PER_FRAME  每帧抽取的 token 数（默认 8；越小 trace 越小）
#   NUM_PROMPTS    采集的 prompt 条数（默认 1）
#   OVERWRITE      覆盖已有 trace（0/1，默认 0）
#   DRY_RUN        只打印命令与体积预估（0/1，默认 0）
#
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  python_bin="$project_root/.venv/bin/python"
else
  python_bin="python3"
fi

resolve_path() {
  case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$project_root/$1" ;; esac
}

[[ -n "${CHECKPOINT:-}" ]] || die "缺少必需变量 CHECKPOINT"
[[ -n "${WAN_MODEL_DIR:-}" ]] || die "缺少必需变量 WAN_MODEL_DIR"
[[ -n "${PROMPTS:-}" ]] || die "缺少必需变量 PROMPTS"

HORIZON_FRAMES="${HORIZON_FRAMES:-1401}"
TOKENS_PER_FRAME="${TOKENS_PER_FRAME:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
for flag in OVERWRITE DRY_RUN; do
  case "${!flag}" in 0|1) ;; *) die "$flag 必须是 0 或 1（当前值：${!flag}）" ;; esac
done

# The runtime's rule, restated once: a latent block covers 4 video frames and
# Self-Forcing requires whole 3-frame latent blocks, i.e. (FRAMES + 3) % 12 == 0.
# Doing it here keeps the script usable before the model runtime is importable.
shifted=$(( HORIZON_FRAMES + 3 ))
if (( HORIZON_FRAMES < 1 )) || (( shifted % 12 != 0 )); then
  die "HORIZON_FRAMES 必须为正且满足 (FRAMES+3)%12==0（当前值：${HORIZON_FRAMES}）"
fi
latent_frames=$(( shifted / 4 ))
if (( latent_frames % 3 != 0 )); then
  die "换算出的 latent 帧数 ${latent_frames} 不是 3 的倍数，采集脚本会拒绝"
fi

trace_dir="$(resolve_path "${TRACE_DIR:-data/long-horizon/${HORIZON_FRAMES}f}")"
trace_file="$(resolve_path "${TRACE_FILE:-$trace_dir/clean_trace.npz}")"
prompts="$(resolve_path "$PROMPTS")"
[[ -f "$prompts" ]] || die "PROMPTS 不存在：$prompts"

# Project the archive size before spending GPU hours on it.  A long horizon
# multiplies the frame axis, so this grows fast and is the first thing to check.
projected="$("$python_bin" -c "
layers, heads, dim = 30, 12, 128
rows = int('$TOKENS_PER_FRAME') * 2 * int('$NUM_PROMPTS')
frames = int('$latent_frames')
per_tensor = layers * rows * frames * heads * dim * 4
print(f'{2 * per_tensor / 1024**3:.2f}')
")"

cat <<INFO
[long-horizon trace]
  视频帧      ${HORIZON_FRAMES}
  latent 帧   ${latent_frames}
  prompts     ${NUM_PROMPTS}
  每帧 token  ${TOKENS_PER_FRAME}
  预计体积    约 ${projected} GiB（K+V，float32）
  输出        ${trace_file}
INFO

if [[ "$OVERWRITE" != "1" && -e "$trace_file" ]]; then
  die "已存在 trace：${trace_file}。设置 OVERWRITE=1 覆盖，或换一个 TRACE_FILE。"
fi

# Take the leading NUM_PROMPTS lines; long-horizon collection is expensive and
# the curve is a per-frame property, so more prompts buy little here.
subset="$trace_dir/prompts_${NUM_PROMPTS}.txt"
mkdir -p -- "$trace_dir"
head -n "$NUM_PROMPTS" "$prompts" > "$subset"

collect_env=(
  "CHECKPOINT=$(resolve_path "$CHECKPOINT")"
  "WAN_MODEL_DIR=$(resolve_path "$WAN_MODEL_DIR")"
  "PROMPTS=$subset"
  "TRACE_DIR=$trace_dir"
  "TRACE_FILE=$trace_file"
  "TRACE_FRAMES=$latent_frames"
  "TRACE_TOKENS_PER_FRAME=$TOKENS_PER_FRAME"
  "LEGACY_PROTOCOL=0"
  "NUM_GPUS=1"
  "OVERWRITE=$OVERWRITE"
)
[[ -n "${MODEL_CONFIG:-}" ]] && collect_env+=("MODEL_CONFIG=$(resolve_path "$MODEL_CONFIG")")
[[ -n "${BASE_SEED:-}" ]] && collect_env+=("BASE_SEED=$BASE_SEED")

command=(env "${collect_env[@]}" bash "$project_root/scripts/Self-Forcing/collect_trace.sh")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dry-run] '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

"${command[@]}"
printf '完成：%s\n' "$trace_file"
