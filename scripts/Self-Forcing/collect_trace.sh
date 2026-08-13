#!/usr/bin/env bash
#
# collect_trace.sh — 采集 clean K/V trace 并合并成拟合输入
#
# 两步完成：
#   1. 通过 Tempokv 的 trace bridge 运行 Self-Forcing，采集未量化的逐层 trace
#      <TRACE_DIR>/raw/               （layer_*.npz，逐层 trace）
#      <TRACE_DIR>/output/            （采集器中间产物）
#   2. 调用 prepare_trace.py 把逐层 trace 合并成 <TRACE_FILE>
#      （默认 data/self-forcing/trace/clean_trace.npz）
#
# 用法（所有参数都用环境变量传入）：
#   CHECKPOINT=/models/self_forcing_dmd.pt \
#   WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/fit_prompts.txt \
#   bash scripts/Self-Forcing/collect_trace.sh
#
# 必需变量：
#   CHECKPOINT     Self-Forcing 模型权重
#   WAN_MODEL_DIR  Wan 模型目录
#   PROMPTS        采集用 prompt 列表
#
# 常用可选变量：
#   TRACE_DIR      该模型的 trace 目录（默认 data/self-forcing/trace）
#   TRACE_FILE     合并后的拟合输入（默认 <TRACE_DIR>/clean_trace.npz）
#   MODEL_CONFIG   Self-Forcing 配置（默认使用仓库内置的 self_forcing_dmd.yaml）
#   NUM_GPUS       采集用 GPU 数量（默认 1；多卡按 prompt 分片，数据与单卡一致）
#   LEGACY_PROTOCOL  使用 legacy 原始拟合数据协议（0/1，默认 1）
#   TRACE_FRAMES   每个 prompt 生成的 latent 帧数（legacy 固定 51）
#   TRACE_TOKENS_PER_FRAME  每帧抽取的 token 数（legacy 固定 8）
#   BASE_SEED      随机种子（默认 0）
#   OVERWRITE      已存在 trace 或 TRACE_FILE 时覆盖（0/1，默认 0）
#   DRY_RUN        只打印命令不执行（0/1，默认 0）
#   PYTHON_BIN     Python 解释器（默认 python3）
#
set -euo pipefail

# shellcheck source=scripts/Self-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
script_dir="$self_forcing_dir"

require_env CHECKPOINT WAN_MODEL_DIR PROMPTS

runtime_root="$project_root/experiments/Self-Forcing"
adapter_root="$runtime_root/tempokv_adapter"
if [[ -n "${MODEL_CONFIG:-}" ]]; then
  model_config="$(resolve_path "$MODEL_CONFIG")"
else
  model_config="$runtime_root/configs/self_forcing_dmd.yaml"
fi
checkpoint="$(resolve_path "$CHECKPOINT")"
wan_model_dir="$(resolve_path "$WAN_MODEL_DIR")"
prompts="$(resolve_path "$PROMPTS")"
trace_dir="$(resolve_path "${TRACE_DIR:-data/self-forcing/trace}")"
trace_file="$(resolve_path "${TRACE_FILE:-$trace_dir/clean_trace.npz}")"
raw_trace_dir="$trace_dir/raw"
trace_output_dir="$trace_dir/output"

if [[ "$trace_dir" == "/" ]]; then
  die "TRACE_DIR 不能是文件系统根目录"
fi

LEGACY_PROTOCOL="${LEGACY_PROTOCOL:-1}"
check_flag LEGACY_PROTOCOL
if [[ "$LEGACY_PROTOCOL" == "1" ]]; then
  default_trace_frames=51
else
  default_trace_frames=21
fi
TRACE_FRAMES="${TRACE_FRAMES:-$default_trace_frames}"
TRACE_TOKENS_PER_FRAME="${TRACE_TOKENS_PER_FRAME:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_SEED="${BASE_SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

check_flag OVERWRITE
check_flag DRY_RUN
check_int TRACE_FRAMES 3
(( TRACE_FRAMES % 3 == 0 )) || \
  die "TRACE_FRAMES 必须是 3 的倍数（当前值：${TRACE_FRAMES}）"
check_int TRACE_TOKENS_PER_FRAME 1
(( TRACE_TOKENS_PER_FRAME <= 1560 )) || \
  die "TRACE_TOKENS_PER_FRAME 必须 <= 1560（当前值：${TRACE_TOKENS_PER_FRAME}）"
check_int NUM_GPUS 1
check_int BASE_SEED 0
if [[ "$LEGACY_PROTOCOL" == "1" ]]; then
  [[ "$TRACE_FRAMES" == "51" ]] || die "LEGACY_PROTOCOL 要求 TRACE_FRAMES=51"
  [[ "$TRACE_TOKENS_PER_FRAME" == "8" ]] || \
    die "LEGACY_PROTOCOL 要求 TRACE_TOKENS_PER_FRAME=8"
  [[ "$BASE_SEED" == "0" ]] || die "LEGACY_PROTOCOL 要求 BASE_SEED=0"
fi

trace_environment=(
  env
  "FORCING_WAN_MODEL_DIR=$wan_model_dir"
  "TEMPOKV_TRACE_DIR=$raw_trace_dir"
  "TEMPOKV_TRACE_TOKENS_PER_FRAME=$TRACE_TOKENS_PER_FRAME"
)
if [[ "$LEGACY_PROTOCOL" == "1" ]]; then
  trace_environment+=(
    "TEMPOKV_TRACE_EVENT_FRAMES=24"
    "TEMPOKV_TRACE_ROPE_PRECISION=complex64"
  )
fi
trace_command=(
  "${trace_environment[@]}"
  "$python_bin" -m torch.distributed.run
  --standalone
  "--nproc_per_node=$NUM_GPUS"
  "$adapter_root/worker.py"
  --mode trace
  --
  --config_path "$model_config"
  --checkpoint_path "$checkpoint"
  --data_path "$prompts"
  --output_folder "$trace_output_dir"
  --num_output_frames "$TRACE_FRAMES"
  --num_samples 1
  --seed "$BASE_SEED"
  --save_with_index
  --use_ema
)
if [[ "$LEGACY_PROTOCOL" == "1" ]]; then
  trace_command+=(--local_attn_size 180)
fi

merge_command=(
  "$python_bin" "$adapter_root/prepare_trace.py"
  "$raw_trace_dir" "$trace_file"
)
[[ "$LEGACY_PROTOCOL" == "1" ]] && merge_command+=(--legacy-protocol)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[collect_trace] 干跑模式：打印将要执行的命令，不执行\n'
  printf '  [1/2] '
  printf '%q ' "${trace_command[@]}"
  printf '\n'
  printf '  [2/2] '
  printf '%q ' "${merge_command[@]}"
  printf '\n'
  exit 0
fi

[[ -f "$runtime_root/inference.py" ]] || \
  die "仓库内置的 Self-Forcing 采集代码不完整：$runtime_root"
[[ -f "$model_config" ]] || die "MODEL_CONFIG 不存在：$model_config"
[[ -f "$checkpoint" ]] || die "CHECKPOINT 不存在：$checkpoint"
[[ -d "$wan_model_dir" ]] || die "WAN_MODEL_DIR 不存在：$wan_model_dir"
[[ -f "$prompts" ]] || die "PROMPTS 不存在：$prompts"
if [[ "$LEGACY_PROTOCOL" == "1" ]]; then
  "$python_bin" - "$prompts" "$project_root/assets/moviegen128.txt" <<'PY'
from pathlib import Path
import sys

PROMPT_COUNT = 12
actual = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
reference = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()[:PROMPT_COUNT]
if len(actual) != PROMPT_COUNT or any(not line.strip() for line in actual):
    raise SystemExit("LEGACY_PROTOCOL 的 PROMPTS 必须恰好包含 12 条非空 prompt")
if actual != reference:
    raise SystemExit("LEGACY_PROTOCOL 的 PROMPTS 必须是 moviegen128.txt 的前 12 条")
PY
fi

if [[ "$OVERWRITE" != "1" ]]; then
  if [[ -e "$trace_file" ]]; then
    die "已存在拟合输入：${trace_file}。设置 OVERWRITE=1 覆盖，或换一个 TRACE_FILE。"
  fi
  if [[ -d "$raw_trace_dir" ]]; then
    existing="$(find "$raw_trace_dir" -type f -name 'layer_*.npz' -print -quit)"
    if [[ -n "$existing" ]]; then
      die "已存在逐层 trace：${existing}。设置 OVERWRITE=1 覆盖，或换一个 TRACE_DIR。"
    fi
  fi
fi

if [[ "$OVERWRITE" == "1" && -d "$trace_dir/raw" ]]; then
  rm -rf -- "$trace_dir/raw" "$trace_dir/output"
fi
mkdir -p -- "$raw_trace_dir" "$trace_output_dir"

printf '\n[1/2] 采集 clean trace ...\n'
(
  "${trace_command[@]}"
)

printf '\n[2/2] 合并逐层 trace -> %s ...\n' "$trace_file"
"${merge_command[@]}"
printf '\n完成。拟合输入已写入：\n  %s\n' "$trace_file"
