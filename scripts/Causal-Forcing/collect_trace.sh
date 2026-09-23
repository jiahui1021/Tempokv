#!/usr/bin/env bash
#
# collect_trace.sh — 采集 Causal Forcing clean K/V trace 并合并成拟合输入
#
# 只采 clean/context refresh 留下的 pre-RoPE normalized K 与 raw V，
# 不采 4 个 noisy denoising step 的 current K/V。
#
# 两步完成：
#   1. 通过 trace bridge 运行 Causal Forcing，采集逐层 trace
#      <TRACE_DIR>/raw/     （layer_*.npz）
#      <TRACE_DIR>/output/  （采集器中间产物）
#   2. 调用 prepare_trace.py 合并成 <TRACE_FILE>
#
# 用法：
#   CHECKPOINT=/models/causal_forcing/chunkwise/causal_ode.pt \
#   MODEL_ROOT=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/fit_prompts.txt \
#   bash scripts/Causal-Forcing/collect_trace.sh
#
# 必需变量：
#   CHECKPOINT   Causal Forcing 模型权重
#   MODEL_ROOT   Wan2.1-T2V-1.3B 目录
#   PROMPTS      采集用 prompt 列表
#
# 常用可选变量：
#   GENERATION_MODE  chunkwise（默认）或 framewise
#   TRACE_DIR        trace 目录（默认 data/causal-forcing/trace/<模式>_legacy）
#   TRACE_FILE       合并后的拟合输入（默认 <TRACE_DIR>/clean_trace.npz）
#   OVERWRITE / DRY_RUN
#   USE_EMA          是否读取 checkpoint 的 generator_ema（默认 0，官方权重只有 generator）
#
# 采集固定使用 legacy 协议：PROMPTS 必须是 moviegen128.txt 的前 12 条，每条 51
# 个 latent 帧、每帧 8 个空间位置、seed 0、单卡（多卡 seed 扰动会打乱 event 顺序）。
#
set -euo pipefail

# shellcheck source=scripts/Causal-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

require_env CHECKPOINT MODEL_ROOT PROMPTS

generation_mode="${GENERATION_MODE:-chunkwise}"
case "$generation_mode" in
  chunkwise|framewise) ;;
  *) die "GENERATION_MODE 必须是 chunkwise 或 framewise（当前值：${generation_mode}）" ;;
esac

runtime_root="$project_root/experiments/Causal-Forcing"
adapter_root="$runtime_root/causal_forcing/tempokv_adapter"
if [[ -n "${MODEL_CONFIG:-}" ]]; then
  model_config="$(resolve_path "$MODEL_CONFIG")"
else
  model_config="$runtime_root/configs/causal_forcing_dmd_$generation_mode.yaml"
fi
checkpoint="$(resolve_path "$CHECKPOINT")"
model_root="$(resolve_path "$MODEL_ROOT")"
prompts="$(resolve_path "$PROMPTS")"
trace_dir="$(resolve_path "${TRACE_DIR:-data/causal-forcing/trace/${generation_mode}_legacy}")"
trace_file="$(resolve_path "${TRACE_FILE:-$trace_dir/clean_trace.npz}")"
raw_trace_dir="$trace_dir/raw"
trace_output_dir="$trace_dir/output"

OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
USE_EMA="${USE_EMA:-0}"
check_flag OVERWRITE
check_flag DRY_RUN
check_flag USE_EMA

trace_command=(
  env
  "TEMPOKV_TRACE_DIR=$raw_trace_dir"
  "TEMPOKV_TRACE_TOKENS_PER_FRAME=8"
  "TEMPOKV_TRACE_EVENT_FRAMES=24"
  "$python_bin" -m torch.distributed.run
  --standalone
  --nproc_per_node=1
  "$adapter_root/worker.py"
  --mode trace
  --
  --config_path "$model_config"
  --checkpoint_path "$checkpoint"
  --model_root "$model_root"
  --data_path "$prompts"
  --output_folder "$trace_output_dir"
  --num_output_frames 51
  --num_samples 1
  --seed 0
  --save_with_index
  --local_attn_size 180
)
[[ "$USE_EMA" == "1" ]] && trace_command+=(--use_ema)

merge_command=(
  "$python_bin" "$adapter_root/prepare_trace.py"
  "$raw_trace_dir" "$trace_file" --legacy-protocol
)

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

"$python_bin" - "$prompts" "$project_root/assets/moviegen128.txt" <<'PY'
from pathlib import Path
import sys

actual = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
reference = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()[:12]
if actual != reference:
    raise SystemExit("PROMPTS 必须是 moviegen128.txt 的前 12 条（legacy 采集协议）")
PY

if [[ "$OVERWRITE" == "1" ]]; then
  rm -rf -- "$raw_trace_dir" "$trace_output_dir"
else
  [[ -e "$trace_file" ]] && die "已存在拟合输入：${trace_file}。设置 OVERWRITE=1 覆盖，或换一个 TRACE_FILE。"
  if [[ -d "$raw_trace_dir" ]]; then
    existing="$(find "$raw_trace_dir" -type f -name 'layer_*.npz' -print -quit)"
    [[ -z "$existing" ]] || die "已存在逐层 trace：${existing}。设置 OVERWRITE=1 覆盖，或换一个 TRACE_DIR。"
  fi
fi
mkdir -p -- "$raw_trace_dir" "$trace_output_dir"

printf '\n[1/2] 采集 clean trace ...\n'
"${trace_command[@]}"

printf '\n[2/2] 合并逐层 trace -> %s ...\n' "$trace_file"
"${merge_command[@]}"
printf '\n完成。拟合输入已写入：\n  %s\n' "$trace_file"
