#!/usr/bin/env bash
#
# download_models.sh — 下载 Causal Forcing 推理所需的全部权重
#
# 下载到（权重不进入 Git）：
#   <CHECKPOINT_ROOT>/wan2.1-t2v-1.3b/          Wan2.1 base（VAE / T5 / 模型配置）
#   <CHECKPOINT_ROOT>/chunkwise/causal_forcing.pt
#   <CHECKPOINT_ROOT>/framewise/causal_forcing.pt
#   <CHECKPOINT_ROOT>/causal-forcing++/framewise-2step.pt   （EXTRAS=1）
#   <CHECKPOINT_ROOT>/causal-forcing++/framewise-1step.pt   （EXTRAS=1）
#   <CHECKPOINT_ROOT>/chunkwise/longvideo.pt                （LONG_VIDEO=1）
#
# 用法：
#   bash scripts/Causal-Forcing/download_models.sh
#
# 常用可选变量：
#   CHECKPOINT_ROOT  下载目录（默认 checkpoints/causal_forcing）
#   EXTRAS           同时下载 Causal Forcing++ 1step/2step（0/1，默认 0）
#   LONG_VIDEO       同时下载 long video 权重（0/1，默认 0）
#   DRY_RUN          只打印命令不执行（0/1，默认 0）
#
set -euo pipefail

# shellcheck source=scripts/Causal-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

checkpoint_root="$(resolve_path "${CHECKPOINT_ROOT:-checkpoints/causal_forcing}")"
EXTRAS="${EXTRAS:-0}"
LONG_VIDEO="${LONG_VIDEO:-0}"
DRY_RUN="${DRY_RUN:-0}"

check_flag EXTRAS
check_flag LONG_VIDEO
check_flag DRY_RUN

if [[ "$checkpoint_root" == "/" ]]; then
  die "CHECKPOINT_ROOT 不能是文件系统根目录"
fi

if command -v hf >/dev/null 2>&1; then
  hf_bin=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  hf_bin=(huggingface-cli download)
else
  die "需要 Hugging Face CLI：pip install -U 'huggingface_hub[cli]'"
fi

declare -a commands=()
commands+=("${hf_bin[*]} Wan-AI/Wan2.1-T2V-1.3B --local-dir $checkpoint_root/wan2.1-t2v-1.3b")
commands+=("${hf_bin[*]} zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt --local-dir $checkpoint_root")
commands+=("${hf_bin[*]} zhuhz22/Causal-Forcing framewise/causal_forcing.pt --local-dir $checkpoint_root")
if [[ "$EXTRAS" == "1" ]]; then
  commands+=("${hf_bin[*]} zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir $checkpoint_root")
  commands+=("${hf_bin[*]} zhuhz22/Causal-Forcing causal-forcing++/framewise-1step.pt --local-dir $checkpoint_root")
fi
if [[ "$LONG_VIDEO" == "1" ]]; then
  commands+=("${hf_bin[*]} zhuhz22/Causal-Forcing chunkwise/longvideo.pt --local-dir $checkpoint_root")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[download_models] 干跑模式：打印将要执行的命令，不执行\n'
  for command in "${commands[@]}"; do
    printf '  $ %s\n' "$command"
  done
  exit 0
fi

mkdir -p -- "$checkpoint_root"
total="${#commands[@]}"
index=0
for command in "${commands[@]}"; do
  index=$(( index + 1 ))
  printf '\n[%d/%d] %s\n' "$index" "$total" "$command"
  # shellcheck disable=SC2086
  eval $command
done

printf '\n完成。权重目录：\n  %s\n\n' "$checkpoint_root"
printf 'BF16 生成：\n'
printf '  CHECKPOINT=%s/chunkwise/causal_forcing.pt \\\n' "$checkpoint_root"
printf '  MODEL_ROOT=%s/wan2.1-t2v-1.3b \\\n' "$checkpoint_root"
printf '  PROMPTS=assets/generation_prompts.txt \\\n'
printf '  bash scripts/Causal-Forcing/run_bf16.sh\n'
