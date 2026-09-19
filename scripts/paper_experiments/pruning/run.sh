#!/usr/bin/env bash
# Token pruning pilot（reports/pruning.md）：BF16、未剪枝 TempoKV、5 个准则 × 4 个比例。
#
# 用法：
#   CHECKPOINT=/models/self_forcing_dmd.pt \
#   WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/moviegen128.txt \
#   PARAM_DIR=data/self-forcing/params/legacy-kh2-vh1-int4-layershared \
#   bash scripts/paper_experiments/pruning/run.sh
#
# 可选变量：OUTPUT_ROOT、FRAMES（默认 237 = 60 latent 帧）、START_INDEX / END_INDEX
#   （默认前 8 条 prompt）、CRITERIA、RATIOS、TEMPOKV_PRUNE_PROTECT（默认 6 帧）、
#   TEMPOKV_PRUNE_LAMBDA、QUALITY_FID=1。CONFIG / KH / BITS 等照常传给 generate.sh。
# 已完成的 run 和质量报告会跳过，中断后直接重跑即可续上。
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$here/../../.." && pwd -P)"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin=python3
: "${CHECKPOINT:?}" "${WAN_MODEL_DIR:?}" "${PROMPTS:?}" "${PARAM_DIR:?}"

out="${OUTPUT_ROOT:-$root/results/self_forcing/pruning_pilot}"
criteria="${CRITERIA:-random attention similarity residual attn_residual}"
ratios="${RATIOS:-0.10 0.25 0.50 0.75}"
export FRAMES="${FRAMES:-237}" START_INDEX="${START_INDEX:-0}" END_INDEX="${END_INDEX:-8}"
export SEED="${SEED:-0}" NUM_GPUS=1 SKIP_EXISTING=0
mkdir -p "$out/logs"

generate() {  # generate <run name> <mode> [criterion ratio]
  local name="$1" run="$out/runs/$1"
  if grep -q '"status": "complete"' "$run/pipeline_report.json" 2>/dev/null; then
    return
  fi
  echo "[pruning] $name"
  local -a prune=()
  if [[ $# -eq 4 ]]; then
    rm -rf "$run/prune_stats"
    prune=(TEMPOKV_PRUNE_CRITERION="$3" TEMPOKV_PRUNE_RATIO="$4"
           TEMPOKV_PRUNE_STATS_DIR="$run/prune_stats")
  fi
  env MODE="$2" OUTPUT_DIR="$run" ${prune[@]+"${prune[@]}"} \
    bash "$root/scripts/Self-Forcing/generate.sh" >"$out/logs/$name.log" 2>&1
}

quality() {  # quality <reference run> <candidate run>
  local report="$out/quality/vs_$1/$2.json"
  [[ -f "$report" ]] && return
  mkdir -p "$(dirname "$report")"
  local -a fid=(--no-fid)
  [[ "${QUALITY_FID:-0}" == "1" ]] && fid=()
  "$python_bin" "$root/scripts/evaluation/evaluate_quality.py" \
    "$out/runs/$1/videos" "$out/runs/$2/videos" --output "$report" ${fid[@]+"${fid[@]}"} \
    >"$out/logs/quality-vs_$1-$2.log" 2>&1
}

generate bf16 bf16
generate none tempokv none 0
runs=(none)
for criterion in $criteria; do
  for ratio in $ratios; do
    generate "${criterion}_$ratio" tempokv "$criterion" "$ratio"
    runs+=("${criterion}_$ratio")
  done
done

for name in "${runs[@]}"; do
  quality bf16 "$name"
  [[ "$name" == none ]] || quality none "$name"
done

"$python_bin" "$here/analyze.py" "$out"
