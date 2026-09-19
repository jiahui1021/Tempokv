#!/usr/bin/env bash
# run_mirror_serial.sh — 在单张卡上串行跑非对称镜像方向 v_to_k: k4v2 / k5v3 / k6v4。
#   (codec 的 v_to_k⇒k_bits==v_bits 约束已按需放宽)
# 用法: bash run_mirror_serial.sh <gpu-uuid>
# 每组: 先起 eval_stream_watch(逐条流式测评 vs SF BF16 idx0-29),再 fit+generate(并发于评测),
#       生成完 watcher 收尾全部 30 条后进下一组。幂等: summary complete 或 30 eval 齐则跳过。
set -uo pipefail
GPU="${1:-GPU-801636c6-7759-36df-3ee7-ad18e0d5cde9}"
TEMPOKV=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
QB=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin; QVG="$QB/python"
CKPT=/mnt/cpfs/epic-user/hujiahui-20260612/checkpoints/self_forcing_dmd.pt
WAN=/mnt/cpfs/epic-user/hujiahui-20260612/models--Wan-AI--Wan2.1-T2V-1.3B
REF=/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/experiments/self-forcing/Self-Forcing/bf16
export PYTHON_BIN="$QVG"
cd "$TEMPOKV"; mkdir -p logs
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$TEMPOKV/logs/mirror_serial.log"; }
log "serial driver 启动 gpu=$GPU 队列: k4v2 k5v3 k6v4"

for PAIR in "4 2" "5 3" "6 4"; do
  KB=${PAIR% *}; VB=${PAIR#* }
  TAG="k${KB}v${VB}"; ROOT="$TEMPOKV/results/direction/v_to_k_${TAG}"
  if grep -q '"status": "complete"' "$ROOT/summary.json" 2>/dev/null; then
    log "$TAG 已 complete,跳过"; continue
  fi
  mkdir -p "$ROOT"
  # 起流式评测 watcher(先起,边生成边评)
  VDIR="$ROOT/runs/v_to_k/videos" OUTDIR="$ROOT" BF16_REF="$REF" N=30 \
    LOG="$TEMPOKV/logs/mirror_${TAG}_watch.log" \
    setsid bash scripts/Self-Forcing/eval_stream_watch.sh "$TAG" "$GPU" 150 \
    >/dev/null 2>&1 &
  WPID=$!
  log "$TAG: watcher($WPID) 起, 开始 fit+generate k=$KB v=$VB"
  CUDA_VISIBLE_DEVICES="$GPU" $QVG scripts/paper_experiments/direction/run_quality.py \
    --checkpoint "$CKPT" --wan-model-dir "$WAN" \
    --prompts assets/moviegen128_first30.txt \
    --trace data/self-forcing/trace_legacy/clean_trace.npz \
    --config configs/legacy.json \
    --output-root "results/direction/v_to_k_${TAG}" \
    --arms v_to_k --frames 717 --seed 0 --start-index 0 --end-index 30 \
    --k-bits "$KB" --v-bits "$VB" --no-include-bf16 --no-quality \
    >>"$TEMPOKV/logs/mirror_${TAG}_gen.log" 2>&1
  log "$TAG: generate 结束 rc=$? , 等 watcher 收尾流式评测…"
  wait "$WPID" 2>/dev/null
  dn=$(ls "$ROOT"/eval_*.json 2>/dev/null | wc -l)
  log "$TAG 完成, 已评 $dn/30"
done
log "ALL SERIAL MIRROR DONE"
