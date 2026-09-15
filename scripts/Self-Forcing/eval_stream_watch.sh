#!/usr/bin/env bash
# eval_stream_watch.sh — 流式评测 watcher：盯某个组的 videos/ 目录，新 mp4 落盘立刻评测
# 单条评测输出 eval_<idx>.json（幂等：已有则跳过）。
# 用法: bash eval_stream_watch.sh <exp_name> <gpu_uuid> [timeout_min]
# 终止: 全部 30 条 eval 完成 或 无新 mp4 超过 timeout 分钟（默认 45）。
set -uo pipefail

EXP="$1"
GPU="$2"
IDLE_MIN="${3:-45}"

TEMPOKV=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
# VDIR/OUTDIR/BF16_REF/N/LOG 可用环境变量覆盖（默认保持 hybrid-bits 行为不变），
# 以便方向研究等新实验复用同一 watcher。
BF16_REF="${BF16_REF:-/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/experiments/self-forcing/Self-Forcing/bf16}"
QVG_BIN=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin
HB="$TEMPOKV/results/paper_experiments/hybrid-bits/$EXP"
VDIR="${VDIR:-$HB/videos}"
OUTDIR="${OUTDIR:-$HB}"
LOG="${LOG:-$TEMPOKV/logs/hyb_watch_$EXP.log}"
N="${N:-30}"
mkdir -p "$OUTDIR" "$(dirname "$LOG")"

echo "[watch] $EXP 开始盯 $VDIR (gpu=$GPU, idle_timeout=${IDLE_MIN}min) $(date '+%F %T')" >>"$LOG"

last_new=$(date +%s)
while true; do
  # 收集正式落盘的 idx（排除 .tmp- 前缀的写一半文件）
  idxs=()
  for f in "$VDIR"/*_0.mp4; do
    [[ -e "$f" ]] || continue
    b=$(basename "$f")
    [[ "$b" == .tmp-* ]] && continue
    if [[ -s "$f" ]]; then
      idxs+=("${b%%_0.mp4}")
    fi
  done

  # 对未评过的逐条评测
  new_this_round=0
  for idx in "${idxs[@]}"; do
    out="$OUTDIR/eval_${idx}.json"
    cand="$VDIR/${idx}_0.mp4"
    ref="$BF16_REF/${idx}_0.mp4"
    [[ -s "$ref" ]] || { echo "[watch] $EXP 缺参考 ${idx}_0.mp4" >>"$LOG"; continue; }
    [[ -s "$out" ]] && continue
    if [[ ! -s "$cand" ]]; then continue; fi

    tmp_ref="/tmp/hybw_ref_${EXP}_${idx}"
    tmp_cand="/tmp/hybw_cand_${EXP}_${idx}"
    rm -rf "$tmp_ref" "$tmp_cand"
    mkdir -p "$tmp_ref" "$tmp_cand"
    ln -sf "$ref" "$tmp_ref/video.mp4"
    ln -sf "$cand" "$tmp_cand/video.mp4"
    echo "[watch] $EXP 评测 idx=$idx 开始 $(date '+%T')" >>"$LOG"
    ( cd "$TEMPOKV"
      CUDA_VISIBLE_DEVICES="$GPU" "$QVG_BIN/python" scripts/evaluation/evaluate_quality.py \
        "$tmp_ref" "$tmp_cand" --no-fid --device cuda:0 --output "$out" \
        >"$TEMPOKV/logs/hyb_watch_eval_${EXP}_${idx}.log" 2>&1
    )
    rc=$?
    rm -rf "$tmp_ref" "$tmp_cand"
    if [[ -s "$out" ]]; then
      echo "[watch] $EXP idx=$idx 完成 rc=$rc $(date '+%T')" >>"$LOG"
    else
      echo "[watch] $EXP idx=$idx FAILED rc=$rc $(date '+%T')" >>"$LOG"
    fi
    new_this_round=$((new_this_round+1))
  done
  if [[ "$new_this_round" -gt 0 ]]; then
    last_new=$(date +%s)
  fi

  done_n=$(ls "$OUTDIR"/eval_*.json 2>/dev/null | wc -l)
  if [[ "$done_n" -ge "$N" ]]; then
    echo "[watch] $EXP 全部 $N 条 eval 完成，退出 $(date '+%T')" >>"$LOG"
    exit 0
  fi

  now=$(date +%s)
  idle=$(( (now - last_new) / 60 ))
  if [[ "$idle" -ge "$IDLE_MIN" ]] && [[ "$done_n" -gt 0 ]]; then
    echo "[watch] $EXP 无新 mp4 达 ${IDLE_MIN}min（已评 $done_n 条），退出 $(date '+%T')" >>"$LOG"
    exit 0
  fi
  sleep 15
done
