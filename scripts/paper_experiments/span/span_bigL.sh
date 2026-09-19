#!/usr/bin/env bash
# span_bigL.sh — 大 L span 消融 worker: L>24 复用主表 k2v2(24帧窗)参数,仅把运行时 span 设大。
# 用法: bash span_bigL.sh <L> <gpu-uuid>
#   generate(k2v2 params, span=L, idx0-29) 与 流式评测 watcher 并发; 完事导 per-video CSV。
set -uo pipefail
L="$1"; GPU="$2"
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
QVG=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
KV2="${PARAM_DIR:-$T/data/self-forcing/params/legacy-kh2-vh1-k2-v2-layershared}"
CFG="${CFG:-$T/results/paper/span-int2/configs/residual_int2_anchor_int4.json}"
PROMPTS="${PROMPTS:-$T/assets/moviegen128_first30.txt}"
START="${START:-0}"; END="${END:-30}"
CKPT=/mnt/cpfs/epic-user/hujiahui-20260612/checkpoints/self_forcing_dmd.pt
WAN=/mnt/cpfs/epic-user/hujiahui-20260612/models--Wan-AI--Wan2.1-T2V-1.3B
REF=/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/experiments/self-forcing/Self-Forcing/bf16
LABEL="${LABEL:-int2_anchor4_span${L}}"
RUN="$T/results/paper/span-int2/runs/${LABEL}/pixel_717"
LOG="$RUN/span_bigL.log"   # 日志放该 run 的视频输出文件夹内(硬规则)
mkdir -p "$RUN/videos" "$(dirname "$LOG")"; cd "$T"
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }
if [[ -s "$RUN/pipeline_report.json" && "${FORCE:-0}" != 1 ]]; then log "span$L 已完成,跳过"; exit 0; fi
log "start span=$L on $GPU (reuse k2v2 params, runtime span=L, fit/nmse skipped)"
# 流式评测 watcher(先起,边生成边评 idx0-29 vs BF16)
VDIR="$RUN/videos" OUTDIR="$RUN" BF16_REF="$REF" N="$END" LOG="$RUN/eval_watch.log" \
  setsid bash "$T/scripts/Self-Forcing/eval_stream_watch.sh" "${EXP:-bigL${L}_${LABEL}}" "$GPU" "${WATCH_IDLE:-150}" >/dev/null 2>&1 &
WPID=$!
export WAN_MODEL_DIR="$WAN" FORCING_WAN_MODEL_DIR="$WAN" CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$T/src:$T/experiments/Self-Forcing" TRITON_CACHE_DIR="/dev/shm/triton_${LABEL}"
$QVG experiments/Self-Forcing/tempokv_adapter/generate.py --mode tempokv \
  --checkpoint "$CKPT" --prompts "$PROMPTS" --output-dir "$RUN" \
  --frames 717 --samples 1 --seed 0 --start-index "$START" --end-index "$END" --num-gpus 1 --skip-existing \
  --config "$CFG" --k-params "$KV2/k_temporal.npz" --v-params "$KV2/v_cross_kv.npz" \
  --span-frames "$L" >>"$LOG" 2>&1
log "generate done rc=$?, 等 watcher 收尾流式评测"
wait "$WPID" 2>/dev/null
# 目录级 quality(含 FID)
$QVG scripts/evaluation/evaluate_quality.py "$REF" "$RUN/videos" --output "$RUN/quality.json" --device cuda:0 >>"$LOG" 2>&1 && log "quality.json(FID) done" || log "quality FAIL"
log "span$L ALL DONE"
