#!/usr/bin/env bash
# span30_full_eval.sh — 717f span 扫描扩到每组 30 条(idx0-29, lag=18, int4 anchor)。
# 全指标:PSNR/SSIM/LPIPS 流式 CSV(评一条进一条)+ 目录 FID + VBench-Long 8 维。
# 断点续:per-point done_L{L};vbench 见 evaluation.json 即跳;生成 skip-existing;idx0 复用旧扫描。
# 用法:主入口直接跑(内部起两卡链并 wait);单卡链:bash 本文件 --chain1|--chain2
set -uo pipefail
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
QVG=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
VPY=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/vbench/bin/python
VBENCH_ROOT=/mnt/cpfs/epic-user/hujiahui-20260612/qvg/forcing/vbench
export VBENCH_CACHE_DIR=/mnt/cpfs/epic-user/hujiahui-20260612/vbench_cache
cd "$T"
R=$T/results/paper/span-int2/runs
D=$R/span30sweep; mkdir -p "$D"
REFS30=$D/refs30; PROV=$D/prompts_first30.json
BF16SRC=/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/experiments/self-forcing/Self-Forcing/bf16
# ---- 公共准备(幂等)----
mkdir -p "$REFS30"
for i in $(seq 0 29); do [[ -e $BF16SRC/${i}_0.mp4 ]] && ln -sf $BF16SRC/${i}_0.mp4 $REFS30/${i}_0.mp4; done
[[ -s $PROV ]] || $QVG - <<'PY'
import json
from pathlib import Path
pl=Path('/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/assets/moviegen128_first30.txt').read_text().splitlines()
Path('/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/paper/span-int2/runs/span30sweep/prompts_first30.json').write_text(
    json.dumps({f'{i}_0.mp4': pl[i] for i in range(30)}, ensure_ascii=False, indent=1))
PY
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$D/driver.log"; }
G1=GPU-2ff1acd3-71e7-4032-9632-3880ba5ee31c
G2=GPU-c8e19653-0138-0223-68fe-5b7236baf470
DIMS=subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality,overall_consistency,clip_score

run_point(){
  local LL=$1 GPU=$2 LAB="spansweep30_lag18_L$1"
  [[ -e $D/done_L${LL} ]] && { log "L$LL 已完成,跳过"; return 0; }
  local RUN=$R/$LAB/pixel_717; mkdir -p "$RUN/videos"
  local old=$R/_archive/spansweep1_idx0_0919/spansweep_lag18_L${LL}/pixel_717/videos/0_0.mp4
  [[ -s $old && ! -s $RUN/videos/0_0.mp4 ]] && cp "$old" "$RUN/videos/0_0.mp4"
  log "L$LL 生成+流式+FID 开始(gpu=$GPU)"
  BF16_REF=$REFS30 TEMPOKV_BF16_LAG_FRAMES=18 LABEL=$LAB WATCH_IDLE=90 START=0 END=30 \
    bash scripts/paper_experiments/span/span_bigL.sh "$LL" "$GPU" >>"$D/L${LL}.log" 2>&1
  log "L$LL 生成/FID rc=$?"
  local VOUT=$R/$LAB/vbench_long8; mkdir -p "$VOUT"
  if [[ ! -s $VOUT/evaluation.json ]]; then
    log "L$LL vbench8 开始"
    PYTHONPATH="$VBENCH_ROOT:$T/scripts/evaluation" CUDA_VISIBLE_DEVICES="$GPU" \
      "$VPY" "$T/scripts/evaluation/evaluate_vbench.py" "$RUN/videos" \
      --output-dir "$VOUT" --name "$LAB" --prompt-file "$PROV" \
      --mode long_custom_input --dimensions "$DIMS" \
      --device cuda --local-checkpoints --reuse-complete --config-out "$VOUT/evaluation.json" \
      >>"$D/L${LL}.log" 2>&1
    log "L$LL vbench8 rc=$?"
  fi
  touch "$D/done_L${LL}"
}

case "${1:-main}" in
  --chain1) for LL in 4 180; do run_point "$LL" "$G1"; done; touch "$D/chain1.done"; exit 0 ;;
  --chain2) for LL in 8 64; do run_point "$LL" "$G2"; done; touch "$D/chain2.done"; exit 0 ;;
  --points) shift; PTS="$1"; shift; GPU="$1"; shift || true
            for LL in $PTS; do run_point "$LL" "$GPU"; done
            touch "$D/points_${PTS// /_}.done"; exit 0 ;;
esac

log "span30 driver start"
nohup bash "$T/scripts/oneoff/span30_full_eval.sh" --chain1 >>"$D/chain1.out" 2>&1 &
C1=$!
nohup bash "$T/scripts/oneoff/span30_full_eval.sh" --chain2 >>"$D/chain2.out" 2>&1 &
C2=$!
log "chains: $C1(card1 L4,12,24,48,96,180) $C2(card2 L8,16,36,64,128)"
wait "$C1"; wait "$C2"
log "span30 ALL DONE"
