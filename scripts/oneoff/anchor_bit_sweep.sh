#!/usr/bin/env bash
# anchor_bit_sweep.sh — anchor 位宽全面对比:k2v2/lag18/span24 固定,anchor∈{2..8,16},每组 30 条。
#   流式 PSNR/SSIM/LPIPS + FID(refs30) + VBench8。anchor4 不跑(聚合 spansweep30_lag18_L24)。
# 用法:
#   bash anchor_bit_sweep.sh                      # 主入口:等 pruning 退出后双卡接力
#   bash anchor_bit_sweep.sh --anchors "2 5" GPU-xxxx   # 任意机器/卡跑指定档(断点续)
# 前提:anchor{b} 档自动用 fit_params.sh 在 trace_legacy 上重拟合(签名含 anchor_bits)。
set -uo pipefail
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
QVG=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
VPY=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/vbench/bin/python
cd "$T"
R=$T/results/paper/span-int2/runs; D=$R/anchorsweep; mkdir -p "$D/configs"
REFS30=$R/span30sweep/refs30; PROV=$R/span30sweep/prompts_first30.json
export VBENCH_CACHE_DIR=/mnt/cpfs/epic-user/hujiahui-20260612/vbench_cache
G1=GPU-2ff1acd3-71e7-4032-9632-3880ba5ee31c
G2=GPU-c8e19653-0138-0223-68fe-5b7236baf470
DIMS=subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality,overall_consistency,clip_score
mkcfg(){ cat > "$D/configs/k2v2_anchor$1.json" <<J
{ "k_bits": 2, "v_bits": 2, "k_history_length": 2, "v_history_length": 1,
  "k_predictor_mode": "shared", "cross_kv_weight_mode": "layer_shared",
  "cross_matmul_precision": "ieee", "fit_mode": "legacy", "anchor_bits": $1 }
J
}
run_anchor(){
  local B=$1; local GPU=$2
  local LAB=anchorsweep_a${B}_30p; local RUN=$R/$LAB/pixel_717
  [[ -e $D/done_a${B} ]] && { echo "[$(date '+%F %T')] a$B 已完成,跳过" >>$D/driver.log; return 0; }
  local cfg param
  if [[ $B == 16 ]]; then
    cfg=$T/results/paper/span-int2/configs/residual_int2_anchor_bf16.json
    param=$T/results/paper/span-int2/params/int2_anchorBF16
  else
    mkcfg "$B"; cfg=$D/configs/k2v2_anchor$B.json; param=$T/data/self-forcing/params/k2v2-anchor$B
    if [[ ! -s $param/k_temporal.npz ]]; then
      echo "[$(date '+%F %T')] a$B 拟合开始" >>$D/driver.log
      CONFIG=$cfg TRACE_FILE=data/self-forcing/trace_legacy/clean_trace.npz SPAN_FRAMES=24 \
        PYTHON_BIN=$QVG PARAM_NAME=k2v2-anchor$B bash scripts/Self-Forcing/fit_params.sh >>"$D/fit_a${B}.log" 2>&1 || { echo "[$(date '+%F %T')] a$B 拟合失败" >>$D/driver.log; return 1; }
    fi
  fi
  mkdir -p "$RUN/videos"
  # 复用已跑过的同配置视频(anchor16 ← anchorcmp 10 条;idx0 ← 单条扫/anchorcmp)
  if [[ $B == 16 ]]; then for f in $R/anchorcmp_abf16_lag18_10p/pixel_717/videos/*_0.mp4; do [[ -e $f && ! -e $RUN/videos/$(basename $f) ]] && cp "$f" "$RUN/videos/"; done; fi
  echo "[$(date '+%F %T')] a$B 生成+流式+FID 开始(gpu=$GPU)" >>$D/driver.log
  CONFIG=$cfg BF16_REF=$REFS30 PARAM_DIR=$param TEMPOKV_BF16_LAG_FRAMES=18 LABEL=$LAB WATCH_IDLE=90 START=0 END=30 \
    bash scripts/paper_experiments/span/span_bigL.sh 24 "$GPU" >>"$D/a${B}.log" 2>&1
  echo "[$(date '+%F %T')] a$B 生成 rc=$?" >>$D/driver.log
  local V=$R/$LAB/vbench_long8
  if [[ ! -s $V/evaluation.json ]] && [[ $(ls $RUN/videos/*_0.mp4 2>/dev/null|wc -l) -ge 30 ]]; then
    mkdir -p "$V"
    PYTHONPATH="/mnt/cpfs/epic-user/hujiahui-20260612/qvg/forcing/vbench:$T/scripts/evaluation" \
    CUDA_VISIBLE_DEVICES="$GPU" "$VPY" scripts/evaluation/evaluate_vbench.py "$RUN/videos" \
      --output-dir "$V" --name "$LAB" --prompt-file "$PROV" --mode long_custom_input \
      --dimensions "$DIMS" --device cuda --local-checkpoints --reuse-complete --config-out "$V/evaluation.json" >>"$D/a${B}.log" 2>&1
    echo "[$(date '+%F %T')] a$B vbench rc=$?" >>$D/driver.log
  fi
  touch "$D/done_a${B}"
}
if [[ "${1:-}" == "--anchors" ]]; then for b in $2; do run_anchor "$b" "$3"; done; exit 0; fi
echo "[$(date '+%F %T')] relay: waiting pruning exit" >>$D/driver.log
while pgrep -f "pruning/run.sh" >/dev/null 2>&1; do sleep 120; done
echo "[$(date '+%F %T')] pruning done, anchor sweep start" >>$D/driver.log
nohup bash "$T/scripts/oneoff/anchor_bit_sweep.sh" --anchors "2 5 7" $G1 >>"$D/c1.out" 2>&1 &
nohup bash "$T/scripts/oneoff/anchor_bit_sweep.sh" --anchors "3 6 8 16" $G2 >>"$D/c2.out" 2>&1 &
wait; echo "[$(date '+%F %T')] ANCHOR SWEEP ALL DONE" >>$D/driver.log
