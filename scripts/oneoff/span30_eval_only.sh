#!/usr/bin/env bash
# span30_eval_only.sh <L> <gpu-uuid> — 只做已生成完点的 FID + VBench8(带锁防双跑、断点续)。
# 适用:L16/48/128(30 条视频已在盘上,等评测)。生成未完成勿用(等 done 或走 --points)。
set -uo pipefail
LL="$1"; GPU="$2"
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
QVG=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
VPY=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/vbench/bin/python
cd "$T"
R=$T/results/paper/span-int2/runs; D=$R/span30sweep; RUN=$R/spansweep30_lag18_L${LL}/pixel_717
V=$R/spansweep30_lag18_L${LL}/vbench_long8
LOCK=$D/eval_L${LL}.lock
if [[ -e $LOCK ]] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "L$LL 评测已被 PID $(cat $LOCK) 占用,退出"; exit 0; fi
echo $$ > "$LOCK"
n=$(ls "$RUN/videos/"*_0.mp4 2>/dev/null | wc -l)
echo "[evalonly] L$LL videos=$n/30 $(date '+%T')" >> "$D/evalonly.log"
if [[ $n -lt 30 ]]; then echo "L$LL 生成未齐($n/30),跳过"; rm -f "$LOCK"; exit 1; fi
[[ -s $RUN/quality.json ]] || {
  echo "[evalonly] L$LL FID 开始 $(date '+%T')" >> "$D/evalonly.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$QVG" scripts/evaluation/evaluate_quality.py "$D/refs30" "$RUN/videos" \
    --device cuda:0 --frame-stride 1 --fid-batch-size 8 --output "$RUN/quality.json" >> "$D/evalonly.log" 2>&1
  echo "[evalonly] L$LL FID rc=$? $(date '+%T')" >> "$D/evalonly.log"; }
RECENT_VB=$(find "$V" -name "*.json" -mmin -30 2>/dev/null | head -1)
if [[ -n $RECENT_VB ]]; then echo "[evalonly] L$LL vbench 30分钟内有新产出,判定他链在跑,跳过" >> "$D/evalonly.log"; fi
if [[ ! -s $V/evaluation.json && -z $RECENT_VB ]]; then
  mkdir -p "$V"
  echo "[evalonly] L$LL vbench8 开始 $(date '+%T')" >> "$D/evalonly.log"
  PYTHONPATH="/mnt/cpfs/epic-user/hujiahui-20260612/qvg/forcing/vbench:$T/scripts/evaluation" \
  VBENCH_CACHE_DIR=/mnt/cpfs/epic-user/hujiahui-20260612/vbench_cache \
  CUDA_VISIBLE_DEVICES="$GPU" "$VPY" scripts/evaluation/evaluate_vbench.py "$RUN/videos" \
    --output-dir "$V" --name "spansweep30_lag18_L${LL}" --prompt-file "$D/prompts_first30.json" \
    --mode long_custom_input \
    --dimensions subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality,overall_consistency,clip_score \
    --device cuda --local-checkpoints --reuse-complete --config-out "$V/evaluation.json" >> "$D/evalonly.log" 2>&1
  echo "[evalonly] L$LL vbench rc=$? $(date '+%T')" >> "$D/evalonly.log"; fi
rm -f "$LOCK"; echo "[evalonly] L$LL ALL DONE $(date '+%T')" >> "$D/evalonly.log"
