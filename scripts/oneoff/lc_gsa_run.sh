#!/bin/bash
# LongCat "整段仅1 anchor"(global_single_anchor):cumulative cond 窗,A=单span(1 anchor),B=分段span(每段anchor)
# 协议对齐 hybrid-bits:idx 固定 prompt、seed 0、9段273f、INIT=LongCat/base/<idx>/0-0.mp4、sa_* config + legacy-lh 参数
# 用法: bash lc_gsa_run.sh (nohup setsid 启动;重复启动可续排队列)
set -u
W=/mnt/cpfs/epic-user/hujiahui-20260612
T=$W/Tempokv
OUT=$W/kv-quant-result/LongCat/global_single_anchor
CKPT=$W/checkpoints/LongCat-Video
Q=$W/miniconda3/envs/qvg/bin
mkdir -p $OUT

PROMPT_0="$(sed -n 1p $W/kv-quant-result/shared/prompts/moviegen_128.txt)"
PROMPT_1="$(sed -n 2p $W/kv-quant-result/shared/prompts/moviegen_128.txt)"

job(){ # tag arm card idx
  local tag=$1 arm=$2 card=$3 idx=$4
  local pdir; case $tag in
    k2v2) pdir=legacy-lh-kh2-vh1-k2-v2 ;;
    k4v2) pdir=legacy-lh-kh2-vh1-k4-v2 ;;
    k6v4) pdir=legacy-lh-kh2-vh1-k6-v4-a6 ;;
    *) echo "bad tag $tag" >&2; return 2 ;;
  esac
  local dir=$OUT/${tag}_${arm}_$idx
  mkdir -p "$dir"
  cd "$T"
  CUDA_VISIBLE_DEVICES=$card TRITON_CACHE_DIR=/dev/shm/triton_home_gsa_${tag}_${arm}_$idx TMPDIR=/dev/shm/gsa_${tag}_${arm}_$idx \
  PYTHON_BIN=$Q/python TORCHRUN_BIN=$Q/torchrun PATH=$Q:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CHECKPOINT_DIR=$CKPT PROMPT="$( [ "$idx" = 0 ] && echo "$PROMPT_0" || echo "$PROMPT_1" )" PROMPT_IDX=$idx SEED=0 MODE=tempokv \
  CONFIG=results/paper/span-int2/configs/sa_$tag.json PARAM_DIR=data/longcat/params/$pdir \
  NUM_SEGMENTS=9 NUM_FRAMES=93 NUM_COND_FRAMES=73 \
  INIT_VIDEO_PATH=$W/kv-quant-result/LongCat/base/$idx/0-0.mp4 \
  CUMULATIVE_COND=1 OFFLOAD_KV_CACHE=1 \
  TEMPOKV_SPAN_SPLITS="$( [ "$arm" = B ] && echo "24+5" || echo "" )" \
  OUTPUT_DIR=$dir TRITON_CACHE=/dev/shm/triton_gsa_${tag}_${arm}_$idx \
  bash scripts/LongCat/generate.sh >> $dir/generate.log 2>&1
  echo "$(date '+%F %T') ${tag}_${arm}_$idx rc=$?" >> $OUT/launch_status.log
}

IFS=, read -ra UD <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#UD[@]}" -lt 4 ]; then echo "need 4 GPUs, got ${#UD[@]}: $CUDA_VISIBLE_DEVICES" >&2; exit 1; fi
c1=${UD[0]}; c2=${UD[1]}; c3=${UD[2]}; c4=${UD[3]}
( job k2v2 A $c1 0; job k6v4 A $c1 0 ) &
( job k2v2 B $c2 0; job k6v4 B $c2 0 ) &
( job k4v2 A $c3 0; job k4v2 A $c3 1 ) &
( job k4v2 B $c4 0; job k4v2 B $c4 1 ) &
wait
echo "$(date '+%F %T') ALL GSA END" >> $OUT/launch_status.log
