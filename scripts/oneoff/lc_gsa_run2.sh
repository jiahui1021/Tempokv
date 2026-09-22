#!/bin/bash
# global_single_anchor 第二批:k2v2 AB×idx1-3, k4v2 AB×idx2-3, k6v4 AB×idx1-3(16 条,拼成 3tag×2arm×idx0-3=24)
# 每张卡一个 worker,等卡空闲(<15GB)再接活;与 lc_gsa_run.sh 的在跑队列共享卡,不冲突。
set -u
W=/mnt/cpfs/epic-user/hujiahui-20260612
T=$W/Tempokv
OUT=$W/kv-quant-result/LongCat/global_single_anchor
CKPT=$W/checkpoints/LongCat-Video
Q=$W/miniconda3/envs/qvg/bin
PROMPTS_F=$W/kv-quant-result/shared/prompts/moviegen_128.txt
mkdir -p $OUT

job(){ # tag arm card idx
  local tag=$1 arm=$2 card=$3 idx=$4
  local pdir; case $tag in
    k2v2) pdir=legacy-lh-kh2-vh1-k2-v2 ;;
    k4v2) pdir=legacy-lh-kh2-vh1-k4-v2 ;;
    k6v4) pdir=legacy-lh-kh2-vh1-k6-v4-a6 ;;
    *) echo "bad tag $tag" >&2; return 2 ;;
  esac
  # 双查防抢卡:run1 排队作业刚启动 90s 内显存还没吃满,单查会误判空闲
  while :; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$card" 2>/dev/null || echo 999999)
    if [ "$u" -lt 15000 ]; then
      sleep 75
      u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$card" 2>/dev/null || echo 999999)
      [ "$u2" -lt 15000 ] && break
    fi
    sleep 120
  done
  local dir=$OUT/${tag}_${arm}_$idx
  mkdir -p "$dir"
  cd "$T"
  CUDA_VISIBLE_DEVICES=$card TRITON_CACHE_DIR=/dev/shm/triton_home_gsa_${tag}_${arm}_$idx TMPDIR=/dev/shm/gsa_${tag}_${arm}_$idx \
  PYTHON_BIN=$Q/python TORCHRUN_BIN=$Q/torchrun PATH=$Q:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CHECKPOINT_DIR=$CKPT PROMPT="$(sed -n $((idx+1))p $PROMPTS_F)" PROMPT_IDX=$idx SEED=0 MODE=tempokv \
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
[ "${#UD[@]}" -ge 4 ] || { echo "need 4 GPUs" >&2; exit 1; }
JOBS=(
  "k2v2 A 1" "k2v2 B 1" "k4v2 A 2" "k4v2 B 2"
  "k2v2 A 2" "k2v2 B 2" "k6v4 A 1" "k6v4 B 1"
  "k2v2 A 3" "k2v2 B 3" "k4v2 A 3" "k4v2 B 3"
  "k6v4 A 2" "k6v4 B 2" "k6v4 A 3" "k6v4 B 3"
)
for w in 0 1 2 3; do
  (
    sleep $((w * 80))   # worker 错峰轮询,避免同卡误判
    for ((i=w; i<${#JOBS[@]}; i+=4)); do
      set -- ${JOBS[i]}
      job $1 $2 ${UD[w]} $3
    done
  ) &
done
wait
echo "$(date '+%F %T') ALL GSA2 END" >> $OUT/launch_status.log
