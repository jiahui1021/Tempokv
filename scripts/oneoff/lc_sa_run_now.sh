#!/bin/bash
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
OUT=/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/LongCat/single_anchor
CKPT=/mnt/cpfs/epic-user/hujiahui-20260612/checkpoints/LongCat-Video
Q=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin
PROMPT="$(head -1 /mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/shared/prompts/moviegen_128.txt)"
run_job(){
  local tag=$1 card=$2 pdir=$3
  for r in 1 2; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$card" 2>/dev/null || echo 999999)
    [ "$u" -lt 15000 ] || { sleep 180; return 1; }
    [ $r = 1 ] && sleep 170
  done
  cd "$T"
  # 单 UUID:整段 CVD 传给单卡生成器会让 4 个任务全挤到列表首卡
  # TRITON_CACHE_DIR/TMPDIR 离开 CPFS:重启后 fuse 旧 fd 报 Stale file handle
  CUDA_VISIBLE_DEVICES=$card TRITON_CACHE_DIR=/dev/shm/triton_home_$tag TMPDIR=/dev/shm/sa_$tag \
  PYTHON_BIN=$Q/python TORCHRUN_BIN=$Q/torchrun PATH=$Q:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CHECKPOINT_DIR=$CKPT PROMPT="$PROMPT" PROMPT_IDX=0 SEED=0 MODE=tempokv \
  CONFIG=results/paper/span-int2/configs/sa_$tag.json PARAM_DIR=data/longcat/params/$pdir \
  NUM_SEGMENTS=1 NUM_FRAMES=273 NUM_COND_FRAMES=53 \
  OUTPUT_DIR=$OUT/$tag TRITON_CACHE=/dev/shm/triton_sa_$tag \
  bash scripts/LongCat/generate.sh >> $OUT/launch_$tag.log 2>&1
  echo "$(date '+%F %T') $tag rc=$?" >> $OUT/launch_status.log
}
# 每卡一个,失败(卡忙)重试
UD=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); i=1
for spec in "k2v2 legacy-lh-kh2-vh1-k2-v2" "k4v2 legacy-lh-kh2-vh1-k4-v2" "k5v3 legacy-lh-kh2-vh1-k5-v3-a5" "k6v4 legacy-lh-kh2-vh1-k6-v4-a6"; do
  set -- $spec; card=$(echo $UD | cut -d' ' -f$i); i=$((i+1))
  ( for a in 1 2 3 4 5 6; do run_job $1 "$card" $2 && break; sleep 300; done ) &
done
wait
echo "$(date '+%F %T') ALL SA END" >> $OUT/launch_status.log
