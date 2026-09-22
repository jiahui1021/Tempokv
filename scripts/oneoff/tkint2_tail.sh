#!/bin/bash
L=/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/experiments/longcat/LongCat-720p
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
QVG=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
VPY=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/vbench/bin/python
UD=$(echo $CUDA_VISIBLE_DEVICES); G1=$(echo $UD|cut -d, -f1); DEV=$(echo $UD|cut -d, -f2,3,4)
M=TEMPOKV_KH2VH1_INT2; LOG(){ echo "$(date '+%F %T') $*" >> /tmp/tkint2_tail.log; }
LOG armed
while [ "$(ls -d $L/$M/*/0-0/refine_9.mp4 2>/dev/null|wc -l)" -lt 30 ]; do sleep 180; done
LOG "gen 30/30, streaming tail"
for i in 1 2 3; do
  c=$(ls $L/$M/.eval/eval_*.json 2>/dev/null|wc -l); [ $c -ge 30 ] && break
  GPU=GPU-76c6e561-7347-d298-3a78-d28a28490bc7 VDIR=$L/$M OUTDIR=$L/$M BF16_REF=$L/BF16 N=30 LOG=$L/$M/.eval_watch.log \
    IDLE_MIN=1200 bash $L/watch_refine720_int2_eval.sh >/dev/null 2>&1 & wp=$!
  for j in $(seq 80); do c=$(ls $L/$M/.eval/eval_*.json 2>/dev/null|wc -l); [ $c -ge 30 ] && break; sleep 30; done
  kill $wp 2>/dev/null
done
LOG "eval=$(ls $L/$M/.eval/eval_*.json|wc -l)/30"
mkdir -p $L/fid/$M; for i in $(seq 0 29); do ln -sf $L/$M/$i/0-0/refine_9.mp4 $L/fid/$M/${i}_0.mp4; done
[ -s $L/fid_results/${M}_fid.json ] || CUDA_VISIBLE_DEVICES=GPU-98a5a0dd-6859-105b-bb38-edef45706ca5 $QVG $T/scripts/evaluation/evaluate_quality.py \
  $L/fid/ref $L/fid/$M --output $L/fid_results/${M}_fid.json --device cuda --frame-stride 8 --fid-batch-size 32 >>/tmp/tkint2_tail.log 2>&1 && LOG "FID ok"
ST=$L/vbench720/stage_$M; mkdir -p $ST; for i in $(seq 0 29); do ln -sf $L/$M/$i/0-0/refine_9.mp4 $ST/${i}_0.mp4; done
cat > $L/vbench_long_manifest_tkint2.json <<J
{"schema_version":1,"expected_count":30,"prompt_text":"$T/assets/moviegen128_first30.txt",
 "entries":[{"name":"${M}_720p","lane":0,"videos":"$ST"}]}
J
LOG "vbench lane"
cd $T; export VBENCH_CACHE_DIR=/mnt/cpfs/epic-user/hujiahui-20260612/vbench_cache
CUDA_VISIBLE_DEVICES=$DEV $VPY scripts/evaluation/run_vbench_long_matrix.py --manifest $L/vbench_long_manifest_tkint2.json \
  --output-root $L/vbench720 --vbench-root /mnt/cpfs/epic-user/hujiahui-20260612/qvg/forcing/vbench \
  --cache-dir /mnt/cpfs/epic-user/hujiahui-20260612/vbench_cache --python $VPY --devices $DEV --idle-memory-mib 70000 >>/tmp/tkint2_tail.log 2>&1
LOG "done vb=$(ls $L/vbench720/full/${M}_720p/*_eval_results.json 2>/dev/null|wc -l)/8"; touch /tmp/tkint2_fill.done
