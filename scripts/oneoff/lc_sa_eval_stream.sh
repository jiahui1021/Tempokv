#!/bin/bash
# 出一组评一组:每个 tag 的 SA 视频 vs BF16/0 与 vs 3-anchor,写 single_anchor/eval/*.json
set -u
W=/mnt/cpfs/epic-user/hujiahui-20260612
T=$W/Tempokv
R=$W/kv-quant-result/LongCat
OUT=$R/single_anchor
E=$OUT/.eval
Q=$W/miniconda3/envs/qvg/bin/python
UD=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')
LOG(){ echo "$(date '+%F %T') $*" >> $OUT/eval_after.log; }
vid(){ find $OUT/$1 -name longcat.mp4 -size +100k 2>/dev/null | head -1; }
ref3(){ case $1 in k2v2) echo $R/TEMPOKV_KH2VH1_INT2/gen/0/longcat.mp4;; *) echo $R/hybrid-bits/$1/gen/0/longcat.mp4;; esac; }
free_card(){ # 挑一张占用 <15GB 的租约卡,没有就等
  while :; do for c in $UD; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 999999); [ "$u" -lt 15000 ] && { echo "$c"; return; }; done; sleep 60; done
}
run_eval(){ # $1=tag $2=ref名 $3=card
  CUDA_VISIBLE_DEVICES=$3 TRITON_CACHE_DIR=/dev/shm/triton_home_eval TMPDIR=/dev/shm/sa_eval \
  $Q $T/scripts/evaluation/evaluate_quality.py $E/$1/$2 $E/$1/cand \
    --output $OUT/eval/$1_vs_$2.json --device cuda --frame-stride 8 --max-frames 35 --no-fid \
    >> $OUT/eval_after.log 2>&1
  LOG "eval $1 vs $2 rc=$?"
}
mkdir -p /dev/shm/triton_home_eval /dev/shm/sa_eval
done_tags=""
while :; do
  for t in k2v2 k4v2 k5v3 k6v4; do
    case " $done_tags " in *" $t "*) continue;; esac
    v=$(vid $t) || true
    [ -n "$v" ] || continue
    mkdir -p $E/$t/cand $E/$t/bf16 $E/$t/3a $OUT/eval
    ln -sf "$v" $E/$t/cand/0_0.mp4
    ln -sf $R/BF16/0/longcat.mp4 $E/$t/bf16/0_0.mp4
    ln -sf "$(ref3 $t)" $E/$t/3a/0_0.mp4
    c=$(free_card); run_eval $t bf16 "$c"
    c=$(free_card); run_eval $t 3a "$c"
    done_tags="$done_tags $t"; LOG "tag $t done ($done_tags)"
  done
  [ $(echo $done_tags | wc -w) -eq 4 ] && break
  pgrep -f "longcat/generate.py" >/dev/null || { LOG "generators gone, tags done:$done_tags"; break; }
  sleep 60
done
LOG "paired evals=$(ls $OUT/eval/*.json 2>/dev/null|wc -l)/8 — 交给 lc_sa_summary.sh 出表"
