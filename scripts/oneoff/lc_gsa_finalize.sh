#!/bin/bash
# GSA 收尾:每个 A 臂 run 收片后自动评 vs BF16 / vs 滑窗基线 / (k2v2) A vs B,最后写 SUMMARY.md
# 日志与输出全部落在各 run 目录内。setsid nohup 长跑,卡空闲才起评测。
set -u
W=${WORKSPACE:?set WORKSPACE to the experiment root}
T=$W/Tempokv
R=$W/kv-quant-result/LongCat
O=$R/global_single_anchor
Q=$W/miniconda3/envs/qvg/bin/python
UD=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')

free_card(){ while :; do for c in $UD; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 999999); [ "$u" -lt 15000 ] && { echo "$c"; return; }; done; sleep 120; done; }

evalpair(){ # runname reffile outjson
  local run=$1 ref=$2 out=$3 d=$O/$run
  mkdir -p $d/.eval/cand $d/.eval/ref
  ln -sf $d/longcat.mp4 $d/.eval/cand/0_0.mp4
  ln -sf "$ref" $d/.eval/ref/0_0.mp4
  local c; c=$(free_card)
  CUDA_VISIBLE_DEVICES=$c TRITON_CACHE_DIR=/dev/shm/triton_home_eval_$run TMPDIR=/dev/shm/gsa_eval_$run \
  $Q $T/scripts/evaluation/evaluate_quality.py $d/.eval/ref $d/.eval/cand \
    --output "$out" --device cuda --frame-stride 8 --no-fid >> $d/eval.log 2>&1
  echo "$(date '+%F %T') eval $run -> $(basename $out) rc=$?" >> $d/eval.log
}

base_ref(){ case $1 in
  k2v2) echo $R/TEMPOKV_KH2VH1_INT2/gen/$2/longcat.mp4 ;;
  *) echo $R/hybrid-bits/$1/gen/$2/longcat.mp4 ;;
esac; }

seen=""
while :; do
  for d in $O/k*_*/; do
    run=$(basename $d); case " $seen " in *" $run "*) continue;; esac
    mp4=$(find $d -maxdepth 1 -name longcat.mp4 -size +100k | head -1); [ -n "$mp4" ] || continue
    tag=${run%_*}; arm=${run#*_}; arm=${arm%_*}; idx=${run##*_}
    seen="$seen $run"
    echo "$(date '+%F %T') eval start $run" >> $O/finalize.log
    mkdir -p $d/eval
    evalpair $run "$R/BF16/$idx/longcat.mp4" $d/eval/vs_bf16.json
    [ -f "$(base_ref $tag $idx)" ] && evalpair $run "$(base_ref $tag $idx)" $d/eval/vs_sliding_anchor.json
    [ "$tag" = k2v2 ] && [ "$arm" = A ] && [ -s $O/k2v2_B_0/longcat.mp4 ] && evalpair $run $O/k2v2_B_0/longcat.mp4 $d/eval/vs_B_pair.json
    echo "$(date '+%F %T') eval done $run" >> $O/finalize.log
  done
  pgrep -f "lc_gsa_run.sh|longcat/generate.py" >/dev/null || { echo "$(date '+%F %T') generators gone, finalize end" >> $O/finalize.log; break; }
  sleep 120
done

$Q - <<'PY'
import json, os, time
from pathlib import Path
O = Path(os.environ["WORKSPACE"]) / "kv-quant-result/LongCat/global_single_anchor"
lines = ["# LongCat 整段全局单 anchor(global_single_anchor)— A 臂每组 1 条",
         "",
         "协议:idx0(seed0,INIT=base/0/0-0.mp4,9段273f,cumulative cond 窗,OFFLOAD);A=单 span(全片 1 anchor),B=[24+5×n] 分段 span(每段 anchor)。",
         "k2v2 B 臂已收片,idx0 有 A/B 配对;其余仅 A。", "",
         "| run | ratio | peak GB | vs BF16 PSNR/SSIM/LPIPS | vs 滑窗9-anchor基线 | vs B(配对) |", "|---|---|---|---|---|---|"]
for d in sorted(O.glob("k*_*")):
    rep = list(d.glob("tempokv_bridge_report*.json"))
    if not (d/"longcat.mp4").is_file(): continue
    ratio = gb = ""
    if rep:
        r = json.loads(rep[0].read_text())
        ratio = f"{r['compression_ratio']:.3f}"; gb = f"{r['peak_gpu_allocated_bytes']/2**30:.1f}"
    def e(n):
        p = d/"eval"/n
        if not p.is_file(): return "—"
        j = json.loads(p.read_text())
        return f"{j.get('psnr_db',0):.2f}/{j.get('ssim',0):.3f}/{j.get('lpips',0):.4f}"
    lines.append(f"| {d.name} | {ratio} | {gb} | {e('vs_bf16.json')} | {e('vs_sliding_anchor.json')} | {e('vs_B_pair.json')} |")
lines += ["", "注意:cumulative 窗与滑窗基线的注意力语义不同,vs BF16/基线的绝对值含内容漂移成分,只能当量级参考;",
          "k2v2 的 A vs B 是唯一严格配对的 anchor 数消融。", "更新:" + time.strftime("%F %H:%M") + "(lc_gsa_finalize.sh)"]
(O/"SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo "$(date '+%F %T') SUMMARY written" >> $O/finalize.log
