#!/bin/bash
# single_anchor 生成完成后:每组 vs BF16/0 与 vs 3-anchor 基线,汇总 SUMMARY.md,再接 tkint2 VBench
set -u
W=/mnt/cpfs/epic-user/hujiahui-20260612
T=$W/Tempokv
R=$W/kv-quant-result/LongCat
OUT=$R/single_anchor
E=$OUT/.eval
Q=$W/miniconda3/envs/qvg/bin/python
TAGS="k2v2 k4v2 k5v3 k6v4"
UD=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')
LOG(){ echo "$(date '+%F %T') $*" >> $OUT/eval_after.log; }
vid(){ find $OUT/$1 -name longcat.mp4 -size +100k 2>/dev/null | head -1; }
ref3(){ case $1 in k2v2) echo $R/TEMPOKV_KH2VH1_INT2/gen/0/longcat.mp4;; *) echo $R/hybrid-bits/$1/videos/0_0.mp4;; esac; }

LOG "armed; waiting for 4 videos"
for i in $(seq 720); do
  n=0; for t in $TAGS; do [ -n "$(vid $t)" ] && n=$((n+1)); done
  [ $n -eq 4 ] && break
  # 生成器全部退出且缺片 → 用已有的收工
  if ! pgrep -f "lc_sa_run_now.sh|longcat/generate.py" >/dev/null; then LOG "generators gone at n=$n"; break; fi
  sleep 60
done
LOG "videos ready: $(for t in $TAGS; do printf '%s=%s ' $t "$([ -n "$(vid $t)" ] && echo y || echo n)"; done)"

# 摊平成同名 0_0.mp4，evaluate_quality 按文件名配对
for t in $TAGS; do
  mkdir -p $E/$t/cand $E/$t/bf16 $E/$t/3a $OUT/eval
  ln -sf "$(vid $t)" $E/$t/cand/0_0.mp4
  ln -sf $R/BF16/0/longcat.mp4 $E/$t/bf16/0_0.mp4
  ln -sf "$(ref3 $t)" $E/$t/3a/0_0.mp4
done

run_eval(){ # $1=tag $2=ref名 $3=card uuid
  CUDA_VISIBLE_DEVICES=$3 TRITON_CACHE_DIR=/dev/shm/triton_home_eval TMPDIR=/dev/shm/sa_eval \
  $Q $T/scripts/evaluation/evaluate_quality.py $E/$1/$2 $E/$1/cand \
    --output $OUT/eval/$1_vs_$2.json --device cuda --frame-stride 8 --max-frames 35 --no-fid \
    >> $OUT/eval_after.log 2>&1
  echo "$(date '+%F %T') eval $1 vs $2 rc=$?" >> $OUT/eval_after.log
}
mkdir -p /dev/shm/triton_home_eval /dev/shm/sa_eval
i=1
for ref in bf16 3a; do
  for t in $TAGS; do
    card=$(echo $UD | cut -d' ' -f$i)
    [ -s $OUT/eval/${t}_vs_${ref}.json ] || { run_eval $t $ref "$card" & }
    i=$((i % 4 + 1))
  done
  wait
done
LOG "eval jsons=$(ls $OUT/eval/*.json 2>/dev/null | wc -l)/8"

$Q - "$OUT" "$R" <<'PY' >> $OUT/eval_after.log 2>&1
import json, sys, time
from pathlib import Path

out, r = Path(sys.argv[1]), Path(sys.argv[2])
TAGS = ("k2v2", "k4v2", "k5v3", "k6v4")
BASE3 = {"k2v2": "TEMPOKV_KH2VH1_INT2/gen/0", "k4v2": "hybrid-bits/k4v2/gen/0",
         "k5v3": "hybrid-bits/k5v3/gen/0", "k6v4": "hybrid-bits/k6v4/gen/0"}


def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def rep(dirpath):
    files = sorted(dirpath.rglob("tempokv_bridge_report*.json"))
    return load(files[0]) if files else {}


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


head = ("| 组 | K/V/anchor | 未压缩缓存 GB | 压缩比 | 峰值显存 GB | "
        "vs BF16 PSNR | SSIM | LPIPS | vs 3-anchor PSNR | SSIM | LPIPS | 3-anchor 压缩比 |")
sep = "|---" * (len(head.split("|")) - 2) + "|"
lines = [
    "# LongCat single_anchor(单 anchor)汇总",
    "",
    "口径:idx0 单条、480p、NUM_SEGMENTS=1、kh2-vh1 + layer_head 协议;"
    "SA 出片为 493 帧,参考为 273 帧,paired 指标取前 273 帧前缀、MSE-pooled、frame-stride 8(35 帧);"
    "对照 = 同条 `BF16/0/longcat.mp4` 与同位宽 3-anchor 基线。n=1 条,FID/VBench 不适用。",
    "",
    head, sep,
]
for tag in TAGS:
    sa = rep(out / tag)
    bf = load(out / "eval" / f"{tag}_vs_bf16.json")
    th = load(out / "eval" / f"{tag}_vs_3a.json")
    b3 = rep(r / BASE3[tag])
    cells = [
        tag,
        "{}/{}/{}".format(sa.get("k_bits", "?"), sa.get("v_bits", "?"), sa.get("anchor_bits", "?")),
        f((sa.get("uncompressed_cache_bytes") or 0) / 2**30, 1),
        f(sa.get("compression_ratio")),
        f((sa.get("peak_gpu_allocated_bytes") or 0) / 2**30, 1),
        f(bf.get("psnr_db")), f(bf.get("ssim"), 4), f(bf.get("lpips"), 4),
        f(th.get("psnr_db")), f(th.get("ssim"), 4), f(th.get("lpips"), 4),
        f(b3.get("compression_ratio")),
    ]
    lines.append("| " + " | ".join(str(c) for c in cells) + " |")
lines += ["", "更新:" + time.strftime("%Y-%m-%d %H:%M")]
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
LOG "SUMMARY.md written"

# 接力:tkint2 720p 还差 VBench(eval/FID 已在,脚本内会自动跳过)
V=$W/kv-quant-result/experiments/longcat/LongCat-720p/vbench720/full/TEMPOKV_KH2VH1_INT2_720p
if [ "$(ls $V/*_eval_results.json 2>/dev/null | wc -l)" -lt 8 ]; then
  LOG "start tkint2 vbench tail ($(ls $V/*_eval_results.json 2>/dev/null | wc -l)/8)"
  TRITON_CACHE_DIR=/dev/shm/triton_home_vb TMPDIR=/dev/shm/vb_tail \
    bash $T/scripts/oneoff/tkint2_tail.sh >> $OUT/eval_after.log 2>&1
  LOG "tkint2 tail rc=$?"
fi
LOG "ALL SA EVAL CHAIN END"
