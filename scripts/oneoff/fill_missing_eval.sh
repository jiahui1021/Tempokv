#!/usr/bin/env bash
# fill_missing_eval.sh <run_dir_name> <gpu-uuid> — 对 streaming_psnr_ssim.csv 缺的 idx 逐条补
# evaluate_quality --no-fid 并追加行(按唯一 video 判断,不重复)。
set -uo pipefail
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
Q=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
cd "$T"; R=$T/results/paper/span-int2/runs; LAB=$1; GPU=$2
P=$R/$LAB/pixel_717; CSV=$P/streaming_psnr_ssim.csv; REFS=$R/span30sweep/refs30
[[ -s $CSV ]] || { echo "$LAB 无CSV"; exit 1; }
for i in $(seq 0 29); do
  grep -q "^${i}_0.mp4," "$CSV" && continue
  [[ -s $P/videos/${i}_0.mp4 && -s $REFS/${i}_0.mp4 ]] || continue
  out=/tmp/fme_${LAB}_${i}.json; rm -f $out
  d1=/tmp/fme_ref_${LAB}_${i}; d2=/tmp/fme_cand_${LAB}_${i}; rm -rf $d1 $d2; mkdir -p $d1 $d2
  ln -sf $REFS/${i}_0.mp4 $d1/video.mp4; ln -sf $P/videos/${i}_0.mp4 $d2/video.mp4
  CUDA_VISIBLE_DEVICES=$GPU timeout 500 $Q scripts/evaluation/evaluate_quality.py \
    $d1 $d2 --no-fid --device cuda:0 --output $out >>$R/anchorsweep/fill_$LAB.log 2>&1
  if [[ -s $out ]]; then
    $Q - $out $CSV $i <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
open(sys.argv[2],'a').write(f"{sys.argv[3]}_0.mp4,{d['psnr_db']:.6f},{d['ssim']:.6f},{d['lpips']:.6f},{d.get('frames','')},{d['mse']:.4f}\n")
PY
    echo "appended $i" >> $R/anchorsweep/fill_$LAB.log
  fi
  rm -rf $d1 $d2 $out
done
echo "FILL DONE $LAB $(date '+%T')" >> $R/anchorsweep/fill_$LAB.log
