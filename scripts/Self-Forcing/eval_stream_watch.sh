#!/usr/bin/env bash
# eval_stream_watch.sh — 流式评测 watcher：盯某个 videos/ 目录，新 mp4 落盘立刻逐条评测，
#   结果**追加进单一 CSV**(video,psnr,ssim,lpips,frames,mse)，不再产 per-video json。
#   幂等:已在 CSV 里的 idx 跳过。终止:满 N 条 或 无新 mp4 超过 idle 分钟。
# 用法: bash eval_stream_watch.sh <exp_name> <gpu_uuid> [timeout_min]
#   可 env 覆盖: VDIR OUTDIR CSV BF16_REF N LOG
set -uo pipefail

EXP="$1"
GPU="$2"
IDLE_MIN="${3:-45}"

TEMPOKV=${WORKSPACE:?set WORKSPACE to the experiment root}/Tempokv
BF16_REF="${BF16_REF:-${WORKSPACE}/kv-quant-result/experiments/self-forcing/Self-Forcing/bf16}"
QVG_BIN=${WORKSPACE}/miniconda3/envs/qvg/bin
HB="$TEMPOKV/results/paper_experiments/hybrid-bits/$EXP"
VDIR="${VDIR:-$HB/videos}"
OUTDIR="${OUTDIR:-$HB}"
CSV="${CSV:-$OUTDIR/streaming_psnr_ssim.csv}"
LOG="${LOG:-$OUTDIR/watch_$EXP.log}"   # 日志放 run 目录(硬规则)
N="${N:-30}"
mkdir -p "$OUTDIR" "$(dirname "$LOG")" "$(dirname "$CSV")"
[[ -s "$CSV" ]] || echo "video,psnr,ssim,lpips,frames,mse" > "$CSV"

echo "[watch] $EXP 盯 $VDIR -> $CSV (gpu=$GPU idle=${IDLE_MIN}min) $(date '+%F %T')" >>"$LOG"

csv_has() { awk -F, -v v="${1}_0.mp4" 'NR>1&&$1==v{f=1}END{exit !f}' "$CSV" 2>/dev/null; }
# 行式计数(CSV 逗号分隔无空格,NF>=6 的 awk 判断恒 0 是历史 bug)
done_count() { local n; n=$(wc -l < "$CSV" 2>/dev/null); echo $(( n>0 ? n-1 : 0 )); }

last_new=$(date +%s)
while true; do
  idxs=()
  for f in "$VDIR"/*_0.mp4; do
    [[ -e "$f" ]] || continue
    b=$(basename "$f"); [[ "$b" == .tmp-* ]] && continue
    [[ -s "$f" ]] && idxs+=("${b%%_0.mp4}")
  done

  new_this_round=0
  for idx in "${idxs[@]}"; do
    csv_has "$idx" && continue                       # 已评(在 CSV 里)
    cand="$VDIR/${idx}_0.mp4"; ref="$BF16_REF/${idx}_0.mp4"
    [[ -s "$ref" ]] || { echo "[watch] $EXP 缺参考 ${idx}_0.mp4" >>"$LOG"; continue; }
    [[ -s "$cand" ]] || continue
    tmpj="/tmp/esw_${EXP}_${idx}.json"
    tr="/tmp/esw_ref_${EXP}_${idx}"; tc="/tmp/esw_cand_${EXP}_${idx}"
    rm -rf "$tmpj" "$tr" "$tc"; mkdir -p "$tr" "$tc"
    ln -sf "$ref" "$tr/video.mp4"; ln -sf "$cand" "$tc/video.mp4"
    echo "[watch] $EXP 评测 idx=$idx 开始 $(date '+%T')" >>"$LOG"
    ( cd "$TEMPOKV"; CUDA_VISIBLE_DEVICES="$GPU" "$QVG_BIN/python" scripts/evaluation/evaluate_quality.py \
        "$tr" "$tc" --no-fid --device cuda:0 --output "$tmpj" \
        >"$OUTDIR/esw_eval_${EXP}_${idx}.log" 2>&1 )   # 日志放 run 目录(硬规则)
    rc=$?; rm -rf "$tr" "$tc"
    if [[ -s "$tmpj" ]]; then
      new_this_round=$((new_this_round+1))   # 只有成功才算活动,失败不再无限续命 idle
      "$QVG_BIN/python" - "$tmpj" "$CSV" "$idx" <<'PY'
import json,sys,csv
d=json.load(open(sys.argv[1]))
with open(sys.argv[2],'a',newline='') as h:
    csv.writer(h,lineterminator='\n').writerow([f"{sys.argv[3]}_0.mp4",
        f"{d['psnr_db']:.6f}",f"{d['ssim']:.6f}",f"{d['lpips']:.6f}",d.get('frames',''),f"{d['mse']:.4f}"])
PY
      rm -f "$tmpj"
      echo "[watch] $EXP idx=$idx 完成 rc=$rc $(date '+%T')" >>"$LOG"
    else
      echo "[watch] $EXP idx=$idx FAILED rc=$rc $(date '+%T')" >>"$LOG"
    fi
  done
  [[ $new_this_round -gt 0 ]] && last_new=$(date +%s)

  dn=$(done_count)
  if [[ "$dn" -ge "$N" ]]; then echo "[watch] $EXP 全部 $N 条完成 -> $CSV $(date '+%T')" >>"$LOG"; exit 0; fi
  now=$(date +%s); idle=$(( (now-last_new)/60 ))
  # 空闲退出不要求 dn>0:生成失败/无视频时也必须放行,否则整链卡死
  if [[ "$idle" -ge "$IDLE_MIN" ]]; then echo "[watch] $EXP 空闲超时(已评 $dn)退出 $(date '+%T')" >>"$LOG"; exit 0; fi
  sleep 15
done
