#!/usr/bin/env bash
# Wait for the predictor_ablation orchestrator (generation + PSNR/SSIM/LPIPS
# quality vs bf16/tempokv) to exit, then compute set-level FID vs bf16_clean
# for all three variants, one per GPU.
set -uo pipefail
B=/mnt/cpfs/epic-user/hujiahui-20260612
PY=$B/miniconda3/envs/qvg/bin/python
OUT=$B/Tempokv/results/self_forcing/predictor_ablation_60p
BF16=$B/kv-quant-result/Self-Forcing/bf16_clean
LOG=$OUT/logs/fid_after_orchestrator.log
GPUS=(GPU-7bd90749-1ac0-c9b0-be0e-5c95cbc6bfcc
      GPU-801636c6-7759-36df-3ee7-ad18e0d5cde9
      GPU-e2e44f02-eb52-04b0-8485-740e3d2d5c1c)

echo "[fid] $(date) waiting for orchestrator to exit" | tee -a "$LOG"
while pgrep -f "predictor_ablation/run.py --checkpoint" >/dev/null 2>&1; do
  sleep 60
done
while pgrep -f "tempokv_adapter/worker.py" >/dev/null 2>&1; do
  sleep 20
done

cd "$B/Tempokv"
pids=()
i=0
for v in identity_vh2 taylor_vh2 taylor2_vh3; do
  echo "[fid] $(date) $v vs bf16 on ${GPUS[$i]}" | tee -a "$LOG"
  env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" TMPDIR=$B/.tmp/tempokv \
    "$PY" scripts/paper_experiments/predictor_ablation/run.py \
      --quality-worker \
      --reference-dir "$BF16" \
      --candidate-dir "$OUT/runs/$v/videos" \
      --output-json "$OUT/quality/$v/vs_bf16_fid.json" \
      --output-csv "$OUT/quality/$v/vs_bf16_fid.csv" \
      --quality-fid \
      >> "$OUT/logs/fid_${v}.log" 2>&1 &
  pids+=($!)
  i=$((i+1))
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
echo "[fid] $(date) all FID workers done rc=$rc" | tee -a "$LOG"
exit "$rc"
