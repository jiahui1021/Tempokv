#!/usr/bin/env bash
# Wait for the running predictor_ablation orchestrator to exit (it will fail on
# taylor_vh2's stale-file-handle generation error), then relaunch with --resume
# so completed generations are skipped and only the missing work (taylor_vh2
# generation + all quality) runs.
set -uo pipefail
B=/mnt/cpfs/epic-user/hujiahui-20260612
PY=$B/miniconda3/envs/qvg/bin/python
ROOT=$B/kv-quant-result/Self-Forcing/rerun_baselines60_717f
LOG=$ROOT/logs/predictor_ablation.log
GPU_UUIDS="GPU-7bd90749-1ac0-c9b0-be0e-5c95cbc6bfcc,GPU-801636c6-7759-36df-3ee7-ad18e0d5cde9,GPU-e2e44f02-eb52-04b0-8485-740e3d2d5c1c"
OUT=$B/Tempokv/results/self_forcing/predictor_ablation_60p

echo "[resume] $(date) waiting for current predictor to exit" | tee -a "$LOG"
while pgrep -f "predictor_ablation/run.py" >/dev/null 2>&1; do
  sleep 20
done
# also wait for any leftover generation workers
while pgrep -f "tempokv_adapter/worker.py" >/dev/null 2>&1; do
  sleep 20
done
echo "[resume] $(date) relaunching predictor_ablation --resume" | tee -a "$LOG"

cd "$B/Tempokv"
nohup env CUDA_VISIBLE_DEVICES="$GPU_UUIDS" $PY scripts/paper_experiments/predictor_ablation/run.py \
  --checkpoint "$B/checkpoints/self_forcing_dmd.pt" \
  --wan-model-dir "$B/kv-quant-4-videogen/wan_models/Wan2.1-T2V-1.3B" \
  --trace "$B/Tempokv/data/self-forcing/trace_legacy/clean_trace.npz" \
  --bf16-videos "$B/kv-quant-result/Self-Forcing/bf16_clean" \
  --tempokv-videos "$B/kv-quant-result/Self-Forcing/kh2-vh1-int2/videos" \
  --output-root "$OUT" \
  --gpus "$GPU_UUIDS" \
  --resume \
  >> "$LOG" 2>&1 < /dev/null &
echo "[resume] relaunched pid=$!" | tee -a "$LOG"
