#!/usr/bin/env bash
# Run the full legacy TempoKV flow (fit -> generate -> stream-eval) for a given
# CONFIG, timing each stage in whole seconds. Writes flow_timing.json next to
# the output videos and prints the per-video metrics.
set -euo pipefail

QVG=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
cd /mnt/cpfs/epic-user/hujiahui-20260612/Tempokv

CONFIG="${CONFIG:?need CONFIG}"
KH="${KH:-}"
VH="${VH:-}"
TRACE_FILE="${TRACE_FILE:-data/self-forcing/trace_legacy/clean_trace.npz}"
LABEL="${LABEL:?need LABEL}"

start_s=$(date +%s)

# --- 1. fit ---
t0=$(date +%s)
env PYTHON_BIN=$QVG CONFIG="$CONFIG" TRACE_FILE="$TRACE_FILE" KH="$KH" VH="$VH" \
  bash scripts/Self-Forcing/fit_params.sh
t1=$(date +%s)
fit_s=$((t1 - t0))

# --- 2. generate 1 video ---
t0=$(date +%s)
rm -rf "results/self_forcing/$LABEL"
mkdir -p "results/self_forcing/$LABEL"
# The persistent launch batch is the one remaining decode tuning knob; the
# temporal V predictor removed the CrossKV projection that used to dominate.
env PYTHON_BIN=$QVG CONFIG="$CONFIG" KH="$KH" VH="$VH" \
  TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH=16 \
  CHECKPOINT=/mnt/cpfs/epic-user/hujiahui-20260612/checkpoints/self_forcing_dmd.pt \
  WAN_MODEL_DIR=/mnt/cpfs/epic-user/hujiahui-20260612/models--Wan-AI--Wan2.1-T2V-1.3B \
  PROMPTS=assets/moviegen128.txt FRAMES=717 START_INDEX=0 END_INDEX=1 \
  bash scripts/Self-Forcing/run_tempokv.sh > "results/self_forcing/$LABEL/gen_timed.log" 2>&1
t1=$(date +%s)
gen_s=$((t1 - t0))

# --- 3. stream eval (single video) ---
t0=$(date +%s)
env PYTHON_BIN=$QVG $QVG scripts/streaming_eval_loop.py \
  --ref-dir /mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/Self-Forcing/bf16 \
  --target-dir "results/self_forcing/$LABEL/videos" \
  --output "results/self_forcing/$LABEL/streaming_psnr_ssim.csv" \
  --expected 1 --device cuda --frame-stride 4 \
  > "results/self_forcing/$LABEL/eval_timed.log" 2>&1 &
EVAL_PID=$!
for i in $(seq 1 90); do
  if [ -s "results/self_forcing/$LABEL/streaming_psnr_ssim.csv" ] && \
     [ "$(wc -l < "results/self_forcing/$LABEL/streaming_psnr_ssim.csv")" -ge 2 ]; then
    break
  fi
  sleep 10
done
kill "$EVAL_PID" 2>/dev/null || true
t1=$(date +%s)
eval_s=$((t1 - t0))

end_s=$(date +%s)
total_s=$((end_s - start_s))

PARAM_DIR="$(ls -d data/self-forcing/params/${LABEL} 2>/dev/null || echo unknown)"
cat > "results/self_forcing/$LABEL/flow_timing.json" <<EOF
{
  "label": "$LABEL",
  "config": "$CONFIG",
  "fit_seconds": $fit_s,
  "generate_seconds": $gen_s,
  "eval_seconds": $eval_s,
  "total_seconds": $total_s,
  "param_dir": "$PARAM_DIR"
}
EOF

echo "=== $LABEL flow done ==="
echo "fit=${fit_s}s generate=${gen_s}s eval=${eval_s}s total=${total_s}s"
cat "results/self_forcing/$LABEL/streaming_psnr_ssim.csv"
