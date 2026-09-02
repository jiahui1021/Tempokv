#!/usr/bin/env bash
# Take over the evaluation phase of predictor_ablation so all metrics run in
# one fully parallel wave (user approved stacking eval jobs on the same GPUs):
#   1. wait for the last taylor_vh2 video, then kill the orchestrator before
#      it can start its serialized 2-batch quality stage
#   2. launch 6 quality workers (3 variants x {bf16+FID, tempokv}) writing the
#      orchestrator's standard paths, stacked two-per-GPU on the 3 ablation GPUs
#   3. relaunch run.py --resume, which skips finished stages and writes the
#      summary CSV
set -uo pipefail
B=/mnt/cpfs/epic-user/hujiahui-20260612
PY=$B/miniconda3/envs/qvg/bin/python
OUT=$B/Tempokv/results/self_forcing/predictor_ablation_60p
LOG=$OUT/logs/eval_takeover.log
BF16=$B/kv-quant-result/Self-Forcing/bf16_clean
TMPKV=$B/kv-quant-result/Self-Forcing/kh2-vh1-int2/videos
GPUS=(GPU-7bd90749-1ac0-c9b0-be0e-5c95cbc6bfcc
      GPU-801636c6-7759-36df-3ee7-ad18e0d5cde9
      GPU-e2e44f02-eb52-04b0-8485-740e3d2d5c1c)
ORCH=3490516

log() { echo "[takeover] $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

cd "$B/Tempokv"

log "waiting for taylor_vh2 60/60"
n=0
while true; do
  n=$(ls "$OUT/runs/taylor_vh2/videos"/*.mp4 2>/dev/null | wc -l)
  (( n >= 60 )) && break
  pgrep -f "predictor_ablation/run.py --checkpoint" >/dev/null || { log "orchestrator already gone"; break; }
  sleep 30
done
log "taylor_vh2 videos=$n; killing orchestrator process group $ORCH"
kill -TERM -- -"$ORCH" 2>/dev/null
sleep 5
kill -KILL -- -"$ORCH" 2>/dev/null
while pgrep -f "tempokv_adapter/worker.py --mode tempokv" >/dev/null 2>&1; do
  sleep 10
done
log "generation workers exited; launching 6 quality workers (2 per GPU)"

launch() {
  local gpu="$1" variant="$2" refname="$3" refdir="$4" extra="$5"
  env CUDA_VISIBLE_DEVICES="$gpu" TMPDIR=$B/.tmp/tempokv \
    "$PY" scripts/paper_experiments/predictor_ablation/run.py \
      --quality-worker \
      --reference-dir "$refdir" \
      --candidate-dir "$OUT/runs/$variant/videos" \
      --output-json "$OUT/quality/$variant/vs_$refname.json" \
      --output-csv "$OUT/quality/$variant/vs_$refname.csv" \
      --quality-device cuda --quality-frame-stride 1 \
      --fid-batch-size 32 --lpips-network alex $extra \
      >> "$OUT/logs/quality_${variant}_vs_$refname.log" 2>&1 &
  log "  $variant vs $refname on $gpu ${extra:+(with FID)}"
}

pids=()
variants=(identity_vh2 taylor_vh2 taylor2_vh3)
for i in 0 1 2; do
  variant="${variants[$i]}"
  launch "${GPUS[$i]}" "$variant" bf16 "$BF16" "--quality-fid"
  pids+=($!)
  launch "${GPUS[$i]}" "$variant" tempokv "$TMPKV" ""
  pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
log "quality wave done rc=$rc; relaunching orchestrator --resume for summary"
"$PY" scripts/paper_experiments/predictor_ablation/run.py \
  --checkpoint "$B/checkpoints/self_forcing_dmd.pt" \
  --wan-model-dir "$B/kv-quant-4-videogen/wan_models/Wan2.1-T2V-1.3B" \
  --trace "$B/Tempokv/data/self-forcing/trace_legacy/clean_trace.npz" \
  --bf16-videos "$BF16" --tempokv-videos "$TMPKV" \
  --output-root "$OUT" --gpus "${GPUS[0]}" --resume \
  >> "$LOG" 2>&1 || rc=1
log "all done rc=$rc"
exit "$rc"
