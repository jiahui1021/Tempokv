#!/usr/bin/env bash
# Predictor ablation v2, no stacking: one generation job per GPU.
#   pass 1: identity (resume, GPU-cbea873b) + taylor (free GPU-6f1aaf89)
#   pass 2: taylor2 on the card identity frees, then quality + summary over all 4 rows
set -u
cd /mnt/cpfs/epic-user/hujiahui-20260612/Tempokv

PYTHON_BIN=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
RUN=scripts/paper_experiments/predictor_ablation/run.py
ROOT=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/self_forcing/predictor_ablation_v2_60p
CARD_A=GPU-cbea873b-5a71-8923-44ec-109b3b8706c9   # identity resumes here
CARD_B=GPU-6f1aaf89-db19-84dc-8ce4-28a8bd9bcad4   # empty card, taylor
export TMPDIR=/mnt/cpfs/epic-user/hujiahui-20260612/.tmp/tempokv

common=(
  --checkpoint /mnt/cpfs/epic-user/hujiahui-20260612/checkpoints/self_forcing_dmd.pt
  --wan-model-dir /mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-4-videogen/wan_models/Wan2.1-T2V-1.3B
  --trace /mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/data/self-forcing/trace_legacy/clean_trace.npz
  --bf16-videos /mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/Self-Forcing/bf16_clean
  --tempokv-videos /mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/self_forcing/predictor_ablation_60p/quality/ref_tempokv_flat
  --output-root "$ROOT"
  --quality-fid
  --resume
)

# `identity_cross_vh2` is deliberately absent from pass A: its pipeline_report says
# status=all_skipped (all 60 videos existed on disk), which _generation_complete does
# not accept, so the runner re-dispatches it — and _run_jobs treats every GPU-sized
# batch as a barrier. Including it would hold a card slot until taylor's last video
# landed. Pass A's generation batch is therefore exactly taylor + taylor2, one per card.
echo "=== pass A: taylor + taylor2 in one batch, then their quality ==="
"$PYTHON_BIN" "$RUN" "${common[@]}" \
  --gpus "$CARD_A,$CARD_B" \
  --variants learned_cross_vh2,taylor_cross_vh2,taylor2_cross_vh3
echo "pass A rc=$?"

echo "=== pass B: identity quality + summary.csv over all 4 rows ==="
"$PYTHON_BIN" "$RUN" "${common[@]}" \
  --gpus "$CARD_A,$CARD_B" \
  --variants learned_cross_vh2,identity_cross_vh2,taylor_cross_vh2,taylor2_cross_vh3
echo "pass B rc=$?"
