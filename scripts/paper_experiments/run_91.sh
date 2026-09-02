#!/usr/bin/env bash
# Run todo.md "9.1实验" two experiments on the 4 leased GPUs, after the
# rerun_baselines60_717f generation finishes and frees the cards.
#
#   1. observation run_study (采集 -> 校验 -> 分解阶梯 -> 图表)
#   2. predictor_ablation (4 variants, fit -> generate 60p -> quality)
#
# Usage: bash run_91.sh [study|ablation|all]
set -uo pipefail

B=/mnt/cpfs/epic-user/hujiahui-20260612
PY=$B/miniconda3/envs/qvg/bin/python
ROOT_PROJECT=$B/Tempokv
CHECKPOINT=$B/checkpoints/self_forcing_dmd.pt
WAN_MODEL_DIR=$B/kv-quant-4-videogen/wan_models/Wan2.1-T2V-1.3B
TRACE=$B/kv-quant-result/innovation_ablation/trace180/clean_trace.npz
BF16_VIDEOS=$B/kv-quant-result/Self-Forcing/bf16_clean
TEMPOKV_VIDEOS=$B/kv-quant-result/Self-Forcing/kh2-vh1-int2/videos
OUT_ROOT=$ROOT_PROJECT/results/self_forcing/predictor_ablation_60p

# Leased GPU UUIDs
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?need CUDA_VISIBLE_DEVICES}"
if ((${#GPUS[@]} < 4)); then echo "need 4 leased GPUs" >&2; exit 2; fi
GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

run_study() {
  echo "===== [9.1] observation run_study ====="
  cd "$ROOT_PROJECT" || exit 1
  CHECKPOINT="$CHECKPOINT" \
  WAN_MODEL_DIR="$WAN_MODEL_DIR" \
  CAL_PROMPTS=assets/calibration/cal_prompts.txt \
  EVAL_PROMPTS=assets/calibration/heldout_prompts.txt \
  PYTHON_BIN="$PY" \
  bash scripts/paper_experiments/observation/run_study.sh
}

run_ablation() {
  echo "===== [9.1] predictor_ablation (4 variants, 60p) ====="
  cd "$ROOT_PROJECT" || exit 1
  "$PY" scripts/paper_experiments/predictor_ablation/run.py \
    --checkpoint "$CHECKPOINT" \
    --wan-model-dir "$WAN_MODEL_DIR" \
    --trace "$TRACE" \
    --bf16-videos "$BF16_VIDEOS" \
    --tempokv-videos "$TEMPOKV_VIDEOS" \
    --output-root "$OUT_ROOT" \
    --gpus "$GPU_LIST"
}

case "${1:-all}" in
  study)   run_study ;;
  ablation) run_ablation ;;
  all)     run_study; run_ablation ;;
  *) echo "usage: run_91.sh [study|ablation|all]"; exit 2 ;;
esac
