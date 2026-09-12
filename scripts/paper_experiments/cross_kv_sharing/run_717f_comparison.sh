#!/usr/bin/env bash
# CrossKV weight-sharing 717-frame generation comparison, 3-GPU variant parallelism.
set -euo pipefail

ROOT=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
PY=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
GEN=$ROOT/experiments/Self-Forcing/tempokv_adapter/generate.py
CKPT=/mnt/cpfs/epic-user/hujiahui-20260612/checkpoints/self_forcing_dmd.pt
WAN_DIR=/mnt/cpfs/epic-user/hujiahui-20260612/models--Wan-AI--Wan2.1-T2V-1.3B
PROMPTS=$ROOT/assets/moviegen128_first30.txt
SMOKE=$ROOT/results/paper/cross_kv_sharing_smoke
OUT=$ROOT/results/paper/cross_kv_sharing_717f
LOGDIR=$OUT/logs
FRAMES=717
END_INDEX=2
SEED=0

GPUS=(
  GPU-2e0caa24-0070-8bb7-5d3c-e22aaa28fd07
  GPU-c7df48b3-ad9a-363b-9af4-62821edcbc9a
  GPU-51d8de1f-3455-fc7a-f4e3-d0e6daee337b
  GPU-6f1aaf89-db19-84dc-8ce4-28a8bd9bcad4
)

mkdir -p "$OUT" "$LOGDIR"

# ---- wait until smoke has fitted all 4 variants (fits are frame-independent) ----
for v in layer_head layer_shared head_shared shared; do
  while [[ ! -f "$SMOKE/params/$v/v_cross_kv.npz" ]]; do
    echo "[wait] params/$v not ready; sleeping 60s"
    sleep 60
  done
done
echo "[info] all 4 param sets ready"

declare -A CFG KP VP
MODES=(bf16 layer_head layer_shared head_shared shared)
for v in layer_head layer_shared head_shared shared; do
  CFG[$v]=$SMOKE/configs/cross_kv_${v}.json
  KP[$v]=$SMOKE/params/$v/k_temporal.npz
  VP[$v]=$SMOKE/params/$v/v_cross_kv.npz
done

run_gen() {  # $1=gpu-uuid $2=mode
  local gpu=$1 mode=$2
  local outdir=$OUT/runs/$mode
  local log=$LOGDIR/gen_$mode.log
  local extra=()
  local gen_mode=bf16
  if [[ $mode != bf16 ]]; then
    gen_mode=tempokv
    extra+=(--config "${CFG[$mode]}" --k-params "${KP[$mode]}" --v-params "${VP[$mode]}")
  fi
  echo "[$(date +%H:%M:%S)] start $mode on ${gpu:0:13}"
  CUDA_VISIBLE_DEVICES=$gpu FORCING_WAN_MODEL_DIR=$WAN_DIR \
  "$PY" "$GEN" --mode "$gen_mode" --checkpoint "$CKPT" --prompts "$PROMPTS" \
    --output-dir "$outdir" --frames $FRAMES --samples 1 --seed $SEED \
    --start-index 0 --end-index $END_INDEX --num-gpus 1 --benchmark --skip-existing \
    "${extra[@]}" >"$log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $mode (rc=$?)"
}

# ---- wave scheduler over 3 GPUs ----
declare -A GPUPID
gpu_launch() {
  for i in "${!GPUS[@]}"; do
    local g=${GPUS[$i]}
    if [[ -z ${GPUPID[$g]:-} ]] || ! kill -0 "${GPUPID[$g]}" 2>/dev/null; then
      run_gen "$g" "$1" &
      GPUPID[$g]=$!
      return 0
    fi
  done
  return 1
}

for mode in "${MODES[@]}"; do
  until gpu_launch "$mode"; do sleep 30; done
done
wait
echo "[info] all generations finished"

# ---- quality vs bf16, sequential on first GPU ----
for v in layer_head layer_shared head_shared shared; do
  echo "[$(date +%H:%M:%S)] quality $v"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$ROOT/scripts/evaluation/evaluate_quality.py" \
    "$OUT/runs/bf16/videos" "$OUT/runs/$v/videos" \
    --output "$OUT/runs/$v/quality_vs_bf16.json" \
    --device auto --fid-batch-size 32 \
    >"$LOGDIR/quality_$v.log" 2>&1
done
echo "[$(date +%H:%M:%S)] ALL DONE"
