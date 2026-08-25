#!/usr/bin/env bash
# Causal Forcing QVG 生成（kv-quant-4-videogen 入口），128 条 moviegen。
set -euo pipefail

export PATH="$HOME/miniconda3/envs/qvg/bin:$PATH"
QVG_ROOT="$HOME/kv-quant-4-videogen"
CHECKPOINT_PATH="$HOME/checkpoints/chunkwise/longvideo.pt"
DATA_PATH="$HOME/kv-quant-result/moviegen_128.txt"
OUT="$HOME/kv-quant-result/Causal Forcing"
PYTHON_BIN="$HOME/miniconda3/envs/qvg/bin/python"

cd "$QVG_ROOT"

run_qvg() {
  local method="$1" gpu_uuid="$2"
  local out_dir="$OUT/$method"
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
    CONFIG_PATH="$QVG_ROOT/Causal-Forcing/configs/causal_forcing_dmd_chunkwise.yaml" \
    CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    DATA_PATH="$DATA_PATH" \
    METHOD="$method" \
    NUM_OUTPUT_FRAMES=180 \
    LOCAL_ATTN_SIZE=180 \
    OUTPUT_FOLDER="$out_dir" \
    PYTHON_BIN="$PYTHON_BIN" \
    RETAIN_FINAL_CACHE=1 \
    bash Causal-Forcing/run_qvg.sh >> "$HOME/Tempokv/logs/cf_qvg_$method.log" 2>&1
  )
}

run_qvg "$1" "$2" &
echo "qvg $1 launched pid=$!"
wait
