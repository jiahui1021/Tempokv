#!/usr/bin/env bash
# Causal Forcing todo 流水线: bf16 -> tempokv int2/int4 (legacy shared kh2)
# 128 条 moviegen, 717 帧。QVG 单独用 kv-quant-4-videogen 跑。
set -euo pipefail

export PATH="$HOME/miniconda3/envs/qvg/bin:$PATH"
export PYTHON_BIN="$HOME/miniconda3/envs/qvg/bin/python"
export CHECKPOINT="$HOME/checkpoints/chunkwise/longvideo.pt"
export MODEL_ROOT="$HOME/models--Wan-AI--Wan2.1-T2V-1.3B"
export PROMPTS="$HOME/kv-quant-result/moviegen_128.txt"
export FRAMES=717
export NUM_GPUS=1
export GENERATION_MODE=chunkwise

OUT="$HOME/kv-quant-result/Causal Forcing"
mkdir -p "$OUT"

# 参数目录
CF_PARAMS="$HOME/Tempokv/data/causal-forcing/params/chunkwise"

run_cf() {
  local label="$1" mode="$2" gpu_uuid="$3" k_bits="$4" v_bits="$5" param_dir="$6"
  local log="$HOME/Tempokv/logs/cf_todo_$label.log"
  (
    cd "$HOME/Tempokv"
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
    export CONFIG="$HOME/Tempokv/configs/legacy.json"
    if [[ "$mode" == "tempokv" ]]; then
      export K_BITS="$k_bits" V_BITS="$v_bits"
      export PARAM_DIR="$param_dir"
      export OUTPUT_DIR="$OUT/$label"
      bash scripts/Causal-Forcing/run_tempokv.sh >> "$log" 2>&1
    else
      export OUTPUT_DIR="$OUT/$label"
      bash scripts/Causal-Forcing/run_bf16.sh >> "$log" 2>&1
    fi
  )
}

# bf16 占 GPU4
run_cf "bf16" "bf16" "GPU-2e0caa24-0070-8bb7-5d3c-e22aaa28fd07" "" "" "" &
echo "cf bf16 launched (GPU4)"

wait
echo "ALL CF DONE"
