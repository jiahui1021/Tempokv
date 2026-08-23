#!/usr/bin/env bash
# Parallel launch of legacy+shared exp5 (k4v4/k2v4/k4v2) across 3 GPUs.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../.." && pwd -P)"

export PATH="$HOME/miniconda3/envs/qvg/bin:$PATH"
export PYTHON_BIN="$HOME/miniconda3/envs/qvg/bin/python"
export CHECKPOINT="$HOME/checkpoints/self_forcing_dmd.pt"
export WAN_MODEL_DIR="$HOME/models--Wan-AI--Wan2.1-T2V-1.3B"
export PROMPTS="$project_root/assets/moviegen128_first30.txt"
export NUM_GPUS=1
export CONFIG=configs/legacy.json
export TRACE_FILE=data/self-forcing/trace_legacy/clean_trace.npz

PARAM_ROOT="$project_root/data/self-forcing/params/paper_exp5_legacy"
RESULT_ROOT="$project_root/results/paper_exp5_legacy"

run_one() {
  local label="$1" gpu_uuid="$2" k_bits="$3" v_bits="$4"
  local log="$project_root/logs/exp5_legacy_$label.log"
  (
    cd "$project_root"
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
    export PARAM_DIR="$PARAM_ROOT/$label"
    export OUTPUT_DIR="$RESULT_ROOT/$label"
    export K_BITS="$k_bits" V_BITS="$v_bits"
    export KH=2 VH=1 V_PREDICTOR=cross_kv
    export FRAMES=717 NUM_SAMPLES=1 SEED=0
    export SKIP_EXISTING=1 BENCHMARK=0
    export OVERWRITE=1
    export TRITON_CACHE="data/self-forcing/triton_cache/exp5_legacy_$label"
    export MODE=tempokv
    echo "=== [$label] fit ===" >> "$log"
    bash scripts/Self-Forcing/fit_params.sh >> "$log" 2>&1
    echo "=== [$label] generate ===" >> "$log"
    bash scripts/Self-Forcing/generate.sh >> "$log" 2>&1
  )
}

# GPU5 c7df48b3 -> k4v4, GPU6 51d8de1f -> k2v4, GPU7 6f1aaf89 -> k4v2
run_one k4v4 "GPU-c7df48b3-ad9a-363b-9af4-62821edcbc9a" 4 4 &
run_one k2v4 "GPU-51d8de1f-3455-fc7a-f4e3-d0e6daee337b" 2 4 &
run_one k4v2 "GPU-6f1aaf89-db19-84dc-8ce4-28a8bd9bcad4" 4 2 &

wait
echo "ALL DONE"
