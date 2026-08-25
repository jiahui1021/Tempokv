#!/usr/bin/env bash
# Causal Forcing todo 调度器：轮询 exp5 legacy 各组完成，把释放的 GPU 分配给 todo 步骤。
# todo: bf16(GPU4已跑) -> qvg int2 -> tempokv int2 -> qvg int4 + tempokv int4
set -uo pipefail

PROJECT="$HOME/Tempokv"
OUT="$HOME/kv-quant-result/Causal Forcing"
mkdir -p "$OUT"
export PATH="$HOME/miniconda3/envs/qvg/bin:$PATH"
export PYTHON_BIN="$HOME/miniconda3/envs/qvg/bin/python"

CF_CHKPT="$HOME/checkpoints/chunkwise/longvideo.pt"
CF_MODEL_ROOT="$HOME/models--Wan-AI--Wan2.1-T2V-1.3B"
CF_PROMPTS="$HOME/kv-quant-result/moviegen_128.txt"
CF_PARAMS="$PROJECT/data/causal-forcing/params/chunkwise"

EXP5_RESULTS="$PROJECT/results/paper_exp5_legacy"

launch_cf_qvg() {
  local method="$1" gpu_uuid="$2"
  (
    cd "$HOME/kv-quant-4-videogen"
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
    CONFIG_PATH="$HOME/kv-quant-4-videogen/Causal-Forcing/configs/causal_forcing_dmd_chunkwise.yaml" \
    CHECKPOINT_PATH="$CF_CHKPT" DATA_PATH="$CF_PROMPTS" METHOD="$method" \
    NUM_OUTPUT_FRAMES=180 LOCAL_ATTN_SIZE=180 \
    OUTPUT_FOLDER="$OUT/$method" PYTHON_BIN="$PYTHON_BIN" RETAIN_FINAL_CACHE=1 \
    bash Causal-Forcing/run_qvg.sh >> "$PROJECT/logs/cf_qvg_$method.log" 2>&1
  )
}

launch_cf_tempokv() {
  local label="$1" gpu_uuid="$2" k_bits="$3" v_bits="$4" param_dir="$5"
  (
    cd "$PROJECT"
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
    export CHECKPOINT="$CF_CHKPT" MODEL_ROOT="$CF_MODEL_ROOT" PROMPTS="$CF_PROMPTS"
    export FRAMES=717 NUM_GPUS=1 GENERATION_MODE=chunkwise
    export CONFIG="$PROJECT/configs/legacy.json"
    export K_BITS="$k_bits" V_BITS="$v_bits"
    export PARAM_DIR="$param_dir"
    export OUTPUT_DIR="$OUT/$label"
    export TRITON_CACHE="$PROJECT/data/self-forcing/triton_cache/cf_$label"
    bash scripts/Causal-Forcing/run_tempokv.sh >> "$PROJECT/logs/cf_$label.log" 2>&1
  )
}

count_videos() {
  find "$EXP5_RESULTS/$1/videos" -name "*.mp4" 2>/dev/null | wc -l
}

started_qvg2=0; started_tk2=0; started_tk4=0

echo "scheduler started $(date +%H:%M:%S)"
while true; do
  k4v4=$(count_videos k4v4)
  k2v4=$(count_videos k2v4)
  k4v2=$(count_videos k4v2)

  # exp5 k4v4 完成 -> GPU5 跑 qvg int2
  if [[ "$k4v4" -ge 30 && "$started_qvg2" -eq 0 ]]; then
    echo "exp5 k4v4 done, launching qvg int2 on GPU5 $(date +%H:%M:%S)"
    launch_cf_qvg "QVG_INT2" "GPU-c7df48b3-ad9a-363b-9af4-62821edcbc9a" &
    started_qvg2=1
  fi
  # exp5 k2v4 完成 -> GPU6 跑 tempokv int2（先拟合 legacy int2）
  if [[ "$k2v4" -ge 30 && "$started_tk2" -eq 0 ]]; then
    echo "exp5 k2v4 done, fitting tempokv int2 on GPU6 $(date +%H:%M:%S)"
    (
      cd "$PROJECT"
      export CUDA_VISIBLE_DEVICES="GPU-51d8de1f-3455-fc7a-f4e3-d0e6daee337b"
      export PYTHON_BIN="$HOME/miniconda3/envs/qvg/bin/python"
      export CONFIG="$PROJECT/configs/legacy.json"
      export TRACE_FILE="$PROJECT/data/causal-forcing/trace/chunkwise_legacy/clean_trace.npz"
      export GENERATION_MODE=chunkwise
      export K_BITS=2 V_BITS=2
      export OUTPUT_DIR="$CF_PARAMS/legacy-kh2-vh1-int2"
      export OVERWRITE=1
      bash scripts/Causal-Forcing/fit_params.sh >> "$PROJECT/logs/cf_fit_int2.log" 2>&1
    )
    launch_cf_tempokv "tempokv-int2" "GPU-51d8de1f-3455-fc7a-f4e3-d0e6daee337b" 2 2 "$CF_PARAMS/legacy-kh2-vh1-int2" &
    started_tk2=1
  fi
  # exp5 k4v2 完成 -> GPU7 跑 tempokv int4
  if [[ "$k4v2" -ge 30 && "$started_tk4" -eq 0 ]]; then
    echo "exp5 k4v2 done, launching tempokv int4 on GPU7 $(date +%H:%M:%S)"
    launch_cf_tempokv "tempokv-int4" "GPU-6f1aaf89-db19-84dc-8ce4-28a8bd9bcad4" 4 4 "$CF_PARAMS/legacy-kh2-vh1-int4" &
    started_tk4=1
  fi

  if [[ "$started_qvg2" -eq 1 && "$started_tk2" -eq 1 && "$started_tk4" -eq 1 ]]; then
    echo "all exp5 done, all todo steps launched $(date +%H:%M:%S)"
    break
  fi
  sleep 60
done
echo "scheduler finished $(date +%H:%M:%S)"
