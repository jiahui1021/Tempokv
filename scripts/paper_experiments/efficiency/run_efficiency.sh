#!/usr/bin/env bash
# Final Self-Forcing launcher for the three TempoKV-owned rows:
# BF16, TempoKV INT2, and TempoKV INT4.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../../.." && pwd -P)"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="${PYTHON_BIN:-python3}"

mode="${EFFICIENCY_MODE:-both}"
only="${ONLY:-}"
skip="${SKIP:-}"
output_root="${OUTPUT_ROOT:-$repo_root/results/paper_experiments/efficiency-self-forcing/tempokv}"
checkpoint="${CHECKPOINT:-}"
wan_model_dir="${WAN_MODEL_DIR:-}"
prompts="${PROMPTS:-}"
config="${CONFIG:-configs/native.json}"
param_dir="${PARAM_DIR:-}"
frames="${FRAMES:-717}"
horizon_frames="${HORIZON_FRAMES:-3009}"
seed="${SEED:-0}"
num_gpus="${NUM_GPUS:-1}"
start_index="${START_INDEX:-0}"
end_index="${END_INDEX:-3}"
alloc_conf="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
dry_run=0
overwrite=0
collect=1

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'EOF'
Usage: scripts/paper_experiments/efficiency/run_efficiency.sh [options]

Options:
  --mode efficiency|horizon|both   Workload to run (default: both)
  --only NAME[,NAME...]             BF16,TEMPOKV_INT2,TEMPOKV_INT4
  --skip NAME[,NAME...]             Skip selected rows
  --output-root PATH                Result directory
  --dry-run                         Print launches without using CUDA
  --overwrite                       Allow existing efficiency/horizon records
  --no-collect                      Do not render the local CSV/TeX summary

Required for a real run: CHECKPOINT, WAN_MODEL_DIR, PROMPTS.  Quantized rows
also require CONFIG and matching fitted parameters under PARAM_DIR.
EOF
}

while (($#)); do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    --only) only="$2"; shift 2 ;;
    --skip) skip="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --wan-model-dir) wan_model_dir="$2"; shift 2 ;;
    --prompts) prompts="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --param-dir) param_dir="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --horizon-frames) horizon_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --num-gpus) num_gpus="$2"; shift 2 ;;
    --start-index) start_index="$2"; shift 2 ;;
    --end-index) end_index="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    --no-collect) collect=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$mode" in efficiency|horizon|both) ;; *) die "--mode must be efficiency, horizon, or both" ;; esac
(( (frames + 3) % 12 == 0 )) || die "FRAMES must satisfy (FRAMES + 3) % 12 == 0"
(( (horizon_frames + 3) % 12 == 0 )) || die "HORIZON_FRAMES must satisfy (FRAMES + 3) % 12 == 0"
mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"

contains() {
  local list=",${1:-}," item="$2"
  [[ "$list" == *",$item,"* ]]
}
selected() {
  if [[ -n "$only" ]] && ! contains "$only" "$1"; then return 1; fi
  if [[ -n "$skip" ]] && contains "$skip" "$1"; then return 1; fi
  return 0
}
require_real_inputs() {
  (( dry_run )) && return 0
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || die "CHECKPOINT must point to an existing file"
  [[ -n "$wan_model_dir" && -d "$wan_model_dir" ]] || die "WAN_MODEL_DIR must point to an existing directory"
  [[ -n "$prompts" && -f "$prompts" ]] || die "PROMPTS must point to an existing file"
  [[ -f "$repo_root/$config" || -f "$config" ]] || die "CONFIG does not exist: $config"
}
check_record() {
  local name="$1" file="$output_root/$1/$2"
  if [[ -f "$file" && "$overwrite" != 1 ]]; then
    die "$name already has $2; use --overwrite or --skip $name"
  fi
}
run_cmd() {
  local label="$1"; shift
  printf '[%s] $' "$label"
  printf ' %q' "$@"
  printf '\n'
  (( dry_run )) && return 0
  "$@"
}

run_main() {
  local name="$1" bits="$2" launcher="$3" out="$output_root/$1"
  selected "$name" || return 0
  check_record "$name" efficiency.json
  local -a command=(
    env
    "PYTORCH_CUDA_ALLOC_CONF=$alloc_conf"
    "CHECKPOINT=$checkpoint"
    "WAN_MODEL_DIR=$wan_model_dir"
    "PROMPTS=$prompts"
    "OUTPUT_DIR=$out"
    "FRAMES=$frames"
    "NUM_SAMPLES=1"
    "SEED=$seed"
    "START_INDEX=$start_index"
    "END_INDEX=$end_index"
    "NUM_GPUS=$num_gpus"
    "PROFILE=1"
    "BENCHMARK=1"
    "SKIP_EXISTING=0"
  )
  if [[ -n "$bits" ]]; then
    command+=("BITS=$bits" "CONFIG=$config")
    [[ -n "$param_dir" ]] && command+=("PARAM_DIR=$param_dir")
  fi
  command+=(bash "$repo_root/scripts/Self-Forcing/$launcher")
  run_cmd "$name" "${command[@]}"
}

run_horizon() {
  local name="$1" bits="$2" launcher="$3" out="$output_root/$1"
  selected "$name" || return 0
  check_record "$name" horizon.json
  local -a command=(
    env
    "PYTORCH_CUDA_ALLOC_CONF=$alloc_conf"
    "SELF_FORCING_HORIZON_PROBE=1"
    "CHECKPOINT=$checkpoint"
    "WAN_MODEL_DIR=$wan_model_dir"
    "PROMPTS=$prompts"
    "OUTPUT_DIR=$out"
    "FRAMES=$horizon_frames"
    "LOCAL_ATTN_SIZE=$(( (horizon_frames + 3) / 4 ))"
    "SEED=$seed"
    "START_INDEX=0"
    "END_INDEX=0"
    "NUM_GPUS=$num_gpus"
    "SKIP_EXISTING=0"
  )
  if [[ -n "$bits" ]]; then
    command+=("BITS=$bits" "CONFIG=$config")
    [[ -n "$param_dir" ]] && command+=("PARAM_DIR=$param_dir")
  fi
  command+=(bash "$repo_root/scripts/Self-Forcing/$launcher")
  run_cmd "$name horizon" "${command[@]}"
}

require_real_inputs
if [[ "$mode" == efficiency || "$mode" == both ]]; then
  run_main BF16 "" run_bf16.sh
  run_main TEMPOKV_INT2 2 run_tempokv.sh
  run_main TEMPOKV_INT4 4 run_tempokv.sh
fi
if [[ "$mode" == horizon || "$mode" == both ]]; then
  run_horizon BF16 "" run_bf16.sh
  run_horizon TEMPOKV_INT2 2 run_tempokv.sh
  run_horizon TEMPOKV_INT4 4 run_tempokv.sh
fi

if (( ! dry_run && collect )); then
  "$python_bin" "$script_dir/collect_efficiency.py" \
    --output-root "$output_root" --local-only
fi
