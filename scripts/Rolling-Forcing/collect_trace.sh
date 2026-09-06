#!/usr/bin/env bash
# Collect clean first-block K/V traces from the upstream Rolling Forcing loop.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
runtime_root="${ROLLING_FORCING_ROOT:?Set ROLLING_FORCING_ROOT to a pinned RollingForcing checkout}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to the Rolling Forcing checkpoint}"
prompts="${PROMPTS:?Set PROMPTS to a prompt file}"
trace_file="${TRACE_FILE:-$repo_root/data/rolling-forcing/trace/clean_trace.npz}"
trace_tokens="${TRACE_TOKENS_PER_FRAME:-8}"
trace_frames="${TRACE_LATENT_FRAMES:-51}"
output_dir="${OUTPUT_DIR:-$repo_root/results/rolling-forcing/trace}"
overwrite="${OVERWRITE:-0}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  python_bin="python3"
fi
[[ "${NUM_GPUS:-1}" == "1" ]] || {
  echo "Rolling Forcing trace collection currently supports NUM_GPUS=1" >&2
  exit 2
}
[[ "$overwrite" == "0" || "$overwrite" == "1" ]] || {
  echo "OVERWRITE must be 0 or 1" >&2
  exit 2
}

[[ -f "$runtime_root/inference.py" ]] || { echo "missing Rolling Forcing inference.py: $runtime_root" >&2; exit 2; }
[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
[[ -f "$prompts" ]] || { echo "missing prompts: $prompts" >&2; exit 2; }
[[ "$trace_frames" =~ ^[0-9]+$ ]] || { echo "TRACE_LATENT_FRAMES must be an integer" >&2; exit 2; }
(( trace_frames > 0 && trace_frames % 3 == 0 )) || { echo "TRACE_LATENT_FRAMES must be a positive multiple of 3" >&2; exit 2; }

mkdir -p -- "$(dirname -- "$trace_file")" "$output_dir"
if [[ -e "$trace_file" && "$overwrite" != "1" ]]; then
  echo "trace already exists: $trace_file (set OVERWRITE=1 to replace it)" >&2
  exit 2
fi
[[ "$overwrite" == "1" ]] && rm -f -- "$trace_file"
"$python_bin" "$repo_root/scripts/common/gpu_lease_guard.py" "${NUM_GPUS:-1}"
export TEMPOKV_TRACE_TOKENS_PER_FRAME="$trace_tokens"
"$python_bin" "$repo_root/experiments/Rolling-Forcing/inference.py" \
  --mode trace \
  --runtime-root "$runtime_root" \
  --trace-output "$trace_file" \
  -- \
  --checkpoint_path "$checkpoint" \
  --prompt_path "$prompts" \
  --output_path "$output_dir" \
  --num_latent_frames "$trace_frames" \
  --num_samples 1 \
  --save_with_index

printf 'Rolling Forcing clean trace: %s\n' "$trace_file"
