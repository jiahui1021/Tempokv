#!/usr/bin/env bash
# Fit TempoKV parameters from a Rolling Forcing clean_trace.npz archive.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
trace_file="${TRACE_FILE:-$repo_root/data/rolling-forcing/trace/clean_trace.npz}"
config="${CONFIG:-$repo_root/configs/legacy.json}"
k_history="${K_HISTORY_LENGTH:-${KH:-2}}"
v_history="${V_HISTORY_LENGTH:-${VH:-1}}"
k_bits="${K_BITS:-4}"
v_bits="${V_BITS:-$k_bits}"
if [[ "$k_bits" == "$v_bits" ]]; then
  bits_label="int${k_bits}"
else
  bits_label="k${k_bits}-v${v_bits}"
fi
param_name="${PARAM_NAME:-legacy-kh${k_history}-vh${v_history}-${bits_label}-layershared}"
output_dir="${OUTPUT_DIR:-$repo_root/data/rolling-forcing/params/$param_name}"
fit_device="${FIT_DEVICE:-auto}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  python_bin="python3"
fi

fit_command=(
  "$python_bin" "$repo_root/scripts/common/fit_params.py" "$trace_file"
  --config "$config"
  --output-dir "$output_dir"
  --coordinate-system pre_rope_normalized
  --fit-history-source "${FIT_HISTORY_SOURCE:-reconstructed}"
  --device "$fit_device"
  --stride 1560
  --k-history-length "$k_history"
  --v-history-length "$v_history"
  --v-predictor-mode "${V_PREDICTOR:-cross_kv}"
)
[[ -n "${K_BITS:-}" ]] && fit_command+=(--k-bits "$K_BITS")
[[ -n "${V_BITS:-}" ]] && fit_command+=(--v-bits "$V_BITS")
[[ "${DRY_RUN:-0}" == "1" ]] && { printf '%q ' "${fit_command[@]}" "$@"; printf '\n'; exit 0; }
[[ -f "$trace_file" ]] || { echo "TRACE_FILE does not exist: $trace_file" >&2; exit 2; }
[[ -f "$config" ]] || { echo "CONFIG does not exist: $config" >&2; exit 2; }
mkdir -p -- "$output_dir"
exec "${fit_command[@]}" "$@"
