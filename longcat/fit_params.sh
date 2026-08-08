#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"

trace="${TRACE:-$project_root/parameters/longcat_trace.npz}"
config="${CONFIG:-$project_root/configs/longcat_int4.json}"
output_dir="${OUTPUT_DIR:-$project_root/parameters/longcat/kh2-vh2-int4}"

exec "$python_bin" "$project_root/scripts/fit_params.py" \
  "$trace" \
  --config "$config" \
  --output-dir "$output_dir" \
  --coordinate-system pre_rope_normalized \
  "$@"
