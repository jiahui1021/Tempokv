#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/LongCat/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

trace="${TRACE:-$project_root/data/longcat/trace/clean_trace.npz}"
config="${CONFIG:-$project_root/configs/longcat_int4.json}"
output_dir="${OUTPUT_DIR:-$project_root/data/longcat/params/kh2-vh2-int4}"

exec "$python_bin" "$project_root/scripts/common/fit_params.py" \
  "$trace" \
  --config "$config" \
  --output-dir "$output_dir" \
  --coordinate-system pre_rope_normalized \
  "$@"
