#!/usr/bin/env bash
# Run only the max-feasible-horizon probes and render horizon.csv/horizon.tex.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$script_dir/run_efficiency.sh" --mode horizon "$@"
