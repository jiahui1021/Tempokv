#!/usr/bin/env bash
# Self-Forcing BF16 baseline.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=bf16 exec bash "$script_dir/generate.sh" "$@"
