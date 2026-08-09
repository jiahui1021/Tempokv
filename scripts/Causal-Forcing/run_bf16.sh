#!/usr/bin/env bash
# Causal Forcing bf16 baseline, for parity and quality comparison.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=bf16 exec bash "$script_dir/generate.sh" "$@"
