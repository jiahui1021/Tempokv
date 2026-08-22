#!/usr/bin/env bash
# Exp. 5: compare K/V mixed bit-widths at the same cache setting.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/paper_experiments/_self_forcing_runner.sh
source "$script_dir/_self_forcing_runner.sh"

PARAM_ROOT="${PARAM_ROOT:-data/self-forcing/params/paper_exp5_mixed_bits}"
RESULT_ROOT="${RESULT_ROOT:-results/paper_exp5_mixed_bits}"
paper_init

# BF16 is the existing reference; this script only adds the quantized variants.
paper_run_tempokv k4v4 4 4
paper_run_tempokv k2v4 2 4
paper_run_tempokv k4v2 4 2

printf '\n[exp5] K3V3 skipped: current codec only supports 2/4/8-bit quantization.\n'
