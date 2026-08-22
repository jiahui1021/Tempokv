#!/usr/bin/env bash
# Exp. 7: compare key prediction in pre-RoPE and post-RoPE coordinates.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/paper_experiments/_self_forcing_runner.sh
source "$script_dir/_self_forcing_runner.sh"

COORDINATE="${COORDINATE:-pre}"
case "$COORDINATE" in
  pre)
    PARAM_ROOT="${PARAM_ROOT:-data/self-forcing/params/paper_exp7_rope}"
    RESULT_ROOT="${RESULT_ROOT:-results/paper_exp7_rope}"
    paper_init
    V_PREDICTOR=temporal paper_run_tempokv pre_rope 4 4
    ;;
  post)
    die "post-RoPE 尚未接入当前 runtime；当前拟合、加载和解码路径固定为 pre_rope_normalized"
    ;;
  *)
    die "COORDINATE 必须是 pre 或 post"
    ;;
esac
