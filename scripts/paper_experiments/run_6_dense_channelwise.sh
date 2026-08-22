#!/usr/bin/env bash
# Exp. 6: compare dense W_K->V with channel-wise W_K->V.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/paper_experiments/_self_forcing_runner.sh
source "$script_dir/_self_forcing_runner.sh"

MAP="${MAP:-dense}"
case "$MAP" in
  dense)
    PARAM_ROOT="${PARAM_ROOT:-data/self-forcing/params/paper_exp6_dense_channelwise}"
    RESULT_ROOT="${RESULT_ROOT:-results/paper_exp6_dense_channelwise}"
    paper_init
    V_PREDICTOR=cross_kv paper_run_tempokv dense 4 4
    ;;
  channelwise)
    die "channel-wise W_K→V 尚未接入当前 runtime；--v-history-mode diagonal 只改变 V-history，不能代替本实验"
    ;;
  *)
    die "MAP 必须是 dense 或 channelwise"
    ;;
esac
