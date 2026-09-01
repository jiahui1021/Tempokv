#!/usr/bin/env bash
# One fitting entry point for Self-Forcing, Causal-Forcing, and LongCat.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/fit_params.sh <model>

Models:
  self-forcing    Self-Forcing
  causal-forcing  Causal Forcing
  longcat         LongCat

Common overrides:
  CONFIG=...              Codec config (default: configs/legacy.json)
  TRACE_FILE=...          Clean trace archive
  OUTPUT_DIR=...          Parameter output directory
  BITS=2|4|8              Set K/V bit width together
  K_BITS=... V_BITS=...   Set K/V bit width separately
  KH=... VH=...           K/V history length
  V_PREDICTOR=cross_kv|temporal|identity|taylor|taylor2|cross_kv_identity|cross_kv_taylor|cross_kv_taylor2|innovation|direct
  FIT_DEVICE=auto|cuda|cpu
  DRY_RUN=1               Print the resolved fitting command

Read scripts/<Model>/README.md for advanced fitting controls. Existing
scripts/<Model>/fit_params.sh entry points remain supported.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  self|self-forcing|Self-Forcing)
    model_dir="Self-Forcing"
    ;;
  causal|causal-forcing|Causal-Forcing)
    model_dir="Causal-Forcing"
    ;;
  longcat|LongCat)
    model_dir="LongCat"
    ;;
  *)
    printf 'Unknown model: %s\n\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$script_dir/$model_dir/fit_params.sh" "$@"
