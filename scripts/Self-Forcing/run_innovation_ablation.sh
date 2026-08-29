#!/usr/bin/env bash
# Small, paired Self-Forcing comparison of temporal, Cross+temporal, and innovation+temporal V predictors.
set -euo pipefail

# shellcheck source=scripts/Self-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

require_env CHECKPOINT WAN_MODEL_DIR PROMPTS

DRY_RUN="${DRY_RUN:-0}"
INCLUDE_BF16="${INCLUDE_BF16:-1}"
RUN_QUALITY="${RUN_QUALITY:-1}"
QUALITY_FID="${QUALITY_FID:-0}"
RESUME="${RESUME:-0}"
check_flag DRY_RUN
check_flag INCLUDE_BF16
check_flag RUN_QUALITY
check_flag QUALITY_FID
check_flag RESUME

command=(
  "$python_bin"
  "$project_root/scripts/paper_experiments/innovation_ablation/run.py"
  --checkpoint "$(resolve_path "$CHECKPOINT")"
  --wan-model-dir "$(resolve_path "$WAN_MODEL_DIR")"
  --prompts "$(resolve_path "$PROMPTS")"
  --trace "$(resolve_path "${TRACE_FILE:-data/self-forcing/trace/clean_trace.npz}")"
  --config "$(resolve_path "${CONFIG:-configs/legacy.json}")"
  --output-root "$(resolve_path "${OUTPUT_ROOT:-results/self_forcing/innovation_ablation_105f}")"
  --frames "${FRAMES:-105}"
  --seed "${SEED:-0}"
  --start-index "${START_INDEX:-0}"
  --end-index "${END_INDEX:-1}"
  --samples "${NUM_SAMPLES:-1}"
  --k-history-length "${KH:-${K_HISTORY_LENGTH:-2}}"
  --k-bits "${K_BITS:-${BITS:-4}}"
  --v-bits "${V_BITS:-${BITS:-4}}"
  --fit-device "${FIT_DEVICE:-auto}"
  --iterations "${ITERATIONS:-2}"
  --quality-device "${QUALITY_DEVICE:-auto}"
)
[[ -n "${MODEL_CONFIG:-}" ]] && command+=(--model-config "$(resolve_path "$MODEL_CONFIG")")
[[ -n "${LOCAL_ATTN_SIZE:-}" ]] && command+=(--local-attn-size "$LOCAL_ATTN_SIZE")
[[ -n "${MAX_QUALITY_FRAMES:-}" ]] && command+=(--max-quality-frames "$MAX_QUALITY_FRAMES")
[[ "$INCLUDE_BF16" == "0" ]] && command+=(--no-include-bf16)
[[ "$RUN_QUALITY" == "0" ]] && command+=(--no-quality)
[[ "$QUALITY_FID" == "1" ]] && command+=(--quality-fid)
[[ "$RESUME" == "1" ]] && command+=(--resume)
[[ "$DRY_RUN" == "1" ]] && command+=(--dry-run)

exec "${command[@]}"
