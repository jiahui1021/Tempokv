#!/usr/bin/env bash
# Shared launcher for the three paper ablations.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Error: this file is a helper and must be sourced\n' >&2
  exit 2
fi

set -euo pipefail

# shellcheck source=scripts/Self-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../Self-Forcing" && pwd -P)/_common.sh"

paper_init() {
  TRACE_FILE="${TRACE_FILE:-data/self-forcing/trace/clean_trace.npz}"
  require_env CHECKPOINT WAN_MODEL_DIR PROMPTS

  CONFIG="$(resolve_path "${CONFIG:-configs/legacy.json}")"
  CHECKPOINT="$(resolve_path "$CHECKPOINT")"
  WAN_MODEL_DIR="$(resolve_path "$WAN_MODEL_DIR")"
  PROMPTS="$(resolve_path "$PROMPTS")"
  TRACE_FILE="$(resolve_path "$TRACE_FILE")"
  PARAM_ROOT="$(resolve_path "${PARAM_ROOT:-data/self-forcing/params/paper_567}")"
  RESULT_ROOT="$(resolve_path "${RESULT_ROOT:-results/paper_567}")"

  KH="${KH:-2}"
  VH="${VH:-1}"
  V_PREDICTOR="${V_PREDICTOR:-cross_kv}"
  FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
  FIT_DEVICE="${FIT_DEVICE:-auto}"
  ITERATIONS="${ITERATIONS:-2}"
  FRAMES="${FRAMES:-717}"
  NUM_SAMPLES="${NUM_SAMPLES:-1}"
  SEED="${SEED:-0}"
  START_INDEX="${START_INDEX:-0}"
  END_INDEX="${END_INDEX:--1}"
  NUM_GPUS="${NUM_GPUS:-1}"
  DRY_RUN="${DRY_RUN:-0}"
  SPAN_FRAMES="${SPAN_FRAMES:-}"
  ATTENTION_STEPS="${ATTENTION_STEPS:-}"

  check_flag DRY_RUN
  check_int KH 1
  check_int VH 0
  check_int ITERATIONS 1
  check_int FRAMES 1
  check_int NUM_SAMPLES 1
  check_int SEED 0
  check_int START_INDEX 0
  check_int END_INDEX -1
  check_int NUM_GPUS 1
  [[ -z "$SPAN_FRAMES" ]] || check_int SPAN_FRAMES 2
  [[ -z "$ATTENTION_STEPS" ]] || check_int ATTENTION_STEPS 1

  case "$V_PREDICTOR" in
    cross_kv|temporal) ;;
    *) die "V_PREDICTOR 必须是 cross_kv 或 temporal" ;;
  esac

  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$CONFIG" ]] || die "CONFIG 不存在：$CONFIG"
    [[ -f "$TRACE_FILE" ]] || die "TRACE_FILE 不存在：$TRACE_FILE"
    [[ -f "$CHECKPOINT" ]] || die "CHECKPOINT 不存在：$CHECKPOINT"
    [[ -d "$WAN_MODEL_DIR" ]] || die "WAN_MODEL_DIR 不存在：$WAN_MODEL_DIR"
    [[ -f "$PROMPTS" ]] || die "PROMPTS 不存在：$PROMPTS"
  fi
}

paper_guard_target() {
  local path="$1" label="$2"
  [[ "$DRY_RUN" == "1" || ! -e "$path" ]] || \
    die "$label 已存在：$path；请更换 PARAM_ROOT/RESULT_ROOT"
}

paper_print_command() {
  printf '[paper]'
  printf ' %q' "$@"
  printf '\n'
}

paper_run_bf16() {
  local label="$1"
  local output_dir="$RESULT_ROOT/$label"
  paper_guard_target "$output_dir" "BF16 输出目录"

  local -a command=(
    env \
    MODE=bf16 \
    CHECKPOINT="$CHECKPOINT" \
    WAN_MODEL_DIR="$WAN_MODEL_DIR" \
    PROMPTS="$PROMPTS" \
    OUTPUT_DIR="$output_dir" \
    FRAMES="$FRAMES" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    SEED="$SEED" \
    START_INDEX="$START_INDEX" \
    END_INDEX="$END_INDEX" \
    NUM_GPUS="$NUM_GPUS" \
    SKIP_EXISTING=1 \
    BENCHMARK=0 \
    DRY_RUN="$DRY_RUN" \
    bash "$project_root/scripts/Self-Forcing/generate.sh"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    paper_print_command "${command[@]}"
  else
    "${command[@]}"
  fi
}

paper_run_tempokv() {
  local label="$1" k_bits="$2" v_bits="$3"
  local param_dir="$PARAM_ROOT/$label"
  local output_dir="$RESULT_ROOT/$label"
  paper_guard_target "$param_dir" "参数目录"
  paper_guard_target "$output_dir" "视频输出目录"

  local -a fit_env=(
    CONFIG="$CONFIG"
    TRACE_FILE="$TRACE_FILE"
    OUTPUT_DIR="$param_dir"
    KH="$KH"
    VH="$VH"
    K_BITS="$k_bits"
    V_BITS="$v_bits"
    V_PREDICTOR="$V_PREDICTOR"
    FIT_HISTORY_SOURCE="$FIT_HISTORY_SOURCE"
    FIT_DEVICE="$FIT_DEVICE"
    ITERATIONS="$ITERATIONS"
    DRY_RUN="$DRY_RUN"
  )
  [[ -z "$SPAN_FRAMES" ]] || fit_env+=(SPAN_FRAMES="$SPAN_FRAMES")
  [[ -z "$ATTENTION_STEPS" ]] || fit_env+=(ATTENTION_STEPS="$ATTENTION_STEPS")
  local -a fit_command=(
    env "${fit_env[@]}" bash "$project_root/scripts/Self-Forcing/fit_params.sh"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    paper_print_command "${fit_command[@]}"
  else
    "${fit_command[@]}"
  fi

  local -a generate_command=(
    env \
    MODE=tempokv \
    CHECKPOINT="$CHECKPOINT" \
    WAN_MODEL_DIR="$WAN_MODEL_DIR" \
    PROMPTS="$PROMPTS" \
    CONFIG="$CONFIG" \
    PARAM_DIR="$param_dir" \
    OUTPUT_DIR="$output_dir" \
    KH="$KH" \
    VH="$VH" \
    K_BITS="$k_bits" \
    V_BITS="$v_bits" \
    V_PREDICTOR="$V_PREDICTOR" \
    FRAMES="$FRAMES" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    SEED="$SEED" \
    START_INDEX="$START_INDEX" \
    END_INDEX="$END_INDEX" \
    NUM_GPUS="$NUM_GPUS" \
    SKIP_EXISTING=1 \
    BENCHMARK=0 \
    DRY_RUN="$DRY_RUN" \
    bash "$project_root/scripts/Self-Forcing/generate.sh"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    paper_print_command "${generate_command[@]}"
  else
    "${generate_command[@]}"
  fi
}
