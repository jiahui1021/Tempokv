#!/usr/bin/env bash
# Run paired quality/FID and VBench-Long for one candidate against a BF16 baseline.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || die "缺少必需变量 $name"
  done
}

check_flag() {
  local name="$1" value="${!1:-0}"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name 必须是 0 或 1"
}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${project_root%/}" "${1#./}" ;;
  esac
}

require_env BASELINE_DIR CANDIDATE_DIR CANDIDATE_NAME OUTPUT_DIR VBENCH_CACHE_DIR
[[ "$CANDIDATE_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || \
  die "CANDIDATE_NAME 只能包含字母、数字、点、下划线和连字符"
[[ "$CANDIDATE_NAME" != "." && "$CANDIDATE_NAME" != ".." ]] || \
  die "CANDIDATE_NAME 不能是 . 或 .."

baseline_dir="$(resolve_path "$BASELINE_DIR")"
candidate_dir="$(resolve_path "$CANDIDATE_DIR")"
output_dir="$(resolve_path "$OUTPUT_DIR")"
[[ -d "$baseline_dir" ]] || die "BASELINE_DIR 不存在：$baseline_dir"
[[ -d "$candidate_dir" ]] || die "CANDIDATE_DIR 不存在：$candidate_dir"
[[ -d "$VBENCH_CACHE_DIR" ]] || die "VBENCH_CACHE_DIR 不存在：$VBENCH_CACHE_DIR"
[[ "$output_dir" != "/" ]] || die "OUTPUT_DIR 不能是文件系统根目录"

DEVICE="${DEVICE:-auto}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
VBENCH_MODE="${VBENCH_MODE:-long_custom_input}"
VBENCH_READ_FRAME="${VBENCH_READ_FRAME:-0}"
FORCE="${FORCE:-0}"
check_flag VBENCH_READ_FRAME
check_flag FORCE
[[ "$FID_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "FID_BATCH_SIZE 必须是正整数"
[[ "$FRAME_STRIDE" =~ ^[1-9][0-9]*$ ]] || die "FRAME_STRIDE 必须是正整数"
if [[ -n "${MAX_FRAMES:-}" && ! "$MAX_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
  die "MAX_FRAMES 必须是正整数"
fi
case "$VBENCH_MODE" in
  long_custom_input|long_vbench_standard) ;;
  *) die "VBENCH_MODE 必须是 long_custom_input 或 long_vbench_standard" ;;
esac
if [[ -n "${PROMPT_FILE:-}" ]]; then
  prompt_file="$(resolve_path "$PROMPT_FILE")"
  [[ -f "$prompt_file" ]] || die "PROMPT_FILE 不存在：$prompt_file"
  [[ "$VBENCH_MODE" == "long_custom_input" ]] || \
    die "PROMPT_FILE 只能用于 VBENCH_MODE=long_custom_input"
elif [[ "$VBENCH_MODE" == "long_custom_input" ]]; then
  die "VBENCH_MODE=long_custom_input 时必须设置 PROMPT_FILE"
fi
if [[ -n "${VBENCH_FULL_INFO:-}" ]]; then
  vbench_full_info="$(resolve_path "$VBENCH_FULL_INFO")"
  [[ -f "$vbench_full_info" ]] || die "VBENCH_FULL_INFO 不存在：$vbench_full_info"
fi
if [[ "$VBENCH_MODE" == "long_vbench_standard" && -z "${VBENCH_FULL_INFO:-}" ]]; then
  die "VBENCH_MODE=long_vbench_standard 时必须设置 VBENCH_FULL_INFO"
fi

candidate_output="$output_dir/$CANDIDATE_NAME"
baseline_vbench_output="$output_dir/baseline/vbench"
candidate_vbench_output="$candidate_output/vbench"
mkdir -p -- "$candidate_output" "$baseline_vbench_output" "$candidate_vbench_output"

quality_command=(
  "$python_bin" "$script_dir/evaluate_quality.py"
  "$baseline_dir" "$candidate_dir"
  --device "$DEVICE"
  --fid-batch-size "$FID_BATCH_SIZE"
  --frame-stride "$FRAME_STRIDE"
  --output "$candidate_output/quality.json"
)
[[ -n "${MAX_FRAMES:-}" ]] && quality_command+=(--max-frames "$MAX_FRAMES")
[[ -n "${LPIPS_NETWORK:-}" ]] && quality_command+=(--lpips-network "$LPIPS_NETWORK")

vbench_common=(--device "$DEVICE" --mode "$VBENCH_MODE")
[[ -n "${PROMPT_FILE:-}" ]] && vbench_common+=(--prompt-file "$prompt_file")
[[ -n "${VBENCH_DIMENSIONS:-}" ]] && vbench_common+=(--dimensions "$VBENCH_DIMENSIONS")
[[ -n "${VBENCH_FULL_INFO:-}" ]] && vbench_common+=(--full-info "$vbench_full_info")
[[ "$VBENCH_READ_FRAME" == "1" ]] && vbench_common+=(--read-frame)
[[ "$FORCE" == "0" ]] && vbench_common+=(--reuse-complete)

printf '[evaluate_all] quality + FID: %s\n' "$CANDIDATE_NAME"
"${quality_command[@]}"

printf '[evaluate_all] VBench-Long: bf16 baseline\n'
"$python_bin" "$script_dir/evaluate_vbench.py" "$baseline_dir" \
  --output-dir "$baseline_vbench_output" \
  --name bf16 \
  --config-out "$baseline_vbench_output/evaluation.json" \
  "${vbench_common[@]}"

printf '[evaluate_all] VBench-Long: %s\n' "$CANDIDATE_NAME"
"$python_bin" "$script_dir/evaluate_vbench.py" "$candidate_dir" \
  --output-dir "$candidate_vbench_output" \
  --name "$CANDIDATE_NAME" \
  --config-out "$candidate_vbench_output/evaluation.json" \
  "${vbench_common[@]}"

printf '[evaluate_all] 完成：%s\n' "$candidate_output"
