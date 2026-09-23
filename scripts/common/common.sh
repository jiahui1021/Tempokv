#!/usr/bin/env bash
# Shared paths, argument checks and codec helpers for every model's entry
# points.  scripts/<Model>/_common.sh sources this after naming its own
# directory.

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
default_codec_config="$project_root/configs/legacy.json"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  python_bin="$project_root/.venv/bin/python"
else
  python_bin="python3"
fi

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || die "缺少必需变量 ${name}（请用环境变量传入）"
  done
}

# KH / VH are short aliases for K_HISTORY_LENGTH / V_HISTORY_LENGTH; setting
# both to different values is a mistake, not a preference.
resolve_history() {
  local long="$1" short="$2"
  local long_value="${!long:-}" short_value="${!short:-}"
  if [[ -n "$long_value" && -n "$short_value" && "$long_value" != "$short_value" ]]; then
    die "$long=$long_value 与 $short=$short_value 冲突，只设置其中一个"
  fi
  printf '%s\n' "${long_value:-$short_value}"
}

# BITS sets both sides; K_BITS / V_BITS set one for a mixed-precision run.
resolve_bits() {
  local side="$1" side_value="${!1:-}" both_value="${BITS:-}"
  if [[ -n "$both_value" && -n "$side_value" && "$both_value" != "$side_value" ]]; then
    die "BITS=$both_value 与 $side=$side_value 冲突，只设置其中一个"
  fi
  printf '%s\n' "${side_value:-$both_value}"
}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$project_root" "${1#./}" ;;
  esac
}

check_flag() {
  local name="$1" value="${!1:-0}"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name 必须是 0 或 1（当前值：${value}）"
}

check_int() {
  local name="$1" min="$2" value="${!1:-}"
  if [[ "$value" =~ ^-?[0-9]+$ ]] && (( value >= min )); then
    return 0
  fi
  die "$name 必须是整数且 >= ${min}（当前值：${value}）"
}

# Parameter directory name; see scripts/common/codec_slug.py for the fields.
codec_slug() {
  "$python_bin" "$project_root/scripts/common/codec_slug.py" \
    "$1" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}" \
    "${8:-reconstructed}" "${9:-}"
}

# Print "<fit_mode> <v_predictor_mode> <k_predictor_mode>" for a codec config.
# A non-empty second argument overrides the config's V predictor.
codec_modes() {
  "$python_bin" - "$1" "${2:-}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
config = config.get("codec", config)
fit_mode = config.get("fit_mode", "legacy")
print(
    fit_mode,
    sys.argv[2] or config.get("v_predictor_mode", "cross_kv"),
    config.get("k_predictor_mode", "shared" if fit_mode == "legacy" else "layer_head"),
)
PY
}

# V parameter file written for a V predictor.  Anything unrecognised maps to
# the CrossKV name so command-only dry runs (PYTHON_BIN=/bin/echo) still work;
# the Python side validates the mode in real runs.
v_artifact_name() {
  case "$1" in
    temporal|identity|taylor|taylor2|direct) printf 'v_temporal.npz\n' ;;
    innovation) printf 'v_innovation.npz\n' ;;
    *) printf 'v_cross_kv.npz\n' ;;
  esac
}

# Whether a V predictor reads the reconstructed K through a Cross projection.
is_cross_v_predictor() {
  case "$1" in
    cross_kv|cross_kv_identity|cross_kv_taylor|cross_kv_taylor2|innovation) return 0 ;;
    *) return 1 ;;
  esac
}

# The Cross production kernel is validated at eight recurrent units per launch;
# temporal predictors keep the runtime default.  Explicit overrides win.
default_units_per_launch() {
  if is_cross_v_predictor "$1"; then
    export TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH="${TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH:-8}"
  fi
}

# Append the attention-refinement flags to fit_command when ATTENTION_STEPS is
# set.  CrossKV V refines K only by default; every other predictor refines both.
append_attention_refinement() {
  local v_predictor_mode="$1"
  [[ -n "${ATTENTION_STEPS:-}" ]] || return 0
  local refine_tensors="${ATTENTION_REFINE_TENSORS:-}"
  if [[ -z "$refine_tensors" ]]; then
    refine_tensors="key_value"
    [[ "$v_predictor_mode" == "cross_kv" ]] && refine_tensors="key"
  fi
  local -a step_sizes
  read -r -a step_sizes <<<"${ATTENTION_SIGN_STEP_SIZES:-0.01}"
  fit_command+=(
    --attention-steps "$ATTENTION_STEPS"
    --attention-lr "${ATTENTION_LR:-0.001}"
    --attention-patience "${ATTENTION_PATIENCE:-10}"
    --attention-min-relative-improvement "${ATTENTION_MIN_RELATIVE_IMPROVEMENT:-0.0001}"
    --attention-layer-chunk "${ATTENTION_LAYER_CHUNK:-4}"
    --attention-optimizer "${ATTENTION_OPTIMIZER:-adam}"
    --attention-reconstruction-weight "${ATTENTION_RECONSTRUCTION_WEIGHT:-0.5}"
    --attention-output-weight "${ATTENTION_OUTPUT_WEIGHT:-0.5}"
    --attention-refine-tensors "$refine_tensors"
    --attention-sign-step-sizes "${step_sizes[@]}"
  )
  if [[ -n "${ATTENTION_MAX_TOKENS:-}" ]]; then
    fit_command+=(--attention-max-tokens "$ATTENTION_MAX_TOKENS")
  fi
}

# Refuse to overwrite fitted parameters unless OVERWRITE=1.
refuse_existing_params() {
  [[ "${OVERWRITE:-0}" == "1" ]] && return 0
  local artifact
  for artifact in "$@"; do
    [[ -e "$artifact" ]] && die "已存在参数：${artifact}。设置 OVERWRITE=1 覆盖，或换一个 OUTPUT_DIR。"
  done
  return 0
}

# Fail when a fit finished without writing its parameter files.
require_params_written() {
  local artifact
  for artifact in "$@"; do
    [[ -s "$artifact" ]] || die "拟合后未生成参数文件：$artifact"
  done
}

# Print a command array for DRY_RUN=1.
print_command() {
  printf '  $'
  printf ' %q' "$@"
  printf '\n'
}
