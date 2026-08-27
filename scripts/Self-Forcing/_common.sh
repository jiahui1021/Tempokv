#!/usr/bin/env bash
# Shared path resolution and argument checks for Self-Forcing entry points.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Error: scripts/Self-Forcing/_common.sh 只能由其他脚本加载\n' >&2
  exit 2
fi

self_forcing_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$self_forcing_dir/../.." && pwd -P)"
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
    if [[ -z "${!name:-}" ]]; then
      die "缺少必需变量 ${name}（请用环境变量传入）"
    fi
  done
}

# KH / VH are short aliases for K_HISTORY_LENGTH / V_HISTORY_LENGTH. Both names
# are accepted; setting both to different values is a mistake, not a preference.
resolve_history() {
  local long="$1" short="$2"
  local long_value="${!long:-}" short_value="${!short:-}"
  if [[ -n "$long_value" && -n "$short_value" && "$long_value" != "$short_value" ]]; then
    die "$long=$long_value 与 $short=$short_value 冲突，只设置其中一个"
  fi
  printf '%s\n' "${long_value:-$short_value}"
}

# BITS sets both sides; K_BITS / V_BITS set one for a mixed-precision run.
# Setting BITS and a side to different values is a mistake, not a preference.
resolve_bits() {
  local side="$1" side_value="${!1:-}" both_value="${BITS:-}"
  if [[ -n "$both_value" && -n "$side_value" && "$both_value" != "$side_value" ]]; then
    die "BITS=$both_value 与 $side=$side_value 冲突，只设置其中一个"
  fi
  printf '%s\n' "${side_value:-$both_value}"
}

resolve_path() {
  local value="$1"
  case "$value" in
    /*) printf '%s\n' "$value" ;;
    *)  printf '%s/%s\n' "${project_root%/}" "${value#./}" ;;
  esac
}

check_flag() {
  local name="$1" value="${!1:-0}"
  [[ "$value" == "0" || "$value" == "1" ]] || \
    die "$name 必须是 0 或 1（当前值：${value}）"
}

check_int() {
  local name="$1" min="$2" value="${!1:-}"
  if [[ "$value" =~ ^-?[0-9]+$ ]] && (( value >= min )); then
    return 0
  fi
  die "$name 必须是整数且 >= ${min}（当前值：${value}）"
}

codec_slug() {
  local config_path="$1" k_override="${2:-}" v_override="${3:-}"
  local k_bits_override="${4:-}" v_bits_override="${5:-}" stride_override="${6:-}"
  local v_predictor_override="${7:-}"
  local fit_history_source="${8:-reconstructed}"
  "$python_bin" - "$config_path" "$k_override" "$v_override" \
    "$k_bits_override" "$v_bits_override" "$stride_override" \
    "$v_predictor_override" "$fit_history_source" <<'SLUGPY'
import json
import sys

# Kept to the standard library on purpose: this runs on every fit and generate
# invocation, and importing tempokv here would pull in torch just to build a
# directory name.  tests/test_configs.py asserts these defaults still match
# CodecConfig, so they cannot drift silently.
DEFAULT_K_BITS = 2
DEFAULT_V_BITS = 2
DEFAULT_STRIDE = 1560
DEFAULT_V_HISTORY_LENGTH = 0
DEFAULT_V_PREDICTOR_MODE = "cross_kv"


def override(index):
    raw = sys.argv[index]
    return int(raw) if raw else None


config = json.load(open(sys.argv[1], encoding="utf-8"))
config = config.get("codec", config)

k_bits = override(4) or int(config.get("k_bits", DEFAULT_K_BITS))
v_bits = override(5) or int(config.get("v_bits", DEFAULT_V_BITS))
stride = override(6) or int(config.get("stride", DEFAULT_STRIDE))
k_override = override(2)
if k_override is None:
    k_history = config.get("k_history_length")
    if k_history is None:
        k_history = 3
else:
    k_history = k_override
v_history = override(3)
if v_history is None:
    v_history = config.get("v_history_length", DEFAULT_V_HISTORY_LENGTH)
    if v_history is None:
        v_history = 0

# All shipped TempoKV parameter sets use the legacy fixed-point protocol.
if config.get("fit_mode", "legacy") != "legacy":
    raise SystemExit("only fit_mode=legacy is supported")
mode_label = "legacy-"
k_label = f"kh{int(k_history)}"
# The V segment records the predictor structure: parameters fitted for the
# temporal V predictor are not interchangeable with CrossKV ones.
v_predictor = sys.argv[7] or config.get(
    "v_predictor_mode", DEFAULT_V_PREDICTOR_MODE
)
v_prefix = "vt" if v_predictor == "temporal" else "vh"
v_label = f"{v_prefix}{int(v_history)}"
cross_weight_mode = config.get("cross_kv_weight_mode", "layer_shared")
if cross_weight_mode != "layer_shared":
    raise SystemExit(
        "production CrossKV parameters must use cross_kv_weight_mode=layer_shared"
    )
cross_weight_label = "-layershared"
if k_bits == v_bits:
    precision_label = f"int{k_bits}"
else:
    # Keep the historical intN name for equal precision while making mixed
    # K/V precision directories unambiguous.
    precision_label = f"k{k_bits}-v{v_bits}"
# stride is part of the codec signature, so a context-parallel run fits its own
# parameters.  Without this segment every CP degree wrote to one directory and
# silently overwrote the others.
stride_label = "" if stride == DEFAULT_STRIDE else f"-s{stride}"
fit_history_source = sys.argv[8]
if fit_history_source not in {"reconstructed", "clean"}:
    raise SystemExit(
        "FIT_HISTORY_SOURCE must be reconstructed or clean, got "
        f"{fit_history_source!r}"
    )
fit_label = "" if fit_history_source == "reconstructed" else "-fitclean"
print(
    f"{mode_label}{k_label}-{v_label}-{precision_label}"
    f"{stride_label}{cross_weight_label}{fit_label}"
)
SLUGPY
}
