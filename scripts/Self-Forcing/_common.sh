#!/usr/bin/env bash
# Shared path resolution and argument checks for Self-Forcing entry points.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Error: scripts/Self-Forcing/_common.sh 只能由其他脚本加载\n' >&2
  exit 2
fi

self_forcing_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$self_forcing_dir/../.." && pwd -P)"
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
    die "$name 必须是 0 或 1（当前值：$value）"
}

check_int() {
  local name="$1" min="$2" value="${!1:-}"
  if [[ "$value" =~ ^-?[0-9]+$ ]] && (( value >= min )); then
    return 0
  fi
  die "$name 必须是整数且 >= $min（当前值：$value）"
}

codec_slug() {
  local config_path="$1" k_override="${2:-}" v_override="${3:-}"
  "$python_bin" - "$config_path" "$k_override" "$v_override" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
config = config.get("codec", config)
k_bits = int(config["k_bits"])
v_bits = int(config["v_bits"])
mode_label = "legacy-" if config.get("fit_mode") == "legacy" else ""
shared_history = (
    config.get("k_max_history_length") is not None
    or config.get("v_max_history_length") is not None
)
if shared_history:
    k_max = config.get("k_max_history_length")
    if k_max is None:
        k_max = config.get("k_history_length", 3)
    v_max = config.get("v_max_history_length")
    if v_max is None:
        v_max = config.get("v_history_length", 0)
    k_label = f"khshared{int(k_max)}"
    v_label = f"vhshared{int(v_max)}"
else:
    k_history = int(sys.argv[2]) if sys.argv[2] else config.get("k_history_length")
    if k_history is None:
        k_history = 3
    v_history = int(sys.argv[3]) if sys.argv[3] else config.get("v_history_length", 0)
    if v_history is None:
        v_history = 0
    k_label = f"kh{int(k_history)}"
    # The V segment records the predictor structure: parameters fitted for the
    # temporal V predictor are not interchangeable with CrossKV ones.
    v_prefix = "vt" if config.get("v_predictor_mode") == "temporal" else "vh"
    v_label = f"{v_prefix}{int(v_history)}"
if k_bits == v_bits:
    precision_label = f"int{k_bits}"
else:
    # Keep the historical intN name for equal precision while making mixed
    # K/V precision directories unambiguous.
    precision_label = f"k{k_bits}-v{v_bits}"
print(f"{mode_label}{k_label}-{v_label}-{precision_label}")
PY
}
