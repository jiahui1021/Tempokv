#!/usr/bin/env bash
# Shared path resolution and argument checks for Self-Forcing entry points.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Error: scripts/Self-Forcing/_common.sh 只能由其他脚本加载\n' >&2
  exit 2
fi

self_forcing_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$self_forcing_dir/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"

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
k_history = int(sys.argv[2]) if sys.argv[2] else int(config["k_history_length"])
v_history = int(sys.argv[3]) if sys.argv[3] else int(config["v_history_length"])
bits = int(config["k_bits"])
print(f"kh{k_history}-vh{v_history}-int{bits}")
PY
}
