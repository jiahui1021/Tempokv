#!/usr/bin/env bash
# Shared paths and argument checks for LongCat entry points.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Error: scripts/LongCat/_common.sh 只能由其他脚本加载\n' >&2
  exit 2
fi

longcat_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$longcat_script_dir/../.." && pwd -P)"
adapter_root="$project_root/experiments/LongCat/longcat"
python_bin="${PYTHON_BIN:-python3}"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"

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
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$project_root" "${1#./}" ;;
  esac
}
