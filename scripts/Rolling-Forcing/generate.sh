#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
mode="${MODE:-tempokv}"
runtime_root="${ROLLING_FORCING_ROOT:?Set ROLLING_FORCING_ROOT to a pinned RollingForcing checkout}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  python_bin="python3"
fi
[[ "${NUM_GPUS:-1}" == "1" ]] || {
  echo "Rolling Forcing adapter currently supports NUM_GPUS=1; use one leased GPU" >&2
  exit 2
}

if [[ "$mode" == "tempokv" ]]; then
  config="${CONFIG:-$repo_root/configs/legacy.json}"
  parameter_dir="${PARAM_DIR:-$repo_root/data/rolling-forcing/params/legacy-kh2-vh1-int4-layershared}"
  case "$config" in /*) ;; *) config="$repo_root/${config#./}" ;; esac
  case "$parameter_dir" in /*) ;; *) parameter_dir="$repo_root/${parameter_dir#./}" ;; esac
  v_predictor="${V_PREDICTOR:-cross_kv}"
  case "$v_predictor" in
    cross_kv) v_params="$parameter_dir/v_cross_kv.npz" ;;
    temporal) v_params="$parameter_dir/v_temporal.npz" ;;
    *) echo "V_PREDICTOR must be cross_kv or temporal" >&2; exit 2 ;;
  esac
  [[ -f "$config" ]] || { echo "missing CONFIG: $config" >&2; exit 2; }
  [[ -f "$parameter_dir/k_temporal.npz" ]] || { echo "missing K parameters: $parameter_dir/k_temporal.npz" >&2; exit 2; }
  [[ -f "$v_params" ]] || { echo "missing V parameters: $v_params" >&2; exit 2; }
  export TEMPOKV_CONFIG_PATH="$config"
  export TEMPOKV_K_PARAMS_PATH="$parameter_dir/k_temporal.npz"
  export TEMPOKV_V_PARAMS_PATH="$v_params"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "$python_bin" "$repo_root/scripts/common/gpu_lease_guard.py" "${NUM_GPUS:-1}"
fi

exec "$python_bin" "$repo_root/experiments/Rolling-Forcing/inference.py" \
  --mode "$mode" \
  --runtime-root "$runtime_root" \
  "$@"
