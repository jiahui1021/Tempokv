#!/usr/bin/env bash
#
# run_study.sh — 跑完整的 calibration study（三个模块 + 合成图）
#
# 前置条件：每个模型各采集一份 calibration pool trace（见本目录 README.md）。
# 主实验（模块 1、2）只用 Self-Forcing 的 pool；模块 3 需要三个模型的 pool。
#
# 必需变量：
#   SF_POOL_TRACE      Self-Forcing pool trace（.npz）
#   PROMPT_GROUPS      经人工确认的 prompt_groups.json（默认用仓库内已提交的
#                      assets/calibration/prompt_groups.json）
#
# 可选变量：
#   CF_POOL_TRACE      Causal-Forcing pool trace；缺省则模块 3 跳过该模型
#   LONGCAT_POOL_TRACE LongCat-Video pool trace；同上
#   CF_PROMPT_GROUPS / LONGCAT_PROMPT_GROUPS
#                      各模型自己的 prompt_groups.json（默认与 PROMPT_GROUPS 相同）
#   OUTPUT_ROOT        输出根目录（默认 results/calibration）
#   SIZES              模块 1 的 N 网格（默认 "1 2 4 8 16 32"）
#   MODEL_SIZES        模块 3 的 N 网格（默认 "2 4 8 32"）
#   REPEATS            每个 N 的抽样次数（默认 3）
#   SEED               抽样种子（默认 0）
#   FIT_DEVICE         拟合设备（默认 auto）
#   EVAL_DEVICE        重建评估设备（默认 cpu）
#   KEEP_ARTIFACTS     保留中间 trace 与参数（0/1，默认 0；打开会占用大量磁盘）
#   PYTHON_BIN         Python 解释器
#
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  python_bin="$project_root/.venv/bin/python"
else
  python_bin="python3"
fi

resolve_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$project_root/$value"
  fi
}

[[ -n "${SF_POOL_TRACE:-}" ]] || die "缺少必需变量 SF_POOL_TRACE"
# The prompt pool and its groups are committed, so a fresh checkout on a GPU
# host already has the reviewed manifest that the collected trace must match.
PROMPT_GROUPS="${PROMPT_GROUPS:-assets/calibration/prompt_groups.json}"

sf_pool="$(resolve_path "$SF_POOL_TRACE")"
groups="$(resolve_path "$PROMPT_GROUPS")"
[[ -f "$sf_pool" ]] || die "pool trace 不存在：$sf_pool"
[[ -f "$groups" ]] || die "prompt groups 不存在：$groups"

output_root="$(resolve_path "${OUTPUT_ROOT:-results/calibration}")"
sizes="${SIZES:-1 2 4 8 16 32}"
model_sizes="${MODEL_SIZES:-2 4 8 32}"
repeats="${REPEATS:-3}"
seed="${SEED:-0}"
fit_device="${FIT_DEVICE:-auto}"
eval_device="${EVAL_DEVICE:-cpu}"
keep_artifacts="${KEEP_ARTIFACTS:-0}"
case "$keep_artifacts" in 0|1) ;; *) die "KEEP_ARTIFACTS 必须是 0 或 1" ;; esac

shared=(
  --prompt-groups "$groups"
  --repeats "$repeats"
  --seed "$seed"
  --fit-device "$fit_device"
  --eval-device "$eval_device"
  --python-bin "$python_bin"
)
[[ "$keep_artifacts" == "1" ]] && shared+=(--keep-artifacts)

echo "== 模块 1：calibration set size (Self-Forcing) =="
# shellcheck disable=SC2086
"$python_bin" "$script_dir/run_size.py" \
  --pool-trace "$sf_pool" \
  --output-dir "$output_root/size" \
  --sizes $sizes \
  "${shared[@]}"

echo "== 模块 2：calibration distribution shift (Self-Forcing) =="
"$python_bin" "$script_dir/run_shift.py" \
  --pool-trace "$sf_pool" \
  --output-dir "$output_root/shift" \
  "${shared[@]}"

echo "== 模块 3：calibration efficiency across models =="
model_args=(--model "self-forcing=$sf_pool:$groups")
if [[ -n "${CF_POOL_TRACE:-}" ]]; then
  cf_pool="$(resolve_path "$CF_POOL_TRACE")"
  cf_groups="$(resolve_path "${CF_PROMPT_GROUPS:-$PROMPT_GROUPS}")"
  [[ -f "$cf_pool" ]] || die "pool trace 不存在：$cf_pool"
  model_args+=(--model "causal-forcing=$cf_pool:$cf_groups")
else
  echo "  (跳过 Causal-Forcing：未设置 CF_POOL_TRACE)"
fi
if [[ -n "${LONGCAT_POOL_TRACE:-}" ]]; then
  longcat_pool="$(resolve_path "$LONGCAT_POOL_TRACE")"
  longcat_groups="$(resolve_path "${LONGCAT_PROMPT_GROUPS:-$PROMPT_GROUPS}")"
  [[ -f "$longcat_pool" ]] || die "pool trace 不存在：$longcat_pool"
  model_args+=(--model "longcat=$longcat_pool:$longcat_groups")
else
  echo "  (跳过 LongCat-Video：未设置 LONGCAT_POOL_TRACE)"
fi

models_command=(
  "$python_bin" "$script_dir/run_models.py"
  --output-dir "$output_root/models"
  "${model_args[@]}"
  --repeats "$repeats"
  --seed "$seed"
  --fit-device "$fit_device"
  --eval-device "$eval_device"
  --python-bin "$python_bin"
)
[[ "$keep_artifacts" == "1" ]] && models_command+=(--keep-artifacts)
# shellcheck disable=SC2086
"${models_command[@]}" --sizes $model_sizes

echo "== 合成图 (a)(b)(c) =="
"$python_bin" "$script_dir/plot_figure.py" \
  --size-report "$output_root/size/calibration_size.json" \
  --shift-report "$output_root/shift/calibration_shift.json" \
  --models-report "$output_root/models/calibration_across_models.json" \
  --output-dir "$output_root/figure"

echo "完成：$output_root"
