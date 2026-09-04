#!/usr/bin/env bash
#
# generate.sh — 读取已拟合参数，生成 Self-Forcing 视频
#
# 不采集 trace、不重新拟合，只使用 PARAM_DIR 下的参数。更换 PROMPTS /
# OUTPUT_DIR 后可以重复运行。
#
# 两种模式：
#   MODE=tempokv  使用 TempoKV 量化参数（默认），参数从 PARAM_DIR 读取
#   MODE=bf16     不量化的对照基线，不需要 CONFIG / PARAM_DIR
#
# 用法（所有参数都用环境变量传入）：
#   CHECKPOINT=/models/self_forcing_dmd.pt \
#   WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
#   PROMPTS=assets/generation_prompts.txt \
#   CONFIG=configs/legacy.json \
#   KH=2 \
#   VH=1 \
#   FRAMES=717 \
#   bash scripts/Self-Forcing/generate.sh
#
# 必需变量：
#   CHECKPOINT    Self-Forcing 模型权重
#   WAN_MODEL_DIR Wan 模型目录
#   PROMPTS       prompt 列表
#
# 常用可选变量：
#   CONFIG          codec 配置 JSON（默认 configs/legacy.json）
#   PARAM_DIR       参数目录（默认 data/self-forcing/params/<参数方案>）
#   OUTPUT_DIR      结果输出目录（默认 results/$MODEL_TYPE/<量化标签>）
#   FRAMES          最终视频帧数（两种模式默认均为 717）
#   NUM_SAMPLES     每条 prompt 采样数（默认 1）
#   SEED            随机种子（默认 0）
#   START_INDEX / END_INDEX  prompt 序号范围（默认 0 .. -1 全部）
#   NUM_GPUS        torchrun 使用的 GPU 数（默认 1）
#   LOCAL_ATTN_SIZE 局部注意力窗口（latent 帧数；小于总帧数时启用滑动窗口）
#   SINK_SIZE       注意力 sink 帧数（窗口前固定的历史帧，默认 0）
#   K_HISTORY_LENGTH  K 使用的历史帧数（1 到 8；默认读取 CONFIG）
#   V_HISTORY_LENGTH  V 使用的历史帧数（0 到 5；默认读取 CONFIG）
#   BITS            K/V residual 位宽（1 到 8；默认读取 CONFIG）；K_BITS / V_BITS 可分别设
#   ANCHOR_BITS     anchor 位宽（1 到 16；默认读取 CONFIG，即 INT4）
#   TORCH_DECODER   用 PyTorch 参考解码器代替 Triton，0 或 1（默认 0）
#   BENCHMARK       要求本次完整生成且不允许跳过已有视频，0 或 1（默认 0）
#   PROFILE         记录每段 CUDA-event 耗时分解，0 或 1（默认 0）；结果写入
#                   $OUTPUT_DIR/profile/rank-N.json，用
#                   scripts/tools/profile_tempokv_runtime.py 查看
#   TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH  单次持续解码单元数（Cross 默认 8，
#                                       temporal 默认 16；范围 1 到 16）
#
set -euo pipefail

# shellcheck source=scripts/Self-Forcing/_common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

K_HISTORY_LENGTH="$(resolve_history K_HISTORY_LENGTH KH)"
V_HISTORY_LENGTH="$(resolve_history V_HISTORY_LENGTH VH)"
K_BITS="$(resolve_bits K_BITS)"
V_BITS="$(resolve_bits V_BITS)"
CONFIG="${CONFIG:-$default_codec_config}"
script_dir="$self_forcing_dir"
mode="${MODE:-tempokv}"
k_predictor_mode="shared"

# ---------- 模式与必需变量 ----------

case "$mode" in
  tempokv) ;;
  bf16) ;;
  *) die "MODE 必须是 tempokv 或 bf16（当前值：${mode}）" ;;
esac

require_env CHECKPOINT WAN_MODEL_DIR PROMPTS

# ---------- 目录 ----------

if [[ "$mode" == "tempokv" ]]; then
  # This only selects the matching fitted parameter directory; runtime always
  # consumes reconstructed cache state regardless of the fitting ablation.
  FIT_HISTORY_SOURCE="${FIT_HISTORY_SOURCE:-reconstructed}"
  parameter_name="${PARAM_NAME:-$(codec_slug "$(resolve_path "$CONFIG")" "${K_HISTORY_LENGTH:-}" "${V_HISTORY_LENGTH:-}" "${K_BITS:-}" "${V_BITS:-}" "" "${V_PREDICTOR:-}" "$FIT_HISTORY_SOURCE" "${ANCHOR_BITS:-}")}"
  param_dir="$(resolve_path "${PARAM_DIR:-data/self-forcing/params/$parameter_name}")"
  result_label="$parameter_name"
  v_predictor_mode="$("$python_bin" - "$(resolve_path "$CONFIG")" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config.get("codec", config).get("v_predictor_mode", "cross_kv"))
PY
)"
  k_predictor_mode="$("$python_bin" - "$(resolve_path "$CONFIG")" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config.get("codec", config).get("k_predictor_mode", "shared"))
PY
)"
  # V_PREDICTOR overrides the config so one file covers all supported predictors.
  if [[ -n "${V_PREDICTOR:-}" ]]; then
    case "$V_PREDICTOR" in
      cross_kv|temporal|identity|taylor|taylor2|cross_kv_identity|cross_kv_taylor|cross_kv_taylor2|innovation|direct) ;;
      *) die "V_PREDICTOR 必须是 cross_kv、temporal、identity、taylor、taylor2、cross_kv_identity、cross_kv_taylor、cross_kv_taylor2、innovation 或 direct（当前值：${V_PREDICTOR}）" ;;
    esac
    v_predictor_mode="$V_PREDICTOR"
  fi
  # DIRECTION mirrors the K/V roles; the artifacts are named for the role each
  # parameter set plays, and --k-params/--v-params are positional by role
  # (temporal first, cross projection second) rather than by tensor name.
  predictor_direction="${DIRECTION:-k_to_v}"
  case "$predictor_direction" in
    k_to_v|v_to_k) ;;
    *) die "DIRECTION 必须是 k_to_v 或 v_to_k（当前值：${predictor_direction}）" ;;
  esac
  if [[ "$predictor_direction" == "v_to_k" ]]; then
    [[ "$v_predictor_mode" == "cross_kv" ]] \
      || die "DIRECTION=v_to_k 需要 V_PREDICTOR=cross_kv（当前值：${v_predictor_mode}）"
    k_artifact_name="v_temporal_primary.npz"
    v_artifact_name="k_cross_from_v.npz"
  else
    k_artifact_name="k_temporal.npz"
    case "$v_predictor_mode" in
      # Direct V has no predictor; its empty temporal map shares the temporal layout.
      temporal|identity|taylor|taylor2|direct) v_artifact_name="v_temporal.npz" ;;
      cross_kv_identity|cross_kv_taylor|cross_kv_taylor2) v_artifact_name="v_cross_kv.npz" ;;
      innovation) v_artifact_name="v_innovation.npz" ;;
      cross_kv) v_artifact_name="v_cross_kv.npz" ;;
      # Keep command-only dry runs usable with PYTHON_BIN=/bin/echo; the Python
      # generation preflight performs authoritative config validation.
      *) v_artifact_name="v_cross_kv.npz" ;;
    esac
  fi
else
  param_dir=""
  result_label="bf16"
fi

# The Cross production kernel is validated at eight recurrent units per
# launch.  Keep temporal's existing default and preserve explicit overrides.
if [[ "$mode" == "tempokv" ]]; then
  case "$v_predictor_mode" in
    temporal|identity|taylor|taylor2|direct) ;;
    *) export TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH="${TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH:-8}" ;;
  esac
fi

model_type="${MODEL_TYPE:-self_forcing}"
model_type="${model_type%/}"
output_dir="$(resolve_path "${OUTPUT_DIR:-$project_root/results/$model_type/$result_label}")"
if [[ "$output_dir" == "/" ]]; then
  die "OUTPUT_DIR 不能是文件系统根目录"
fi

# ---------- 生成参数 ----------

FRAMES="${FRAMES:-717}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-0}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_TRITON_FALLBACK="${ALLOW_TRITON_FALLBACK:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BENCHMARK="${BENCHMARK:-0}"
PROFILE="${PROFILE:-0}"
TORCH_DECODER="${TORCH_DECODER:-0}"

check_flag DRY_RUN
check_flag ALLOW_TRITON_FALLBACK
check_flag SKIP_EXISTING
check_flag BENCHMARK
check_flag PROFILE
check_flag TORCH_DECODER
check_int FRAMES 1
check_int NUM_SAMPLES 1
check_int SEED 0
check_int START_INDEX 0
check_int END_INDEX -1
check_int NUM_GPUS 1
if [[ "$BENCHMARK" == "1" && "$SKIP_EXISTING" == "1" ]]; then
  die "BENCHMARK=1 不允许 SKIP_EXISTING=1"
fi
if (( (FRAMES + 3) % 12 != 0 )); then
  die "FRAMES 必须满足 (FRAMES + 3) % 12 == 0，保证视频帧精确对应完整的 3-frame latent block（当前值：${FRAMES}）"
fi
if [[ -n "${K_HISTORY_LENGTH:-}" ]]; then
  if [[ "$mode" == "tempokv" && "$k_predictor_mode" == "direct" ]]; then
    check_int K_HISTORY_LENGTH 0
  else
    check_int K_HISTORY_LENGTH 1
  fi
  (( K_HISTORY_LENGTH <= 8 )) || die "K_HISTORY_LENGTH 必须 <= 8（当前值：${K_HISTORY_LENGTH}）"
fi
if [[ -n "${V_HISTORY_LENGTH:-}" ]]; then
  check_int V_HISTORY_LENGTH 0
  (( V_HISTORY_LENGTH <= 5 )) || die "V_HISTORY_LENGTH 必须 <= 5（当前值：${V_HISTORY_LENGTH}）"
fi

latent_frames=$(( (FRAMES + 3) / 4 ))
if [[ "$mode" == "tempokv" && "$latent_frames" -lt 27 ]]; then
  die "tempokv 模式至少需要 27 个 latent 帧（约 105 个视频帧）才能触发量化，当前 FRAMES=$FRAMES"
fi

# ---------- 组装命令 ----------

generate_command=(
  "$python_bin" "$project_root/experiments/Self-Forcing/tempokv_adapter/generate.py"
  --mode "$mode"
  --checkpoint "$(resolve_path "$CHECKPOINT")"
  --prompts "$(resolve_path "$PROMPTS")"
  --output-dir "$output_dir"
  --frames "$FRAMES"
  --samples "$NUM_SAMPLES"
  --seed "$SEED"
  --start-index "$START_INDEX"
  --end-index "$END_INDEX"
  --num-gpus "$NUM_GPUS"
)
if [[ "$mode" == "tempokv" ]]; then
  generate_command+=(
    --config "$(resolve_path "$CONFIG")"
    --k-params "$param_dir/$k_artifact_name"
    --v-params "$param_dir/$v_artifact_name"
  )
  [[ -n "${K_HISTORY_LENGTH:-}" ]] && generate_command+=(--k-history-length "$K_HISTORY_LENGTH")
  [[ -n "${V_HISTORY_LENGTH:-}" ]] && generate_command+=(--v-history-length "$V_HISTORY_LENGTH")
  [[ -n "${K_BITS:-}" ]] && generate_command+=(--k-bits "$K_BITS")
  [[ -n "${V_BITS:-}" ]] && generate_command+=(--v-bits "$V_BITS")
  [[ -n "${ANCHOR_BITS:-}" ]] && generate_command+=(--anchor-bits "$ANCHOR_BITS")
  [[ -n "${V_PREDICTOR:-}" ]] && generate_command+=(--v-predictor-mode "$V_PREDICTOR")
  [[ -n "${DIRECTION:-}" ]] && generate_command+=(--predictor-direction "$predictor_direction")
  [[ "$TORCH_DECODER" == "1" ]] && generate_command+=(--torch-decoder)
fi

[[ -n "${MODEL_CONFIG:-}" ]] && generate_command+=(--model-config "$(resolve_path "$MODEL_CONFIG")")
[[ -n "${LOCAL_ATTN_SIZE:-}" ]] && generate_command+=(--local-attn-size "$LOCAL_ATTN_SIZE")
[[ -n "${SPAN_FRAMES:-}" ]] && generate_command+=(--span-frames "$SPAN_FRAMES")
[[ -n "${SINK_SIZE:-}" ]] && generate_command+=(--sink-size "$SINK_SIZE")
[[ -n "${TRITON_CACHE:-}" ]] && generate_command+=(--triton-cache "$(resolve_path "$TRITON_CACHE")")
[[ "$ALLOW_TRITON_FALLBACK" == "1" ]] && generate_command+=(--allow-triton-fallback)
[[ "$SKIP_EXISTING" == "1" ]] && generate_command+=(--skip-existing)
[[ "$BENCHMARK" == "1" ]] && generate_command+=(--benchmark)
[[ "$PROFILE" == "1" ]] && generate_command+=(--profile)
[[ "$DRY_RUN" == "1" ]] && generate_command+=(--dry-run)

export FORCING_WAN_MODEL_DIR="$(resolve_path "$WAN_MODEL_DIR")"
if [[ "$DRY_RUN" != "1" ]]; then
  [[ -d "$FORCING_WAN_MODEL_DIR" ]] || \
    die "WAN_MODEL_DIR 不存在：$FORCING_WAN_MODEL_DIR"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[generate] 干跑模式：打印最终命令，不执行\n'
  printf '  $'
  printf ' %q' "${generate_command[@]}"
  printf '\n'
  "${generate_command[@]}"
  exit 0
fi

mkdir -p -- "$output_dir"
printf '[generate] 输出目录：%s\n' "$output_dir"
"${generate_command[@]}" 2>&1 | tee "$output_dir/generate.log"
