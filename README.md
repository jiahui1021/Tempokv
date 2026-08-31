# TempoKV

TempoKV 用时间预测和 CrossKV 预测压缩视频生成模型的 KV cache。仓库包含三个完整接入：
Self-Forcing、Causal Forcing 和 LongCat。

## 安装

```bash
# 基础代码
pip install -e .

# 按模型选择一个推理环境
pip install -e ".[cuda,self-forcing]"
pip install -e ".[cuda,causal-forcing]"
pip install -e ".[cuda,longcat]"

# 需要质量评测时安装
pip install -e ".[metrics]"
```

Self-Forcing 与 LongCat 依赖的 `diffusers`、`transformers` 和 `av` 版本不同，建议使用
独立虚拟环境。

## 完整流程

每个模型都按同一顺序运行：

```text
采集 clean trace → 拟合参数 → TempoKV 生成 → BF16 对照
```

### 1. 采集 trace

Self-Forcing：

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Self-Forcing/collect_trace.sh
```

Causal Forcing：

```bash
# 首次使用可先运行：bash scripts/Causal-Forcing/download_models.sh
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Causal-Forcing/collect_trace.sh
```

LongCat：

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/collect_trace.sh
```

### 2. 拟合参数

三个模型使用同一个入口：

```bash
FIT_DEVICE=cuda bash scripts/fit_params.sh self-forcing
FIT_DEVICE=cuda bash scripts/fit_params.sh causal-forcing
FIT_DEVICE=cuda bash scripts/fit_params.sh longcat
```

默认配置是 `configs/legacy.json`：INT4、K history 2、CrossKV V history 1、
layer-shared Cross 权重。常用覆盖项：

| 变量 | 作用 |
|---|---|
| `CONFIG` | 选择 `configs/legacy.json` |
| `TRACE_FILE` | 指定采集好的 `clean_trace.npz` |
| `OUTPUT_DIR` | 指定参数输出目录 |
| `BITS=2/4/8` | 同时设置 K/V 位宽 |
| `K_BITS`、`V_BITS` | 分别设置 K/V 位宽 |
| `KH`、`VH` | 设置 K/V history length |
| `V_PREDICTOR` | `cross_kv`、`temporal`、`identity`、`taylor`、`taylor2`、`innovation` 或 `direct` |
| `CACHE_STRIDE` | LongCat Context Parallel 的 local stride |
| `FIT_DEVICE` | `cuda`、`cpu` 或 `auto` |
| `DRY_RUN=1` | 只显示最终命令，不执行拟合 |

例如：

```bash
BITS=2 FIT_DEVICE=cuda bash scripts/fit_params.sh self-forcing
KH=3 VH=1 FIT_DEVICE=cuda bash scripts/fit_params.sh causal-forcing
CACHE_STRIDE=780 FIT_DEVICE=cuda bash scripts/fit_params.sh longcat
```

默认参数目录：

```text
data/self-forcing/params/<参数方案>/
data/causal-forcing/params/<模式>/<参数方案>/
data/longcat/params/<参数方案>/
```

每个目录包含 `k_temporal.npz`、一个 V 参数文件和 `fit_report.json`。参数不能跨模型、
stride 或配置复用。模型专用的旧入口仍可使用：
`scripts/<Model>/fit_params.sh`。

旧的 native 或非 layer-shared 参数不能与当前配置混用，需要按新的 legacy 配置重新拟合。

高级拟合选项可查看：

```bash
python scripts/common/fit_params.py --help
```

### 3. 生成视频

Self-Forcing：

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Self-Forcing/run_tempokv.sh

# 同一组 prompt 的 BF16 对照
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Self-Forcing/run_bf16.sh
```

Causal Forcing：

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Causal-Forcing/run_tempokv.sh

CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Causal-Forcing/run_bf16.sh
```

LongCat：

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/run_tempokv.sh

CHECKPOINT_DIR=/models/LongCat-Video \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/run_bf16.sh
```

TempoKV 生成会自动选择与拟合命令相同的参数目录。使用 `BITS`、`KH`、`VH`、
`V_PREDICTOR` 或 `CACHE_STRIDE` 时，拟合和生成必须传入相同的值。

## 查看结果

生成结果写入 `results/<模型>/`。至少检查：

- 视频是否完整生成；
- `tempokv_report.json` 中 encode/decode 计数是否非零；
- BF16 与 TempoKV 是否使用相同 prompt、seed 和帧数；
- 峰值显存、完整生成时间、质量指标和压缩比。

Cross kernel microbenchmark 需要 NVIDIA GPU：

```bash
python scripts/tools/benchmark_tempokv_kernels.py \
  --route v --sweep production --cross-matmul-precision ieee
```

查看完整生成的耗时分解：

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
PROFILE=1 bash scripts/Self-Forcing/run_tempokv.sh

python scripts/tools/profile_tempokv_runtime.py \
  results/self_forcing/legacy-kh2-vh1-int4-layershared
```

比较 BF16 与 TempoKV 视频质量：

```bash
python scripts/evaluation/evaluate_quality.py \
  results/self_forcing/bf16/videos \
  results/self_forcing/legacy-kh2-vh1-int4-layershared/videos \
  --device cuda \
  --output results/self_forcing/evaluation/quality.json
```

### Self-Forcing 最终四卡消融

需要把 K/V history、独立 temporal-V 对照、50 条视频和 cache/视频指标放在同一张表时，
使用 [`run_history_ablation_4gpu.sh`](scripts/Self-Forcing/run_history_ablation_4gpu.sh)。
默认配置是 K 轴 `h_K=0..5`、固定 CrossKV `h_V=1`，V 轴 CrossKV
`h_V=0,1,2,4,5`、固定 `h_K=3`，以及 `w/o cross` 的 temporal V `h_V=1`。
脚本要求四张卡，每个 variant 使用一张卡，并在
[`history_ablation/README.md`](scripts/paper_experiments/history_ablation/README.md)
中保存配置、参数、视频、日志和汇总格式。

## 详细说明

- [Self-Forcing](scripts/Self-Forcing/README.md)
- [Causal Forcing](scripts/Causal-Forcing/README.md)
- [LongCat](scripts/LongCat/README.md)
- [脚本目录](scripts/README.md)
- [CrossKV sharing 实验](scripts/paper_experiments/cross_kv_sharing/README.md)
- [Predictor replacement 实验](scripts/paper_experiments/predictor_ablation/README.md)
- [latency 实验](scripts/paper_experiments/latency/README.md)
