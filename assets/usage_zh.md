# TempoKV 中文使用指南

TempoKV 的代码按职责分为三层：

- `experiments/`：Self-Forcing、LongCat 等模型侧代码。
- `src/tempokv/`：TempoKV 核心算法、参数与 Triton 算子。
- `scripts/<模型>/`：采集、拟合和生成入口。

可复用数据统一存入 `data/<模型>/`，生成视频和报告存入 `results/`。

## 安装

```bash
pip install -e ".[cuda]"
```

## Self-Forcing

采集并合并 clean trace：

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Self-Forcing/collect_trace.sh
```

输出位于：

```text
data/self-forcing/trace/raw/
data/self-forcing/trace/clean_trace.npz
```

拟合参数：

```bash
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=2 \
V_HISTORY_LENGTH=2 \
bash scripts/Self-Forcing/fit_params.sh
```

输出位于：

```text
data/self-forcing/params/kh2-vh2-int2/
```

TempoKV 生成：

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=2 \
V_HISTORY_LENGTH=2 \
FRAMES=717 \
bash scripts/Self-Forcing/run_tempokv.sh
```

BF16 生成：

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
FRAMES=81 \
bash scripts/Self-Forcing/run_bf16.sh
```

Self-Forcing 三种模式都使用
`experiments/Self-Forcing/inference.py`，不需要 `MODEL_REPO`、
`TRACE_RUNNER` 或 QVG。

详细说明见 [scripts/Self-Forcing/README.md](../scripts/Self-Forcing/README.md)。

## LongCat

LongCat 运行代码已经包含在本仓库，只需提供模型权重目录：

```bash
pip install -e ".[longcat]"

CHECKPOINT_DIR=/models/LongCat-Video \
bash scripts/LongCat/collect_trace.sh

CONFIG=configs/longcat_int4.json \
bash scripts/LongCat/fit_params.sh

CHECKPOINT_DIR=/models/LongCat-Video \
bash scripts/LongCat/run_bf16.sh

CHECKPOINT_DIR=/models/LongCat-Video \
CONFIG=configs/longcat_int4.json \
K_PARAMS=data/longcat/params/kh2-vh2-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh2-int4/v_cross_kv.npz \
bash scripts/LongCat/run_tempokv.sh
```

默认会根据 prompt 生成首段，再继续生成长视频。`INIT_VIDEO_PATH` 仅用于从已有
视频继续生成，不再是必需参数。
