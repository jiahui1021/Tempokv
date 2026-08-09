# TempoKV

TempoKV 使用时间预测与 CrossKV 预测压缩视频生成模型的 KV cache。仓库包含
PyTorch 参考实现、可选 Triton 算子、参数拟合工具，以及 Self-Forcing、
Causal Forcing 和 LongCat 的模型接入。

## 目录

```text
Tempokv/
├── experiments/
│   ├── Self-Forcing/          Self-Forcing 模型与推理代码
│   ├── Causal-Forcing/        Causal Forcing 模型与推理代码
│   └── LongCat/               LongCat 模型接入代码
├── src/tempokv/               TempoKV 核心算法、参数格式与 Triton 算子
│   └── runtime/               因果视频模型共用的 KV cache、RoPE、流式编码与指标
├── scripts/
│   ├── Self-Forcing/          Self-Forcing 采集、拟合与生成入口
│   ├── Causal-Forcing/        Causal Forcing 下载、采集、拟合与生成入口
│   ├── LongCat/               LongCat 采集、拟合与生成入口
│   ├── common/                多模型共用的参数拟合入口
│   ├── evaluation/            视频质量评测
│   └── tools/                 通用分析与流水线工具
├── data/
│   ├── self-forcing/
│   │   ├── trace/             raw trace 与合并后的 clean trace
│   │   └── params/            按参数方案分组的拟合参数
│   ├── causal-forcing/
│   │   ├── trace/<模式>/
│   │   └── params/<模式>/
│   └── longcat/
│       ├── trace/
│       └── params/
├── configs/                   TempoKV 配置
└── results/                   生成视频、报告与评测结果
```

目录职责固定：

- `experiments/` 只保存模型侧代码。
- `src/tempokv/` 只保存 TempoKV 自己的算法和算子；`src/tempokv/runtime/` 保存
  Self-Forcing 与 Causal Forcing 共用的因果 KV cache 运行时。
- `scripts/` 只保存最终调用入口与通用工具。
- `data/` 保存可重复使用的 trace 和拟合参数。
- `results/` 保存每次运行产生的视频与报告。

## 安装

PyTorch 参考实现：

```bash
pip install -e .
```

NVIDIA GPU Triton 解码：

```bash
pip install -e ".[cuda]"
```

Self-Forcing 推理环境：

```bash
pip install -e ".[cuda,self-forcing]"
```

Causal Forcing 推理环境：

```bash
pip install -e ".[cuda,causal-forcing]"
```

如果还需要 Self-Forcing 训练工具，再安装：

```bash
pip install -e ".[cuda,self-forcing,self-forcing-train]"
```

Self-Forcing 与 LongCat 对 `diffusers`、`transformers` 和 `av` 的版本要求不同，
建议分别使用独立的虚拟环境。

仓库提供 `configs/int2.json` 和 `configs/int4.json`。每一种
`k_history_length`、`v_history_length` 与位宽组合都需要一套对应的拟合参数。
`residual_mode=symmetric` 且位宽为 2 时使用的是 `-1, 0, 1` 三个量化值；默认的
`residual_mode=asymmetric` 才使用 2 bit 的四个编码值。

## Self-Forcing

Self-Forcing 的 BF16、clean trace 和 TempoKV 使用同一份本地模型代码：
`experiments/Self-Forcing/inference.py`。不需要 `MODEL_REPO`、`TRACE_RUNNER`
或 QVG 仓库。`CHECKPOINT` 与 `WAN_MODEL_DIR` 只负责指定本机模型权重。

### 1. 采集并合并 clean trace

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Self-Forcing/collect_trace.sh
```

默认输出：

```text
data/self-forcing/trace/raw/
data/self-forcing/trace/output/
data/self-forcing/trace/clean_trace.npz
```

已有 raw trace 时只执行合并：

```bash
python experiments/Self-Forcing/tempokv_adapter/prepare_trace.py \
  data/self-forcing/trace/raw \
  data/self-forcing/trace/clean_trace.npz
```

### 2. 拟合一套参数

```bash
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=2 \
V_HISTORY_LENGTH=2 \
bash scripts/Self-Forcing/fit_params.sh
```

默认输出：

```text
data/self-forcing/params/kh2-vh2-int2/k_temporal.npz
data/self-forcing/params/kh2-vh2-int2/v_cross_kv.npz
```

K/V 使用不同位宽时，目录名会明确写出两者，例如
`kh2-vh2-k2-v4`，避免 `K2/V2` 与 `K2/V4` 覆盖到同一个目录。

### 3. 使用 TempoKV 生成

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

脚本会按配置自动选择 `data/self-forcing/params/<参数方案>/`。也可以用
`PARAM_DIR` 指定其他参数目录。

### 4. 生成 BF16 对照

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
FRAMES=717 \
bash scripts/Self-Forcing/run_bf16.sh
```

BF16 不读取 TempoKV 配置或参数。BF16 与 TempoKV 默认都生成 717 帧，保证
耗时、显存和视频质量可以直接对照。717 帧对应 60 个生成 block，因此两种
模式的进度条都显示 `60/60`。两种生成方式都会在 `results/self_forcing/`
中写入视频、耗时和显存报告。

完整变量说明见 [Self-Forcing 使用说明](scripts/Self-Forcing/README.md)。

## Causal Forcing

Causal Forcing 的模型、pipeline、VAE、scheduler 和 TempoKV 接入代码都位于
`experiments/Causal-Forcing/`，入口位于 `scripts/Causal-Forcing/`。不需要 clone
外部 `thu-ml/Causal-Forcing`，不需要外部 `PYTHONPATH`，权重由 `MODEL_ROOT` 与
`CHECKPOINT` 指定。

下载权重：

```bash
bash scripts/Causal-Forcing/download_models.sh
```

BF16 基线：

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Causal-Forcing/run_bf16.sh
```

采集 clean trace 并拟合 INT4：

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Causal-Forcing/collect_trace.sh

CONFIG=configs/causal_forcing_int4.json \
bash scripts/Causal-Forcing/fit_params.sh
```

TempoKV 生成：

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/causal_forcing_int4.json \
bash scripts/Causal-Forcing/run_tempokv.sh
```

`GENERATION_MODE=framewise` 切换到 frame-wise 模型，trace、参数和结果分开存放。
Causal Forcing 与 Self-Forcing 的 30 层 / 12 头 / D=128 / stride=1560 完全相同，
但 checkpoint 与 cache 分布不同，**参数不能互相复用**，必须各自采 trace 各自拟合。

### 已接入范围

已接入的是**标准 Causal Forcing 推理**（chunk-wise 与 frame-wise）的 BF16、
trace、INT4/INT2 全流程。上游的 Rolling Forcing / long-video 扩展
（`long_video/`）**尚未接入**，rolling eviction 与 sink 保留也未实现：cache 已经
保存绝对帧位置和独立的 predictor span，为这些功能留了口子，但在实现之前
`sink_size > 0` 与局部注意力窗口同时使用会直接报错，而不是给出与上游不同的结果。

完整变量说明见 [Causal Forcing 使用说明](scripts/Causal-Forcing/README.md)。

## LongCat

LongCat 的模型、VAE、scheduler、Context Parallel 和 TempoKV 接入代码都位于
`experiments/LongCat/`，最终入口位于 `scripts/LongCat/`。不需要
`LONGCAT_REPO`、外部 runner 或预先生成的 init 视频。模型权重仍由
`CHECKPOINT_DIR` 指定。

安装 LongCat 依赖：

```bash
pip install -e ".[cuda,longcat]"
```

采集并合并 clean trace：

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
bash scripts/LongCat/collect_trace.sh
```

BF16：

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
bash scripts/LongCat/run_bf16.sh
```

TempoKV：

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
CONFIG=configs/longcat_int4.json \
K_PARAMS=data/longcat/params/kh2-vh2-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh2-int4/v_cross_kv.npz \
bash scripts/LongCat/run_tempokv.sh
```

默认会从 prompt 生成首段，再执行 LongCat video continuation。只有希望从已有
视频继续生成时才需要设置 `INIT_VIDEO_PATH`。

## 评测

```bash
pip install -e ".[metrics]"

python scripts/evaluation/evaluate_quality.py \
  /data/reference_videos \
  results/self_forcing/videos \
  --device cuda \
  --output results/self_forcing/quality.json
```
