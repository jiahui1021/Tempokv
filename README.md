# TempoKV

TempoKV 使用时间预测与 CrossKV 预测压缩视频生成模型的 KV cache。仓库包含
PyTorch 参考实现、可选 Triton 算子、参数拟合工具，以及 Self-Forcing 和
LongCat 的模型接入。

## 目录

```text
Tempokv/
├── experiments/
│   ├── Self-Forcing/          Self-Forcing 模型与推理代码
│   └── LongCat/               LongCat 模型接入代码
├── src/tempokv/               TempoKV 核心算法、参数格式与 Triton 算子
├── scripts/
│   ├── Self-Forcing/          Self-Forcing 采集、拟合与生成入口
│   ├── LongCat/               LongCat 采集、拟合与生成入口
│   ├── common/                多模型共用的参数拟合入口
│   ├── evaluation/            视频质量评测
│   └── tools/                 通用分析与流水线工具
├── data/
│   ├── self-forcing/
│   │   ├── trace/             raw trace 与合并后的 clean trace
│   │   └── params/            按参数方案分组的拟合参数
│   └── longcat/
│       ├── trace/
│       └── params/
├── configs/                   TempoKV 配置
├── results/                   生成视频、报告与评测结果
└── tests/
```

目录职责固定：

- `experiments/` 只保存模型侧代码。
- `src/tempokv/` 只保存 TempoKV 自己的算法和算子。
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

仓库提供 `configs/int2.json` 和 `configs/int4.json`。每一种
`k_history_length`、`v_history_length` 与位宽组合都需要一套对应的拟合参数。

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
FRAMES=81 \
bash scripts/Self-Forcing/run_bf16.sh
```

BF16 不读取 TempoKV 配置或参数。两种生成方式都会在 `results/` 中写入视频、
耗时和显存报告。

完整变量说明见 [Self-Forcing 使用说明](scripts/Self-Forcing/README.md)。

## LongCat

LongCat 的模型接入代码位于 `experiments/LongCat/longcat/`，最终入口位于
`scripts/LongCat/`。现有 LongCat 接入仍使用 `LONGCAT_REPO` 指向已安装的
LongCat 模型运行环境；它与 Self-Forcing 的本地运行方式互不混用。

## 评测

```bash
pip install -e ".[metrics]"

python scripts/evaluation/evaluate_quality.py \
  /data/reference_videos \
  results/self-forcing/videos \
  --device cuda \
  --output results/self-forcing/quality.json
```
