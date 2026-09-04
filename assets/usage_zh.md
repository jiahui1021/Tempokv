# TempoKV 中文使用指南

TempoKV 的代码按职责分为三层：

- `experiments/`：Self-Forcing、Causal Forcing、LongCat 的模型侧代码。
- `src/tempokv/`：TempoKV 核心算法、参数与 Triton 算子；`src/tempokv/runtime/`
  是 Self-Forcing 与 Causal Forcing 共用的因果 KV cache 运行时。
- `scripts/<模型>/`：采集、拟合和生成入口。

可复用数据统一存入 `data/<模型>/`，生成视频和报告存入 `results/`。

三个模型的流程完全同构，都是四步：

```text
1 采集并合并 clean trace  →  2 拟合参数  →  3 TempoKV 生成  →  4 BF16 对照
```

参数目录名由 codec 配置推导（如 `legacy-kh2-vh1-int4-layershared`），换位宽或换历史长度都不会互相
覆盖；参数文件记录拟合时的 codec signature，加载时校验，用错参数会直接报错而不是
产出错误结果。

旧的 native 或非 layer-shared 参数不能与当前配置混用，需要按新的 legacy 配置重新拟合。

历史帧数默认 `kh=2`、`vh=1`，写在各模型的 codec 配置里。命令行可以用 `KH=` / `VH=`
覆盖（等价于长名 `K_HISTORY_LENGTH` / `V_HISTORY_LENGTH`），目录名会跟着变。

## 安装

三个模型的 `diffusers`、`transformers`、`av` 版本要求不同，建议分别用独立虚拟环境。

```bash
pip install -e ".[cuda,self-forcing]"      # Self-Forcing
pip install -e ".[cuda,causal-forcing]"    # Causal Forcing
pip install -e ".[cuda,longcat]"           # LongCat
```

Self-Forcing 训练工具另需：

```bash
pip install -e ".[cuda,self-forcing,self-forcing-train]"
```

## Self-Forcing

### 1. 采集并合并 clean trace

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Self-Forcing/collect_trace.sh
```

输出 `data/self-forcing/trace/{raw/,clean_trace.npz}`。

### 2. 拟合参数

```bash
CONFIG=configs/legacy.json \
KH=2 \
VH=1 \
bash scripts/Self-Forcing/fit_params.sh
```

输出 `data/self-forcing/params/legacy-kh2-vh1-int2-layershared/`。

### 3. TempoKV 生成

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/legacy.json \
KH=2 \
VH=1 \
FRAMES=717 \
bash scripts/Self-Forcing/run_tempokv.sh
```

### 4. BF16 对照

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
FRAMES=717 \
bash scripts/Self-Forcing/run_bf16.sh
```


详细说明见 [scripts/Self-Forcing/README.md](../scripts/Self-Forcing/README.md)。

## Causal Forcing


### 0. 下载权重

```bash
bash scripts/Causal-Forcing/download_models.sh
```

默认下载到 `checkpoints/causal_forcing/`，含 Wan2.1-T2V-1.3B base 与chunk-wise / frame-wise 两个权重

### 1. 采集并合并 clean trace

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Causal-Forcing/collect_trace.sh
```

输出 `data/causal-forcing/trace/chunkwise_legacy/{raw/,clean_trace.npz}`。只采
clean/context refresh 留下的 pre-RoPE normalized K 与 raw V，4 次 noisy
denoising step 不入 trace。


### 2. 拟合参数

```bash
CONFIG=configs/legacy.json \
bash scripts/Causal-Forcing/fit_params.sh
```

输出 `data/causal-forcing/params/chunkwise/legacy-kh2-vh1-int4-layershared/`。INT2 加 `BITS=2` 即可


### 3. TempoKV 生成

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/legacy.json \
bash scripts/Causal-Forcing/run_tempokv.sh
```


### 4. BF16 对照

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Causal-Forcing/run_bf16.sh
```


## LongCat

### 1. 采集并合并 clean trace

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPTS=assets/fit_prompts.txt \
bash scripts/LongCat/collect_trace.sh
```

### 2. 拟合参数

```bash
CONFIG=configs/legacy.json \
bash scripts/LongCat/fit_params.sh
```

输出 `data/longcat/params/legacy-kh2-vh1-int4-layershared/`。加 `BITS=2` 会写到
`legacy-kh2-vh1-int2-layershared`，不会覆盖 INT4。

### 3. TempoKV 生成

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
CONFIG=configs/legacy.json \
K_PARAMS=data/longcat/params/legacy-kh2-vh1-int4-layershared/k_temporal.npz \
V_PARAMS=data/longcat/params/legacy-kh2-vh1-int4-layershared/v_cross_kv.npz \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/run_tempokv.sh
```

输出到 `results/longcat/tempokv/`。

### 4. BF16 对照

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPTS=Tempokv/assets/moviegen128.txt \
bash scripts/LongCat/run_bf16.sh
```

输出到 `results/longcat/bf16/`。

