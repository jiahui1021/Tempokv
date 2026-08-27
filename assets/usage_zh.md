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

## 历史帧数敏感性实验

三个模型都有，入口同名：

```bash
bash scripts/Self-Forcing/sweep_history.sh
bash scripts/Causal-Forcing/sweep_history.sh      
bash scripts/LongCat/sweep_history.sh
```

先跑小规模确认能跑通：

```bash
MAX_SAMPLES=64 MAX_FRAMES=8 LAYERS=0,15,29 \
bash scripts/Self-Forcing/sweep_history.sh
```

每个取值报告三项：

| 指标 | 含义 |
|---|---|
| 重建误差 | `||rebuilt - clean||² / ||clean||²`，模型真正读到的 cache 的闭环误差 |
| Attention 输出误差 | 同一份重建过 causal softmax 后的相对输出误差 |
| 预测项占比 | `1 - ||clean - prediction||² / ||clean||²`，其补集是量化器要花比特编的残差 |

K 扫 `k_history_length`（1 到 8），V 在 CrossKV 项之上再扫 `v_history_length`
（0 到 5），并额外拆出 V 预测里 CrossKV 项与 history 项各自的能量占比——两项相关，
所以并排报告而不是把总量切开。

每个 vh 取值还会做一次 **CrossKV 项的留一消融**：把 V 预测里的 CrossKV 项或
history 项从**已拟合好的**模型里去掉，看残差各涨多少。

```text
vh       both  cross only  history only  cross buys  history buys
 0    0.01493     0.01493       0.99618     0.98125       0.00000
 1    0.00489     0.48567       0.12061     0.11573       0.48078
```

`cross buys` = 去掉 CrossKV 项后的残差减去两项都在时的残差，`history buys` 同理。
这是**留一消融**而不是"不带该项重新拟合"，所以说的是该项在这个预测器里贡献了多少，
不是"不用它训出来的模型能做到多好"。两项相关，所以两个残差不构成对总残差的分解。

**每个取值都会重新拟合一次。** 非 shared 参数的历史长度在拟合时就固定了，拿一套
参数改 N 得到的不是那个 N 的模型。位宽由 `CONFIG` 固定，扫描过程中不变。

三个模型共用 `scripts/common/sweep_history.sh`，变量说明见该文件头部。

拟合和扫描都默认在 GPU 上跑（`FIT_DEVICE=auto`）。没有 CUDA 时回退 CPU 并告警——
CPU 拟合会同时降到 float64，比看上去慢得多；`FIT_DEVICE=cuda` 让它直接报错。

## 评测

```bash
pip install -e ".[metrics]"

python scripts/evaluation/evaluate_quality.py \
  results/self_forcing/bf16/videos \
  results/self_forcing/legacy-kh2-vh1-int2-layershared/videos \
  --device cuda \
  --output results/self_forcing/evaluation/legacy-kh2-vh1-int2-layershared/quality.json
```

该命令在一次配对解码中计算 PSNR、SSIM、LPIPS 和 2048 维 Inception frame FID。
默认评测全部帧；`--frame-stride`、`--max-frames` 和 `--fid-batch-size` 可用于
smoke。这里的 FID 以同 prompt、同 seed 的 BF16 视频为 reference，只表示量化前后
的帧分布偏移，不是相对真实视频数据集的标准 FID。

一键评测一个候选方法：

```bash
BASELINE_DIR=results/self_forcing/bf16/videos \
CANDIDATE_DIR=results/self_forcing/legacy-kh2-vh1-int2-layershared/videos \
CANDIDATE_NAME=legacy-kh2-vh1-int2-layershared \
OUTPUT_DIR=results/self_forcing/evaluation \
DEVICE=cuda \
bash scripts/evaluation/evaluate_all.sh
```

脚本先生成候选相对 BF16 的 `quality.json`，再运行 BF16 和候选的 VBench。BF16
VBench 按视频清单与配置指纹复用；切换候选时不会重复计算，输入变化或 `FORCE=1`
时才会重算。可选的 `PROMPT_FILE` 必须是覆盖当前视频文件名的 JSON 映射。
