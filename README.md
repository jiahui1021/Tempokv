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
`k_history_length`、`v_history_length` 与位宽组合都需要一套对应的拟合参数，
参数目录名会带上这三者（如 `kh2-vh1-int4`），互不覆盖。历史帧数默认
`kh=2`、`vh=1`；命令行用 `KH=` / `VH=` 覆盖即可（等价于长名
`K_HISTORY_LENGTH` / `V_HISTORY_LENGTH`，两个都设且不同会报错）。
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
KH=2 \
VH=1 \
bash scripts/Self-Forcing/fit_params.sh
```

默认输出：

```text
data/self-forcing/params/kh2-vh1-int2/k_temporal.npz
data/self-forcing/params/kh2-vh1-int2/v_cross_kv.npz
```

K/V 使用不同位宽时，目录名会明确写出两者，例如
`kh2-vh1-k2-v4`，避免 `K2/V2` 与 `K2/V4` 覆盖到同一个目录。

### 3. 使用 TempoKV 生成

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/int2.json \
KH=2 \
VH=1 \
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

安装 Causal Forcing 依赖：

```bash
pip install -e ".[cuda,causal-forcing]"
```

### 0. 下载权重

```bash
bash scripts/Causal-Forcing/download_models.sh
```

默认下载到 `checkpoints/causal_forcing/`，包含 Wan2.1-T2V-1.3B base 与
chunk-wise / frame-wise 两个 Causal Forcing 权重。权重不进入 Git。

### 1. 采集并合并 clean trace

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Causal-Forcing/collect_trace.sh
```

默认输出：

```text
data/causal-forcing/trace/chunkwise/raw/
data/causal-forcing/trace/chunkwise/output/
data/causal-forcing/trace/chunkwise/clean_trace.npz
```

只采 clean/context refresh 留下的 pre-RoPE normalized K 与 raw V，4 次 noisy
denoising step 的 current K/V 不入 trace。

### 2. 拟合一套参数

```bash
CONFIG=configs/causal_forcing_int4.json \
bash scripts/Causal-Forcing/fit_params.sh
```

默认输出：

```text
data/causal-forcing/params/chunkwise/kh2-vh1-int4/k_temporal.npz
data/causal-forcing/params/chunkwise/kh2-vh1-int4/v_cross_kv.npz
```

INT2 只需要换 `configs/causal_forcing_int2.json`，目录名会自动带上位宽。

### 3. 使用 TempoKV 生成

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/causal_forcing_int4.json \
bash scripts/Causal-Forcing/run_tempokv.sh
```

脚本按配置自动选择 `data/causal-forcing/params/<模式>/<参数方案>/`，也可以用
`PARAM_DIR` 指定其他目录。生成结束会写出 `tempokv_report.json`；`encode_key_calls`、
`encode_value_calls`、`decode_tensor_calls`、`finalized_frames` 任何一个为 0 都会
让本次运行直接失败，避免"视频正常生成但其实没走 TempoKV"。

### 4. 生成 BF16 对照

```bash
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Causal-Forcing/run_bf16.sh
```

BF16 不读取 TempoKV 配置或参数。两种模式默认都生成 81 帧，耗时、显存和视频质量
可以直接对照。

### chunk-wise 与 frame-wise

`GENERATION_MODE=framewise` 切换到 frame-wise 模型（`num_frame_per_block=1`），
上面四步的命令完全不变，trace、参数和结果分别存放在 `framewise/` 子目录下，
两种模式互不覆盖。两者共用同一套 cache 与 runtime，没有单独的算法分支。

Causal Forcing 与 Self-Forcing 的 30 层 / 12 头 / D=128 / stride=1560 完全相同，
但 checkpoint、训练方式和 cache 分布不同，**参数不能互相复用**，必须各自采 trace
各自拟合。`run_tempokv.sh` 会检查参数文件的 `coordinate_system` 与 codec
signature，用错参数会在加载模型之前直接失败。

### 已接入范围

已接入的是**标准 Causal Forcing 推理**（chunk-wise 与 frame-wise）的 BF16、
trace、INT4/INT2 全流程。上游的 Rolling Forcing / long-video 扩展
（`long_video/`）**尚未接入**，rolling eviction 与 sink 保留也未实现：cache 已经
保存绝对帧位置和独立的 predictor span，为这些功能留了口子，但在实现之前
`sink_size > 0` 与局部注意力窗口同时使用会直接报错，而不是给出与上游不同的结果。

完整变量说明见 [Causal Forcing 使用说明](scripts/Causal-Forcing/README.md)。

## LongCat

LongCat 的模型、VAE、scheduler、Context Parallel 和 TempoKV 接入代码都位于
`experiments/LongCat/`，入口位于 `scripts/LongCat/`。不需要 `LONGCAT_REPO`、
外部 runner 或预先生成的 init 视频。模型权重由 `CHECKPOINT_DIR` 指定。

安装 LongCat 依赖：

```bash
pip install -e ".[cuda,longcat]"
```

### 1. 采集并合并 clean trace

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/collect_trace.sh
```

默认输出：

```text
data/longcat/trace/raw/
data/longcat/trace/output/
data/longcat/trace/clean_trace.npz
```

合并阶段从 trace 文件里的 `trace_stride` 读取空间步长，并校验所有 Context
Parallel rank 的 metadata 一致，因此 `CONTEXT_PARALLEL_SIZE=1/2/4`
（对应 stride `1560/780/390`）都不需要改合并命令。

### 2. 拟合一套参数

```bash
CONFIG=configs/longcat_int4.json \
bash scripts/LongCat/fit_params.sh
```

默认输出：

```text
data/longcat/params/kh2-vh1-int4/k_temporal.npz
data/longcat/params/kh2-vh1-int4/v_cross_kv.npz
```

换配置时目录名会自动带上位宽，INT4 与 INT2 不会互相覆盖：

```bash
CONFIG=configs/longcat_int2.json \
bash scripts/LongCat/fit_params.sh    # -> data/longcat/params/kh2-vh1-int2
```

三个模型用同一套目录命名规则。参数文件会记录拟合时的 codec signature，加载时
校验，用错位宽或用另一次拟合的半套参数都会直接报错而不是产出错误结果。

### 3. 使用 TempoKV 生成

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
CONFIG=configs/longcat_int4.json \
K_PARAMS=data/longcat/params/kh2-vh1-int4/k_temporal.npz \
V_PARAMS=data/longcat/params/kh2-vh1-int4/v_cross_kv.npz \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/run_tempokv.sh
```

默认输出到 `results/longcat/tempokv/`。

### 4. 生成 BF16 对照

```bash
CHECKPOINT_DIR=/models/LongCat-Video \
PROMPT="a cinematic mountain road" \
bash scripts/LongCat/run_bf16.sh
```

默认输出到 `results/longcat/bf16/`。BF16 不读取 TempoKV 配置或参数。

三种模式都先从 prompt 生成首段，再执行 video continuation。只有希望从已有视频
继续生成时才设置 `INIT_VIDEO_PATH`。常用可选变量：`NUM_SEGMENTS`、
`NUM_FRAMES`、`NUM_COND_FRAMES`、`NUM_INFERENCE_STEPS`、
`CONTEXT_PARALLEL_SIZE`、`NUM_GPUS`、`SEED`、`OUTPUT_DIR`、`DRY_RUN`。

完整变量说明见 [LongCat 使用说明](scripts/LongCat/README.md)。

## 历史帧数敏感性实验

三个模型都可以扫 K 与 V 的历史帧数，报告重建误差、attention 输出误差和预测项占比：

```bash
bash scripts/Self-Forcing/sweep_history.sh
bash scripts/Causal-Forcing/sweep_history.sh      # GENERATION_MODE 分开存放
bash scripts/LongCat/sweep_history.sh
```

先跑小规模确认能跑通：

```bash
MAX_SAMPLES=64 MAX_FRAMES=8 LAYERS=0,15,29 \
bash scripts/Self-Forcing/sweep_history.sh
```

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
参数改 N 得到的不是那个 N 的模型，只能靠重拟合来比较。位宽、group_size 和 stride
由 `CONFIG` 固定，扫描过程中不变。

三个模型共用 `scripts/common/sweep_history.sh`，变量说明见该文件头部。单点分析
（不扫描、直接读已有参数）用 `scripts/tools/analyze_predictor.py`。

拟合默认在 GPU 上跑（`FIT_DEVICE=auto`）。没有 CUDA 时会回退 CPU 并打出告警——
CPU 拟合同时降到 float64，比看上去慢得多；`FIT_DEVICE=cuda` 可以让它直接报错
而不是悄悄回退。

## 评测

```bash
pip install -e ".[metrics]"

python scripts/evaluation/evaluate_quality.py \
  results/self_forcing/bf16/videos \
  results/self_forcing/kh2-vh1-int2/videos \
  --device cuda \
  --output results/self_forcing/evaluation/kh2-vh1-int2/quality.json
```

`evaluate_quality.py` 在同一次配对解码中计算 pooled RGB PSNR、平均 SSIM、
平均 LPIPS 和 2048 维 Inception frame FID。默认使用全部帧；smoke 可以用
`--frame-stride`、`--max-frames` 和 `--fid-batch-size` 控制工作量。这里的 FID
以同 prompt、同 seed 的 BF16 视频为 reference，衡量量化后帧分布偏移，不是生成视频
相对真实数据集的标准 FID，也不替代时间一致性评测。

对一个候选方法一键运行 quality/FID 和 BF16、候选各自的 VBench：

```bash
BASELINE_DIR=results/self_forcing/bf16/videos \
CANDIDATE_DIR=results/self_forcing/kh2-vh1-int2/videos \
CANDIDATE_NAME=kh2-vh1-int2 \
OUTPUT_DIR=results/self_forcing/evaluation \
DEVICE=cuda \
bash scripts/evaluation/evaluate_all.sh
```

再次评测其他方法时只需更换 `CANDIDATE_DIR` 与 `CANDIDATE_NAME`。脚本根据视频
清单和 VBench 配置指纹复用已经完成的 BF16 VBench；输入或配置变化会自动重算，
`FORCE=1` 可以强制重算。可选变量还有 `FRAME_STRIDE`、`MAX_FRAMES`、
`FID_BATCH_SIZE`、`LPIPS_NETWORK`、`VBENCH_MODE`、`VBENCH_DIMENSIONS`、
`VBENCH_FULL_INFO`、`VBENCH_LOCAL_CHECKPOINTS` 和 `VBENCH_READ_FRAME`。
`PROMPT_FILE` 如需使用，必须是覆盖当前视频文件名的 JSON 映射。
