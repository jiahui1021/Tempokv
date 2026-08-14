# TempoKV

TempoKV 使用时间预测与 CrossKV 预测压缩视频生成模型的 KV cache。仓库包含 PyTorch
参考实现、可选 Triton 算子、参数拟合工具，以及 Self-Forcing、Causal Forcing 和
LongCat 的模型接入。

## 目录

```text
Tempokv/
├── experiments/               模型侧代码（Self-Forcing / Causal-Forcing / LongCat）
├── src/tempokv/               核心算法、参数格式与 Triton 算子
│   └── runtime/               因果视频模型共用的 KV cache、RoPE、流式编码与指标
├── scripts/                   各模型入口、共用拟合入口、评测与分析工具
├── configs/                   codec 配置
├── data/                      可复用的 trace 与拟合参数
└── results/                   生成视频、报告与评测结果
```

## 安装

| 用途 | 命令 |
|---|---|
| PyTorch 参考实现 | `pip install -e .` |
| NVIDIA GPU Triton 解码 | `pip install -e ".[cuda]"` |
| Self-Forcing 推理 | `pip install -e ".[cuda,self-forcing]"` |
| Causal Forcing 推理 | `pip install -e ".[cuda,causal-forcing]"` |
| LongCat 推理 | `pip install -e ".[cuda,longcat]"` |
| 视频质量评测 | `pip install -e ".[metrics]"` |

Self-Forcing 与 LongCat 对 `diffusers`、`transformers` 和 `av` 的版本要求不同，
建议分别使用独立的虚拟环境。

## 快速开始

每个模型都是四步：采 trace → 拟合参数 → TempoKV 生成 → BF16 对照。完整变量说明见
各自的使用说明，下面只列入口。

| 模型 | 入口 | 使用说明 |
|---|---|---|
| Self-Forcing | `scripts/Self-Forcing/{collect_trace,fit_params,run_tempokv,run_bf16}.sh` | [说明](scripts/Self-Forcing/README.md) |
| Causal Forcing | `scripts/Causal-Forcing/{download_models,collect_trace,fit_params,run_tempokv,run_bf16}.sh` | [说明](scripts/Causal-Forcing/README.md) |
| LongCat | `scripts/LongCat/{collect_trace,fit_params,run_tempokv,run_bf16}.sh` | [说明](scripts/LongCat/README.md) |

```bash
# Self-Forcing 为例
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Self-Forcing/collect_trace.sh

CONFIG=configs/legacy.json bash scripts/Self-Forcing/fit_params.sh

CHECKPOINT=... WAN_MODEL_DIR=... PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/legacy.json FRAMES=717 \
bash scripts/Self-Forcing/run_tempokv.sh
```

BF16 对照不读 TempoKV 配置或参数，默认与 TempoKV 生成相同帧数，所以耗时、显存和
视频质量可以直接对照。两者都会在 `results/<模型>/` 写入视频与报告。

## 配置

`configs/` 有两个 family 配置和一个显式兼容 preset。通常只需选择 family，
位宽、历史长度与 V predictor 用命令行 override：

```text
configs/
├── native.json    native 拟合（Causal Forcing、LongCat）
├── legacy.json    legacy/shared-K 拟合 family
└── legacy_kh3_vt1_int4.json  KH3 + temporal-V 的完整兼容 preset
```

K 预测器的粒度是最显眼的差别：native 每层每头一套系数（`[L,P,P,H,D]`），legacy 把
全部层和头池化成一套通道系数（`[P,P,D]`，参数量少 360 倍）。但 `fit_mode` 不只管
K——它同时改 V 预测的累加顺序、编解码的 BF16 截断点，并强制 `strict_triton`，是一个
**全局的逐位兼容模式**（目标是复现 QVG 参考实现），不是单纯的 K 预测器开关。它与
`k_predictor_mode` 互为必需（legacy ⟺ shared K），所以是一个不可分的 bit，也是文件
唯一需要编码的东西；legacy 另外钉死拟合控制参数（`iterations=2`、
`temporal_ridge=1.0`、`cross_ridge=0.1`、`max_gain=1.0`），脚本按家族自动取默认值。
其余字段取 [`CodecConfig`](src/tempokv/config.py) 的默认值（`anchor_bits=4`、
`group_size=64`、`residual_mode=asymmetric`、
`scale_dtype`/`reconstruction_dtype=bfloat16`、`stride=1560`）。

### K 家族 × V 预测器：四种组合都可用

两个轴独立，`V_PREDICTOR` 在任一家族下都成立：

| | `V_PREDICTOR=cross_kv` | `V_PREDICTOR=temporal` |
|---|---|---|
| `native.json` | V = Cross(K) + 对角 V history | V 只读自己的 V history，`[L,P,P,H,D]` |
| `legacy.json` | Cross(K) 仍是 per-layer per-head | V history 池化成 `[P,P,D]` |

注意 legacy 池化的是 **K 与 temporal V**；它的 CrossKV V 仍然是 per-layer per-head。
`v_predictor_mode=temporal` 需要 `v_history_length >= 1`——没有历史就没有可预测的东西。
`legacy.json` 本身不钉死 V family：Self-Forcing wrapper 保持 CrossKV 默认，Causal-Forcing
wrapper 为 attention-refinement 协议选择 temporal V。两者都会把解析后的 family 显式传给
fitter/runtime，并编码进参数目录，不能共享参数。

### 命令行 override

| 变量 | 作用 | 适用 |
|---|---|---|
| `BITS` | 同时设 K/V 位宽（2/4/8） | 全部 |
| `K_BITS` / `V_BITS` | 分别设，用于混合精度 | 全部 |
| `KH` / `VH` | K/V 历史长度（长名 `K_HISTORY_LENGTH` / `V_HISTORY_LENGTH`） | 全部 |
| `V_PREDICTOR` | V 预测器：`cross_kv` 或 `temporal` | 全部 |
| `FIT_HISTORY_SOURCE` | predictor 拟合历史：`reconstructed`（默认）或 `clean` | 全部 |
| `CACHE_STRIDE` | 本次运行的 local cache stride（CP>1 时用） | LongCat |
| `TORCH_DECODER=1` | 用 PyTorch 参考解码器而不是 Triton | Self-Forcing |
| `ATTENTION_STEPS` | 每个 K/V phase 的最大 Adam 步数；留空不精调 | Self-Forcing / Causal Forcing |
| `ATTENTION_OPTIMIZER` | `adam`（默认）或历史 `sign_gradient` 对照 | Self-Forcing / Causal Forcing |
| `ATTENTION_RECONSTRUCTION_WEIGHT` / `ATTENTION_OUTPUT_WEIGHT` | 归一化 objective 权重，和必须为 1 | Self-Forcing / Causal Forcing |
| `ATTENTION_REFINE_TENSORS` | `key`、`value` 或 `key_value` | Self-Forcing / Causal Forcing |
| `ATTENTION_LR` / `ATTENTION_PATIENCE` | Adam 学习率与早停 patience | Self-Forcing / Causal Forcing |
| `ATTENTION_LAYER_CHUNK` | refinement 每批处理的层数 | Self-Forcing / Causal Forcing |

冲突会直接报错，不会悄悄取一个：`BITS` 与 `K_BITS`/`V_BITS` 不一致，或 `KH` 与
`K_HISTORY_LENGTH` 不一致，都会失败。

```bash
CONFIG=configs/legacy.json BITS=2 bash scripts/Self-Forcing/fit_params.sh
CONFIG=configs/legacy.json V_PREDICTOR=temporal bash scripts/Self-Forcing/fit_params.sh
CONFIG=configs/legacy.json FIT_HISTORY_SOURCE=clean bash scripts/Self-Forcing/fit_params.sh
CONFIG=configs/native.json CACHE_STRIDE=780 bash scripts/LongCat/fit_params.sh
```

两个 family 配置的历史长度都是 `kh2-vh1`。Self-Forcing 复现参考实现的 KH3 要显式写
`KH=3`；Causal Forcing 可按实验选择 native CrossKV 或 legacy temporal V，LongCat 使用
native family。attention refinement 支持 legacy/shared 与 native/layered temporal K；V 可选
temporal 或 `CrossKV + V-history`。
启用 refinement 时，同一次拟合还会保存 `k_temporal_ls.npz` 和对应的 `v_*_ls.npz`，供
LS-only / refined 严格消融；常规文件名仍指向 refinement 后参数。
最终方法的训练 loss 在 K/V 两个 phase 中分别使用当前 tensor relative SSE 与 attention-output
relative SSE 的归一化 `0.5/0.5` 加权，并由 Adam 多步优化、guardrail 选取最佳检查点。
消融接口可在同一 rollout/trace/参数布局下切换 reconstruction-only `1/0`、历史
`sign_gradient`、K-only、V-only 与 K+V；候选选择和 guardrail 使用与训练相同的非零权重，
避免 reconstruction-only 被隐藏的 attention guardrail 污染。

### 参数目录名

每一种位宽、历史长度、predictor 结构与 stride 的组合都需要各自的一套拟合参数，
目录名把它们全部编码进去，互不覆盖：

```text
legacy-kh2-vh1-int4        Self-Forcing legacy.json 默认（CrossKV V）
legacy-kh2-vt1-int4        Causal legacy.json 默认（temporal V）
legacy-kh3-vh1-int4        + KH=3（Self-Forcing 参考实现）
legacy-kh2-vh1-int2        + BITS=2
kh2-vh1-int4               native.json 默认
kh2-vh1-int4-s780          + CACHE_STRIDE=780
kh2-vh1-k2-v4              K/V 位宽不同
kh2-vh1-int4-fitclean       clean-history fitting 消融
```

`FIT_HISTORY_SOURCE` 只改变 predictor 拟合时可见的历史；target 始终是 clean current K/V，
生成时仍只读取量化闭环重建 cache。默认 `reconstructed` 保留历史目录名，`clean` 增加
`-fitclean` 后缀，避免覆盖。`fit_report.json` 会记录请求迭代数、实际重建轮数和 K/V
feature source。

参数文件记录拟合时的 codec signature，加载时校验，用错位宽、用错 stride 或用另一次
拟合的半套参数都会直接报错，而不是产出错误结果。

### 两条约束

**legacy 家族只接受固定的 legacy event trace 协议。**
`scripts/common/fit_params.py` 会校验 `trace_layout=legacy_event_bpfhd_v1`、
`trace_stride=1560` 和固定的 30 层张量形状。Self-Forcing 与 Causal-Forcing 的 trace bridge
都能产出该协议，但 checkpoint 与 cache 分布不同，参数仍不能跨 backend 复用。LongCat
目前只提供 native trace 入口。

**stride 随模型而变，不是常量。** 它是模型的 frame token 数（Self-Forcing /
Causal Forcing 是 1560；LongCat 在 CP>1 时按 CP 度数切成 780 / 390），只是
`CodecConfig` 的默认值恰好取了 1560。stride 进 codec signature，所以每个 stride
要各自拟合；设错不会静默跑，bridge 会把 codec 的 stride 对着**从实际 cache 形状
算出来的** `sequence // temporal_units` 比。

## 各模型注意事项

**Self-Forcing** 的 BF16、trace 与 TempoKV 共用 `experiments/Self-Forcing/inference.py`，
不需要外部仓库。`FRAMES=717` 对应 60 个生成 block，两种模式的进度条都显示 `60/60`。

**Causal Forcing** 用 `GENERATION_MODE=framewise` 切到 frame-wise 模型
（`num_frame_per_block=1`），四步命令完全不变，trace、参数和结果分别存放在
`framewise/` 子目录，两种模式互不覆盖。两者共用同一套 cache 与 runtime，没有单独的
算法分支。

它与 Self-Forcing 的 30 层 / 12 头 / D=128 / stride=1560 完全相同，但 checkpoint、
训练方式和 cache 分布不同，**参数不能互相复用**，必须各自采 trace 各自拟合；
`run_tempokv.sh` 会检查 `coordinate_system` 与 codec signature，用错参数在加载模型
之前就失败。生成结束写出 `tempokv_report.json`，其中 `encode_key_calls`、
`encode_value_calls`、`decode_tensor_calls`、`finalized_frames` 任一为 0 都会让本次
运行直接失败，避免"视频正常生成但其实没走 TempoKV"。

已接入的是**标准 Causal Forcing 推理**（chunk-wise 与 frame-wise）的 BF16、trace、
INT4/INT2 全流程。上游的 Rolling Forcing / long-video 扩展（`long_video/`）**尚未
接入**，rolling eviction 与 sink 保留也未实现：cache 已经保存绝对帧位置和独立的
predictor span，为这些功能留了口子，但在实现之前 `sink_size > 0` 与局部注意力窗口
同时使用会直接报错，而不是给出与上游不同的结果。

**LongCat** 的模型、VAE、scheduler、Context Parallel 与 TempoKV 接入都在
`experiments/LongCat/`，权重由 `CHECKPOINT_DIR` 指定，不需要外部 runner 或预生成的
init 视频。

## 历史帧数敏感性实验

扫 K 与 V 的历史帧数，报告重建误差、sampled temporal-track attention proxy 误差和
预测项占比：

```bash
bash scripts/Self-Forcing/sweep_history.sh          # 三个模型各有一个同名入口
MAX_SAMPLES=64 MAX_FRAMES=8 LAYERS=0,15,29 \
bash scripts/Self-Forcing/sweep_history.sh          # 先跑小规模确认能跑通
```

K 扫 `k_history_length`（1 到 8），V 在 CrossKV 项之上再扫 `v_history_length`
（0 到 5）。每个 vh 取值还会做一次 **CrossKV 项的留一消融**：把 V 预测里的 CrossKV
项或 history 项从**已拟合好的**模型里去掉，看残差各涨多少。

```text
vh       both  cross only  history only  cross buys  history buys
 0    0.01493     0.01493       0.99618     0.98125       0.00000
 1    0.00489     0.48567       0.12061     0.11573       0.48078
```

`cross buys` = 去掉 CrossKV 项后的残差减去两项都在时的残差，`history buys` 同理。
这是留一消融而不是"不带该项重新拟合"，所以说的是该项在这个预测器里贡献了多少；两项
相关，所以两个残差不构成对总残差的分解。

**每个取值都会重新拟合一次**——非 shared 参数的历史长度在拟合时就固定了，拿一套
参数改 N 得到的不是那个 N 的模型。拟合默认在 GPU 上跑（`FIT_DEVICE=auto`）；没有
CUDA 时会回退 CPU 并告警，而 CPU 拟合同时降到 float64，比看上去慢得多，
`FIT_DEVICE=cuda` 可以让它直接报错。

三个模型共用 `scripts/common/sweep_history.sh`，变量说明见该文件头部。单点分析
（不扫描、直接读已有参数）用 `scripts/tools/analyze_predictor.py`。

## 评测

```bash
python scripts/evaluation/evaluate_quality.py \
  results/self_forcing/bf16/videos \
  results/self_forcing/legacy-kh2-vh1-int4/videos \
  --device cuda \
  --output results/self_forcing/evaluation/quality.json
```

在同一次配对解码中计算 pooled RGB PSNR、平均 SSIM、平均 LPIPS 和 2048 维 Inception
frame FID。这里的 FID 以同 prompt、同 seed 的 BF16 视频为 reference，衡量量化后的帧
分布偏移，**不是**生成视频相对真实数据集的标准 FID，也不替代时间一致性评测。

一键跑 quality/FID 加 BF16 与候选各自的 VBench：

```bash
BASELINE_DIR=results/self_forcing/bf16/videos \
CANDIDATE_DIR=results/self_forcing/legacy-kh2-vh1-int4/videos \
CANDIDATE_NAME=legacy-kh2-vh1-int4 \
OUTPUT_DIR=results/self_forcing/evaluation \
DEVICE=cuda \
bash scripts/evaluation/evaluate_all.sh
```

脚本按视频清单与 VBench 配置指纹复用已完成的 BF16 VBench，输入或配置变化自动重算，
`FORCE=1` 强制重算。其余可选变量见
[`scripts/evaluation/evaluate_all.sh`](scripts/evaluation/evaluate_all.sh) 头部。

## 运行时加速路径

TempoKV 的 CUDA 路径不是“先用 PyTorch 做 predictor，再单独反量化”。Triton decoder
在同一个 persistent kernel 中完成低比特 unpack、scale/zero-point 反量化、temporal 或
CrossKV prediction、residual add 和 BF16 截断；temporal 路径一次 launch 最多推进 16 个
unit，CrossKV 路径因寄存器压力上限为 8 个。输出可直接写入 backend 需要的 BSHD/BHSD
cache-read slice，避免中间 layout tensor 和额外 copy。

编码侧会对一个 span 的 CrossKV projection 做一次 batched GEMM，并让 K/V encoder 复用
输入 buffer 完成闭环重建。解码侧直接读取 reconstructed-K source view，不为每个 packed
span 复制完整 K。每个 rank 使用可跨运行复用的 Triton compilation cache，避免把重复编译
计入稳态 latency。

这些是**已实现的机制**，不是端到端加速结论。当前每次读取 packed span 仍会重新构造完整
BF16 K/V read buffer，K 解码后再单独执行 RoPE；尚未实现 decoded-span cache、decode+RoPE
融合、decode+attention 融合或 compressed-domain attention。因此论文和报告必须分别给出
encode/decode、attention、完整 generation 的实测结果，不能从 KV bytes 或 kernel
microbenchmark 推导端到端加速。

## 性能分解

加速工作从测量开始。给任意一次生成加上 `PROFILE=1`，会用 CUDA event 记录每一段的
耗时，写到 `$OUTPUT_DIR/profile/rank-N.json`；记录 event 不做同步，插桩后的运行保持
原来的 launch 模式。

```bash
PROFILE=1 CONFIG=configs/legacy.json ... bash scripts/Self-Forcing/run_tempokv.sh
python scripts/tools/profile_tempokv_runtime.py results/self_forcing/legacy-kh2-vh1-int4
```

报告按调用树给出每段的 total、self（减去子段）与占 generation 的比例，分段包括
`attention_kernel`、`rope`、`cache_read_k/v`、`decode_k/v`、`encode_k/v` 与
`vae_decode`；同一段在不同父段下分别统计。`--baseline` 对比两次运行。BF16 对照同样
支持 `PROFILE=1`，它没有 codec，所以 `decode_*` / `encode_*` 会列为未插桩而不是 0。

kernel 级别的扫描用 `scripts/tools/benchmark_tempokv_kernels.py`：在生产 shape 下
构造真实 packed span，逐配置对比 Torch decoder 确认数值正确，再报告 median/p90 延迟、
峰值显存与寄存器/spill。microbenchmark 变快不等于端到端变快，结论要用完整 717 帧生成
确认。
