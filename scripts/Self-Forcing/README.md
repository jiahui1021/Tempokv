# Self-Forcing + TempoKV

## 文件职责

```text
experiments/Self-Forcing/
├── Self-Forcing 模型、pipeline、Wan 模块与本地 KV cache
└── tempokv_adapter/
    ├── bridge.py           TempoKV cache 接入
    ├── trace_bridge.py     clean pre-RoPE K/V 采集
    ├── worker.py           torchrun 进程入口
    ├── generate.py         生成检查与报告
    └── prepare_trace.py    raw trace 合并

scripts/Self-Forcing/
├── README.md
├── _common.sh          公共路径与参数检查
├── collect_trace.sh    采集并合并 clean trace
├── fit_params.sh       拟合 TempoKV 参数
├── generate.sh         两种生成模式的公共入口
├── run_bf16.sh         BF16 生成入口
└── run_tempokv.sh      TempoKV 生成入口
```

BF16、trace、TempoKV 最终都执行
`experiments/Self-Forcing/inference.py`。三条路径使用相同的 checkpoint、Wan
模型、attention 和 prompt 读取逻辑。

## 运行环境

推理依赖可以单独安装：

```bash
pip install -e ".[cuda,self-forcing]"
```

训练或数据工具依赖另装：

```bash
pip install -e ".[cuda,self-forcing,self-forcing-train]"
```

Self-Forcing 和 LongCat 的 `diffusers`、`transformers`、`av` 版本不同，建议
分别建立虚拟环境。

## 数据目录

```text
data/self-forcing/
├── trace/
│   ├── raw/                         逐层采集文件
│   ├── output/                      采集过程输出
│   └── clean_trace.npz              合并后的拟合输入
└── params/
    ├── legacy-kh2-vh1-int2/
    │   ├── k_temporal.npz
    │   └── v_cross_kv.npz
    ├── legacy-kh2-vh1-k2-v4/      K/V 使用不同位宽时
    ├── legacy-kh3-vt1-int4/       temporal-V 兼容 preset
    └── <其他参数方案>/
```

`results/self_forcing/` 只保存视频与运行报告，不保存 trace 或拟合参数。

## 1. 采集 clean trace

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Self-Forcing/collect_trace.sh
```

默认使用固定的 legacy event trace 协议：`assets/fit_prompts.txt` 中的 12 条
prompt、每条 51 个 latent 帧、每帧 8 个 token。每条 prompt 保存两个独立的
24 帧记录并分别重置历史，末尾 3 帧不进入拟合。保存全部 30 层和 12 个
attention head。K 来自 `norm_k` 输出，坐标为 normalized、pre-RoPE；V 来自同
一次 clean attention 计算。同时记录模型实际用于注意力的旋转后 Q/K——直接取运行时
算好的张量，不重算 RoPE——供拟合阶段的 attention 目标使用。采集过程不启用 TempoKV。

常用覆盖项：

```bash
TRACE_DIR=data/self-forcing/trace
TRACE_FILE=data/self-forcing/trace/clean_trace.npz
LEGACY_PROTOCOL=1
TRACE_FRAMES=51
TRACE_TOKENS_PER_FRAME=8
NUM_GPUS=1
BASE_SEED=0
OVERWRITE=1
```

已有 raw trace 时直接合并：

```bash
python experiments/Self-Forcing/tempokv_adapter/prepare_trace.py \
  data/self-forcing/trace/raw \
  data/self-forcing/trace/clean_trace.npz \
  --legacy-protocol
```

合并文件包含：

```text
keys                 [30,24,1,8,24,12,128]   pre-RoPE K
values               [30,24,1,8,24,12,128]   V
q_post_rope          [30,24,1,8,24,12,128]   旋转后的 Q（attention 目标用）
k_post_rope          [30,24,1,8,24,12,128]   旋转后的 K（attention 目标用）
layer_ids            [0..29]
stride               1560
coordinate_system    pre_rope_normalized
cross_source_kind    normalized
```

## 2. 拟合参数

```bash
CONFIG=configs/legacy.json \
KH=2 \
VH=1 \
bash scripts/Self-Forcing/fit_params.sh
```

脚本根据配置自动生成参数方案名，例如 `legacy-kh2-vh1-int2`。默认读取
`data/self-forcing/trace/clean_trace.npz`，并写入：

```text
data/self-forcing/params/legacy-kh2-vh1-int2/k_temporal.npz
data/self-forcing/params/legacy-kh2-vh1-int2/v_cross_kv.npz
```

可用 `TRACE_FILE`、`PARAM_NAME` 或 `OUTPUT_DIR` 覆盖默认位置。

如果配置中的 K/V 位宽不同，方案名会使用 `legacy-kh2-vh1-k2-v4` 这样的格式。shared
参数会保留配置中的最大 history length，并可在运行时选择不超过该上限的历史长度。
拟合目录同时写入 `fit_report.json`，其中包含 `codec_config_sha256`，用于确认
参数来自哪一份 codec 配置。

legacy temporal V 可在基础 fitting 后启用 sampled attention proxy refinement：

```bash
CONFIG=configs/legacy.json V_PREDICTOR=temporal \
PARAM_NAME=legacy-temporal-adam-r05-a05-kv \
ATTENTION_STEPS=100 ATTENTION_LR=0.001 ATTENTION_PATIENCE=10 \
ATTENTION_LAYER_CHUNK=4 \
bash scripts/Self-Forcing/fit_params.sh
```

该目标使用 recorded post-RoPE Q/K，但每个采样空间位置独立、只沿帧轴做 SDPA；它不是完整
1560-token backend attention 的重放。`fit_report.json` 会记录 refinement 是否启用及 phase 结果。
同一入口还支持 `ATTENTION_OPTIMIZER=sign_gradient`、归一化
`ATTENTION_RECONSTRUCTION_WEIGHT` / `ATTENTION_OUTPUT_WEIGHT`，以及
`ATTENTION_REFINE_TENSORS=key|value|key_value`；这些只用于同协议消融，最终方法默认仍是
Adam、`0.5/0.5`、K+V。

## 3. TempoKV 生成

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

TempoKV 生成至少需要 27 个 latent 帧，约等于 105 个最终视频帧。`FRAMES`
必须满足 `(FRAMES + 3) % 12 == 0`，避免脚本静默增减视频帧。默认的 717 帧
对应 180 个 latent 帧、60 个 3-frame block，所以 BF16 与 TempoKV 的生成进度
都会显示 `60/60`。每个 block 内部仍执行原模型的 4 次去噪和 1 次 clean cache
更新，进度条不会把内部 4 次去噪重复计数。

## 4. BF16 生成

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
FRAMES=717 \
bash scripts/Self-Forcing/run_bf16.sh
```

BF16 不需要 `CONFIG`、`PARAM_DIR`、K 参数或 V 参数。它与 TempoKV 使用同一
份内置 Self-Forcing 推理代码。两种模式默认都生成 717 帧，用于耗时、显存
和视频质量对照。

## 常用生成变量

```text
OUTPUT_DIR       输出目录
NUM_SAMPLES      每条 prompt 的样本数
SEED             随机种子
START_INDEX      起始 prompt 序号
END_INDEX        结束 prompt 序号
NUM_GPUS         GPU 数量
SPAN_FRAMES      每个独立 anchor span 的 latent 帧数；不设置时保持原来的 24
SKIP_EXISTING    跳过已有视频，0 或 1（默认 1）
BENCHMARK        要求本次完整生成，不能跳过已有视频，0 或 1（默认 0）
DRY_RUN          只检查并打印命令，0 或 1
```

做 latency benchmark 时设置 `BENCHMARK=1 SKIP_EXISTING=0`。报告中的
`generation_seconds` 是 prompt batch 的耗时；`NUM_SAMPLES>1` 时另记录每样本
耗时和吞吐量，不会把多个视频的累计保存时间当作单个样本延迟。

默认视频文件名为 `<prompt序号>_<样本序号>.mp4`，当前一次运行的明细写入
`videos/summary.json`。默认结果目录是 `results/self_forcing/<参数方案>/`；BF16
使用 `results/self_forcing/bf16/`。

仓库内没有 `MODEL_REPO` 或 `TRACE_RUNNER` 参数。模型代码固定来自
`experiments/Self-Forcing/`；`CHECKPOINT` 与 `WAN_MODEL_DIR` 仅指定权重位置。

`FIT_DEVICE` 默认 `auto`：有 CUDA 就用 GPU，没有则回退 CPU 并在 stderr 打出告警——
CPU 拟合同时会把方程降到 float64，比看上去慢得多，所以回退不会静默发生。
`FIT_DEVICE=cuda` 让没有 GPU 时直接报错而不是回退。拟合报告里记录实际使用的设备。

`K_HISTORY_LENGTH` / `V_HISTORY_LENGTH` 可以简写成 `KH` / `VH`，效果相同；
两个名字都设置且取值不同会直接报错。不设置时读 `CONFIG` 里的值（默认 `kh=2`、`vh=1`）。

位宽是命令行 override，不是单独的配置文件：`BITS=2` 同时设 K/V，
`K_BITS` / `V_BITS` 分别设（混合精度用）。`BITS` 与某一侧设成不同值会直接报错。
位宽会进入参数目录名，所以 INT2 与 INT4 的参数互不覆盖。

`V_PREDICTOR=cross_kv|temporal` 切换 V 预测器，也不需要单独的配置文件；它会同时
决定参数目录名（`vh1` 与 `vt1`）和参数文件名（`v_cross_kv.npz` 与 `v_temporal.npz`），
两者不会互相覆盖。native 与 legacy family 都能拟合 temporal V；但 sampled attention
refinement 当前要求 legacy/shared temporal K，V 可选 temporal 或 `CrossKV + V-history`；
其他 K predictor shape 会在读 trace 前报错。

`TORCH_DECODER=1` 用 PyTorch 参考解码器代替 Triton，用来做数值对照；它同时关掉
强制 Triton 的检查，所以不需要再配 `ALLOW_TRITON_FALLBACK`。两者共用同一套拟合
参数——解码器是运行时开关，不进 codec signature。

## 历史帧数敏感性实验

扫 K 与 V 的历史帧数，看更长的历史是否值它的参数开销：

```bash
bash scripts/Self-Forcing/sweep_history.sh
```

先跑一遍小规模确认能跑通：

```bash
MAX_SAMPLES=64 MAX_FRAMES=8 LAYERS=0,15,29 \
bash scripts/Self-Forcing/sweep_history.sh
```

每个取值报告三项：

| 指标 | 含义 |
|---|---|
| 重建误差 | `\|\|rebuilt - clean\|\|² / \|\|clean\|\|²`，模型真正读到的 cache 的闭环误差 |
| Attention proxy 误差 | sampled token 各自沿帧轴做 recorded-Q SDPA 的相对输出误差；不是完整 1560-token backend attention |
| 预测项占比 | `1 - \|\|clean - prediction\|\|² / \|\|clean\|\|²`，其补集就是量化器要花比特编的残差 |

K 扫 `k_history_length`（1 到 8），V 在 CrossKV 项之上再扫 `v_history_length`
（0 到 5），并额外拆出 V 预测里 CrossKV 项与 history 项各自的能量占比——两项相关，
所以是并排报告而不是把总量切开。

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

**每个取值都会重新拟合一次。** 非 shared 参数的历史长度在拟合时就固定了，
拿一套参数改 N 得到的不是那个 N 的模型，所以扫描只能靠重拟合。位宽、group_size
和 stride 由 `CONFIG` 固定，整个扫描过程中不变。

报告默认写到 `results/self_forcing/history_sweep.json`。

变量说明见 [scripts/common/sweep_history.sh](../common/sweep_history.sh)，三个模型
共用同一份实现。单点分析（不扫描、直接读已有参数）用
`scripts/tools/analyze_predictor.py`。
