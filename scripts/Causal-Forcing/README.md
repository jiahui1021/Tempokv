# Causal Forcing 使用说明

全部参数用环境变量传入。模型代码在
[experiments/Causal-Forcing/](../../experiments/Causal-Forcing/)，不依赖外部
`thu-ml/Causal-Forcing` 仓库。

```text
scripts/Causal-Forcing/
├── download_models.sh   下载 Wan2.1 base 与 Causal Forcing 权重
├── collect_trace.sh     采集 clean K/V trace 并合并成拟合输入
├── fit_params.sh        调用通用拟合器产出 k_temporal.npz 与 V 参数
│                       （v_cross_kv.npz 或 v_temporal.npz，取决于 v_predictor_mode）
├── run_bf16.sh          BF16 基线
├── run_tempokv.sh       TempoKV 生成
└── generate.sh          run_bf16.sh / run_tempokv.sh 的共同实现
```

## 完整流程

```bash
pip install -e ".[cuda,causal-forcing,metrics]"

bash scripts/Causal-Forcing/download_models.sh

# BF16 基线
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Causal-Forcing/run_bf16.sh

# 采集 clean trace
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Causal-Forcing/collect_trace.sh

# 拟合 INT4
bash scripts/Causal-Forcing/fit_params.sh

# TempoKV 生成
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
bash scripts/Causal-Forcing/run_tempokv.sh
```

INT2 使用同一个配置并设置 `BITS=2`；参数目录会自动切换到 INT2 identity。

## 公共变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `CHECKPOINT` | Causal Forcing 权重 | 必填 |
| `MODEL_ROOT` | Wan2.1-T2V-1.3B 目录 | 必填 |
| `PROMPTS` | prompt 列表 | 必填 |
| `CONFIG` | codec 配置 JSON | `configs/legacy.json` |
| `GENERATION_MODE` | `chunkwise` 或 `framewise` | `chunkwise` |
| `FRAMES` | 最终视频帧数 | `81` |
| `SPAN_FRAMES` | 每个 TempoKV span 的 latent 帧数 | `12` |
| `LOCAL_ATTN_SIZE` | 局部注意力窗口 | `-1` |
| `NUM_GPUS` | torchrun GPU 数 | `1` |
| `DRY_RUN` | 只打印命令不执行 | `0` |

`GENERATION_MODE` 会同时决定 `num_frame_per_block`（chunkwise=3，framewise=1）、
默认的模型配置文件，以及 trace / 参数 / 结果的子目录，两种模式互不覆盖。

采 trace 统一使用 legacy family 并写入 `trace/<模式>_legacy/`。手工设置
`LEGACY_PROTOCOL` 与 legacy protocol 冲突会直接报错。

`FRAMES` 必须让视频帧精确对应完整的 latent block：chunkwise 要求
`(FRAMES + 3) % 12 == 0`，framewise 要求 `(FRAMES + 3) % 4 == 0`。

## 参数不能跨模型复用

Causal Forcing 与 Self-Forcing 的层数、头数、head dim 和 stride 完全相同，但
checkpoint、训练方式和 cache 分布不同，**必须重新采 trace 重新拟合**。
`run_tempokv.sh` 会检查参数文件里的 `coordinate_system` 与 codec signature，
用错参数会在加载模型之前直接失败。

## 目录

```text
data/causal-forcing/
├── trace/<模式>_legacy/{raw,output,clean_trace.npz}
└── params/<模式>/<参数方案>/{k_temporal.npz,v_cross_kv.npz 或 v_temporal.npz}

results/causal_forcing/<模式>/<量化标签>/
├── videos/
├── metrics/
├── tempokv_report.json
└── pipeline_report.json
```

## 报告

TempoKV 生成结束后 `tempokv_report.json` 至少包含 `encode_key_calls`、
`encode_value_calls`、`decode_tensor_calls`、`finalized_frames`、
`ephemeral_writes`，以及 `encode_ms` / `decode_ms` / `generation_ms`。任何一个
计数为 0 都会让本次运行直接失败——视频正常生成但其实没走 TempoKV 是必须暴露的错误。

报告只写真正测过的量。attention 单独耗时**不在报告里**：逐层计时需要 device
sync，会扭曲被测对象本身；需要时用 `generation_ms - encode_ms - decode_ms` 估算。
逐 prompt 的墙钟时间在同目录的 `videos/summary.json`。

内存有两套数字，不要混用：

- `current_compressed_cache_bytes` / `peak_compressed_cache_bytes` 只统计已经打包
  的 span，`compression_ratio` 用的是同一份 cache state 的压缩前后字节数；
- `resident_bf16_bytes` / `resident_total_bytes` / `peak_resident_total_bytes`
  还包含**已 finalize 但尚未攒满一个 span 的 BF16 帧**。这些帧真实占显存，所以
  报告显存峰值时要用 `peak_resident_total_bytes`，只报压缩后的字节会低估实际占用。

多 GPU 时所有 cache 字节都是**单卡口径**（`byte_scope: per_device`），取最差的一张
卡——那才是需要装进一块 GPU 的数字。只有调用计数和 `total_*_cache_bytes`
（整个运行的累计编码量）是跨卡求和的。

## SPAN_FRAMES 怎么选

一个 span 只有一个 anchor，span 越大压缩越好，但 span 必须**在生成结束前就攒满并
被后续 block 读回**，否则打包发生在最后一次 forward 之后，没有任何一次注意力读过
压缩 cache。因此要求：

```text
latent_frames >= SPAN_FRAMES + num_frame_per_block
```

`FRAMES=81` 对应 21 个 latent 帧，chunkwise 下 `SPAN_FRAMES` 最大 18，默认 12。
不满足时 `generate.py` 会在加载模型之前报错，而不是跑完再失败。

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
两者不会互相覆盖。两种 V 预测器都使用 legacy/shared temporal K；V 可选 temporal、
CrossKV only 或 `CrossKV + V-history`，每种结构都必须从同一 trace 独立拟合。

## Attention proxy refinement

`3340753` 起 Causal trace 同时保存采样位置的 post-RoPE Q/K。对 legacy/shared temporal K
以及 temporal 或 CrossKV V，可在基础 closed-loop least-squares fitting 后启用可选
refinement：

```bash
CONFIG=configs/legacy.json PARAM_NAME=legacy-temporal-adam-r05-a05-kv \
ATTENTION_STEPS=100 ATTENTION_LR=0.001 \
ATTENTION_PATIENCE=10 ATTENTION_LAYER_CHUNK=4 \
bash scripts/Causal-Forcing/fit_params.sh
```

启用 refinement 时，同一输出目录会同时保存 refinement 前的 `k_temporal_ls.npz` 与
`v_temporal_ls.npz` / `v_cross_kv_ls.npz`，常规文件名保存 refinement 后参数。
每个 K/V phase 用 Adam 最小化归一化的 `0.5 * own reconstruction + 0.5 * attention`
目标，按 phase 起点做非退化 guardrail，并用 patience 早停、保留最佳检查点。

objective/optimizer 消融使用同一入口，例如 reconstruction-only V phase 与历史 sign-gradient：

```bash
CONFIG=configs/legacy.json PARAM_NAME=legacy-winner-adam-r1-a0-kv ATTENTION_STEPS=100 \
ATTENTION_RECONSTRUCTION_WEIGHT=1 ATTENTION_OUTPUT_WEIGHT=0 \
ATTENTION_REFINE_TENSORS=value \
bash scripts/Causal-Forcing/fit_params.sh

CONFIG=configs/legacy.json PARAM_NAME=legacy-winner-sign-r05-a05-kv ATTENTION_STEPS=1 \
ATTENTION_OPTIMIZER=sign_gradient ATTENTION_SIGN_STEP_SIZES="0.001 0.01" \
bash scripts/Causal-Forcing/fit_params.sh
```

`PARAM_NAME` 必须为每个 optimizer/objective/phase 组合取独立值；默认 codec slug 不编码这些
拟合超参，已有参数默认拒绝覆盖，但不能用 `OVERWRITE=1` 把不同消融写进同一目录。

该目标把每个采样空间位置作为独立 record，只沿帧轴计算 recorded-Q SDPA；它不重放完整
1560-token attention 或 backend mask，因此报告中称为 `recorded-Q temporal-track attention
proxy`，不能等同完整模型 attention-output error。旧 KV-only trace 缺少 Q/K，需要重新采集。

## 历史帧数敏感性实验

扫 K 与 V 的历史帧数，报告重建误差、sampled temporal-track attention proxy 误差和
预测项占比：

```bash
MAX_SAMPLES=64 MAX_FRAMES=8 LAYERS=0,15,29 \
bash scripts/Causal-Forcing/sweep_history.sh    # 先小规模确认能跑通
bash scripts/Causal-Forcing/sweep_history.sh    # 全量
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
参数改 N 得到的不是那个 N 的模型。位宽、group_size 和 stride 由 `CONFIG` 固定。

报告默认写到 `results/causal_forcing/<模式>/history_sweep.json`。变量说明见
[scripts/common/sweep_history.sh](../common/sweep_history.sh)。
