# Causal Forcing 使用说明

全部参数用环境变量传入。模型代码在
[experiments/Causal-Forcing/](../../experiments/Causal-Forcing/)，不依赖外部
`thu-ml/Causal-Forcing` 仓库。

```text
scripts/Causal-Forcing/
├── download_models.sh   下载 Wan2.1 base 与 Causal Forcing 权重
├── collect_trace.sh     采集 clean K/V trace 并合并成拟合输入
├── fit_params.sh        调用通用拟合器产出 k_temporal.npz / v_cross_kv.npz
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
CONFIG=configs/causal_forcing_int4.json \
bash scripts/Causal-Forcing/fit_params.sh

# TempoKV 生成
CHECKPOINT=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
MODEL_ROOT=checkpoints/causal_forcing/wan2.1-t2v-1.3b \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/causal_forcing_int4.json \
bash scripts/Causal-Forcing/run_tempokv.sh
```

INT2 只需要换 `configs/causal_forcing_int2.json` 与对应的参数目录。

## 公共变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `CHECKPOINT` | Causal Forcing 权重 | 必填 |
| `MODEL_ROOT` | Wan2.1-T2V-1.3B 目录 | 必填 |
| `PROMPTS` | prompt 列表 | 必填 |
| `CONFIG` | codec 配置 JSON | TempoKV 模式必填 |
| `GENERATION_MODE` | `chunkwise` 或 `framewise` | `chunkwise` |
| `FRAMES` | 最终视频帧数 | `81` |
| `SPAN_FRAMES` | 每个 TempoKV span 的 latent 帧数 | `24` |
| `LOCAL_ATTN_SIZE` | 局部注意力窗口 | `-1` |
| `NUM_GPUS` | torchrun GPU 数 | `1` |
| `DRY_RUN` | 只打印命令不执行 | `0` |

`GENERATION_MODE` 会同时决定 `num_frame_per_block`（chunkwise=3，framewise=1）、
默认的模型配置文件，以及 trace / 参数 / 结果的子目录，两种模式互不覆盖。

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
├── trace/<模式>/{raw,output,clean_trace.npz}
└── params/<模式>/<参数方案>/{k_temporal.npz,v_cross_kv.npz}

results/causal_forcing/<模式>/<量化标签>/
├── videos/
├── metrics/
├── tempokv_report.json
└── pipeline_report.json
```

## 报告

TempoKV 生成结束后 `tempokv_report.json` 至少包含 `encode_key_calls`、
`encode_value_calls`、`decode_tensor_calls`、`finalized_frames`、
`ephemeral_writes`、resident / peak / total 三组 cache 字节数，以及 encode 与
decode 的分项耗时。任何一个计数为 0 都会让本次运行直接失败——视频正常生成但其实
没走 TempoKV 是必须暴露的错误。
