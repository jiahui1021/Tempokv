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

## 数据目录

```text
data/self-forcing/
├── trace/
│   ├── raw/                         逐层采集文件
│   ├── output/                      采集过程输出
│   └── clean_trace.npz              合并后的拟合输入
└── params/
    ├── kh2-vh2-int2/
    │   ├── k_temporal.npz
    │   └── v_cross_kv.npz
    └── <其他参数方案>/
```

`results/` 只保存视频与运行报告，不保存 trace 或拟合参数。

## 1. 采集 clean trace

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
bash scripts/Self-Forcing/collect_trace.sh
```

默认采集 21 个 latent 帧，每帧抽取 8 个 token，保存全部 30 层和 12 个
attention head。K 来自 `norm_k` 输出，坐标为 normalized、pre-RoPE；V 来自同
一次 clean attention 计算。采集过程不启用 TempoKV。

常用覆盖项：

```bash
TRACE_DIR=data/self-forcing/trace
TRACE_FILE=data/self-forcing/trace/clean_trace.npz
TRACE_FRAMES=21
TRACE_TOKENS_PER_FRAME=8
NUM_GPUS=1
BASE_SEED=0
OVERWRITE=1
```

已有 raw trace 时直接合并：

```bash
python experiments/Self-Forcing/tempokv_adapter/prepare_trace.py \
  data/self-forcing/trace/raw \
  data/self-forcing/trace/clean_trace.npz
```

合并文件包含：

```text
keys                 [30,samples,T,12,128]
values               [30,samples,T,12,128]
layer_ids            [0..29]
stride               1560
coordinate_system    pre_rope_normalized
cross_source_kind    normalized
```

## 2. 拟合参数

```bash
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=2 \
V_HISTORY_LENGTH=2 \
bash scripts/Self-Forcing/fit_params.sh
```

脚本根据配置自动生成参数方案名，例如 `kh2-vh2-int2`。默认读取
`data/self-forcing/trace/clean_trace.npz`，并写入：

```text
data/self-forcing/params/kh2-vh2-int2/k_temporal.npz
data/self-forcing/params/kh2-vh2-int2/v_cross_kv.npz
```

可用 `TRACE_FILE`、`PARAM_NAME` 或 `OUTPUT_DIR` 覆盖默认位置。

## 3. TempoKV 生成

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

TempoKV 生成至少需要 27 个 latent 帧，约等于 105 个最终视频帧。`FRAMES`
必须能换算为 3 的倍数个 latent 帧。生成参数必须与拟合参数的 history、stride
和坐标一致。

## 4. BF16 生成

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
FRAMES=81 \
bash scripts/Self-Forcing/run_bf16.sh
```

BF16 不需要 `CONFIG`、`PARAM_DIR`、K 参数或 V 参数。它与 TempoKV 使用同一
份内置 Self-Forcing 推理代码，用于耗时、显存和视频质量对照。

## 常用生成变量

```text
OUTPUT_DIR       输出目录
NUM_SAMPLES      每条 prompt 的样本数
SEED             随机种子
START_INDEX      起始 prompt 序号
END_INDEX        结束 prompt 序号
NUM_GPUS         GPU 数量
SKIP_EXISTING    跳过已有视频，0 或 1
DRY_RUN          只检查并打印命令，0 或 1
```

仓库内没有 `MODEL_REPO` 或 `TRACE_RUNNER` 参数。模型代码固定来自
`experiments/Self-Forcing/`；`CHECKPOINT` 与 `WAN_MODEL_DIR` 仅指定权重位置。
