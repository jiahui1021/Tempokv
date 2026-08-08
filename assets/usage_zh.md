# TempoKV 中文使用指南

TempoKV 负责对注意力层的 K/V cache 做参数拟合、量化和重建。仓库提供 codec、
PyTorch 参考解码器、可选的 Triton CUDA 解码器。完整英文文档见
[README](../README.md)，Self-Forcing 详细说明见
[self-forcing/README.md](../self-forcing/README.md)。

## 1. 安装与配置

CPU 或 PyTorch 参考路径：

```bash
pip install -e .
```

使用 NVIDIA GPU 上的 Triton 解码器：

```bash
pip install -e ".[cuda]"
```

仓库提供两份配置，两者默认都用最近两帧重建 K 预测当前 K，并用当前重建 K
和最近两帧重建 V 共同预测 V：

- `configs/int2.json`：K/V 残差为 INT2，首个 anchor unit 保持 INT4。
- `configs/int4.json`：K/V 残差和首个 anchor unit 都为 INT4。

`k_history_length` 用于选择 K 历史长度，可设为 `1` 到 `8`。
`v_history_length` 用于选择 V 历史长度：

`2` 使用前两帧，`1` 只使用前一帧，`0` 保留原来的纯 CrossKV 方式。

不同的历史长度配置需要各自对应的拟合参数。由于 GPU 拟合只需约 2 秒，
推荐做法是：每个要运行的 `k_history_length` / `v_history_length` 组合，分别拟合一套专属参数。

## Self-Forcing 完整生成流程

### 第一步：采集 + 合并 clean trace（一次性）

采集需要跑模型推理，约 10-15 分钟。`collect_trace.sh` 采集并合并 clean
trace，产出拟合输入 `self_forcing_trace.npz`；之后生成任意视频都复用同一个
`self_forcing_trace.npz`，无需重新采集。

```bash
TRACE_RUNNER=/path/to/self-forcing-trace-runner.sh \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
WORK_DIR=runs/self_forcing_prepare \
bash self-forcing/collect_trace.sh
```

已有 raw trace 时，也可以不用 `collect_trace.sh`，直接调用外部采集入口再
手动合并：

```bash
SELF_FORCING_CHECKPOINT_PATH=/models/self_forcing_dmd.pt \
FORCING_WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS_PATH=assets/fit_prompts.txt \
CROSS_KV_TRACE_DIR=runs/self_forcing_prepare/raw_trace/traces \
OUTPUT_FOLDER=runs/self_forcing_prepare/raw_trace/output \
bash /path/to/self-forcing-trace-runner.sh

python self-forcing/prepare_trace.py \
  runs/self_forcing_prepare/raw_trace/traces \
  parameters/self_forcing_trace.npz
```

### 第二步：拟合参数

每次生成前先拟合参数（GPU 只需约 1.5 秒）。每个
`k_history_length` / `v_history_length` 配置拟合一套专属参数，用
`K_HISTORY_LENGTH` 和 `V_HISTORY_LENGTH` 显式指定，用不同 `OUTPUT_DIR`
区分，且生成时的 `K_HISTORY_LENGTH` / `V_HISTORY_LENGTH` 必须与拟合值一致

```bash
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=3 \
V_HISTORY_LENGTH=1 \
OUTPUT_DIR=parameters/self_forcing/kh2-vh2-int2 \
bash self-forcing/fit_params.sh
```

### 第三步：生成视频

生成时 `PARAM_DIR` 指向第二步的 `OUTPUT_DIR`，`K_HISTORY_LENGTH` /
`V_HISTORY_LENGTH` 必须与拟合值一致：

```bash
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/int2.json \
PARAM_DIR=parameters/self_forcing/kh2-vh2-int2 \
K_HISTORY_LENGTH=3 \
V_HISTORY_LENGTH=1 \
FRAMES=717 \
bash self-forcing/generate.sh
```

`K_HISTORY_LENGTH` 和 `V_HISTORY_LENGTH` 明确指定本次生成使用的历史帧数；
省略时读取 `CONFIG`。示例省略了默认的 `MODE=tempokv` 和输出目录。

设置 `MODE=bf16` 可以运行不量化的对照生成；该模式不需要 `CONFIG` 和
`PARAM_DIR`。`pipeline_report.json` 中保存总耗时、每个 GPU 的耗时和最高显存占用。
、