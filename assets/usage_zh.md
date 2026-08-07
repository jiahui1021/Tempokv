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

仓库提供两份配置，两者默认都用最近三帧重建 K 预测当前 K，并用当前重建 K
和最近两帧重建 V 共同预测 V：

- `configs/int2.json`：K/V 残差为 INT2，首个 anchor unit 保持 INT4。
- `configs/int4.json`：K/V 残差和首个 anchor unit 都为 INT4。

`k_history_length` 用于选择 K 历史长度，可设为 `1` 到 `8`。
每一帧对应的预测系数由拟合过程自动学习。
`v_history_length` 用于选择 V 历史长度：
`2` 使用前两帧，`1` 只使用前一帧，`0` 保留原来的纯 CrossKV 方式。
修改任一历史长度后都需要重新拟合参数。

## Self-Forcing 完整生成流程
流程分成两个 SH 脚本。所有参数都用环境变量传入，第一个脚本运行一次，
第二个脚本可以更换 prompt 后重复运行。

### 第一步：采集 clean trace 和拟合参数（只需运行一次）

```bash
TRACE_RUNNER=/path/to/self-forcing-trace-runner.sh \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
WORK_DIR=runs/self_forcing_prepare \
CONFIG=configs/int2.json \
bash self-forcing/prepare.sh
```

### 第二步：读取参数生成视频（可更换 prompt 后重复运行）

```bash
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=3 \
V_HISTORY_LENGTH=2 \
FRAMES=717 \
bash self-forcing/generate.sh
```

`K_HISTORY_LENGTH` 和 `V_HISTORY_LENGTH` 明确指定本次生成使用的历史帧数；
省略时读取 `CONFIG`。示例省略了默认的 `MODE=tempokv`、`PARAM_DIR=parameters`
和输出目录。

设置 `MODE=bf16` 可以运行不量化的对照生成；该模式不需要 `CONFIG` 和
`PARAM_DIR`。`pipeline_report.json` 中保存总耗时、每个 GPU 的耗时和最高显存占用。

两个入口都会检查 `stride=1560`、30 层、12 heads、head dimension 128 和
pre-RoPE normalized K。生成长度还会检查是否真正触发 TempoKV。
