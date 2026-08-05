# Self-Forcing 使用说明

这个目录只放 Self-Forcing 相关代码：

- `prepare.sh`：一次性采集 clean trace 并拟合参数。
- `generate.sh`：读取已有参数生成视频。
- `generate.py`：生成脚本使用的内部入口。
- `worker.py`：在 `torchrun` 进程中加载外部 Self-Forcing。
- `bridge.py`：把 Self-Forcing 的分段 K/V cache 接到 TempoKV。

Self-Forcing、Wan、VAE、tokenizer 和模型权重继续由外部模型仓库提供，不复制到 TempoKV。

## Trace 要求

拟合文件使用 `.npz`，包含：

```text
keys                 [30,...,T,12,128]
values               [30,...,T,12,128]
layer_ids            [0,1,...,29]
stride               1560
coordinate_system    pre_rope_normalized
cross_source_kind    normalized
```

生产 trace 原始层文件如果是 `[B,F,sampled_tokens,H,D]`，合并多层时必须转换为：

```text
[L,B,sampled_tokens,F,H,D]
```

这样 `F` 才是拟合使用的时间轴。

`prepare.sh` 会自动执行转换。已有 raw trace 时也可以单独转换：

```bash
python self-forcing/prepare_trace.py \
  /data/self_forcing_trace_root \
  /data/self_forcing_trace.npz
```

## 第一步：采集 trace 并拟合参数

这一阶段只需要运行一次。`TRACE_RUNNER` 指向外部 Self-Forcing 仓库中
能够采集 clean CrossKV trace 的 SH 入口；它需要支持本节列出的环境变量。

```bash
TRACE_RUNNER=/path/to/self-forcing-trace-runner.sh \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=/data/fit_prompts.txt \
WORK_DIR=runs/self_forcing_prepare \
CONFIG=configs/int2.json \
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
bash self-forcing/prepare.sh
```

输出为：

```text
runs/self_forcing_prepare/raw_trace/
runs/self_forcing_prepare/trace/self_forcing_trace.npz
runs/self_forcing_prepare/parameters/k_temporal.npz
runs/self_forcing_prepare/parameters/v_cross_kv.npz
runs/self_forcing_prepare/parameters/fit_report.json
```

采集固定使用未量化 K/V。默认生成 51 帧、每帧抽取 8 个 token，采集全部
30 层和全部 head。可以用环境变量调整：

```bash
START_INDEX=0
END_INDEX=8
BASE_SEED=0
TRACE_FRAMES=51
TRACE_TOKENS_PER_FRAME=8
HISTORY_LENGTH=3
ITERATIONS=2
TEMPORAL_RIDGE=0.1
CROSS_RIDGE=0.1
MAX_GAIN=1.0
```

量化位宽、anchor 位宽、分组大小和残差方式在 `CONFIG` 指向的 JSON 中调整。
这些配置改变后必须重新运行 `prepare.sh`，避免参数和生成配置不匹配。

采集器需要接受以下环境变量：`SELF_FORCING_CHECKPOINT_PATH`、
`FORCING_WAN_MODEL_DIR`、`CROSS_KV_TRACE_DIR`、`OUTPUT_FOLDER`、
`PROMPTS_PATH`、`START_INDEX`、`END_INDEX`、`BASE_SEED`、
`NUM_OUTPUT_FRAMES`、`TRACE_LAYERS`、`TRACE_HEADS`、
`TRACE_TOKENS_PER_FRAME`、`TRACE_CAPTURE_HIDDEN`、
`TRACE_CAPTURE_ATTENTION` 和 `TRACE_QUANT_TYPE`。

## 第二步：生成视频

该脚本只读取第一步生成的参数，不会采集 trace，也不会重新拟合：

```bash
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=/data/generation_prompts.txt \
PARAM_DIR=runs/self_forcing_prepare/parameters \
OUTPUT_DIR=runs/self_forcing_int2 \
CONFIG=configs/int2.json \
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
FRAMES=27 \
bash self-forcing/generate.sh
```

输出为：

```text
runs/self_forcing_int2/videos/*.mp4
runs/self_forcing_int2/pipeline_report.json
```

`FRAMES` 至少为 27。更短的运行不会触发 Self-Forcing 的旧 cache span 压缩，
入口会直接拒绝。

生成阶段默认强制使用 Triton 严格模式，底层算子不可用时会直接报错。功能调试时可以设置 `ALLOW_TRITON_FALLBACK=1`，此时使用 JSON 中的 `use_triton` 和 `strict_triton` 设置。

无 GPU 时，可以在上述任一组完整命令中额外设置 `DRY_RUN=1`，只检查输入和最终命令，不执行采集、拟合或生成。

## 视频质量评测

安装 PSNR、SSIM、LPIPS 依赖：

```bash
pip install -e ".[metrics]"
```

同时计算 PSNR、SSIM 和 LPIPS：

```bash
python scripts/evaluate_quality.py \
  /data/reference_videos \
  runs/self_forcing_int2/videos \
  --device cuda \
  --output runs/self_forcing_int2/quality.json
```

运行 VBench：

```bash
python scripts/evaluate_vbench.py \
  runs/self_forcing_int2/videos \
  --output-dir runs/self_forcing_int2/vbench \
  --prompt-file /data/video_prompts.json \
  --device cuda
```

`video_prompts.json` 是“视频文件名到 prompt”的 JSON 对象。VBench 使用官方包和模型文件；可以先加 `--dry-run --config-out prepared.json` 检查输入，不加载评测模型。
