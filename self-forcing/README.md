# Self-Forcing 使用说明

- `prepare.sh`：一次性采集 clean trace 并拟合参数。
- `generate.sh`：读取已有参数生成视频。
- `generate.py`：生成脚本使用的内部入口。
- `worker.py`：在 `torchrun` 进程中加载外部 Self-Forcing。
- `bridge.py`：把 Self-Forcing 的分段 K/V cache 接到 TempoKV。


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

所有参数都用环境变量传入。完整示例：

```bash
TRACE_RUNNER=/path/to/self-forcing-trace-runner.sh \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
WORK_DIR=runs/self_forcing_prepare \
CONFIG=configs/int2.json \
bash self-forcing/prepare.sh
```

全部可配置变量见 `prepare.sh` 头部注释。

输出为：

```text
runs/self_forcing_prepare/raw_trace/          trace 采集中间产物
parameters/self_forcing_trace.npz
parameters/k_temporal.npz
parameters/v_cross_kv.npz
parameters/fit_report.json
```

`prepare.sh` 默认把采集的 trace 和拟合参数都写到独立的 `parameters/` 目录
（可用 `PARAM_DIR` 覆盖），`WORK_DIR` 只保留 trace 采集中间产物。

采集固定使用未量化 K/V。默认生成 51 帧、每帧抽取 8 个 token，采集全部
30 层和全部 head。可以用环境变量调整：

```bash
START_INDEX=0
END_INDEX=8
BASE_SEED=0
TRACE_FRAMES=51
TRACE_TOKENS_PER_FRAME=8
ITERATIONS=2
TEMPORAL_RIDGE=0.1
CROSS_RIDGE=0.1
MAX_GAIN=1.0
```

量化位宽、anchor 位宽、分组大小、残差方式和 K/V 历史长度都在 `CONFIG`
指向的 JSON 中调整。`k_history_length` 可设为 `1` 到 `8`，设为 `3` 时使用
前三帧重建 K，各帧系数由拟合过程自动学习；
`v_history_length=2` 使用前两帧重建 V，设为 `1` 时只使用前一帧，设为 `0`
时使用旧的纯 CrossKV 预测。
这些配置改变后必须重新运行 `prepare.sh`，避免参数和生成配置不匹配。

采集器需要接受以下环境变量：`SELF_FORCING_CHECKPOINT_PATH`、
`FORCING_WAN_MODEL_DIR`、`CROSS_KV_TRACE_DIR`、`OUTPUT_FOLDER`、
`PROMPTS_PATH`、`START_INDEX`、`END_INDEX`、`BASE_SEED`、
`NUM_OUTPUT_FRAMES`、`TRACE_LAYERS`、`TRACE_HEADS`、
`TRACE_TOKENS_PER_FRAME`、`TRACE_CAPTURE_HIDDEN`、
`TRACE_CAPTURE_ATTENTION` 和 `TRACE_QUANT_TYPE`。

## 第二步：生成视频

`MODE=tempokv` 读取第一步生成的参数，不会采集 trace，也不会重新拟合。
例如生成 717 帧（= 180 latent 帧）的完整视频：

```bash
MODE=tempokv \
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
PARAM_DIR=parameters \
OUTPUT_DIR=results/self_forcing \
CONFIG=configs/int2.json \
FRAMES=717 \
bash self-forcing/generate.sh
```

> 注意：`MODE=tempokv` 时 `FRAMES` 至少要给到 105 视频帧（27 latent 帧），
> 否则入口会直接拒绝。`MODE=bf16` 没有这个下限。

`MODE=bf16` 运行不量化的基线，不需要 `CONFIG` 和 `PARAM_DIR`：

```bash
MODE=bf16 \
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
OUTPUT_DIR=results/self_forcing_bf16 \
FRAMES=27 \
bash self-forcing/generate.sh
```

输出为：

```text
results/self_forcing/videos/*.mp4
results/self_forcing/pipeline_report.json
```

`pipeline_report.json` 记录端到端生成 latency（`generation_seconds`，从提交
生成到视频写完）、每个 GPU 的推理耗时、以及所有 GPU 中的最高已分配/保留
显存。TempoKV 与 BF16 使用同一套统计方式，可以直接比较。

```text
generation_seconds              端到端生成 latency（秒）
performance.max_worker_seconds
performance.peak_allocated_gib
performance.peak_reserved_gib
performance.decode_routes
performance.ranks[]
```

`FRAMES` 是**最终视频帧数**。脚本内部按 VAE 时间压缩率（1 latent = 4 视频帧）
换算成 latent 帧传给推理器，例如 `FRAMES=717` → 180 latent 帧。`FRAMES` 至少
需要 105 视频帧（27 latent 帧）才会触发 TempoKV 量化，否则入口直接拒绝。

`LOCAL_ATTN_SIZE` 默认是 -1（全局注意力）。TempoKV 接管 cache 压缩，不需要
局部滑窗；比较 TempoKV 与 BF16 时，两次运行必须使用相同的 `FRAMES`、
`LOCAL_ATTN_SIZE`、prompt、seed 和 GPU 数量。

K→V 与历史 V 的 128×128 预测矩阵乘在编码和 Triton 解码端都使用 TF32
（A100 上比 IEEE FP32 明显更快），与旧仓库
`S2_CROSS_MATMUL_PRECISION=tf32` 一致。历史 V 的计算已融合进同一个 Triton
解码 kernel；正式生成仍使用 `cross_kv_persistent_direct` 快速路径。
predictor 参数在 codec 内按 (device, layer) 缓存并全局共享，多个压缩 span
复用同一份参数，不再为每个 span 复制。

`decode_routes` 记录实际调用的 CUDA 路径。严格模式下，TempoKV 生成必须同时
出现 `temporal_persistent_direct` 和 `cross_kv_persistent_direct`；缺少任意一项
都会报错，防止在不知情的情况下使用慢速路径。

TempoKV 和 BF16 应使用不同的 `OUTPUT_DIR`，避免视频及报告互相覆盖。

生成阶段默认强制使用 Triton 严格模式，底层算子不可用时会直接报错。功能调试时可以设置 `ALLOW_TRITON_FALLBACK=1`，此时使用 JSON 中的 `use_triton` 和 `strict_triton` 设置。

### A100 快速验收

108 视频帧（27 latent 帧）、单条 prompt 的运行会触发首次量化，也足以
检查两条直接 CUDA 路径。使用与正式生成相同的模型、参数和配置，只把
输出目录换成独立目录：

```bash
MODE=tempokv \
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
PARAM_DIR=parameters \
OUTPUT_DIR=results/self_forcing_smoke \
CONFIG=configs/int2.json \
TRITON_CACHE=/tmp/tempokv-triton \
NUM_GPUS=1 \
FRAMES=108 \
START_INDEX=0 \
END_INDEX=1 \
bash self-forcing/generate.sh
```

查看耗时、峰值显存和实际 CUDA 路径：

```bash
jq '.performance | {
  max_worker_seconds,
  peak_allocated_gib,
  peak_reserved_gib,
  decode_routes
}' results/self_forcing_smoke/pipeline_report.json
```

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
  results/self_forcing/videos \
  --device cuda \
  --output results/self_forcing/quality.json
```

运行 VBench：

```bash
python scripts/evaluate_vbench.py \
  results/self_forcing/videos \
  --output-dir results/self_forcing/vbench \
  --prompt-file /data/video_prompts.json \
  --device cuda
```

`video_prompts.json` 是“视频文件名到 prompt”的 JSON 对象。VBench 使用官方包和模型文件；可以先加 `--dry-run --config-out prepared.json` 检查输入，不加载评测模型。
