# TempoKV

[中文使用指南](assets/usage_zh.md) · [INT2 配置](configs/int2.json) · [INT4 配置](configs/int4.json)

TempoKV fits, quantizes, and reconstructs attention KV caches. The repository
provides the codec, a plain PyTorch reference decoder, and optional Triton CUDA
decoders. See the [Chinese usage guide](assets/usage_zh.md) for details, and
[self-forcing/README.md](self-forcing/README.md) for the Self-Forcing model
adapter.

## Installation

CPU or PyTorch reference path:

```bash
pip install -e .
```

NVIDIA GPU Triton decoders:

```bash
pip install -e ".[cuda]"
```

Two codec configs are provided:

- `configs/int2.json`: INT2 K/V residuals with an INT4 first anchor unit.
- `configs/int4.json`: INT4 K/V residuals and INT4 first anchor unit.

## Self-Forcing pipeline

Two shell scripts. All settings are passed as environment variables. Run
preparation once to collect a clean trace and fit parameters:

```bash
TRACE_RUNNER=/path/to/self-forcing-trace-runner.sh \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
WORK_DIR=runs/self_forcing_prepare \
CONFIG=configs/int2.json \
bash self-forcing/prepare.sh
```

Generate videos from the fitted parameters, repeatable with different prompts:

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

Set `MODE=bf16` for an unquantized baseline; that mode does not need `CONFIG`
or `PARAM_DIR`. `pipeline_report.json` records total wall-clock time, per-GPU
latency, and peak GPU memory.

Both entry points validate `stride=1560`, 30 layers, 12 heads, head dimension
128, and pre-RoPE normalized K. Generation also checks that TempoKV is actually
triggered.
