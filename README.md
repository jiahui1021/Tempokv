# TempoKV

[中文使用指南](assets/usage_zh.md) 

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

Two codec configs are provided. Both use the two most recent reconstructed K
frames to predict K, and current reconstructed K plus the two most recent
reconstructed V frames to predict V:

- `configs/int2.json`: INT2 K/V residuals with an INT4 first anchor unit.
- `configs/int4.json`: INT4 K/V residuals and INT4 first anchor unit.

`k_history_length` controls how many reconstructed K frames predict the next K
and accepts values from `1` to `8`; fitting learns a separate coefficient for
each selected history frame.
`v_history_length` controls the V predictor: `2` uses the previous two V frames,
`1` uses only the previous frame, and `0` retains the legacy K-only CrossKV
predictor. V history coefficients are fitted per head and channel, so each lag
adds only lightweight multiply-adds instead of another full K-to-V projection.
Predictor parameters must be refitted after changing either value.

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

If `parameters/self_forcing_trace.npz` already exists, refit the fast two-frame
K/V parameters without collecting another trace:

```bash
python scripts/fit_params.py parameters/self_forcing_trace.npz \
  --config configs/int2.json \
  --output-dir parameters \
  --k-history-length 2 \
  --v-history-length 2 \
  --iterations 2 \
  --temporal-ridge 0.1 \
  --cross-ridge 0.1 \
  --max-gain 1.0 \
  --stride 1560 \
  --coordinate-system pre_rope_normalized
```

Generate videos from the fitted parameters, repeatable with different prompts:

```bash
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=2 \
V_HISTORY_LENGTH=2 \
FRAMES=717 \
bash self-forcing/generate.sh
```

`K_HISTORY_LENGTH` and `V_HISTORY_LENGTH` explicitly select the history used
for this run. If omitted, both values come from `CONFIG`; the example also
omits the default mode, parameter directory, and output directory.

The fast K/V history paths decode up to 16 units per launch by default. Override
this for A/B testing without refitting with
`TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH=8`. Values from 1 through 16 are accepted.
Dense V history and K history orders above four are always capped at eight
because their compile-time expansion is substantially larger.

Set `MODE=bf16` for an unquantized baseline; that mode does not need `CONFIG`
or `PARAM_DIR`. `pipeline_report.json` records total wall-clock time, per-GPU
latency, peak GPU memory, and the exact predictor-parameter footprint.

Both entry points validate `stride=1560`, 30 layers, 12 heads, head dimension
128, and pre-RoPE normalized K. Generation also checks that TempoKV is actually
triggered.
