# TempoKV

[中文使用指南](assets/usage_zh.md) 

TempoKV fits, quantizes, and reconstructs attention KV caches. The repository
provides the codec, a plain PyTorch reference decoder, and optional Triton CUDA
decoders. See the [Chinese usage guide](assets/usage_zh.md) for details, and
[self-forcing/README.md](self-forcing/README.md) for the Self-Forcing model
adapter.

The LongCat-Video adapter is documented in
[longcat/README.md](longcat/README.md). It keeps the TempoKV codec unchanged and
connects it to LongCat's clean pre-RoPE KV cache. It supports single-process
and Context Parallel runs, optional KV CPU offload, and independent attention
sink spans.

## Installation

CPU or PyTorch reference path:

```bash
pip install -e .
```

NVIDIA GPU Triton decoders:

```bash
pip install -e ".[cuda]"
```

Two codec configs are provided. Each config's `k_history_length` and
`v_history_length` select the exact number of reconstructed history frames used
at generation time, and the same values drive the parameter fit:

- `configs/int2.json`: INT2 K/V residuals with an INT4 first anchor unit.
- `configs/int4.json`: INT4 K/V residuals and INT4 first anchor unit.

`k_history_length` controls how many reconstructed K frames predict the next K
and accepts values from `1` through `8`.
`v_history_length` controls the V predictor: `2` uses the previous two V frames,
`1` uses only the previous frame, and `0` retains the legacy K-only CrossKV
predictor.

Different history configurations need their own fitted parameter files. Because
fitting is accelerated on GPU and takes only ~2 seconds per config (see below),
the recommended workflow is to fit one dedicated parameter set for every
`k_history_length` / `v_history_length` combination you intend to run.

## Self-Forcing pipeline

Three shell scripts. All settings are passed as environment variables.

**Step 1 — collect and merge a clean trace (once).** Collecting runs model
inference and takes about 10-15 minutes; every subsequent generation reuses the
same `self_forcing_trace.npz`:

```bash
TRACE_RUNNER=/path/to/self-forcing-trace-runner.sh \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/fit_prompts.txt \
WORK_DIR=runs/self_forcing_prepare \
bash self-forcing/collect_trace.sh
```

To collect without the wrapper, call the external trace runner directly, then
merge with `prepare_trace.py`:

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

**Step 2 — fit parameters.** Fit a dedicated parameter set for the exact
`k_history_length` / `v_history_length` you plan to run (takes about 1.5 seconds
on GPU):

```bash
CONFIG=configs/int2.json \
K_HISTORY_LENGTH=2 \
V_HISTORY_LENGTH=2 \
OUTPUT_DIR=parameters/self_forcing/kh2-vh2-int2 \
bash self-forcing/fit_params.sh
```

**Step 3 — generate videos.** `PARAM_DIR` points at step 2's output:

```bash
MODEL_REPO=/path/to/self-forcing-model-repo \
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/generation_prompts.txt \
CONFIG=configs/int2.json \
PARAM_DIR=parameters/self_forcing/kh2-vh2-int2 \
K_HISTORY_LENGTH=2 \
V_HISTORY_LENGTH=2 \
FRAMES=717 \
bash self-forcing/generate.sh
```

`K_HISTORY_LENGTH=2` and `V_HISTORY_LENGTH=2` explicitly select the history used
for this run; they must match the fitted `k_history_length` /
`v_history_length` of the parameter files in `PARAM_DIR`, otherwise parameter
shape validation fails. If omitted, both active values come from `CONFIG`.

`K_HISTORY_LENGTH` / `V_HISTORY_LENGTH` fall back to the config values when
omitted. The remaining fit options have sensible defaults and are usually left
as is: `FIT_DEVICE=auto` uses CUDA when available (float32 fitting),
`ITERATIONS` defaults to 2, `MAX_GAIN` to 1.0, and stride is fixed at 1560
(to match the trace). On an A100, fitting one dedicated config from the
128-prompt trace takes about 1.5 seconds of fit time (roughly 10-20 seconds
wall-clock including trace loading), versus roughly 14 minutes for the same fit
on CPU with float64 equations.

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
