# LongCat + TempoKV

`longcat/` adapts the existing TempoKV codec to the LongCat-Video runtime in
`qvg/Quant-VideoGen`. The LongCat model remains external; this directory only
patches the clean-cache, encode, and attention-decode boundaries.

The active Python environment must contain the dependencies declared by
`Quant-VideoGen` for LongCat, including `einops`, `diffusers`,
`transformers`, and the CUDA attention/runtime packages used by that checkout.
From a Quant-VideoGen checkout, its declared LongCat extras can be installed
with:

```bash
python -m pip install -e '.[longcat]'
```

Check the external environment before loading the checkpoint:

```bash
python longcat/preflight.py \
  --mode trace \
  --longcat-repo /path/to/qvg/Quant-VideoGen \
  --checkpoint-dir /models/LongCat-Video
```

For TempoKV generation, add `--config`, `--k-parameters`, and
`--v-parameters`. The command returns exit code `2` and lists missing
dependencies, checkpoint directories, CUDA, or mismatched stride settings.

The adapter supports the LongCat cache boundary used by the released runner:

- LongCat 48 layers, 32 heads, head dimension 128;
- context parallel size 1 or a configured Context Parallel split;
- KV CPU offload, with only packed quantized storage moved between CPU and GPU;
- attention sink frames represented as independent spans;
- K is decoded before V, and the reconstructed K is passed to the V decoder.

The preflight command also checks the qvg cache-hook source contract and the
parameter metadata. LongCat parameters must use `pre_rope_normalized`
coordinates, the measured local `stride`, and either one layer-specific set for
all 48 layers or a parameter set explicitly marked as shared.

Both shell launchers run this check once before starting `torchrun`. Set
`PREFLIGHT=0` only after the same environment and checkpoint have already been
checked. Set `PYTHON_BIN` when the Python executable used by `torchrun` is not
`python3`.

The codec parameters must be fitted for the local cache stride. For the default
480p CP=1 workload this is `1560`; with CP=2 it is typically `780`, and with
CP=4 it is typically `390`. The bridge validates the actual runtime value and
rejects a mismatched parameter set.

## Trace and fit

Collect clean pre-RoPE normalized K/V traces:

```bash
LONGCAT_REPO=/path/to/qvg/Quant-VideoGen \
CHECKPOINT_DIR=/models/LongCat-Video \
INIT_VIDEO_PATH=results/longcat/base/0-0.mp4 \
TRACE_DIR=parameters/longcat_raw_trace \
bash longcat/collect_trace.sh
```

Trace mode always forwards `--quant_type none` to qvg, so the saved samples are
the unquantized LongCat cache even when extra qvg arguments are appended.

Merge the trace into the fitter layout and fit a dedicated parameter set:

```bash
python longcat/prepare_trace.py \
  parameters/longcat_raw_trace \
  parameters/longcat_trace.npz

CONFIG=configs/longcat_int4.json \
TRACE=parameters/longcat_trace.npz \
OUTPUT_DIR=parameters/longcat/kh2-vh2-int4 \
bash longcat/fit_params.sh
```

The merged trace has shape `[48, samples, temporal_units, 32, 128]` and records
the local stride plus `coordinate_system=pre_rope_normalized`. With Context
Parallel, each rank writes a separate shard under `TRACE_DIR`; the preparation
script merges all shards recursively.

For a CP=2 trace collected at the default 480p shape, pass the local stride
explicitly when merging:

```bash
python longcat/prepare_trace.py \
  parameters/longcat_raw_trace \
  parameters/longcat_cp2_trace.npz \
  --stride 780
```

## Generation

```bash
LONGCAT_REPO=/path/to/qvg/Quant-VideoGen \
CHECKPOINT_DIR=/models/LongCat-Video \
INIT_VIDEO_PATH=results/longcat/base/0-0.mp4 \
OUTPUT_DIR=results/longcat/tempokv-int4 \
CONFIG=configs/longcat_int4.json \
K_PARAMS=parameters/longcat/kh2-vh2-int4/k_temporal.npz \
V_PARAMS=parameters/longcat/kh2-vh2-int4/v_cross_kv.npz \
bash longcat/generate.sh
```

Optional runtime modes are selected with environment variables:

```bash
CONTEXT_PARALLEL_SIZE=2 \
NUM_GPUS=2 \
ATTN_SINK_FRAMES=8 \
OFFLOAD_KV_CACHE=1 \
CONFIG=configs/longcat_cp2_int4.json \
bash longcat/generate.sh
```

For CP=2 or CP=4, fit parameters from traces collected with the same split and
use the matching `longcat_cp2_*.json` or `longcat_cp4_*.json` configuration.

For another LongCat resolution, pass the measured local cache stride explicitly
with `--stride VALUE`, or set `CACHE_STRIDE=VALUE` when using either shell
launcher. The value must match the `stride` in the TempoKV config and fitted
parameter files.

Set `TRITON_CACHE` to reuse a writable Triton compilation directory across
runs. The option is forwarded to TempoKV when the selected config enables the
Triton decoder.

The launcher adds the bridge before model instances are created, then attaches
it to the loaded DIT and pipeline. Existing LongCat command-line arguments can
be appended to `generate.sh`. Each rank writes a
`tempokv_bridge_report_rank_*.json` file containing quantization, decode, and
storage statistics. It also records `encode_key_calls`, `encode_value_calls`,
and `decode_tensor_calls`, so the TempoKV path can be checked directly. The
report separates `compressed_cache_bytes`,
`uncompressed_cache_bytes`, `compression_ratio`,
`predictor_parameter_bytes`, and `temporary_reconstruction_bytes`. For Context
Parallel runs, `stride` records the measured local value and
`configured_stride` records the optional explicit constraint.
