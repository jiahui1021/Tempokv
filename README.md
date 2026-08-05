# TempoKV

TempoKV is a compact implementation of predictive residual quantization for
attention KV caches. It keeps the three pieces needed by the runtime:

- K is predicted from multiple reconstructed history units, with separate
  warm-up parameters for one, two, and three available histories.
- V is predicted from the reconstructed K of the same unit with a per-head
  CrossKV affine projection.
- Only the prediction residual is quantized after the first anchor unit.

The package includes parameter fitting, a plain PyTorch reference decoder, and
fused Triton decoders for NVIDIA GPUs. It does not include model-specific video
generation code, evaluation pipelines, or unrelated cache codecs.

## Installation

```bash
pip install -e ".[test]"
```

For the GPU decoder, install a Triton version compatible with the installed
PyTorch and CUDA runtime:

```bash
pip install -e ".[cuda,test]"
```

## Runtime

```python
import torch

from tempokv import (
    CodecConfig,
    CrossKVParameters,
    TempoKVCodec,
    TemporalParameters,
)

k_parameters = TemporalParameters.load("k_temporal.npz")
v_parameters = CrossKVParameters.load("v_cross_kv.npz")

codec = TempoKVCodec(
    k_parameters,
    v_parameters,
    CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=4,
        group_size=64,
        stride=1560,
        residual_mode="asymmetric",
        reconstruction_dtype="bfloat16",
        use_triton=True,
        strict_triton=True,
    ),
)

packed = codec.encode(key_cache, value_cache, layer=layer_index)
key_rebuilt, value_rebuilt = codec.decode(
    packed, output_dtype=torch.bfloat16
)
```

The order and reconstruction dtype are part of the format: K is encoded and
reconstructed first, then V uses that exact reconstructed K. Decoding follows
the same order. Missing or
mismatched parameters or dtypes raise an error instead of silently changing
the closed loop.

## Parameter fitting

The closed-loop fitting entry point starts from clean traces, repeatedly
reconstructs K with the target quantizer, and fits V from the final
reconstructed K:

```python
from tempokv import fit_predictors

k_parameters, v_parameters = fit_predictors(
    clean_k,
    clean_v,
    config=config,
    history_length=3,
    iterations=2,
    temporal_ridge=0.1,
    cross_ridge=0.1,
)
```

Clean traces use `[L, ..., T, H, D]`. The low-level fitting functions can also
operate directly on prepared tensors:

```python
from tempokv import fit_cross_kv, fit_temporal

k_parameters = fit_temporal(
    reconstructed_k,
    clean_k,
    history_length=3,
    shared=False,
    ridge=0.1,
    max_gain=1.0,
    stride=1560,
)

v_parameters = fit_cross_kv(
    reconstructed_k_by_layer,
    clean_v_by_layer,
    ridge=0.1,
    stride=1560,
)
```

Temporal fitting expects `[..., T, H, D]`; `T` is the unit or frame order.
CrossKV fitting expects `[L, ..., H, D]`; `L` is the layer axis and the middle
dimensions are pooled observations. Fit K from decoder-visible reconstructed
history, then fit V from the same reconstructed K used by the runtime.

Temporal parameters support three sharing levels:

- `[P,P,D]`: shared across layers and heads;
- `[P,P,H,D]`: separate heads, shared across layers;
- `[L,P,P,H,D]`: separate layers and heads.

CrossKV parameters use `[L,H,D,D]` weights and `[L,H,D]` bias. Both parameter
types use numerical-only `.npz` files loaded with pickle disabled.

## Quantization format

- Supported stored widths: INT2, INT4, and INT8.
- INT2 and INT4 values are packed into bytes.
- The anchor is symmetrically quantized.
- Residuals support symmetric quantization and asymmetric zero-point
  quantization. The latter is the default.
- Scale storage supports `float32`, `float16`, and `bfloat16`.

## GPU decoder

The Triton path fuses prediction, residual unpacking, scaling, and cache
reconstruction. It contains dedicated temporal and CrossKV kernels, plus
persistent variants that decode several regular units per launch. Unsupported
layouts fall back to PyTorch unless `strict_triton=True`.

For multi-process jobs, isolate the compilation cache before the first decode:

```python
from tempokv import configure_triton_cache

configure_triton_cache("/fast/local/cache")
```

## Validation

```bash
pytest
```

CPU tests cover fitting, parameter files, all supported bit widths, both
residual modes, warm-up prediction, partial final units, and exact agreement
between encoder reconstruction and the reference decoder. CUDA tests compare
the Triton and PyTorch decoders when a compatible NVIDIA environment is
available.
