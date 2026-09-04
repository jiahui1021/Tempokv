#!/usr/bin/env python3
"""Microbenchmark the TempoKV persistent decode kernels at production shape.

Stage 1 of the acceleration plan tunes launch configuration and nothing else, so
it needs a harness that can vary ``persistent_units``, the tile shape, the warp
count and the pipeline depth *without* touching the decoder.  Two sweeps are
provided:

``--sweep production``
    calls :func:`tempokv.triton_ops.decode` exactly as the runtime does and only
    varies ``TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH``.  This is the number that
    predicts an end-to-end change, because it is the real route.

``--sweep tile``
    launches the persistent kernels directly so ``block_s``, ``block_d``,
    ``num_warps`` and ``num_stages`` can be varied.  These are not yet runtime
    settings; the sweep exists to find the winner that Stage 1 then wires in.

Every row is checked against the Torch decoder for the same packed state.  Tile
rows are also compared bit-for-bit with the current production Triton route, so
a faster tile can be rejected when it changes the established reconstruction.
Registers and spills are read from the compiled kernel when the installed Triton
exposes them; occupancy is deliberately absent rather than guessed — the plan
reads that from Nsight Compute.

    python scripts/tools/benchmark_tempokv_kernels.py --route v --units 24 \\
      --sweep production --persistent-units 2 4 8

A microbenchmark win is not a result.  The plan requires the 717-frame
generation to confirm it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.codec import PackedTensor, TempoKVCodec, decode_tensor  # noqa: E402
from tempokv.config import CodecConfig  # noqa: E402
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402
from tempokv.runtime.launch import PERSISTENT_UNITS_ENV  # noqa: E402

COLUMNS = (
    "route",
    "sweep",
    "persistent_units",
    "block_s",
    "block_d",
    "num_warps",
    "num_stages",
    "median_ms",
    "p90_ms",
    "min_ms",
    "peak_allocated_mib",
    "production_equal",
    "production_max_abs_diff",
    "max_abs_diff",
    "rmse",
    "registers_per_thread",
    "spills",
)


@dataclass(frozen=True)
class Shape:
    """The production cache geometry one packed span covers."""

    batch: int = 1
    heads: int = 12
    head_dim: int = 128
    stride: int = 1560
    units: int = 24

    @property
    def sequence(self) -> int:
        return self.stride * self.units


def _production_config(
    bits: int,
    k_history: int,
    v_history: int,
    stride: int,
    cross_matmul_precision: str,
    v_bits: int | None = None,
    anchor_bits: int = 4,
) -> CodecConfig:
    """The legacy production codec settings the plan freezes for this round.

    ``stride`` is threaded through rather than hardcoded so a mismatch between
    the benchmark geometry and the codec is impossible; ``CodecConfig`` rejects
    any value legacy fitting does not support.  ``bits``/``v_bits`` and
    ``anchor_bits`` are separate because the decode kernels read a different
    packed width for each, and a mixed or wide-anchor run is exactly the case
    where a per-width regression would otherwise go unmeasured.
    """
    return CodecConfig(
        k_bits=bits,
        v_bits=bits if v_bits is None else v_bits,
        anchor_bits=anchor_bits,
        group_size=64,
        stride=stride,
        residual_mode="asymmetric",
        scale_dtype="bfloat16",
        reconstruction_dtype="bfloat16",
        use_triton=True,
        strict_triton=True,
        k_history_length=k_history,
        v_history_length=v_history,
        cross_matmul_precision=cross_matmul_precision,
        k_predictor_mode="shared",
        v_predictor_mode="cross_kv",
        fit_mode="legacy",
        cross_kv_weight_mode="layer_shared",
    )


def _synthetic_codec(
    shape: Shape, config: CodecConfig, device: torch.device, generator: torch.Generator
) -> TempoKVCodec:
    """Predictor parameters with production shapes and arbitrary values.

    Decode cost does not depend on what the coefficients are, and a synthetic set
    keeps the benchmark runnable without a fitted parameter directory.  Values are
    kept small so the closed-loop recurrence cannot diverge over 24 units and
    swamp the parity comparison with meaningless magnitudes.
    """
    history = config.k_history_length
    assert history is not None
    temporal = TemporalParameters(
        history_weights=torch.randn(
            history, history, shape.head_dim, generator=generator, device=device
        )
        * 0.1,
        history_bias=torch.zeros(history, shape.head_dim, device=device),
        stride=config.stride,
    )
    # A zero-length history axis is the diagonal-history shape that means "no V
    # history", so VH=0 needs no separate construction.
    history_weights = (
        torch.randn(
            config.v_history_length,
            shape.heads,
            shape.head_dim,
            generator=generator,
            device=device,
        )
        * 0.1
    )
    cross = CrossKVParameters(
        weight=torch.randn(
            shape.heads,
            shape.head_dim,
            shape.head_dim,
            generator=generator,
            device=device,
        )
        * (shape.head_dim**-0.5),
        bias=torch.zeros(shape.heads, shape.head_dim, device=device),
        stride=config.stride,
        history_weights=history_weights,
    )
    return TempoKVCodec(temporal, cross, config)


def _load_codec(
    shape: Shape,
    config: CodecConfig,
    device: torch.device,
    generator: torch.Generator,
    k_params: Path | None,
    v_params: Path | None,
) -> TempoKVCodec:
    if k_params is None or v_params is None:
        return _synthetic_codec(shape, config, device, generator)
    return TempoKVCodec(
        TemporalParameters.load(k_params),
        CrossKVParameters.load(v_params),
        config,
    )


def _build_state(
    shape: Shape,
    config: CodecConfig,
    codec: TempoKVCodec,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[PackedTensor, torch.Tensor, PackedTensor]:
    """Encode one K and one V span through the real encoder."""
    keys = torch.randn(
        shape.batch,
        shape.heads,
        shape.sequence,
        shape.head_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    packed_key, reconstructed_key = codec.encode_key(keys, layer=0)
    values = torch.randn(
        shape.batch,
        shape.heads,
        shape.sequence,
        shape.head_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    packed_value = codec.encode_value(values, reconstructed_key, layer=0)
    return packed_key, reconstructed_key, packed_value


# -- measurement -------------------------------------------------------------


def _time(
    call: Callable[[], Any], device: torch.device, warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        call()
        ends[index].record()
    torch.cuda.synchronize(device)
    samples = sorted(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )
    # median and p90, never best-of-N: the plan's stopping conditions compare
    # runs, and a minimum hides the variance that decides whether a 3% claim is
    # real.
    return {
        "median_ms": samples[len(samples) // 2],
        "p90_ms": samples[int(round(0.90 * (len(samples) - 1)))],
        "min_ms": samples[0],
    }


def _parity(
    produced: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    difference = produced.float() - reference.float()
    return (
        float(difference.abs().max().item()),
        float(difference.pow(2).mean().sqrt().item()),
    )


def _peak_allocated_mib(
    call: Callable[[], Any], device: torch.device
) -> float:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    call()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024**2)


# -- direct kernel launches --------------------------------------------------


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


@dataclass
class _LaunchInputs:
    """The tensors and scalars both persistent kernels need, resolved once."""

    quant: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    asymmetric: bool
    residual_sequence: int
    residual_packed_width: int
    scale_width: int
    batch_heads: int


def _launch_inputs(packed: PackedTensor) -> _LaunchInputs:
    batch, heads, _, width = packed.shape
    batch_heads = batch * heads
    scale_width = width // packed.group_size
    assert packed.residual_quant is not None and packed.residual_scale is not None
    residual_sequence = int(packed.residual_quant.shape[2])
    residual_packed_width = (
        (width * packed.bits + 7) // 8
    )
    quant = packed.residual_quant.contiguous().view(
        batch_heads, residual_sequence, residual_packed_width
    )
    scale = packed.residual_scale.contiguous().view(
        batch_heads, residual_sequence, scale_width
    )
    asymmetric = packed.residual_mode == "asymmetric"
    zero = (
        packed.residual_zero_point.contiguous().view(
            batch_heads, residual_sequence, scale_width
        )
        if asymmetric and packed.residual_zero_point is not None
        else scale
    )
    return _LaunchInputs(
        quant=quant,
        scale=scale,
        zero=zero,
        asymmetric=asymmetric,
        residual_sequence=residual_sequence,
        residual_packed_width=residual_packed_width,
        scale_width=scale_width,
        batch_heads=batch_heads,
    )


def _launch_anchor(
    packed: PackedTensor, output: torch.Tensor, inputs: _LaunchInputs
) -> None:
    """Decode unit 0, which every recurrent unit reads as its first history."""
    from tempokv.triton_ops import _anchor_decode_kernel

    batch, heads, sequence, width = packed.shape
    anchor_length = packed.unit_lengths[0]
    anchor_packed_width = (
        (width * packed.anchor_bits + 7) // 8
    )
    block_s, block_d = 16, min(128, width)
    strides = output.stride()
    _anchor_decode_kernel[
        (
            inputs.batch_heads,
            _ceil_div(anchor_length, block_s),
            _ceil_div(width, block_d),
        )
    ](
        packed.anchor_quant.contiguous().view(
            inputs.batch_heads, anchor_length, anchor_packed_width
        ),
        packed.anchor_scale.contiguous().view(
            inputs.batch_heads, anchor_length, inputs.scale_width
        ),
        output,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        sequence,
        anchor_length,
        heads,
        width,
        anchor_packed_width,
        inputs.scale_width,
        packed.anchor_bits,
        packed.group_size,
        block_s,
        block_d,
    )


def _launch_temporal(
    packed: PackedTensor,
    output: torch.Tensor,
    inputs: _LaunchInputs,
    *,
    persistent_units: int,
    block_s: int,
    block_d: int,
    num_warps: int,
    num_stages: int,
) -> Any:
    from tempokv.triton_ops import _temporal_persistent_decode_kernel

    _, heads, sequence, width = packed.shape
    parameters = packed.temporal
    assert parameters is not None
    parameters = parameters.to(output.device)
    strides = output.stride()
    grid = (
        inputs.batch_heads,
        _ceil_div(packed.stride, block_s),
        _ceil_div(width, block_d),
    )
    compiled = []
    for start_unit in range(1, len(packed.unit_lengths), persistent_units):
        chunk_units = min(
            persistent_units, len(packed.unit_lengths) - start_unit
        )
        # Keyword arguments so a kernel signature change fails loudly here
        # instead of silently shifting a positional argument.
        compiled.append(
            _temporal_persistent_decode_kernel[grid](
                quant_ptr=inputs.quant,
                scale_ptr=inputs.scale,
                zero_ptr=inputs.zero,
                weight_ptr=parameters.history_weights.contiguous(),
                bias_ptr=parameters.history_bias.contiguous(),
                output_ptr=output,
                output_stride_b=strides[0],
                output_stride_h=strides[1],
                output_stride_s=strides[2],
                output_stride_d=strides[3],
                sequence=sequence,
                residual_sequence=inputs.residual_sequence,
                heads=heads,
                width=width,
                packed_width=inputs.residual_packed_width,
                scale_width=inputs.scale_width,
                stride=packed.stride,
                history_length=parameters.history_length,
                start_unit=start_unit,
                chunk_units=chunk_units,
                bits=packed.bits,
                group_size=packed.group_size,
                per_head=parameters.per_head,
                asymmetric=inputs.asymmetric,
                block_s=block_s,
                block_d=block_d,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    return tuple(compiled)


def _launch_cross(
    packed: PackedTensor,
    source: torch.Tensor,
    output: torch.Tensor,
    inputs: _LaunchInputs,
    *,
    persistent_units: int,
    block_s: int,
    block_d: int,
    num_warps: int,
    num_stages: int,
) -> Any:
    from tempokv.triton_ops import (
        _cross_kv_history_persistent_decode_kernel,
        _cross_kv_persistent_decode_kernel,
    )

    _, heads, sequence, width = packed.shape
    parameters = packed.cross_kv
    assert parameters is not None
    parameters = parameters.to(output.device)
    if parameters.layered:
        raise ValueError("select one layer before benchmarking CrossKV decode")
    weight = parameters.weight
    history_weight = (
        parameters.history_weights
        if parameters.history_length
        else weight
    )
    bias = parameters.bias
    block_k = max(16, _next_power_of_two(width))
    source_strides = source.stride()
    strides = output.stride()
    grid = (
        inputs.batch_heads,
        _ceil_div(packed.stride, block_s),
        _ceil_div(width, block_d),
    )
    compiled = []
    for start_unit in range(1, len(packed.unit_lengths), persistent_units):
        chunk_units = min(
            persistent_units, len(packed.unit_lengths) - start_unit
        )
        shared = dict(
            quant_ptr=inputs.quant,
            scale_ptr=inputs.scale,
            zero_ptr=inputs.zero,
            source_ptr=source,
            weight_ptr=weight,
            bias_ptr=bias,
            output_ptr=output,
            source_stride_b=source_strides[0],
            source_stride_h=source_strides[1],
            source_stride_s=source_strides[2],
            source_stride_d=source_strides[3],
            output_stride_b=strides[0],
            output_stride_h=strides[1],
            output_stride_s=strides[2],
            output_stride_d=strides[3],
            sequence=sequence,
            residual_sequence=inputs.residual_sequence,
            heads=heads,
            width=width,
            packed_width=inputs.residual_packed_width,
            scale_width=inputs.scale_width,
            stride=packed.stride,
            start_unit=start_unit,
            chunk_units=chunk_units,
            bits=packed.bits,
            group_size=packed.group_size,
            asymmetric=inputs.asymmetric,
            cross_input_precision=packed.cross_matmul_precision,
            block_s=block_s,
            block_d=block_d,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if parameters.history_length:
            compiled.append(
                _cross_kv_history_persistent_decode_kernel[grid](
                    history_weight_ptr=history_weight,
                    history_length=parameters.history_length,
                    **shared,
                )
            )
        else:
            compiled.append(_cross_kv_persistent_decode_kernel[grid](**shared))
    return tuple(compiled)


def _next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


def _kernel_metrics(compiled: Any) -> dict[str, int | None]:
    """Maximum registers and spills across every launch in one decode.

    The last launch is commonly shorter than the others (for example 8/8/7
    units), so reporting only its compiled variant can hide the resource peak
    of the main kernel.
    """
    variants = compiled if isinstance(compiled, tuple) else (compiled,)
    registers = [
        value
        for item in variants
        if (value := getattr(item, "n_regs", None)) is not None
    ]
    spills = [
        value
        for item in variants
        if (value := getattr(item, "n_spills", None)) is not None
    ]
    return {
        "registers_per_thread": max(registers) if registers else None,
        "spills": max(spills) if spills else None,
    }


# -- sweeps ------------------------------------------------------------------


def _reference(
    packed: PackedTensor, source: torch.Tensor | None
) -> torch.Tensor:
    return decode_tensor(
        packed,
        source=source,
        output_layout="BHSD",
        use_triton=False,
    )


def _production_rows(
    route: str,
    packed: PackedTensor,
    source: torch.Tensor | None,
    reference: torch.Tensor,
    device: torch.device,
    units_choices: tuple[int, ...],
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    from tempokv.triton_ops import decode as decode_triton

    # A list, not a generator: the sweep sets a process-wide environment
    # variable, and a lazily consumed generator would leave it set.
    rows: list[dict[str, Any]] = []
    output = torch.empty(
        packed.shape, device=device, dtype=torch.bfloat16
    )
    previous = os.environ.get(PERSISTENT_UNITS_ENV)
    try:
        for units in units_choices:
            os.environ[PERSISTENT_UNITS_ENV] = str(units)

            def call() -> None:
                decode_triton(
                    packed,
                    source=source,
                    output_dtype=torch.bfloat16,
                    output_tensor=output,
                    output_layout="BHSD",
                )

            call()
            torch.cuda.synchronize(device)
            max_abs_diff, rmse = _parity(output, reference)
            timing = _time(call, device, warmup, iterations)
            rows.append(
                {
                    "route": route,
                    "sweep": "production",
                    "persistent_units": units,
                    "block_s": None,
                    "block_d": None,
                    "num_warps": None,
                    "num_stages": None,
                    "peak_allocated_mib": _peak_allocated_mib(call, device),
                    "production_equal": True,
                    "production_max_abs_diff": 0.0,
                    "max_abs_diff": max_abs_diff,
                    "rmse": rmse,
                    "registers_per_thread": None,
                    "spills": None,
                    **timing,
                }
            )
    finally:
        if previous is None:
            os.environ.pop(PERSISTENT_UNITS_ENV, None)
        else:
            os.environ[PERSISTENT_UNITS_ENV] = previous
    return rows


def _tile_rows(
    route: str,
    packed: PackedTensor,
    source: torch.Tensor | None,
    reference: torch.Tensor,
    device: torch.device,
    units_choices: tuple[int, ...],
    block_s_choices: tuple[int, ...],
    block_d_choices: tuple[int, ...],
    warp_choices: tuple[int, ...],
    stage_choices: tuple[int, ...],
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    inputs = _launch_inputs(packed)
    output = torch.empty(packed.shape, device=device, dtype=torch.bfloat16)
    production = torch.empty_like(output)
    from tempokv.triton_ops import decode as decode_triton

    decode_triton(
        packed,
        source=source,
        output_dtype=torch.bfloat16,
        output_tensor=production,
        output_layout="BHSD",
    )
    torch.cuda.synchronize(device)
    rows: list[dict[str, Any]] = []
    for units, block_s, block_d, warps, stages in itertools.product(
        units_choices,
        block_s_choices,
        block_d_choices,
        warp_choices,
        stage_choices,
    ):
        def call() -> Any:
            _launch_anchor(packed, output, inputs)
            if route == "k_temporal":
                return _launch_temporal(
                    packed,
                    output,
                    inputs,
                    persistent_units=units,
                    block_s=block_s,
                    block_d=block_d,
                    num_warps=warps,
                    num_stages=stages,
                )
            assert source is not None
            return _launch_cross(
                packed,
                source,
                output,
                inputs,
                persistent_units=units,
                block_s=block_s,
                block_d=block_d,
                num_warps=warps,
                num_stages=stages,
            )

        try:
            compiled = call()
            torch.cuda.synchronize(device)
        except Exception as error:  # a tile can simply not be compilable
            rows.append(
                {
                    "route": route,
                    "sweep": "tile",
                    "persistent_units": units,
                    "block_s": block_s,
                    "block_d": block_d,
                    "num_warps": warps,
                    "num_stages": stages,
                    "median_ms": None,
                    "p90_ms": None,
                    "min_ms": None,
                    "peak_allocated_mib": None,
                    "production_equal": None,
                    "production_max_abs_diff": None,
                    "max_abs_diff": None,
                    "rmse": None,
                    "registers_per_thread": None,
                    "spills": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        max_abs_diff, rmse = _parity(output, reference)
        production_max_abs_diff, _ = _parity(output, production)
        timing = _time(call, device, warmup, iterations)
        rows.append(
            {
                "route": route,
                "sweep": "tile",
                "persistent_units": units,
                "block_s": block_s,
                "block_d": block_d,
                "num_warps": warps,
                "num_stages": stages,
                "peak_allocated_mib": _peak_allocated_mib(call, device),
                "production_equal": bool(torch.equal(output, production)),
                "production_max_abs_diff": production_max_abs_diff,
                "max_abs_diff": max_abs_diff,
                "rmse": rmse,
                **_kernel_metrics(compiled),
                **timing,
            }
        )
    return rows


def _print_rows(rows: list[dict[str, Any]]) -> None:
    widths = {column: max(len(column), 12) for column in COLUMNS}
    print("  ".join(column.rjust(widths[column]) for column in COLUMNS))
    for row in rows:
        cells = []
        for column in COLUMNS:
            value = row.get(column)
            if value is None:
                text = "-"
            elif isinstance(value, float):
                text = f"{value:.5g}"
            else:
                text = str(value)
            cells.append(text.rjust(widths[column]))
        print("  ".join(cells))
        if row.get("error"):
            print(f"    {row['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--route",
        choices=("k", "v", "both"),
        default="both",
        help="k = temporal K decode, v = CrossKV V decode",
    )
    parser.add_argument(
        "--sweep", choices=("production", "tile", "both"), default="production"
    )
    parser.add_argument("--bits", type=int, choices=range(1, 9), default=4)
    parser.add_argument(
        "--v-bits",
        type=int,
        choices=range(1, 9),
        help="V residual width; defaults to --bits",
    )
    parser.add_argument(
        "--anchor-bits", type=int, choices=range(1, 17), default=4
    )
    parser.add_argument("--k-history", type=int, default=3)
    parser.add_argument("--v-history", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1560)
    parser.add_argument(
        "--units",
        type=int,
        default=24,
        help="units per packed span; Self-Forcing packs 24 frames per event",
    )
    parser.add_argument(
        "--persistent-units", type=int, nargs="+", default=(2, 4, 8)
    )
    parser.add_argument(
        "--cross-matmul-precision",
        choices=("ieee", "tf32"),
        default="ieee",
        help="CrossKV tl.dot precision; run both values to measure the TF32 candidate",
    )
    parser.add_argument("--block-s", type=int, nargs="+", default=(8, 16, 32, 64))
    parser.add_argument("--block-d", type=int, nargs="+", default=(16, 32, 64, 128))
    parser.add_argument("--num-warps", type=int, nargs="+", default=(4,))
    parser.add_argument("--num-stages", type=int, nargs="+", default=(3,))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-params", type=Path)
    parser.add_argument("--v-params", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    arguments = parser.parse_args(argv)

    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires a CUDA device")
    if arguments.iterations < 1 or arguments.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")

    shape = Shape(
        heads=arguments.heads,
        head_dim=arguments.head_dim,
        stride=arguments.stride,
        units=arguments.units,
    )
    config = _production_config(
        arguments.bits,
        arguments.k_history,
        arguments.v_history,
        arguments.stride,
        arguments.cross_matmul_precision,
        v_bits=arguments.v_bits,
        anchor_bits=arguments.anchor_bits,
    )
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    codec = _load_codec(
        shape, config, device, generator, arguments.k_params, arguments.v_params
    )
    packed_key, reconstructed_key, packed_value = _build_state(
        shape, config, codec, device, generator
    )

    targets = []
    if arguments.route in ("k", "both"):
        targets.append(("k_temporal", packed_key, None))
    if arguments.route in ("v", "both"):
        targets.append(("v_cross_kv", packed_value, reconstructed_key))

    rows: list[dict[str, Any]] = []
    for route, packed, source in targets:
        reference = _reference(packed, source)
        if arguments.sweep in ("production", "both"):
            rows.extend(
                _production_rows(
                    route,
                    packed,
                    source,
                    reference,
                    device,
                    tuple(arguments.persistent_units),
                    arguments.warmup,
                    arguments.iterations,
                )
            )
        if arguments.sweep in ("tile", "both"):
            rows.extend(
                _tile_rows(
                    route,
                    packed,
                    source,
                    reference,
                    device,
                    tuple(arguments.persistent_units),
                    tuple(arguments.block_s),
                    tuple(arguments.block_d),
                    tuple(arguments.num_warps),
                    tuple(arguments.num_stages),
                    arguments.warmup,
                    arguments.iterations,
                )
            )
        del reference

    _print_rows(rows)
    payload = {
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "shape": {
            "batch": shape.batch,
            "heads": shape.heads,
            "head_dim": shape.head_dim,
            "stride": shape.stride,
            "units": shape.units,
        },
        "bits": config.k_bits if config.k_bits == config.v_bits else None,
        "k_bits": config.k_bits,
        "v_bits": config.v_bits,
        "anchor_bits": config.anchor_bits,
        "k_history": arguments.k_history,
        "v_history": arguments.v_history,
        "cross_matmul_precision": arguments.cross_matmul_precision,
        "parameters": (
            "fitted"
            if arguments.k_params and arguments.v_params
            else "synthetic"
        ),
        "rows": rows,
    }
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {arguments.json}")
    if arguments.csv is not None:
        arguments.csv.parent.mkdir(parents=True, exist_ok=True)
        with arguments.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=(*COLUMNS, "error"), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {arguments.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
