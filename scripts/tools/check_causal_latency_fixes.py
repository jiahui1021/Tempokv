#!/usr/bin/env python3
"""Check the two Causal Forcing latency fixes on a CUDA device.

``rope``
    Read-time RoPE of the whole decoded K cache, once per layer per forward.
    Times the per-block PyTorch float64 reference (out of place, as the
    attention path used to call it) against the fused Triton kernel (in place),
    at the production grid, and counts differing BF16 values.  Any nonzero
    count is a failure.

``decode``
    Replays the one-anchor rollout schedule: a single packed span read at every
    length it reaches (3, 6, ..., 159 units for 717 frames with a BF16 lag of
    18).  Reports how many distinct ``(start_unit, chunk_units)`` launches the
    persistent kernels see, times a first and a second pass (the difference is
    Triton compile/load time), and compares every decode bit for bit with the
    unshifted launch plan.  Run it with a fresh ``TRITON_CACHE_DIR`` to see the
    cold-cache cost a new container pays.

    TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=src \\
      python scripts/tools/check_causal_latency_fixes.py

A microbenchmark is not the end-to-end result: rerun the 717-frame generation
and compare ``diffusion_generation_s`` / ``generation_seconds``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv import triton_ops  # noqa: E402
from tempokv.codec import TempoKVCodec, decode_tensor  # noqa: E402
from tempokv.config import CodecConfig  # noqa: E402
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402
from tempokv.runtime.cache import CacheSpan  # noqa: E402
from tempokv.runtime.rope import ROPE_BACKEND_ENV, apply_rope_by_span  # noqa: E402


def _rope_table(max_frames: int, head_dim: int) -> torch.Tensor:
    """``CausalWanModel.freqs``: complex128, temporal/height/width halves."""

    def params(dim: int) -> torch.Tensor:
        angles = torch.outer(
            torch.arange(max_frames),
            1.0 / torch.pow(10000, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
        )
        return torch.polar(torch.ones_like(angles), angles)

    return torch.cat(
        [
            params(head_dim - 4 * (head_dim // 6)),
            params(2 * (head_dim // 6)),
            params(2 * (head_dim // 6)),
        ],
        dim=1,
    )


def _median_ms(call: Callable[[], Any], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def check_rope(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    freqs = _rope_table(1024, args.head_dim).to(device)
    grid = torch.tensor([[args.block_frames, args.height, args.width]])
    rows = []
    for frames in args.frames:
        tokens = frames * args.height * args.width
        generator = torch.Generator(device=device).manual_seed(frames)
        x = torch.randn(
            1, tokens, args.heads, args.head_dim, generator=generator, device=device
        ).to(torch.bfloat16)
        spans = [CacheSpan(0, 0, frames)]

        os.environ[ROPE_BACKEND_ENV] = "torch"
        expected = apply_rope_by_span(x, grid, freqs, spans)
        reference_ms = _median_ms(
            lambda: apply_rope_by_span(x, grid, freqs, spans),
            args.warmup,
            args.iterations,
        )

        os.environ[ROPE_BACKEND_ENV] = "triton"
        work = x.clone()
        actual = apply_rope_by_span(work, grid, freqs, spans, out=work)
        mismatched = int((actual != expected).sum())
        fused_ms = _median_ms(
            lambda: apply_rope_by_span(work, grid, freqs, spans, out=work),
            args.warmup,
            args.iterations,
        )
        rows.append(
            {
                "frames": frames,
                "reference_ms": reference_ms,
                "fused_ms": fused_ms,
                "speedup": reference_ms / fused_ms,
                "mismatched_values": mismatched,
            }
        )
        del x, expected, work, actual
        torch.cuda.empty_cache()
    os.environ.pop(ROPE_BACKEND_ENV, None)
    return rows


def _codec(args: argparse.Namespace, device: torch.device) -> TempoKVCodec:
    generator = torch.Generator(device=device).manual_seed(0)
    config = CodecConfig(
        k_bits=args.bits,
        v_bits=args.bits,
        anchor_bits=4,
        group_size=64,
        stride=args.stride,
        residual_mode="asymmetric",
        scale_dtype="bfloat16",
        reconstruction_dtype="bfloat16",
        use_triton=True,
        strict_triton=True,
        k_history_length=args.k_history,
        v_history_length=args.v_history,
        cross_matmul_precision="ieee",
        k_predictor_mode="shared",
        v_predictor_mode="cross_kv",
        fit_mode="legacy",
        cross_kv_weight_mode="layer_shared",
    )
    temporal = TemporalParameters(
        history_weights=torch.randn(
            args.k_history, args.k_history, args.head_dim,
            generator=generator, device=device,
        ) * 0.1,
        history_bias=torch.zeros(args.k_history, args.head_dim, device=device),
        stride=args.stride,
    )
    cross = CrossKVParameters(
        weight=torch.randn(
            args.heads, args.head_dim, args.head_dim,
            generator=generator, device=device,
        ) * (args.head_dim ** -0.5),
        bias=torch.zeros(args.heads, args.head_dim, device=device),
        stride=args.stride,
        history_weights=torch.randn(
            args.v_history, args.heads, args.head_dim,
            generator=generator, device=device,
        ) * 0.1,
    )
    return TempoKVCodec(temporal, cross, config)


def _span_lengths(latent_frames: int, block: int, lag: int) -> list[int]:
    """Packed-span lengths the attention reads, one per block (one anchor)."""
    lengths = []
    for finalized in range(block, latent_frames, block):
        units = finalized - lag
        if units > 0:
            lengths.append(units)
    return lengths


def check_decode(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    codec = _codec(args, device)
    lengths = _span_lengths(args.latent_frames, args.block_frames, args.lag)
    generator = torch.Generator(device=device).manual_seed(1)
    states = []
    for units in lengths:
        keys = torch.randn(
            1, args.heads, units * args.stride, args.head_dim,
            generator=generator, device=device,
        ).to(torch.bfloat16)
        values = torch.randn_like(keys)
        packed_key, reconstructed_key = codec.encode_key(keys, layer=0)
        packed_value = codec.encode_value(values, reconstructed_key, layer=0)
        states.append((packed_key, packed_value))
    torch.cuda.synchronize()

    launches: set[tuple[str, int, int]] = set()
    planner = triton_ops.persistent_launch_plan

    def recording_plan(units: int, per_launch: int, lead: int):
        plan = planner(units, per_launch, lead)
        launches.update((f"lead{lead}", start, chunk) for _, start, chunk in plan)
        return plan

    def unshifted_plan(units: int, per_launch: int, lead: int):
        del lead
        return [
            (0, start, min(per_launch, units - start))
            for start in range(1, units, per_launch)
        ]

    def decode_all() -> list[tuple[torch.Tensor, torch.Tensor]]:
        decoded = []
        for packed_key, packed_value in states:
            key = decode_tensor(
                packed_key, output_dtype=torch.bfloat16,
                use_triton=True, strict_triton=True,
            )
            value = decode_tensor(
                packed_value, source=key, output_dtype=torch.bfloat16,
                use_triton=True, strict_triton=True,
            )
            decoded.append((key, value))
        torch.cuda.synchronize()
        return decoded

    triton_ops.persistent_launch_plan = recording_plan
    try:
        started = time.perf_counter()
        shifted = decode_all()
        first_pass_s = time.perf_counter() - started
        started = time.perf_counter()
        decode_all()
        second_pass_s = time.perf_counter() - started
        triton_ops.persistent_launch_plan = unshifted_plan
        unshifted = decode_all()
    finally:
        triton_ops.persistent_launch_plan = planner
    mismatched = sum(
        int((a_key != b_key).sum()) + int((a_value != b_value).sum())
        for (a_key, a_value), (b_key, b_value) in zip(shifted, unshifted)
    )
    return {
        "span_lengths": f"{lengths[0]}..{lengths[-1]} ({len(lengths)} reads)",
        "distinct_launch_shapes": len(launches),
        "first_pass_s": first_pass_s,
        "second_pass_s": second_pass_s,
        "compile_and_load_s": first_pass_s - second_pass_s,
        "mismatched_values_vs_unshifted_plan": mismatched,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", "(Triton default)"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", choices=("rope", "decode", "both"), default="both")
    parser.add_argument("--frames", type=int, nargs="+", default=(21, 90, 180))
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--width", type=int, default=52)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-frames", type=int, default=3)
    parser.add_argument("--latent-frames", type=int, default=180)
    parser.add_argument("--lag", type=int, default=18)
    parser.add_argument(
        "--stride", type=int, default=16,
        help="tokens per unit for the decode check; compile count does not depend on it",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--k-history", type=int, default=2)
    parser.add_argument("--v-history", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("this check needs a CUDA device")

    report: dict[str, Any] = {"device": torch.cuda.get_device_name()}
    failed = False
    if args.check in {"rope", "both"}:
        report["rope"] = check_rope(args)
        failed |= any(row["mismatched_values"] for row in report["rope"])
    if args.check in {"decode", "both"}:
        report["decode"] = check_decode(args)
        failed |= bool(report["decode"]["mismatched_values_vs_unshifted_plan"])
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
