#!/usr/bin/env python3
"""Measure the Torch Cross(K) projection cost at production shapes.

    python scripts/tools/measure_cross_cost.py --reads 7200

Use ``benchmark_tempokv_kernels.py --route v`` for the Triton route.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.codec import cross_projection, predict_cross_kv  # noqa: E402
from tempokv.parameters import CrossKVParameters  # noqa: E402


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(function, device: torch.device, repeats: int) -> float:
    function()
    _synchronize(device)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        _synchronize(device)
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1560)
    parser.add_argument(
        "--units",
        type=int,
        nargs="*",
        default=(8, 45, 90, 180),
        help="span lengths in units to measure",
    )
    parser.add_argument(
        "--reads",
        type=int,
        default=7200,
        help="cache reads per generation, used to extrapolate the total",
    )
    parser.add_argument("--precision", choices=("ieee", "tf32"), default="ieee")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; pass --device cpu to measure there")

    heads, dim, stride = args.heads, args.head_dim, args.stride
    parameters = CrossKVParameters(
        weight=(torch.randn(heads, dim, dim, device=device) * 0.05),
        bias=torch.randn(heads, dim, device=device) * 0.05,
        stride=stride,
        history_weights=torch.rand(1, heads, dim, device=device) * 0.3,
    )

    print(
        f"device={device}  heads={heads}  head_dim={dim}  stride={stride}  "
        f"precision={args.precision}"
    )
    header = (
        f"{'units':>6} {'GFLOP':>8} {'batched':>10} {'per-unit':>10} "
        f"{'speedup':>8} {'x{reads} reads':>14}"
    )
    print(header)
    print("-" * len(header))
    for units in args.units:
        sequence = units * stride
        source = torch.randn(
            1, heads, sequence, dim, device=device, dtype=torch.float32
        ).to(torch.bfloat16)
        previous = torch.randn(
            1, heads, stride, dim, device=device, dtype=torch.float32
        ).to(torch.bfloat16)
        gflop = 2 * sequence * heads * dim * dim / 1e9

        def batched() -> None:
            projection = cross_projection(
                source, parameters, matmul_precision=args.precision
            )
            for index in range(units):
                span = slice(index * stride, (index + 1) * stride)
                predict_cross_kv(
                    source[:, :, span],
                    parameters,
                    [previous],
                    matmul_precision=args.precision,
                    projection=projection[:, :, span],
                )

        def per_unit() -> None:
            for index in range(units):
                span = slice(index * stride, (index + 1) * stride)
                predict_cross_kv(
                    source[:, :, span],
                    parameters,
                    [previous],
                    matmul_precision=args.precision,
                )

        batched_seconds = _time(batched, device, args.repeats)
        per_unit_seconds = _time(per_unit, device, args.repeats)
        total = batched_seconds * args.reads
        print(
            f"{units:>6} {gflop:>8.1f} {batched_seconds * 1e3:>9.2f}ms "
            f"{per_unit_seconds * 1e3:>9.2f}ms "
            f"{per_unit_seconds / max(batched_seconds, 1e-9):>7.2f}x "
            f"{total:>13.1f}s"
        )
        del source, previous

    print(
        "\nThe last column extrapolates one decode of that span to --reads "
        "cache reads.  Read the row nearest your average span, not the "
        "longest one: the span grows from one unit to the full sequence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
