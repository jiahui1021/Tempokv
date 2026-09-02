#!/usr/bin/env python3
"""Measure how K/V reconstruction error grows over a long horizon.

Each method is fitted on a calibration trace and scored on a separate
long-horizon trace, reporting the error at every frame index as well as the
pooled figure.  Fitting on the short trace and scoring on the long one is the
point: it measures what happens when the predictor runs far past the horizon it
was calibrated for, which a same-trace fit cannot show.

    python scripts/paper_experiments/long_horizon/run_reconstruction.py \\
        --calibration-trace data/self-forcing/trace/clean_trace.npz \\
        --horizon-trace data/long-horizon/trace_1401f.npz \\
        --output-dir results/long_horizon/int2
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from methods import (  # noqa: E402
    DEFAULT_METHODS,
    METHODS,
    SCHEMA_VERSION,
    fit,
    frame_curve,
    method,
    reconstruct,
    summarize_curve,
    write_report,
)


def _load_trace(path: Path, *, max_records: int | None = None):
    """Read a fitting trace as [L,...,T,H,D] float32 K/V plus its stride."""
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        keys = torch.from_numpy(np.array(archive["keys"], copy=True)).float()
        values = torch.from_numpy(np.array(archive["values"], copy=True)).float()
        stride = None
        for name in ("trace_stride", "stride"):
            if name in archive:
                stride = int(np.asarray(archive[name]).item())
                break
    if keys.shape != values.shape:
        raise ValueError(f"{path} K and V must share a shape")
    if stride is None:
        raise ValueError(f"{path} does not record a trace stride")
    # Seven-dimensional legacy event traces carry the same records with the
    # event/batch/token axes kept apart; collapse them so both layouts read the
    # same way from here on.
    if keys.ndim == 7:
        layers, events, batch, tokens = keys.shape[:4]
        keys = keys.reshape(layers, events * batch * tokens, *keys.shape[4:])
        values = values.reshape(layers, events * batch * tokens, *values.shape[4:])
    if keys.ndim != 5:
        raise ValueError(
            f"{path} must use [L,S,T,H,D] or the legacy event layout, got "
            f"{tuple(keys.shape)}"
        )
    if max_records is not None:
        if max_records < 1:
            raise ValueError("--max-records must be positive")
        # Records are independent rollouts, so dropping some scales memory down
        # without touching the frame axis the curve is measured along.
        keys = keys[:, :max_records].contiguous()
        values = values[:, :max_records].contiguous()
    return keys, values, stride


def _projected_bytes(shape: tuple[int, ...]) -> int:
    total = 4
    for item in shape:
        total *= int(item)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument(
        "--horizon-trace",
        type=Path,
        required=True,
        help="the long trace to score; must share the calibration stride",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help=f"subset of {', '.join(item.name for item in METHODS)}",
    )
    parser.add_argument("--bits", type=int, default=2, choices=range(1, 9))
    parser.add_argument(
        "--max-records",
        type=int,
        help="keep only this many records of the horizon trace, to bound memory",
    )
    parser.add_argument(
        "--fit-on-horizon",
        action="store_true",
        help=(
            "fit on the horizon trace itself; measures fit rather than "
            "extrapolation and is only meaningful as a reference line"
        ),
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--temporal-ridge", type=float, default=1.0)
    parser.add_argument("--cross-ridge", type=float, default=0.1)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--model", default="self-forcing")
    parser.add_argument("--report-name", default="long_horizon_reconstruction.json")
    return parser


def run(args: argparse.Namespace) -> dict:
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1]")
    horizon_keys, horizon_values, horizon_stride = _load_trace(
        args.horizon_trace, max_records=args.max_records
    )
    if args.fit_on_horizon:
        calibration = (horizon_keys, horizon_values, horizon_stride)
    else:
        calibration = _load_trace(args.calibration_trace)
    if calibration[2] != horizon_stride:
        raise ValueError(
            f"calibration stride {calibration[2]} does not match horizon stride "
            f"{horizon_stride}; the two traces come from different runs"
        )
    frames = int(horizon_keys.shape[-3])
    print(
        f"[long-horizon] horizon trace {tuple(horizon_keys.shape)} "
        f"({_projected_bytes(tuple(horizon_keys.shape)) / 1024**3:.2f} GiB per tensor), "
        f"{frames} frames",
        file=sys.stderr,
        flush=True,
    )

    results = []
    for name in args.methods:
        spec = method(name)
        print(f"[long-horizon] {spec.name}", file=sys.stderr, flush=True)
        config = spec.codec(horizon_stride, args.bits)
        temporal, secondary = fit(
            spec,
            calibration[0],
            calibration[1],
            config,
            iterations=args.iterations,
            temporal_ridge=args.temporal_ridge,
            cross_ridge=args.cross_ridge,
        )
        rebuilt_keys, rebuilt_values = reconstruct(
            spec, horizon_keys, horizon_values, temporal, secondary, config
        )
        curves = {
            "k": frame_curve(horizon_keys, rebuilt_keys),
            "v": frame_curve(horizon_values, rebuilt_values),
        }
        # Release the rebuilt tensors before the next method allocates its own;
        # at 1401 frames these are the largest objects in the process.
        del rebuilt_keys, rebuilt_values
        gc.collect()
        results.append(
            {
                "method": spec.name,
                "label": spec.label,
                "description": spec.description,
                "codec_signature": config.format_signature(),
                "k": curves["k"],
                "v": curves["v"],
                "summary": {
                    tensor: summarize_curve(
                        curves[tensor], tail_fraction=args.tail_fraction
                    )
                    for tensor in ("k", "v")
                },
            }
        )

    payload = {
        "module": "long_horizon_reconstruction",
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "bits": args.bits,
        "frames": frames,
        "stride": horizon_stride,
        "horizon_trace": str(args.horizon_trace.resolve()),
        "calibration_trace": (
            str(args.horizon_trace.resolve())
            if args.fit_on_horizon
            else str(args.calibration_trace.resolve())
        ),
        "fitted_on_horizon": bool(args.fit_on_horizon),
        "horizon_shape": list(horizon_keys.shape),
        "max_records": args.max_records,
        "tail_fraction": args.tail_fraction,
        "methods": results,
    }
    write_report(args.output_dir / args.report_name, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if payload["fitted_on_horizon"]:
        print(
            "warning: fitted on the horizon trace, so these numbers measure fit "
            "rather than extrapolation",
            file=sys.stderr,
        )
    header = f"{'method':<14}{'tensor':>7}{'frame 0':>12}{'last':>12}{'tail mean':>12}{'pooled':>12}"
    print(header)
    for item in payload["methods"]:
        for tensor in ("k", "v"):
            summary = item["summary"][tensor]
            print(
                f"{item['method']:<14}{tensor.upper():>7}"
                f"{summary['first_frame_nmse']:>12.6f}"
                f"{summary['last_frame_nmse']:>12.6f}"
                f"{summary['tail_mean_nmse']:>12.6f}"
                f"{summary['pooled_nmse']:>12.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
