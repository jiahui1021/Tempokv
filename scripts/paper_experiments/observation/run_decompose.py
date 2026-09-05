#!/usr/bin/env python3
"""Run the section 3.2 predictability ladder over a collected clean trace.

Fits every predictor on calibration prompts and reports each tier on strictly
disjoint held-out prompts, in two loops:

  ``clean``        open-loop probe - the predictor reads clean history.
  ``closed_loop``  the deployed structure - anchor INT4 plus quantized
                   residuals, so the predictor reads only what the decoder
                   holds.

Both are written.  The clean numbers describe the signal; only the closed-loop
numbers describe the system, so a design choice must never be argued from the
clean column alone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from decompose import (  # noqa: E402
    TierStats,
    ar_step,
    build_lags,
    channel_mean,
    closed_loop_rollout,
    cross_step,
    fit_ar,
    fit_cross,
    predict_ar,
    predict_cross,
    residual_stats,
)

SCHEMA = "tempokv.observation-decompose.v1"
LOOPS = ("clean", "closed_loop")
# ``K0``/``V0`` are the direct-quantization reference and are identically 1.0;
# they are emitted so the figure and the table can be built from the rows
# alone without a special case for the baseline column.
TIERS = ("K0", "K1", "K2", "V0", "V1", "V2", "V3")
_SCALE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class TraceError(ValueError):
    """A trace archive does not satisfy the observation protocol."""


# ---------------------------------------------------------------------------
# trace loading
# ---------------------------------------------------------------------------


def _int_field(archive: "np.lib.npyio.NpzFile", name: str, default: int | None = None):
    if name not in archive.files:
        if default is None:
            raise TraceError(f"trace is missing the {name!r} field")
        return default
    return int(np.asarray(archive[name]).item())


def load_trace(path: Path) -> dict[str, Any]:
    """Read a ``prepare_trace`` archive into ``[L, S, T, H, D]`` tensors."""
    with np.load(path, allow_pickle=False) as archive:
        for name in ("keys", "values"):
            if name not in archive.files:
                raise TraceError(f"{path} is missing the {name!r} array")
        keys = np.array(archive["keys"], copy=True)
        values = np.array(archive["values"], copy=True)
        if keys.shape != values.shape:
            raise TraceError(f"{path} K and V shapes disagree")
        if keys.ndim != 5:
            raise TraceError(
                f"{path} must use the flat [L, S, T, H, D] layout, got "
                f"{keys.shape}"
            )
        layout = (
            str(np.asarray(archive["trace_layout"]).item())
            if "trace_layout" in archive.files
            else "tempokv_flat_samples_v1"
        )
        if layout != "tempokv_flat_samples_v1":
            raise TraceError(
                f"{path} uses layout {layout!r}; convert it with prepare_trace "
                "without --legacy-protocol"
            )
        coordinates = (
            str(np.asarray(archive["coordinate_system"]).item())
            if "coordinate_system" in archive.files
            else ""
        )
        if coordinates and not coordinates.startswith("pre_rope"):
            raise TraceError(
                f"{path} stores {coordinates!r} K; the ladder requires pre-RoPE K"
            )
        events = _int_field(archive, "trace_event_count", 1)
        token_indices = (
            np.asarray(archive["token_indices"]).tolist()
            if "token_indices" in archive.files
            else []
        )
        layer_ids = (
            np.asarray(archive["layer_ids"]).tolist()
            if "layer_ids" in archive.files
            else list(range(keys.shape[0]))
        )
    return {
        "path": path,
        "keys": torch.from_numpy(keys),
        "values": torch.from_numpy(values),
        "events": events,
        "token_indices": token_indices,
        "layer_ids": layer_ids,
    }


def event_slice(trace: dict[str, Any], first: int, last: int) -> dict[str, Any]:
    """Select whole events, which is the finest prompt-disjoint cut available."""
    events = int(trace["events"])
    samples = int(trace["keys"].shape[1])
    if samples % events:
        raise TraceError(
            f"{samples} samples do not divide into {events} events; the "
            "archive cannot be split on a prompt boundary"
        )
    per_event = samples // events
    if not 0 <= first < last <= events:
        raise TraceError(f"event range [{first}, {last}) is outside 0..{events}")
    start, stop = first * per_event, last * per_event
    selected = dict(trace)
    selected["keys"] = trace["keys"][:, start:stop]
    selected["values"] = trace["values"][:, start:stop]
    selected["events"] = last - first
    return selected


def select_axes(
    trace: dict[str, Any], layers: Sequence[int] | None, heads: Sequence[int] | None
) -> dict[str, Any]:
    selected = dict(trace)
    if layers is not None:
        index = torch.as_tensor(list(layers), dtype=torch.long)
        selected["keys"] = selected["keys"].index_select(0, index)
        selected["values"] = selected["values"].index_select(0, index)
        selected["layer_ids"] = [int(trace["layer_ids"][i]) for i in layers]
    if heads is not None:
        index = torch.as_tensor(list(heads), dtype=torch.long)
        selected["keys"] = selected["keys"].index_select(3, index)
        selected["values"] = selected["values"].index_select(3, index)
    return selected


def _parse_indices(text: str | None, limit: int, name: str) -> list[int] | None:
    if text is None or text.strip().lower() == "all":
        return None
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if not 0 <= value < limit:
            raise ValueError(f"{name} index {value} is outside 0..{limit - 1}")
        values.append(value)
    if not values:
        raise ValueError(f"no {name} indices were selected")
    return values


# ---------------------------------------------------------------------------
# the ladder for one (layer, head)
# ---------------------------------------------------------------------------


class CodecSettings:
    """The closed-loop quantizer settings, read from a codec config."""

    def __init__(
        self,
        *,
        anchor_bits: int,
        k_bits: int,
        v_bits: int,
        group_size: int,
        residual_mode: str,
        scale_dtype: str,
    ) -> None:
        self.anchor_bits = anchor_bits
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_mode = residual_mode
        self.scale_dtype = _SCALE_DTYPES[scale_dtype]

    @property
    def symmetric_residual(self) -> bool:
        return self.residual_mode == "symmetric"

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_bits": self.anchor_bits,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "group_size": self.group_size,
            "residual_mode": self.residual_mode,
        }


def head_ladder(
    calibration_keys: torch.Tensor,
    calibration_values: torch.Tensor,
    evaluation_keys: torch.Tensor,
    evaluation_values: torch.Tensor,
    *,
    k_order: int,
    v_order: int,
    ridge: float,
    codec: CodecSettings,
    chunk: int,
    closed_loop: bool,
) -> dict[str, dict[str, TierStats]]:
    """Every tier of one (layer, head), on ``[S, T, D]`` inputs.

    All tiers share the warm-up ``max(k_order, v_order)`` so they are measured
    on identical rows against one denominator; without that the tier deltas
    would mix "what the term earned" with "how many leading frames the order
    happened to cost".
    """
    warmup = max(k_order, v_order, 1)
    group = codec.group_size

    key_mean = channel_mean(calibration_keys)
    value_mean = channel_mean(calibration_values)
    calibration_keys_centered = calibration_keys - key_mean
    calibration_values_centered = calibration_values - value_mean
    evaluation_keys_centered = evaluation_keys - key_mean
    evaluation_values_centered = evaluation_values - value_mean

    key_ar, key_bias = fit_ar(
        calibration_keys_centered, k_order, warmup=warmup, ridge=ridge, chunk=chunk
    )
    value_ar, value_bias = fit_ar(
        calibration_values_centered, v_order, warmup=warmup, ridge=ridge, chunk=chunk
    )
    cross_ar, cross_projection, cross_bias = fit_cross(
        calibration_values_centered,
        calibration_keys_centered,
        v_order,
        warmup=warmup,
        ridge=ridge,
        chunk=chunk,
    )

    key_reference = build_lags(evaluation_keys, 0, warmup=warmup)[1]
    value_reference = build_lags(evaluation_values, 0, warmup=warmup)[1]
    key_lags, key_target = build_lags(
        evaluation_keys_centered, k_order, warmup=warmup
    )
    value_lags, value_target = build_lags(
        evaluation_values_centered, v_order, warmup=warmup
    )
    current_keys = evaluation_keys_centered[:, warmup:].reshape(
        -1, evaluation_keys.shape[-1]
    )

    def stats(residual: torch.Tensor, reference: torch.Tensor) -> TierStats:
        return residual_stats(residual, reference, group_size=group)

    results: dict[str, dict[str, TierStats]] = {"clean": {}}
    clean = results["clean"]
    clean["K0"] = stats(key_reference, key_reference)
    clean["K1"] = stats(key_target, key_reference)
    clean["K2"] = stats(
        key_target - predict_ar(key_lags, key_ar, key_bias), key_reference
    )
    clean["V0"] = stats(value_reference, value_reference)
    clean["V1"] = stats(value_target, value_reference)
    clean["V2"] = stats(
        value_target - predict_ar(value_lags, value_ar, value_bias), value_reference
    )
    clean["V3"] = stats(
        value_target
        - predict_cross(
            value_lags, current_keys, cross_ar, cross_projection, cross_bias
        ),
        value_reference,
    )
    if not closed_loop:
        return results

    rebuilt_keys = closed_loop_rollout(
        evaluation_keys,
        ar_step(key_ar, key_bias, key_mean),
        anchor_bits=codec.anchor_bits,
        residual_bits=codec.k_bits,
        group_size=group,
        symmetric_residual=codec.symmetric_residual,
        scale_dtype=codec.scale_dtype,
    )
    rebuilt_values_temporal = closed_loop_rollout(
        evaluation_values,
        ar_step(value_ar, value_bias, value_mean),
        anchor_bits=codec.anchor_bits,
        residual_bits=codec.v_bits,
        group_size=group,
        symmetric_residual=codec.symmetric_residual,
        scale_dtype=codec.scale_dtype,
    )
    rebuilt_values_cross = closed_loop_rollout(
        evaluation_values,
        cross_step(
            cross_ar,
            cross_projection,
            cross_bias,
            value_mean,
            rebuilt_keys,
            key_mean,
        ),
        anchor_bits=codec.anchor_bits,
        residual_bits=codec.v_bits,
        group_size=group,
        symmetric_residual=codec.symmetric_residual,
        scale_dtype=codec.scale_dtype,
    )

    rebuilt_key_lags = build_lags(rebuilt_keys - key_mean, k_order, warmup=warmup)[0]
    rebuilt_value_lags = build_lags(
        rebuilt_values_temporal - value_mean, v_order, warmup=warmup
    )[0]
    rebuilt_cross_lags = build_lags(
        rebuilt_values_cross - value_mean, v_order, warmup=warmup
    )[0]
    rebuilt_current_keys = (rebuilt_keys - key_mean)[:, warmup:].reshape(
        -1, evaluation_keys.shape[-1]
    )

    # Tiers 0 and 1 hold no history, so the decoder sees exactly what the
    # clean probe sees and the loop cannot move them.  Copying the clean cell
    # states that rather than paying for a rollout that provably matches.
    results["closed_loop"] = {
        "K0": clean["K0"],
        "K1": clean["K1"],
        "K2": stats(
            key_target - predict_ar(rebuilt_key_lags, key_ar, key_bias),
            key_reference,
        ),
        "V0": clean["V0"],
        "V1": clean["V1"],
        "V2": stats(
            value_target - predict_ar(rebuilt_value_lags, value_ar, value_bias),
            value_reference,
        ),
        "V3": stats(
            value_target
            - predict_cross(
                rebuilt_cross_lags,
                rebuilt_current_keys,
                cross_ar,
                cross_projection,
                cross_bias,
            ),
            value_reference,
        ),
    }
    return results


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def pool(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Energy-weighted pooled ratios plus median-of-heads range statistics.

    Energy pools exactly because it is a ratio of sums.  ``bits_saved`` and
    ``range_over_std`` are medians over per-group values, which do not pool,
    so the reported figure is the median across heads of each head's median
    and is named that way in the output.
    """
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(
            (row["loop"], row["tier"]),
            {"energy": [], "base": [], "bits": [], "tail": []},
        )
        bucket["energy"].append(row["energy_ratio"] * row["reference_energy"])
        bucket["base"].append(row["reference_energy"])
        bucket["bits"].append(row["bits_saved"])
        bucket["tail"].append(row["range_over_std"])
    pooled: dict[str, dict[str, Any]] = {}
    for (loop, tier), bucket in grouped.items():
        base = sum(bucket["base"])
        ratio = sum(bucket["energy"]) / base if base > 0 else float("nan")
        pooled.setdefault(loop, {})[tier] = {
            "energy_ratio": ratio,
            "energy_equivalent_bits": (
                0.5 * math.log2(1.0 / ratio) if ratio > 0 else float("inf")
            ),
            "bits_saved_median_over_heads": float(
                np.median(np.asarray(bucket["bits"]))
            ),
            "range_over_std_median_over_heads": float(
                np.median(np.asarray(bucket["tail"]))
            ),
            "heads": len(bucket["base"]),
        }
    return pooled


def ladder_shares(pooled_loop: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Fig 3(a) segments: each term's share of the original energy."""

    def ratio(tier: str) -> float:
        return float(pooled_loop[tier]["energy_ratio"])

    key = {
        "dc": 1.0 - ratio("K1"),
        "temporal": ratio("K1") - ratio("K2"),
        "cross": 0.0,
        "residual": ratio("K2"),
    }
    value = {
        "dc": 1.0 - ratio("V1"),
        "temporal": ratio("V1") - ratio("V2"),
        "cross": ratio("V2") - ratio("V3"),
        "residual": ratio("V3"),
    }
    return {"k": key, "v": value}


def tier_gains(pooled_loop: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Energy-reduction factors, the numbers the paper text quotes."""

    def ratio(tier: str) -> float:
        return float(pooled_loop[tier]["energy_ratio"])

    return {
        "k_tier1_over_tier2": ratio("K1") / ratio("K2") if ratio("K2") > 0 else float("inf"),
        "v_tier1_over_tier2": ratio("V1") / ratio("V2") if ratio("V2") > 0 else float("inf"),
        "v_tier2_over_tier3": ratio("V2") / ratio("V3") if ratio("V3") > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _load_codec(path: Path | None) -> CodecSettings:
    if path is None:
        return CodecSettings(
            anchor_bits=4,
            k_bits=2,
            v_bits=2,
            group_size=64,
            residual_mode="asymmetric",
            scale_dtype="float16",
        )
    from tempokv.config import config_from_mapping

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "codec" in payload:
        payload = payload["codec"]
    config = config_from_mapping(payload)
    if config.raw_anchor:
        raise SystemExit(
            "this study quantizes the anchor with its own standalone "
            "quantizer, so it cannot model anchor_bits=16, which stores the "
            "anchor unquantized; use a quantized anchor width"
        )
    return CodecSettings(
        anchor_bits=config.anchor_bits,
        k_bits=config.k_bits,
        v_bits=config.v_bits,
        group_size=config.group_size,
        residual_mode=config.residual_mode,
        scale_dtype=config.scale_dtype,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument(
        "--held-out-trace",
        type=Path,
        help="separate held-out archive; omit it and use --split-events instead",
    )
    parser.add_argument(
        "--split-events",
        type=int,
        help=(
            "split one archive at this event index: events before it calibrate, "
            "the rest are held out"
        ),
    )
    parser.add_argument("--layers", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--heads", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--k-order", type=int, default=2)
    parser.add_argument("--v-order", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--chunk", type=int, default=64, help="samples per Gram chunk")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--config",
        type=Path,
        help="codec config supplying the closed-loop bit widths (default INT4/INT2)",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="skip the closed-loop pass (the open-loop probe alone)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/observation/decompose.json")
    )
    return parser


def resolve_split(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    calibration = load_trace(args.calibration_trace)
    if args.held_out_trace is not None and args.split_events is not None:
        raise TraceError("pass either --held-out-trace or --split-events, not both")
    if args.held_out_trace is not None:
        if args.held_out_trace.resolve() == args.calibration_trace.resolve():
            raise TraceError(
                "the held-out archive must not be the calibration archive; "
                "the ladder is only meaningful on disjoint prompts"
            )
        held_out = load_trace(args.held_out_trace)
        # Pooling across prompts assumes every archive sampled the same
        # spatial positions.  Two runs collected with different
        # TRACE_TOKENS_PER_FRAME settings would otherwise be averaged as if
        # they described the same tokens.
        if held_out["token_indices"] != calibration["token_indices"]:
            raise TraceError(
                "the two archives sampled different spatial positions; "
                "collect both with the same TRACE_TOKENS_PER_FRAME"
            )
        provenance = {
            "split": "separate_archives",
            "calibration_trace": str(args.calibration_trace),
            "held_out_trace": str(args.held_out_trace),
        }
        return calibration, held_out, provenance
    if args.split_events is None:
        raise TraceError(
            "held-out prompts are required: pass --held-out-trace or --split-events"
        )
    held_out = event_slice(calibration, args.split_events, calibration["events"])
    calibration = event_slice(calibration, 0, args.split_events)
    provenance = {
        "split": "event_boundary",
        "calibration_trace": str(args.calibration_trace),
        "held_out_trace": str(args.calibration_trace),
        "split_events": args.split_events,
    }
    return calibration, held_out, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        calibration, held_out, provenance = resolve_split(args)
    except (OSError, TraceError) as error:
        parser.error(str(error))
    layer_count, _, frames, head_count, dim = calibration["keys"].shape
    if held_out["keys"].shape[0] != layer_count or held_out["keys"].shape[3] != head_count:
        parser.error("calibration and held-out traces disagree on layers or heads")
    try:
        layers = _parse_indices(args.layers, layer_count, "layer")
        heads = _parse_indices(args.heads, head_count, "head")
        codec = _load_codec(args.config)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    calibration = select_axes(calibration, layers, heads)
    held_out = select_axes(held_out, layers, heads)
    device = torch.device(args.device)

    rows: list[dict[str, Any]] = []
    selected_layers = calibration["keys"].shape[0]
    selected_heads = calibration["keys"].shape[3]
    warmup = max(args.k_order, args.v_order, 1)

    def head_slice(trace: dict[str, Any], name: str, layer: int, head: int):
        return (
            trace[name][layer, :, :, head, :]
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )

    for layer in range(selected_layers):
        for head in range(selected_heads):
            held_out_keys = head_slice(held_out, "keys", layer, head)
            held_out_values = head_slice(held_out, "values", layer, head)
            ladder = head_ladder(
                head_slice(calibration, "keys", layer, head),
                head_slice(calibration, "values", layer, head),
                held_out_keys,
                held_out_values,
                k_order=args.k_order,
                v_order=args.v_order,
                ridge=args.ridge,
                codec=codec,
                chunk=args.chunk,
                closed_loop=not args.clean_only,
            )
            # The tier-0 denominator: held-out energy on the rows every tier
            # is scored on, so the pooled ratio stays a ratio of sums.
            key_energy = float(
                held_out_keys[:, warmup:].double().pow(2).sum().item()
            )
            value_energy = float(
                held_out_values[:, warmup:].double().pow(2).sum().item()
            )
            for loop, tiers in ladder.items():
                for tier, stat in tiers.items():
                    rows.append(
                        {
                            "layer": int(calibration["layer_ids"][layer]),
                            "head": head if heads is None else int(heads[head]),
                            "loop": loop,
                            "tier": tier,
                            "reference_energy": (
                                key_energy if tier.startswith("K") else value_energy
                            ),
                            **stat.as_dict(),
                        }
                    )

    pooled = pool(rows)
    report = {
        "schema": SCHEMA,
        "settings": {
            "k_order": args.k_order,
            "v_order": args.v_order,
            "warmup_frames": max(args.k_order, args.v_order, 1),
            "ridge": args.ridge,
            "layers": calibration["layer_ids"],
            "heads": heads if heads is not None else list(range(selected_heads)),
            "codec": codec.as_dict(),
            "loops": ["clean"] if args.clean_only else list(LOOPS),
            **provenance,
        },
        "trace": {
            "frames": int(frames),
            "head_dim": int(dim),
            "calibration_samples": int(calibration["keys"].shape[1]),
            "held_out_samples": int(held_out["keys"].shape[1]),
            "calibration_events": int(calibration["events"]),
            "held_out_events": int(held_out["events"]),
        },
        "rows": rows,
        "pooled": pooled,
        "ladder_shares": {loop: ladder_shares(pooled[loop]) for loop in pooled},
        "tier_gains": {loop: tier_gains(pooled[loop]) for loop in pooled},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(render_table(report))
    return 0


def render_table(report: dict[str, Any]) -> str:
    """The section 3.2 table, printed for a quick read of a finished run."""
    lines = [f"trace: {report['trace']}", ""]
    for loop, tiers in report["pooled"].items():
        lines.append(f"--- {loop} ---")
        lines.append(
            f"{'tier':6s} {'energy/orig':>12s} {'equiv bits':>11s} "
            f"{'bits saved':>11s} {'range/std':>10s}"
        )
        for tier in TIERS:
            if tier not in tiers:
                continue
            cell = tiers[tier]
            lines.append(
                f"{tier:6s} {cell['energy_ratio']:12.4f} "
                f"{cell['energy_equivalent_bits']:11.3f} "
                f"{cell['bits_saved_median_over_heads']:11.3f} "
                f"{cell['range_over_std_median_over_heads']:10.3f}"
            )
        gains = report["tier_gains"][loop]
        lines.append(
            "gain  K tier1->2 {k:.3f}x  V tier1->2 {v12:.3f}x  "
            "V tier2->3 {v23:.3f}x".format(
                k=gains["k_tier1_over_tier2"],
                v12=gains["v_tier1_over_tier2"],
                v23=gains["v_tier2_over_tier3"],
            )
        )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
