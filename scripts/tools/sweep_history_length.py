#!/usr/bin/env python3
"""Sensitivity of TempoKV quality to the predictor history length.

Driven per model by `bash scripts/<model>/sweep_history.sh`, which supplies
that model's trace, config, and report path. Called directly it falls back to
Self-Forcing's paths.

Sweeps the number of reconstructed history frames the predictors may use and,
at each setting, reports the three quantities that decide whether a longer
history is worth its parameter cost:

  * **reconstruction error** — ``||rebuilt - clean||^2 / ||clean||^2``, the
    end-to-end closed-loop error of the cache the model will actually read;
  * **attention output error** — the same reconstruction pushed through causal
    softmax attention, which is what the error finally does to the model;
  * **prediction share** — how much of the signal the predictor explains, i.e.
    ``1 - ||clean - prediction||^2 / ||clean||^2``. Its complement is the
    residual the quantizer has to spend bits on, so a larger share at the same
    bit width is exactly what buys quality.

Two sweeps are run:

``K``
    ``k_history_length`` = N, using N reconstructed K frames. ``N=0`` is the
    direct-quantization baseline. Dedicated (non shared) parameters are fitted
    per positive length, so **each N is refitted from the trace** — a parameter
    file fitted for one length cannot be re-pointed at another, and reusing one
    would measure the wrong model.

``V``
    ``v_history_length`` = M *on top of* the CrossKV term. V is always predicted
    from the current reconstructed K; M adds reconstructed V history to that.
    The report therefore also breaks the V prediction into its CrossKV term and
    its history term, so a sweep can show what the history adds rather than
    only that the total improved. The two terms are correlated, so their
    energies are reported side by side and do not sum to the total.

Everything is measured on the clean trace with the repo's own fitter and the
same closed-loop reconstruction the codec performs, so the numbers correspond
to a deployed run rather than to an open-loop approximation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from analyze_predictor import (  # noqa: E402
    attention_output_diff,
    energy_ratio,
    reconstruct_k,
    reconstruct_temporal,
    reconstruct_v,
)
from tempokv.config import (  # noqa: E402
    FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY,
    CodecConfig,
    config_from_mapping,
    fixed_temporal_predictor_mode,
    is_cross_kv_v_predictor,
    is_temporal_v_predictor,
)
from tempokv.fitting import fit_predictors  # noqa: E402

K_HISTORY_CHOICES = tuple(range(0, 9))
V_HISTORY_CHOICES = tuple(range(0, 6))


def _load_config(path: Path) -> CodecConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "codec" in payload:
        if len(payload) != 1:
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    return config_from_mapping(payload)


def _int_list(raw: str, *, name: str, allowed: tuple[int, ...]) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{name} must be a comma-separated list of integers, got {item!r}"
            ) from error
        if value not in allowed:
            raise argparse.ArgumentTypeError(
                f"{name}={value} is outside the supported range {allowed}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must name at least one value")
    return tuple(sorted(values))


def load_trace(
    path: Path,
    *,
    max_samples: int | None,
    max_frames: int | None,
    layers: tuple[int, ...] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a clean trace as ``[L, S, T, H, D]`` float32 K and V."""
    with np.load(path, allow_pickle=False) as archive:
        keys = torch.from_numpy(np.array(archive["keys"], copy=True))
        values = torch.from_numpy(np.array(archive["values"], copy=True))
    if keys.shape != values.shape or keys.ndim != 5:
        raise ValueError(
            f"trace must hold matching [L,S,T,H,D] K/V, got {tuple(keys.shape)} "
            f"and {tuple(values.shape)}"
        )
    if layers is not None:
        index = torch.tensor(layers, dtype=torch.long)
        if int(index.max()) >= keys.shape[0]:
            raise ValueError(f"trace has {keys.shape[0]} layers; --layers is out of range")
        keys = keys.index_select(0, index)
        values = values.index_select(0, index)
    if max_samples is not None and max_samples < keys.shape[1]:
        keys = keys[:, :max_samples]
        values = values[:, :max_samples]
    if max_frames is not None and max_frames < keys.shape[2]:
        keys = keys[:, :, :max_frames]
        values = values[:, :, :max_frames]
    return keys.to(device=device, dtype=torch.float32), values.to(
        device=device, dtype=torch.float32
    )


def measure(
    keys: torch.Tensor,
    values: torch.Tensor,
    config: CodecConfig,
    *,
    iterations: int,
    temporal_ridge: float,
    cross_ridge: float,
    max_gain: float,
) -> dict[str, object]:
    """Fit predictors for one configuration and measure what they achieve."""
    started = time.perf_counter()
    temporal, cross = fit_predictors(
        keys,
        values,
        config=config,
        iterations=iterations,
        temporal_ridge=temporal_ridge,
        cross_ridge=cross_ridge,
        max_gain=max_gain,
        v_history_mode="diagonal",
        coordinate_system="pre_rope_normalized",
    )
    fit_seconds = time.perf_counter() - started

    k_pred, k_res, k_rebuilt = reconstruct_k(keys, temporal, config)
    temporal_v = is_temporal_v_predictor(config.v_predictor_mode)
    if temporal_v:
        v_pred, v_res, v_rebuilt = reconstruct_temporal(
            values, cross, config, bits=config.v_bits
        )
        v_cross = v_hist = None
        v_rebuilt_cross_only = v_rebuilt_history_only = None
    else:
        v_pred, v_res, v_rebuilt, v_cross, v_hist = reconstruct_v(
            keys, values, k_rebuilt, cross, config
        )
        # Ablation reconstructions: drop one V term and run the full quantization
        # round-trip again, so each reduced predictor also reports a closed-loop
        # reconstruction error, not just its residual.
        _, _, v_rebuilt_cross_only, _, _ = reconstruct_v(
            keys, values, k_rebuilt, cross, config, use_history=False
        )
        _, _, v_rebuilt_history_only, _, _ = reconstruct_v(
            keys, values, k_rebuilt, cross, config, use_cross=False
        )
    attention_error, _ = attention_output_diff(keys, values, k_rebuilt, v_rebuilt)

    # Frame 0 is the anchor; it has no predictor, so prediction statistics are
    # taken over the predicted frames only. Reconstruction error covers every
    # frame, anchor included, because that is what the model reads.
    later = slice(1, None)
    k_residual = energy_ratio(k_res[:, :, later], keys[:, :, later])
    v_residual = energy_ratio(v_res[:, :, later], values[:, :, later])

    if temporal_v:
        v_residual_cross_only = None
        v_residual_history_only = v_residual
        v_residual_bias_only = None
    else:
        # Leave-one-out ablation of the two V prediction terms. Both are dropped
        # from the *jointly fitted* model rather than refitted without each other,
        # so these say what each term contributes to this predictor, not what a
        # predictor trained without it would achieve. The terms are correlated, so
        # the two residuals do not decompose the full one.
        v_without_history = values[:, :, later] - (
            v_pred[:, :, later] - v_hist[:, :, later]
        )
        v_without_cross = values[:, :, later] - (
            v_pred[:, :, later] - v_cross[:, :, later]
        )
        v_without_terms = values[:, :, later] - (
            v_pred[:, :, later] - v_cross[:, :, later] - v_hist[:, :, later]
        )
        v_residual_cross_only = energy_ratio(v_without_history, values[:, :, later])
        v_residual_history_only = energy_ratio(v_without_cross, values[:, :, later])
        # Pure-bias baseline with both terms dropped: what the V history and the
        # CrossKV term each buy *from nothing*, not just on top of the other term.
        v_residual_bias_only = energy_ratio(v_without_terms, values[:, :, later])
    return {
        "k_history_length": temporal.history_length,
        "v_history_length": cross.history_length,
        "fit_seconds": fit_seconds,
        "k": {
            "prediction_share": 1.0 - k_residual,
            "residual_energy_ratio": k_residual,
            "prediction_energy_ratio": energy_ratio(
                k_pred[:, :, later], keys[:, :, later]
            ),
            "reconstruction_error": energy_ratio(k_rebuilt - keys, keys),
        },
        "v": {
            "prediction_share": 1.0 - v_residual,
            "residual_energy_ratio": v_residual,
            "prediction_energy_ratio": energy_ratio(
                v_pred[:, :, later], values[:, :, later]
            ),
            "reconstruction_error": energy_ratio(v_rebuilt - values, values),
            # Descriptive split of the CrossKV prediction. Temporal-only V has
            # no CrossKV term, so these fields are intentionally null there.
            "cross_term_energy_ratio": (
                None
                if temporal_v
                else energy_ratio(v_cross[:, :, later], values[:, :, later])
            ),
            "history_term_energy_ratio": (
                None
                if temporal_v
                else energy_ratio(v_hist[:, :, later], values[:, :, later])
            ),
            # Ablation: residual left when one term is removed from the fitted
            # predictor. Compare against `residual_energy_ratio` (both terms).
            "residual_cross_only": v_residual_cross_only,
            "residual_history_only": v_residual_history_only,
            "cross_term_gain": (
                None
                if temporal_v
                else v_residual_history_only - v_residual
            ),
            "history_term_gain": (
                None
                if temporal_v
                else v_residual_cross_only - v_residual
            ),
            # Closed-loop reconstruction error of the reduced predictors: drop
            # one V term and rerun the full quantization round-trip. These are
            # the end-to-end errors the model would actually read under that
            # ablation, unlike the residuals above.
            "reconstruction_cross_only": (
                None
                if temporal_v
                else energy_ratio(v_rebuilt_cross_only - values, values)
            ),
            "reconstruction_history_only": (
                energy_ratio(v_rebuilt - values, values)
                if temporal_v
                else energy_ratio(v_rebuilt_history_only - values, values)
            ),
            # Pure-bias baseline (both terms dropped) and each term's gain from
            # nothing. `cross_term_gain_vs_bias` is what CrossKV buys alone;
            # `history_term_gain_vs_bias` is what V history buys alone.
            "residual_bias_only": v_residual_bias_only,
            "cross_term_gain_vs_bias": (
                None
                if temporal_v
                else v_residual_bias_only - v_residual_cross_only
            ),
            "history_term_gain_vs_bias": (
                None
                if temporal_v
                else v_residual_bias_only - v_residual_history_only
            ),
        },
        "attention_output_error": attention_error,
        "v_predictor_mode": config.v_predictor_mode,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=PROJECT_ROOT / "data" / "self-forcing" / "trace" / "clean_trace.npz",
        help="clean trace from collect_trace.sh (default: Self-Forcing's)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "int4.json",
        help="codec config supplying the bit widths held fixed across the sweep",
    )
    parser.add_argument(
        "--k-history",
        default="0,1,2,3,4,5",
        help=f"K history lengths to sweep, from {K_HISTORY_CHOICES}",
    )
    parser.add_argument(
        "--v-history",
        default="0,1,2,4,5",
        help=f"V history lengths to sweep, from {V_HISTORY_CHOICES}",
    )
    parser.add_argument(
        "--k-bits",
        type=int,
        choices=range(1, 9),
        help="override K residual bit width while keeping the config family",
    )
    parser.add_argument(
        "--v-bits",
        type=int,
        choices=range(1, 9),
        help="override V residual bit width while keeping the config family",
    )
    parser.add_argument(
        "--v-predictor-mode",
        choices=(
            "cross_kv",
            "temporal",
            "identity",
            "taylor",
            "taylor2",
            "cross_kv_identity",
            "cross_kv_taylor",
            "cross_kv_taylor2",
            "innovation",
            "direct",
        ),
        help=(
            "V predictor family; temporal/identity/taylor/taylor2 use V's own "
            "history, while cross_kv/innovation use a K-derived source"
        ),
    )
    parser.add_argument(
        "--baseline-k-history",
        type=int,
        help="K length held fixed while V is swept (default: the largest swept K)",
    )
    parser.add_argument(
        "--baseline-v-history",
        type=int,
        help="V length held fixed while K is swept (default: the config's value)",
    )
    parser.add_argument("--max-samples", type=int, help="subsample spatial positions")
    parser.add_argument("--max-frames", type=int, help="truncate the temporal axis")
    parser.add_argument("--layers", help="comma-separated layer indices to include")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--temporal-ridge", type=float, default=0.1)
    parser.add_argument("--cross-ridge", type=float, default=0.1)
    parser.add_argument("--max-gain", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _format_table(title: str, swept: str, rows: list[dict[str, object]]) -> str:
    lines = [
        "",
        f"--- {title} ---",
        f"{swept:>3}  {'K recon':>9} {'V recon':>9} {'attn out':>9} "
        f"{'K pred%':>8} {'V pred%':>8}  {'fit s':>6}",
    ]
    for row in rows:
        lines.append(
            f"{row['swept_value']:>3}  "
            f"{row['k']['reconstruction_error']:9.5f} "
            f"{row['v']['reconstruction_error']:9.5f} "
            f"{row['attention_output_error']:9.5f} "
            f"{100 * row['k']['prediction_share']:7.2f}% "
            f"{100 * row['v']['prediction_share']:7.2f}%  "
            f"{row['fit_seconds']:6.1f}"
        )
    return "\n".join(lines)


def _format_v_terms(rows: list[dict[str, object]]) -> str:
    def value_text(value: object, width: int) -> str:
        return "n/a".rjust(width) if value is None else f"{float(value):{width}.5f}"

    lines = [
        "",
        "--- V prediction terms (energy relative to clean V; correlated, not a partition) ---",
        f"{'vh':>3}  {'CrossKV term':>13} {'history term':>13} {'V residual':>11}",
    ]
    for row in rows:
        lines.append(
            f"{row['swept_value']:>3}  "
            f"{value_text(row['v']['cross_term_energy_ratio'], 13)} "
            f"{value_text(row['v']['history_term_energy_ratio'], 13)} "
            f"{value_text(row['v']['residual_energy_ratio'], 11)}"
        )
    return "\n".join(lines)


def _format_v_ablation(rows: list[dict[str, object]]) -> str:
    """Residual and reconstruction left when one term is dropped from the
    fitted V predictor."""
    def value_text(value: object, width: int) -> str:
        return "n/a".rjust(width) if value is None else f"{float(value):{width}.5f}"

    lines = [
        "",
        "--- V term ablation (residual energy and closed-loop reconstruction; "
        "lower is better) ---",
        f"{'vh':>3}  {'recon(both)':>12} {'recon(cross)':>13} "
        f"{'recon(hist)':>12} {'resid(cross)':>12} {'resid(hist)':>12}",
    ]
    for row in rows:
        v = row["v"]
        lines.append(
            f"{row['swept_value']:>3}  "
            f"{value_text(v['reconstruction_error'], 12)} "
            f"{value_text(v['reconstruction_cross_only'], 13)} "
            f"{value_text(v['reconstruction_history_only'], 12)} "
            f"{value_text(v['residual_cross_only'], 12)} "
            f"{value_text(v['residual_history_only'], 12)}"
        )
    lines.append(
        "    recon(both) = full predictor (history+cross+bias) closed-loop "
        "reconstruction error; recon(cross) = cross+bias (no history); "
        "recon(hist) = history+bias (no cross). resid(*) are the "
        "corresponding residuals."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        k_values = _int_list(
            args.k_history, name="--k-history", allowed=K_HISTORY_CHOICES
        )
        v_values = _int_list(
            args.v_history, name="--v-history", allowed=V_HISTORY_CHOICES
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    layers = (
        tuple(int(item) for item in args.layers.split(",") if item.strip())
        if args.layers
        else None
    )
    device = _resolve_device(args.device)
    base = _load_config(args.config)
    if base.fit_mode != "legacy":
        parser.error("history sweep requires fit_mode=legacy")
    if base.cross_kv_weight_mode != "layer_shared":
        parser.error("history sweep requires cross_kv_weight_mode=layer_shared")
    config_updates = {}
    if args.k_bits is not None:
        config_updates["k_bits"] = args.k_bits
    if args.v_bits is not None:
        config_updates["v_bits"] = args.v_bits
    if args.v_predictor_mode is not None:
        config_updates["v_predictor_mode"] = args.v_predictor_mode
        if is_cross_kv_v_predictor(args.v_predictor_mode) or (
            is_temporal_v_predictor(args.v_predictor_mode)
            and args.v_predictor_mode != "direct"
        ):
            config_updates["cross_kv_weight_mode"] = "layer_shared"
    if config_updates:
        base = replace(base, **config_updates)
    fixed_mode = fixed_temporal_predictor_mode(base.v_predictor_mode)
    if base.v_predictor_mode != "direct" and (
        is_temporal_v_predictor(base.v_predictor_mode)
        or fixed_mode is not None
    ):
        minimum_history = FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY.get(fixed_mode, 1)
        if any(value < minimum_history for value in v_values):
            parser.error(
                f"{base.v_predictor_mode} V requires v-history values >= "
                f"{minimum_history}"
            )

    keys, values = load_trace(
        args.trace,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
        layers=layers,
        device=device,
    )
    frames = int(keys.shape[2])
    predicted_frames = frames - 1
    if predicted_frames < 1:
        parser.error(f"the trace has {frames} frame(s); a sweep needs at least 2")

    # A history longer than the trace can never be exercised: the fitter caps
    # the order at min(8, T-1) and refuses anything longer.
    skipped = {
        "k": [n for n in k_values if n > predicted_frames],
        "v": [m for m in v_values if m > predicted_frames],
    }
    k_values = tuple(n for n in k_values if n <= predicted_frames)
    v_values = tuple(m for m in v_values if m <= predicted_frames)
    if not k_values or not v_values:
        parser.error(
            f"the trace only has {predicted_frames} predicted frame(s); nothing "
            "in the requested sweep can be exercised"
        )

    # Baselines are chosen from the surviving lengths, so the sweep never holds
    # one axis at a length the trace cannot fit.
    minimum_v_history = FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY.get(fixed_mode, 0)
    baseline_v = (
        args.baseline_v_history
        if args.baseline_v_history is not None
        else max(min(base.v_history_length, predicted_frames), minimum_v_history)
    )
    baseline_k = (
        args.baseline_k_history if args.baseline_k_history is not None else max(k_values)
    )
    if baseline_v not in V_HISTORY_CHOICES:
        parser.error(f"--baseline-v-history must be one of {V_HISTORY_CHOICES}")
    if baseline_k not in K_HISTORY_CHOICES:
        parser.error(f"--baseline-k-history must be one of {K_HISTORY_CHOICES}")
    for name, value in (("--baseline-k-history", baseline_k), ("--baseline-v-history", baseline_v)):
        if value > predicted_frames:
            parser.error(
                f"{name}={value} exceeds the {predicted_frames} predicted "
                f"frame(s) this trace holds"
            )
    for axis, dropped in skipped.items():
        if dropped:
            print(
                f"[sweep] skipping {axis}-history {dropped}: the trace has only "
                f"{predicted_frames} predicted frames",
                file=sys.stderr,
            )

    common = {
        "iterations": args.iterations,
        "temporal_ridge": args.temporal_ridge,
        "cross_ridge": args.cross_ridge,
        "max_gain": args.max_gain,
    }

    k_rows: list[dict[str, object]] = []
    for value in k_values:
        print(f"[sweep] K history {value} (V held at {baseline_v}) ...", file=sys.stderr)
        k_predictor_mode = (
            "direct"
            if value == 0
            else (
                "shared"
                if base.k_predictor_mode == "direct"
                else base.k_predictor_mode
            )
        )
        config = replace(
            base,
            k_history_length=value,
            v_history_length=baseline_v,
            k_predictor_mode=k_predictor_mode,
        )
        record = measure(keys, values, config, **common)
        record["swept_value"] = value
        k_rows.append(record)

    v_rows: list[dict[str, object]] = []
    for value in v_values:
        print(f"[sweep] V history {value} (K held at {baseline_k}) ...", file=sys.stderr)
        k_predictor_mode = (
            "direct"
            if baseline_k == 0
            else (
                "shared"
                if base.k_predictor_mode == "direct"
                else base.k_predictor_mode
            )
        )
        config = replace(
            base,
            k_history_length=baseline_k,
            v_history_length=value,
            k_predictor_mode=k_predictor_mode,
        )
        record = measure(keys, values, config, **common)
        record["swept_value"] = value
        v_rows.append(record)

    report = {
        "trace": {
            "path": str(args.trace),
            "layers": int(keys.shape[0]),
            "samples": int(keys.shape[1]),
            "frames": frames,
            "heads": int(keys.shape[3]),
            "dim": int(keys.shape[4]),
        },
        "fixed_config": {
            name: getattr(base, name)
            for name in (
                "k_bits",
                "v_bits",
                "anchor_bits",
                "group_size",
                "stride",
                "residual_mode",
                "k_predictor_mode",
                "v_predictor_mode",
            )
        },
        "fitting": {**common, "device": str(device)},
        "metric_definitions": {
            "reconstruction_error": "||rebuilt - clean||^2 / ||clean||^2 over all frames",
            "attention_output_error": (
                "relative output-energy error of causal softmax attention; clean K "
                "is the query for both branches so only K/V error propagates"
            ),
            "prediction_share": "1 - ||clean - prediction||^2 / ||clean||^2, frames >= 1",
            "cross_term_energy_ratio": "||W k_hat||^2 / ||V||^2, frames >= 1",
            "history_term_energy_ratio": "||sum_l W_l v_hat_l||^2 / ||V||^2, frames >= 1",
            "residual_cross_only": (
                "residual energy with the V history term removed from the fitted "
                "predictor, leaving the CrossKV term and bias"
            ),
            "residual_history_only": (
                "residual energy with the CrossKV term removed, leaving the V "
                "history term and bias"
            ),
            "cross_term_gain": (
                "residual_history_only - residual_energy_ratio: what the CrossKV "
                "term contributes to this fitted predictor"
            ),
            "history_term_gain": (
                "residual_cross_only - residual_energy_ratio: what the V history "
                "contributes to this fitted predictor"
            ),
            "residual_bias_only": (
                "residual energy with both the V history term and the CrossKV term "
                "removed from the fitted predictor, leaving only the bias"
            ),
            "cross_term_gain_vs_bias": (
                "residual_bias_only - residual_cross_only: what the CrossKV term "
                "buys from the no-prediction (bias-only) baseline"
            ),
            "history_term_gain_vs_bias": (
                "residual_bias_only - residual_history_only: what the V history "
                "term buys from the no-prediction (bias-only) baseline"
            ),
            "reconstruction_cross_only": (
                "closed-loop reconstruction error ||rebuilt - clean||^2/||clean||^2 "
                "with the V history term removed (cross+bias only)"
            ),
            "reconstruction_history_only": (
                "closed-loop reconstruction error with the CrossKV term removed "
                "(history+bias only)"
            ),
        },
        "k_sweep": {"baseline_v_history_length": baseline_v, "points": k_rows},
        "v_sweep": {"baseline_k_history_length": baseline_k, "points": v_rows},
        "skipped_lengths": skipped,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[sweep] wrote {args.output}", file=sys.stderr)

    print(
        _format_table(
            f"K history sweep (V history fixed at {baseline_v})", "kh", k_rows
        )
    )
    print(
        _format_table(
            f"V history sweep (K history fixed at {baseline_k})", "vh", v_rows
        )
    )
    print(_format_v_terms(v_rows))
    print(_format_v_ablation(v_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
