#!/usr/bin/env python3
"""Run the whole TempoKV regression verification and emit one report.

Fits a small matrix of predictors on one clean trace, analyzes each with the
production analyzer, and measures Triton decode parity.  Everything lands in
``<workdir>/summary.json``; hand that file back for interpretation.

The matrix isolates the two changes still under investigation:

    cell        v_history_mode   max_gain   meaning
    baseline    diagonal         1.0        behavior before the fixes
    dense       dense            1.0        V history may mix channels
    gain        diagonal         high       predictor may extrapolate
    both        dense            high       both

``baseline`` reproduces what the questioned runs used, so every other cell is
read as a delta against it.

    python scripts/tools/verify_regression.py \
        --trace runs/.../self_forcing_trace.npz \
        --config configs/legacy.json \
        --k-history-length 3 --v-history-length 1 \
        --workdir runs/verify-kh3-vh1-int4

Re-running skips cells that already completed, so an interrupted sweep resumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CELLS = {
    "baseline": {"v_history_mode": "diagonal", "high_gain": False},
    "dense": {"v_history_mode": "dense", "high_gain": False},
    "gain": {"v_history_mode": "diagonal", "high_gain": True},
    "both": {"v_history_mode": "dense", "high_gain": True},
}


def _run(command: list[str], log: Path) -> tuple[bool, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    log.write_text(
        f"$ {' '.join(command)}\n\n--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        return False, "\n".join(tail[-6:]) or f"exit {completed.returncode}"
    return True, f"{elapsed:.0f}s"


def _fit(args, cell: str, spec: dict, directory: Path, log: Path) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "common" / "fit_params.py"),
        str(Path(args.trace).resolve()),
        "--output-dir", str(directory),
        "--v-history-mode", spec["v_history_mode"],
        "--max-gain", str(args.high_max_gain if spec["high_gain"] else args.max_gain),
        "--iterations", str(args.iterations),
        "--device", args.device,
        "--keys-name", args.keys_name,
        "--values-name", args.values_name,
    ]
    if args.config:
        command += ["--config", str(Path(args.config).resolve())]
    if args.k_history_length is not None:
        command += ["--k-history-length", str(args.k_history_length)]
    if args.v_history_length is not None:
        command += ["--v-history-length", str(args.v_history_length)]
    if args.coordinate_system:
        command += ["--coordinate-system", args.coordinate_system]
    return _run(command, log)


def _analyze(args, directory: Path, report: Path, log: Path) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tools" / "analyze_predictor.py"),
        "--trace", str(Path(args.trace).resolve()),
        "--k-params", str(directory / "k_temporal.npz"),
        "--v-params", str(directory / "v_cross_kv.npz"),
        "--output", str(report),
    ]
    if args.config:
        command += ["--config", str(Path(args.config).resolve())]
    return _run(command, log)


def _summarize(report: Path, fit_report: Path) -> dict:
    payload = json.loads(report.read_text(encoding="utf-8"))
    statistics = payload["stats"]
    summary = {
        "k_residual": statistics["k"]["pooled"]["residual_energy_ratio"],
        "v_residual": statistics["v"]["pooled"]["residual_energy_ratio"],
        "k_recon": statistics["k"]["pooled"]["reconstruction_error"],
        "v_recon": statistics["v"]["pooled"]["reconstruction_error"],
        "attention_error": payload["attention_output"]["relative_output_error"],
        "k_history_length": payload.get("k_history_length"),
        "v_history_length": payload.get("v_history_length"),
    }
    if fit_report.is_file():
        summary["fit"] = json.loads(fit_report.read_text(encoding="utf-8")).get("fit")
    return summary


def _gain_saturation(k_params: Path) -> dict:
    """Fraction of channels whose fitted L1 gain across lags sits at the cap."""
    from tempokv.parameters import TemporalParameters

    parameters = TemporalParameters.load(k_params)
    weights = parameters.history_weights
    gains = weights.abs().sum(dim=-3 if weights.ndim >= 4 else -2)
    return {
        "max_l1_gain": float(gains.max()),
        "fraction_at_or_above_one": float((gains >= 1.0 - 1e-6).float().mean()),
    }


def _triton_parity(args, directory: Path) -> dict:
    """Max |Triton decode - encoder reconstruction|, separately for K and V.

    K is asserted bit-exact by the test suite.  V additionally crosses the
    CrossKV matmul, where the encoder uses cuBLAS and the decoder uses Triton,
    so any gap here is error the residual never absorbed.
    """
    import numpy as np
    import torch

    from diagnose_regression import _as_cache, _restride
    from tempokv.codec import TempoKVCodec, decode_tensor
    from tempokv.config import config_from_mapping
    from tempokv.integration import check_deployable_cross_kv_layout
    from tempokv.parameters import CrossKVParameters, TemporalParameters

    if not torch.cuda.is_available():
        return {"skipped": "no CUDA device"}

    payload = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    payload = payload.get("codec", payload)
    if args.k_history_length is not None:
        payload["k_history_length"] = args.k_history_length
    if args.v_history_length is not None:
        payload["v_history_length"] = args.v_history_length

    with np.load(args.trace, allow_pickle=False) as data:
        keys = data[args.keys_name]
        values = data[args.values_name]
        layer_ids = [int(x) for x in data["layer_ids"]] if "layer_ids" in data else None

    stride = int(keys.shape[1])
    temporal = _restride(TemporalParameters.load(directory / "k_temporal.npz"), stride)
    cross = _restride(CrossKVParameters.load(directory / "v_cross_kv.npz"), stride)
    config = config_from_mapping(
        {**payload, "stride": stride, "use_triton": False}
    )
    check_deployable_cross_kv_layout(config)
    codec = TempoKVCodec(temporal, cross, config)

    results = {}
    for layer in [int(x) for x in args.layers.split(",") if x.strip()]:
        index = layer_ids.index(layer) if layer_ids else layer
        key = _as_cache(keys[index], "cuda")
        value = _as_cache(values[index], "cuda")
        packed_key, key_recon = codec.encode_key(key, layer=layer)
        packed_value = codec.encode_value(value, key_recon, layer=layer)
        value_recon = decode_tensor(packed_value, source=key_recon, use_triton=False)
        try:
            triton_key = decode_tensor(packed_key, use_triton=True, strict_triton=True)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            results[str(layer)] = {"error": str(error)}
            continue
        entry: dict[str, object] = {
            "k_max_abs_diff": float((triton_key.float() - key_recon.float()).abs().max()),
            "v_recon_rms": float(value_recon.float().pow(2).mean().sqrt()),
        }
        # Legacy Cross+previous-V decoding is Torch-only by design.  Report K
        # parity on its own there instead of failing the whole layer.
        if (
            packed_value.compatibility_mode == "legacy"
            and packed_value.cross_kv is not None
            and packed_value.cross_kv.history_length > 0
        ):
            entry["v_route"] = "torch"
            entry["v_max_abs_diff"] = None
        else:
            entry["v_route"] = "triton"
            try:
                triton_value = decode_tensor(
                    packed_value, source=triton_key, use_triton=True, strict_triton=True
                )
            except Exception as error:  # noqa: BLE001 - reported, not raised
                entry["v_error"] = str(error)
            else:
                entry["v_max_abs_diff"] = float(
                    (triton_value.float() - value_recon.float()).abs().max()
                )
        results[str(layer)] = entry
    return results


def _preflight(args) -> None:
    """Validate every input before launching any fit.

    Each cell is a full closed-loop fit, so a bad path or too-short trace must
    fail here rather than four times over after minutes of work.
    """
    import numpy as np

    trace = Path(args.trace)
    if not trace.is_file():
        raise SystemExit(
            f"trace not found: {trace}\n"
            "Collect one first:  bash scripts/Self-Forcing/collect_trace.sh\n"
            "It writes data/self-forcing/trace/clean_trace.npz by default; pass "
            "--trace if yours lives elsewhere."
        )
    if args.config and not Path(args.config).is_file():
        raise SystemExit(f"config not found: {args.config}")

    with np.load(trace, allow_pickle=False) as data:
        missing = [n for n in (args.keys_name, args.values_name) if n not in data]
        if missing:
            raise SystemExit(
                f"trace {trace} has no array named {' or '.join(missing)}; "
                f"it contains {sorted(data)}. Use --keys-name/--values-name."
            )
        shape = data[args.keys_name].shape
    if len(shape) != 5:
        raise SystemExit(
            f"expected a [L,N,T,H,D] trace, got shape {tuple(shape)}"
        )
    frames = int(shape[2])
    needed = max(args.k_history_length or 1, args.v_history_length or 0) + 1
    if frames < needed:
        raise SystemExit(
            f"trace has {frames} frames but kh={args.k_history_length} / "
            f"vh={args.v_history_length} needs at least {needed}. Recollect with "
            "a larger TRACE_FRAMES or lower the history lengths."
        )
    print(
        f"trace {tuple(shape)}  layers={shape[0]} tokens/frame={shape[1]} "
        f"frames={frames} heads={shape[3]} dim={shape[4]}"
    )
    # Returned so the report records what was measured; a summary without the
    # trace geometry cannot be compared against an earlier sweep.
    return {
        "shape": [int(x) for x in shape],
        "layers": int(shape[0]),
        "tokens_per_frame": int(shape[1]),
        "frames": frames,
        "heads": int(shape[3]),
        "dim": int(shape[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trace",
        default=str(PROJECT_ROOT / "data" / "self-forcing" / "trace" / "clean_trace.npz"),
        help="clean trace written by collect_trace.sh",
    )
    parser.add_argument("--config")
    parser.add_argument("--workdir", default="runs/verify")
    parser.add_argument("--k-history-length", type=int)
    parser.add_argument("--v-history-length", type=int)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-gain", type=float, default=1.0,
                        help="gain cap for the baseline/dense cells (default: 1.0)")
    parser.add_argument("--high-max-gain", type=float, default=4.0,
                        help="lifted gain cap for the gain/both cells (default: 4.0)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,15,29",
                        help="layers used for the Triton parity check")
    parser.add_argument("--keys-name", default="keys")
    parser.add_argument("--values-name", default="values")
    parser.add_argument("--coordinate-system")
    parser.add_argument("--only", help="comma-separated subset of "
                        + ",".join(CELLS))
    parser.add_argument("--force", action="store_true",
                        help="refit cells that already completed")
    args = parser.parse_args()

    selected = [c.strip() for c in args.only.split(",")] if args.only else list(CELLS)
    unknown = [c for c in selected if c not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells: {', '.join(unknown)}")
    trace_geometry = _preflight(args)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "trace": str(Path(args.trace).resolve()),
        "trace_geometry": trace_geometry,
        "config": str(Path(args.config).resolve()) if args.config else None,
        "k_history_length": args.k_history_length,
        "v_history_length": args.v_history_length,
        "max_gain": {"baseline": args.max_gain, "lifted": args.high_max_gain},
        "cells": {},
    }

    for cell in selected:
        spec = CELLS[cell]
        directory = workdir / cell
        report = directory / "analysis.json"
        gain = args.high_max_gain if spec["high_gain"] else args.max_gain
        label = f"{cell} (v_history={spec['v_history_mode']}, max_gain={gain})"

        if report.is_file() and not args.force:
            print(f"[skip] {label} -- already complete")
        else:
            print(f"[fit ] {label}", flush=True)
            ok, detail = _fit(args, cell, spec, directory, directory / "fit.log")
            if not ok:
                print(f"[FAIL] fit {cell}:\n{detail}")
                summary["cells"][cell] = {"error": f"fit failed: {detail}"}
                continue
            print(f"[ok  ] fit {cell} in {detail}", flush=True)

            print(f"[eval] {label}", flush=True)
            ok, detail = _analyze(args, directory, report, directory / "analyze.log")
            if not ok:
                print(f"[FAIL] analyze {cell}:\n{detail}")
                summary["cells"][cell] = {"error": f"analyze failed: {detail}"}
                continue
            print(f"[ok  ] analyze {cell} in {detail}", flush=True)

        entry = _summarize(report, directory / "fit_report.json")
        entry["v_history_mode"] = spec["v_history_mode"]
        entry["max_gain"] = gain
        entry.update(_gain_saturation(directory / "k_temporal.npz"))
        summary["cells"][cell] = entry

    # Parity is measured per cell, not once: the V-history map differs between
    # cells, so a gap that appears only under one mode localizes the fault to
    # that Triton branch rather than to the shared CrossKV matmul.
    summary["triton_parity"] = {}
    for cell in selected:
        if "error" in summary["cells"].get(cell, {"error": "missing"}):
            continue
        print(f"\n[trit] Triton decode parity for '{cell}'", flush=True)
        try:
            summary["triton_parity"][cell] = _triton_parity(args, workdir / cell)
        except Exception as error:  # noqa: BLE001 - recorded for the report
            summary["triton_parity"][cell] = {"error": str(error)}

    destination = workdir / "summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    columns = f"\n{'cell':>9} {'v_hist':>9} {'gain':>6} {'K resid':>9} {'V resid':>9} {'V recon':>9} {'attn err':>10}"
    print(columns)
    print("-" * len(columns))
    reference = summary["cells"].get("baseline", {})
    for cell in selected:
        entry = summary["cells"].get(cell, {})
        if "error" in entry:
            print(f"{cell:>9}  FAILED: {entry['error'].splitlines()[0][:60]}")
            continue
        print(f"{cell:>9} {entry['v_history_mode']:>9} {entry['max_gain']:>6.1f} "
              f"{entry['k_residual']:>9.5f} {entry['v_residual']:>9.5f} "
              f"{entry['v_recon']:>9.5f} {entry['attention_error']:>10.6f}")
    if reference and "error" not in reference:
        print(f"\nrelative to baseline (>1 means baseline was better):")
        for cell in selected:
            entry = summary["cells"].get(cell, {})
            if cell == "baseline" or "error" in entry:
                continue
            print(f"{cell:>9}  K resid x{entry['k_residual'] / reference['k_residual']:.3f}"
                  f"   V resid x{entry['v_residual'] / reference['v_residual']:.3f}"
                  f"   attn err x{entry['attention_error'] / reference['attention_error']:.3f}")

    parity = summary.get("triton_parity", {})
    if parity:
        print("\nTriton decode vs encoder reconstruction "
              "(K must be 0.0; V is shown relative to its own RMS):")
        for cell, layers in parity.items():
            if not isinstance(layers, dict) or "skipped" in layers or "error" in layers:
                print(f"  {cell}: {layers}")
                continue
            mode = summary["cells"][cell]["v_history_mode"]
            print(f"  {cell} (v_history={mode}):")
            for layer, values in layers.items():
                if "error" in values:
                    print(f"    layer {layer}: {values['error']}")
                    continue
                if values.get("v_error"):
                    print(f"    layer {layer:>3}: K {values['k_max_abs_diff']:.3e}   "
                          f"V failed: {values['v_error'].splitlines()[0][:50]}")
                    continue
                if values.get("v_max_abs_diff") is None:
                    print(f"    layer {layer:>3}: K {values['k_max_abs_diff']:.3e}   "
                          f"V decoded in Torch (Triton declined this layout)")
                    continue
                share = values["v_max_abs_diff"] / max(values["v_recon_rms"], 1e-12)
                print(f"    layer {layer:>3}: K {values['k_max_abs_diff']:.3e}   "
                      f"V {values['v_max_abs_diff']:.3e} ({share:.1%} of RMS)")

    print(f"\nwrote {destination}")
    print("Send that file back -- it carries every number above plus the fit settings.")


if __name__ == "__main__":
    main()
