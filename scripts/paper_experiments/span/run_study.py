#!/usr/bin/env python3
"""Plan or execute the Self-Forcing TempoKV span-length ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPANS = (2, 4, 8, 12, 18)
DEFAULT_BITS = (2, 4)
FIXED_PIXEL_FRAMES = 717
LONG_PIXEL_FRAMES = 1401  # nearest exact Self-Forcing workload to requested 1400


def _read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    codec = payload.get("codec", payload)
    if not isinstance(codec, dict):
        raise ValueError(f"invalid codec config: {path}")
    return dict(codec)


def _write_config(path: Path, codec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"codec": codec}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant_config(
    source: Path, output: Path, *, bits: int, anchor_bits: int
) -> Path:
    codec = _read_config(source)
    codec.update({"k_bits": bits, "v_bits": bits, "anchor_bits": anchor_bits})
    _write_config(output, codec)
    return output


def _v_name(codec: dict[str, Any]) -> str:
    return (
        "v_temporal.npz"
        if codec.get("v_predictor_mode", "cross_kv") == "temporal"
        else "v_cross_kv.npz"
    )


def _fit_command(
    args: argparse.Namespace,
    config: Path,
    params: Path,
    span: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/common/fit_params.py"),
        str(args.trace.resolve()),
        "--config",
        str(config.resolve()),
        "--output-dir",
        str(params.resolve()),
        "--span-frames",
        str(span),
        "--fit-history-source",
        "reconstructed",
        "--v-history-mode",
        "diagonal",
        "--device",
        args.fit_device,
    ]
    if args.attention_steps is not None:
        command.extend(("--attention-steps", str(args.attention_steps)))
    return command


def _generation_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output: Path,
    pixel_frames: int,
    config: Path | None = None,
    params: Path | None = None,
    span: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "experiments/Self-Forcing/tempokv_adapter/generate.py"
        ),
        "--mode",
        mode,
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--prompts",
        str(args.prompts.resolve()),
        "--output-dir",
        str(output.resolve()),
        "--frames",
        str(pixel_frames),
        "--samples",
        "1",
        "--seed",
        str(args.seed),
        "--start-index",
        str(args.prompt_index),
        "--end-index",
        str(args.prompt_index + 1),
        "--num-gpus",
        "1",
        "--benchmark",
        "--profile",
    ]
    if mode == "tempokv":
        assert config is not None and params is not None and span is not None
        codec = _read_config(config)
        command.extend(
            (
                "--config",
                str(config.resolve()),
                "--k-params",
                str((params / "k_temporal.npz").resolve()),
                "--v-params",
                str((params / _v_name(codec)).resolve()),
                "--span-frames",
                str(span),
            )
        )
    return command


def _quality_command(reference: Path, candidate: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluation/evaluate_quality.py"),
        str(reference.resolve()),
        str(candidate.resolve()),
        "--output",
        str(output.resolve()),
        "--no-fid",
    ]


def _nmse_command(
    args: argparse.Namespace, config: Path, params: Path, span: int, output: Path
) -> list[str]:
    codec = _read_config(config)
    return [
        sys.executable,
        str(Path(__file__).with_name("nmse_by_position.py")),
        "--trace",
        str(args.trace.resolve()),
        "--config",
        str(config.resolve()),
        "--k-params",
        str((params / "k_temporal.npz").resolve()),
        "--v-params",
        str((params / _v_name(codec)).resolve()),
        "--span-frames",
        str(span),
        "--output-json",
        str((output / "nmse_by_position.json").resolve()),
        "--output-csv",
        str((output / "nmse_by_position.csv").resolve()),
        "--device",
        args.nmse_device,
    ]


def _load_quality(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        psnr = float(payload["psnr_db"])
        lpips = float(payload["lpips"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"quality report lacks finite PSNR/LPIPS: {path}") from error
    if not math.isfinite(psnr) or not math.isfinite(lpips):
        raise ValueError(f"quality report lacks finite PSNR/LPIPS: {path}")
    return psnr, lpips


def select_long_spans(
    output: Path, bits_values: tuple[int, ...] | list[int], spans: tuple[int, ...] | list[int]
) -> dict[str, Any]:
    """Select each bit width's best/worst span from the fixed-length scan.

    PSNR is the pre-registered primary ranking metric. Lower LPIPS breaks exact
    PSNR ties; the numeric span is a final deterministic tie breaker.
    """
    selections: list[dict[str, Any]] = []
    for bits in bits_values:
        candidates: list[dict[str, Any]] = []
        for span in spans:
            quality_path = (
                output
                / "runs"
                / f"int{bits}_anchor4_span{span}"
                / f"pixel_{FIXED_PIXEL_FRAMES}"
                / "quality.json"
            )
            psnr, lpips = _load_quality(quality_path)
            candidates.append(
                {"span_frames": int(span), "psnr_db": psnr, "lpips": lpips}
            )
        ordered = sorted(
            candidates,
            key=lambda row: (
                float(row["psnr_db"]),
                -float(row["lpips"]),
                -int(row["span_frames"]),
            ),
        )
        worst, best = ordered[0], ordered[-1]
        if int(best["span_frames"]) == int(worst["span_frames"]):
            raise ValueError(f"INT{bits} long-horizon selection collapsed to one span")
        selections.append({"bits": int(bits), "role": "best", **best})
        selections.append({"bits": int(bits), "role": "worst", **worst})
    return {
        "schema_version": 1,
        "fixed_pixel_frames": FIXED_PIXEL_FRAMES,
        "primary_metric": "psnr_db_higher_is_better",
        "tie_breaker": "lpips_lower_is_better_then_shorter_span",
        "selections": selections,
    }


def _long_steps(
    args: argparse.Namespace, output: Path, selection: dict[str, Any]
) -> list[dict[str, Any]]:
    reference = output / "runs/bf16" / f"pixel_{LONG_PIXEL_FRAMES}"
    steps: list[dict[str, Any]] = []
    for selected in selection["selections"]:
        bits = int(selected["bits"])
        span = int(selected["span_frames"])
        role = str(selected["role"])
        label = f"int{bits}_anchor4_span{span}"
        config = output / "configs" / f"residual_int{bits}_anchor_int4.json"
        params = output / "params" / label
        long_run = output / "runs" / label / f"pixel_{LONG_PIXEL_FRAMES}"
        frozen = {
            "bits": bits,
            "anchor_bits": 4,
            "span_frames": span,
            "pixel_frames": LONG_PIXEL_FRAMES,
            "selection_role": role,
        }
        steps.append(
            {
                "id": f"generate_{label}_{role}_pixel{LONG_PIXEL_FRAMES}",
                "kind": "generate_long",
                "command": _generation_command(
                    args,
                    mode="tempokv",
                    output=long_run,
                    pixel_frames=LONG_PIXEL_FRAMES,
                    config=config,
                    params=params,
                    span=span,
                ),
                "expected": str(long_run / "pipeline_report.json"),
                "reference": str(reference / "videos"),
                "frozen": frozen,
            }
        )
        steps.append(
            {
                "id": f"quality_{label}_{role}_pixel{LONG_PIXEL_FRAMES}",
                "kind": "quality_long",
                "command": _quality_command(
                    reference / "videos",
                    long_run / "videos",
                    long_run / "quality.json",
                ),
                "expected": str(long_run / "quality.json"),
                "frozen": frozen,
            }
        )
    return steps


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    configs = output / "configs"
    steps: list[dict[str, Any]] = []
    references: dict[int, Path] = {}
    for frames in (FIXED_PIXEL_FRAMES, LONG_PIXEL_FRAMES):
        reference = output / "runs/bf16" / f"pixel_{frames}"
        references[frames] = reference
        steps.append(
            {
                "id": f"generate_bf16_pixel{frames}",
                "kind": "generate_reference",
                "command": _generation_command(
                    args, mode="bf16", output=reference, pixel_frames=frames
                ),
                "expected": str(reference / "pipeline_report.json"),
            }
        )

    base_codec = _read_config(args.base_config)
    if int(base_codec.get("anchor_bits", 4)) != 4:
        raise ValueError("base config must use the paper's INT4 anchor")
    variants: list[tuple[int, int, Path]] = []
    for bits in args.bits:
        path = _variant_config(
            args.base_config,
            configs / f"residual_int{bits}_anchor_int4.json",
            bits=bits,
            anchor_bits=4,
        )
        variants.append((bits, 4, path))
    # The requested mixed-anchor control is meaningful only for residual INT2:
    # compare all-INT2 against the default INT4-anchor/INT2-residual codec.
    anchor_source = args.anchor_control_config or args.base_config
    anchor_codec = _read_config(anchor_source)
    if anchor_codec.get("fit_mode", "native") == "legacy":
        raise ValueError(
            "anchor INT2 is forbidden by legacy arithmetic; provide a native "
            "--anchor-control-config"
        )
    anchor_path = _variant_config(
        anchor_source,
        configs / "residual_int2_anchor_int2.json",
        bits=2,
        anchor_bits=2,
    )
    variants.append((2, 2, anchor_path))

    for bits, anchor_bits, config in variants:
        for span in args.spans:
            label = f"int{bits}_anchor{anchor_bits}_span{span}"
            params = output / "params" / label
            run = output / "runs" / label / f"pixel_{FIXED_PIXEL_FRAMES}"
            steps.append(
                {
                    "id": f"fit_{label}",
                    "kind": "fit",
                    "command": _fit_command(args, config, params, span),
                    "expected": str(params / "fit_report.json"),
                    "frozen": {
                        "bits": bits,
                        "anchor_bits": anchor_bits,
                        "span_frames": span,
                    },
                }
            )
            steps.append(
                {
                    "id": f"nmse_{label}",
                    "kind": "nmse",
                    "command": _nmse_command(args, config, params, span, params),
                    "expected": str(params / "nmse_by_position.json"),
                    "frozen": {
                        "bits": bits,
                        "anchor_bits": anchor_bits,
                        "span_frames": span,
                    },
                }
            )
            steps.append(
                {
                    "id": f"generate_{label}_pixel{FIXED_PIXEL_FRAMES}",
                    "kind": "generate",
                    "command": _generation_command(
                        args,
                        mode="tempokv",
                        output=run,
                        pixel_frames=FIXED_PIXEL_FRAMES,
                        config=config,
                        params=params,
                        span=span,
                    ),
                    "expected": str(run / "pipeline_report.json"),
                    "reference": str(
                        references[FIXED_PIXEL_FRAMES] / "videos"
                    ),
                    "frozen": {
                        "bits": bits,
                        "anchor_bits": anchor_bits,
                        "span_frames": span,
                        "pixel_frames": FIXED_PIXEL_FRAMES,
                    },
                }
            )
            steps.append(
                {
                    "id": f"quality_{label}_pixel{FIXED_PIXEL_FRAMES}",
                    "kind": "quality",
                    "command": _quality_command(
                        references[FIXED_PIXEL_FRAMES] / "videos",
                        run / "videos",
                        run / "quality.json",
                    ),
                    "expected": str(run / "quality.json"),
                    "frozen": {
                        "bits": bits,
                        "anchor_bits": anchor_bits,
                        "span_frames": span,
                        "pixel_frames": FIXED_PIXEL_FRAMES,
                    },
                }
            )
    return {
        "schema_version": 1,
        "study": "tempokv_span_ablation",
        "trace": str(args.trace.resolve()),
        "trace_sha256": _sha256(args.trace),
        "checkpoint": str(args.checkpoint.resolve()),
        "prompts": str(args.prompts.resolve()),
        "prompt_sha256": _sha256(args.prompts),
        "seed": args.seed,
        "prompt_index": args.prompt_index,
        "resolution": [832, 480],
        "spans": list(args.spans),
        "bits": list(args.bits),
        "fixed_pixel_frames": FIXED_PIXEL_FRAMES,
        "long_requested_pixel_frames": 1400,
        "long_computed_pixel_frames": LONG_PIXEL_FRAMES,
        "long_selection_rule": (
            "per bit width: order 717-frame quality by higher PSNR, then lower "
            "LPIPS, then shorter span; select both ends"
        ),
        "long_steps_deferred_until_fixed_scan": True,
        "steps": steps,
    }


def _execute(manifest: dict[str, Any], output: Path) -> None:
    env = os.environ.copy()
    if "WAN_MODEL_DIR" not in env and "FORCING_WAN_MODEL_DIR" not in env:
        raise ValueError("set WAN_MODEL_DIR or FORCING_WAN_MODEL_DIR")
    if "WAN_MODEL_DIR" in env:
        env["FORCING_WAN_MODEL_DIR"] = env["WAN_MODEL_DIR"]
    for step in manifest["steps"]:
        expected = Path(step["expected"])
        if expected.is_file():
            continue
        log = output / "logs" / f"{step['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                step["command"],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not expected.is_file():
            raise RuntimeError(f"step {step['id']} did not create {expected}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--anchor-control-config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spans", type=int, nargs="+", default=DEFAULT_SPANS)
    parser.add_argument("--bits", type=int, nargs="+", default=DEFAULT_BITS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--nmse-device", default="cpu")
    parser.add_argument("--attention-steps", type=int)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if sorted(set(args.spans)) != sorted(DEFAULT_SPANS):
        raise ValueError(f"formal scan requires spans {DEFAULT_SPANS}")
    if sorted(set(args.bits)) != sorted(DEFAULT_BITS):
        raise ValueError(f"formal scan requires bits {DEFAULT_BITS}")
    for path in (
        args.trace,
        args.base_config,
        args.checkpoint,
        args.prompts,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = build_manifest(args)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.execute:
        _execute(manifest, args.output.resolve())
        selection = select_long_spans(
            args.output.resolve(), list(args.bits), list(args.spans)
        )
        selection_path = args.output / "long_selection.json"
        selection_path.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["long_selection"] = selection
        manifest["steps"].extend(
            _long_steps(args, args.output.resolve(), selection)
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _execute(manifest, args.output.resolve())
    else:
        print(f"wrote {manifest_path} ({len(manifest['steps'])} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
