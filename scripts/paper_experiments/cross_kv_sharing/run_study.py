#!/usr/bin/env python3
"""Plan or execute the Self-Forcing shared-CrossKV ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODES = ("layer_head", "layer_shared", "head_shared", "shared")
METHODS = ("bf16", *MODES)
TF32_METHOD = "layer_shared_tf32"


def _read_codec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    codec = payload.get("codec", payload)
    if not isinstance(codec, dict):
        raise ValueError(f"invalid codec config: {path}")
    return dict(codec)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_variant_configs(base_config: Path, output: Path) -> dict[str, Path]:
    codec = _read_codec(base_config)
    if codec.get("fit_mode", "legacy") != "legacy":
        raise ValueError("base config must use fit_mode=legacy")
    if codec.get("k_predictor_mode", "shared") != "shared":
        raise ValueError("base config must use k_predictor_mode=shared")
    if codec.get("v_predictor_mode", "cross_kv") != "cross_kv":
        raise ValueError("base config must use v_predictor_mode=cross_kv")
    if codec.get("cross_kv_weight_mode", "layer_shared") != "layer_shared":
        raise ValueError(
            "base config must use cross_kv_weight_mode=layer_shared"
        )
    variants: dict[str, Path] = {}
    for mode in MODES:
        path = output / "configs" / f"cross_kv_{mode}.json"
        _write_json(path, {"codec": {**codec, "cross_kv_weight_mode": mode}})
        variants[mode] = path
    baseline = _read_codec(variants["layer_head"])
    for mode in MODES[1:]:
        candidate = _read_codec(variants[mode])
        differences = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key) != candidate.get(key)
        }
        if differences != {"cross_kv_weight_mode"}:
            raise AssertionError(
                f"{mode} config differs in unexpected fields: {differences}"
            )
    return variants


def add_tf32_latency_candidate(
    variants: dict[str, Path], output: Path
) -> dict[str, Path]:
    """Add a layer_shared candidate that changes only CrossKV dot precision."""
    codec = _read_codec(variants["layer_shared"])
    codec["cross_matmul_precision"] = "tf32"
    path = output / "configs" / f"cross_kv_{TF32_METHOD}.json"
    _write_json(path, {"codec": codec})
    return {**variants, TF32_METHOD: path}


def _fit_command(args: argparse.Namespace, mode: str, config: Path, params: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/common/fit_params.py"),
        str(args.calibration_trace.resolve()),
        "--config",
        str(config.resolve()),
        "--output-dir",
        str(params.resolve()),
        "--fit-history-source",
        "reconstructed",
        "--v-history-mode",
        "diagonal",
        "--device",
        args.fit_device,
    ]
    return command


def _reconstruction_command(
    args: argparse.Namespace,
    config: Path,
    params: Path,
    output: Path,
) -> list[str]:
    trace = args.held_out_trace or args.calibration_trace
    return [
        sys.executable,
        str(Path(__file__).with_name("evaluate_reconstruction.py")),
        "--trace",
        str(trace.resolve()),
        "--config",
        str(config.resolve()),
        "--k-params",
        str((params / "k_temporal.npz").resolve()),
        "--v-params",
        str((params / "v_cross_kv.npz").resolve()),
        "--output-json",
        str(output.with_suffix(".json").resolve()),
        "--output-csv",
        str(output.with_suffix(".csv").resolve()),
        "--device",
        args.reconstruction_device,
    ]


def _generation_command(
    args: argparse.Namespace,
    method: str,
    output: Path,
    *,
    config: Path | None = None,
    params: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/Self-Forcing/tempokv_adapter/generate.py"),
        "--mode",
        "bf16" if method == "bf16" else "tempokv",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--prompts",
        str(args.prompts.resolve()),
        "--output-dir",
        str(output.resolve()),
        "--frames",
        str(args.frames),
        "--samples",
        str(args.samples),
        "--seed",
        str(args.seed),
        "--start-index",
        str(args.start_index),
        "--end-index",
        str(args.end_index),
        "--num-gpus",
        str(args.num_gpus),
        "--benchmark",
        "--profile",
    ]
    if method != "bf16":
        assert config is not None and params is not None
        command.extend(
            (
                "--config",
                str(config.resolve()),
                "--k-params",
                str((params / "k_temporal.npz").resolve()),
                "--v-params",
                str((params / "v_cross_kv.npz").resolve()),
            )
        )
    if args.model_config is not None:
        command.extend(("--model-config", str(args.model_config.resolve())))
    if args.local_attn_size is not None:
        command.extend(("--local-attn-size", str(args.local_attn_size)))
    if args.span_frames is not None:
        command.extend(("--span-frames", str(args.span_frames)))
    return command


def _quality_command(
    reference: Path,
    candidate: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluation/evaluate_quality.py"),
        str((reference / "videos").resolve()),
        str((candidate / "videos").resolve()),
        "--output",
        str(output.resolve()),
        "--device",
        args.quality_device,
        "--fid-batch-size",
        str(args.fid_batch_size),
    ]
    if args.max_quality_frames is not None:
        command.extend(("--max-frames", str(args.max_quality_frames)))
    return command


def _fair_fields(args: argparse.Namespace) -> dict[str, Any]:
    wan_model_dir = os.environ.get("WAN_MODEL_DIR") or os.environ.get(
        "FORCING_WAN_MODEL_DIR"
    )
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "wan_model_dir": (
            str(Path(wan_model_dir).expanduser().resolve())
            if wan_model_dir
            else None
        ),
        "prompts": str(args.prompts.resolve()),
        "prompt_sha256": _sha256_if_file(args.prompts),
        "seed": args.seed,
        "frames": args.frames,
        "samples": args.samples,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_gpus": args.num_gpus,
        "model_config": str(args.model_config.resolve()) if args.model_config else None,
        "local_attn_size": args.local_attn_size,
        "span_frames": args.span_frames,
    }


def _validate_one_video_protocol(args: argparse.Namespace) -> None:
    if args.samples != 1:
        raise ValueError("the formal one-video study requires samples=1")
    if args.start_index < 0 or args.end_index <= args.start_index:
        raise ValueError("prompt range must satisfy 0 <= start-index < end-index")
    if args.end_index - args.start_index != 1:
        raise ValueError("the formal one-video study requires exactly one prompt")


def _vbench_files(
    args: argparse.Namespace,
    output: Path,
    runs: dict[str, Path],
    methods: tuple[str, ...],
) -> tuple[Path, list[str]]:
    if args.samples != 1 or args.start_index != 0 or args.end_index <= 0:
        raise ValueError("VBench-Long requires samples=1 and prompt range 0..N")
    if not args.prompts.is_file():
        raise FileNotFoundError(args.prompts)
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.end_index > len(prompts):
        raise ValueError("VBench-Long end-index exceeds the prompt file")
    selected_prompts = prompts[: args.end_index]
    prompt_path = output / "vbench_long" / "prompts.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("\n".join(selected_prompts) + "\n", encoding="utf-8")
    manifest_path = output / "vbench_long" / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "expected_count": args.end_index,
            "prompt_text": str(prompt_path.resolve()),
            "entries": [
                {"name": method, "videos": str((runs[method] / "videos").resolve()), "lane": 0}
                for method in methods
            ],
        },
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluation/run_vbench_long_matrix.py"),
        "--manifest",
        str(manifest_path.resolve()),
        "--output-root",
        str((output / "vbench_long/results").resolve()),
        "--vbench-root",
        str(args.vbench_root.resolve()),
        "--cache-dir",
        str(args.vbench_cache_dir.resolve()),
        "--devices",
        args.vbench_devices,
        "--smoke-count",
        "1",
    ]
    return manifest_path, command


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    _validate_one_video_protocol(args)
    if args.attention_steps is not None:
        raise ValueError(
            "--attention-steps is not supported: value refinement would break "
            "the shared CrossKV treatment"
        )
    output = args.output.resolve()
    variants = make_variant_configs(args.base_config, output)
    if getattr(args, "include_tf32_latency_candidate", False):
        variants = add_tf32_latency_candidate(variants, output)
    tempokv_methods = tuple(variants)
    methods = ("bf16", *tempokv_methods)
    fair = _fair_fields(args)
    params = {mode: output / "params" / mode for mode in tempokv_methods}
    runs = {method: output / "runs" / method for method in methods}
    reconstruction_trace = args.held_out_trace or args.calibration_trace
    steps: list[dict[str, Any]] = []
    for mode in tempokv_methods:
        steps.append(
            {
                "id": f"fit_{mode}",
                "kind": "fit",
                "method": mode,
                "command": _fit_command(args, mode, variants[mode], params[mode]),
                "expected": str((params[mode] / "fit_report.json").resolve()),
            }
        )
        reconstruction_base = output / "reconstruction" / mode
        steps.append(
            {
                "id": f"reconstruction_{mode}",
                "kind": "reconstruction",
                "method": mode,
                "command": _reconstruction_command(
                    args, variants[mode], params[mode], reconstruction_base
                ),
                "expected": str(reconstruction_base.with_suffix(".json").resolve()),
            }
        )
    for method in methods:
        steps.append(
            {
                "id": f"generate_{method}",
                "kind": "generation",
                "method": method,
                "command": _generation_command(
                    args,
                    method,
                    runs[method],
                    config=variants.get(method),
                    params=params.get(method),
                ),
                "expected": str((runs[method] / "pipeline_report.json").resolve()),
                "fairness": fair,
            }
        )
    for mode in tempokv_methods:
        quality = runs[mode] / "quality_vs_bf16.json"
        steps.append(
            {
                "id": f"quality_{mode}_vs_bf16",
                "kind": "quality",
                "method": mode,
                "reference": "bf16",
                "command": _quality_command(runs["bf16"], runs[mode], quality, args),
                "expected": str(quality.resolve()),
            }
        )
    vbench_manifest = None
    if args.vbench_long:
        if args.vbench_root is None or args.vbench_cache_dir is None or not args.vbench_devices:
            raise ValueError("--vbench-long requires root, cache dir, and devices")
        vbench_manifest, command = _vbench_files(args, output, runs, methods)
        steps.append(
            {
                "id": "vbench_long_matrix",
                "kind": "vbench_long",
                "command": command,
                "expected": str((output / "vbench_long/results/acceptance.json").resolve()),
            }
        )
    return {
        "schema_version": 1,
        "study": "self_forcing_cross_kv_weight_sharing",
        "hypothesis": (
            "CrossKV projections can share weights across layers, heads, or "
            "both without unacceptable reconstruction or generation loss"
        ),
        "sharing_ablation_difference": "codec.cross_kv_weight_mode",
        "latency_candidate_difference": (
            "codec.cross_matmul_precision"
            if TF32_METHOD in tempokv_methods
            else None
        ),
        "modes": list(MODES),
        "methods": list(methods),
        "base_config": str(args.base_config.resolve()),
        "variant_configs": {mode: str(path.resolve()) for mode, path in variants.items()},
        "calibration_trace": str(args.calibration_trace.resolve()),
        "calibration_trace_sha256": _sha256_if_file(args.calibration_trace),
        "reconstruction_trace": str(reconstruction_trace.resolve()),
        "reconstruction_trace_role": "held_out" if args.held_out_trace else "calibration",
        "reconstruction_trace_sha256": _sha256_if_file(reconstruction_trace),
        "fair_generation_protocol": fair,
        "quality_reference": "bf16",
        "quality_metrics": ["psnr_db", "ssim", "lpips", "fid"],
        "vbench_long_manifest": str(vbench_manifest.resolve()) if vbench_manifest else None,
        "steps": steps,
    }


def _comparison_methods(output: Path) -> tuple[str, ...]:
    candidate = output / "configs" / f"cross_kv_{TF32_METHOD}.json"
    return (*METHODS, TF32_METHOD) if candidate.is_file() else METHODS


def _profile_total(report: dict[str, Any], section_name: str) -> float | None:
    total = 0.0
    found = False
    for profile in report.get("runtime_profile", []) or []:
        for section in profile.get("sections", []) or []:
            if section.get("name") == section_name:
                total += float(section["total_ms"])
                found = True
    return total if found else None


def _comparison_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in _comparison_methods(output):
        row: dict[str, Any] = {
            "method": method,
            "status": "reference" if method == "bf16" else "planned",
            "k_pooled_nmse": None,
            "v_pooled_nmse": None,
            "psnr_db": None,
            "ssim": None,
            "lpips": None,
            "fid": None,
            "materialized_parameter_mib": None,
            "logical_parameter_mib": None,
            "generation_seconds": None,
            "decode_v_ms": None,
            "cross_kv_triton_calls": None,
            "generation_speedup_vs_layer_shared": None,
            "decode_v_speedup_vs_layer_shared": None,
            "vbench_visual_quality": None,
            "vbench_temporal_quality": None,
            "vbench_frame_wise_quality": None,
            "vbench_text_alignment": None,
        }
        vbench_summary = (
            output
            / "vbench_long"
            / "results"
            / "full"
            / method
            / f"{method}_vbench_long_summary.json"
        )
        if vbench_summary.is_file():
            payload = json.loads(vbench_summary.read_text(encoding="utf-8"))
            display = payload.get("display_scores", {})
            official = payload.get("official_competition_aspects", {})
            row["vbench_visual_quality"] = display.get("visual_quality")
            row["vbench_temporal_quality"] = official.get("temporal_quality")
            row["vbench_frame_wise_quality"] = official.get("frame_wise_quality")
            row["vbench_text_alignment"] = official.get("text_alignment")
        pipeline = output / "runs" / method / "pipeline_report.json"
        if pipeline.is_file():
            payload = json.loads(pipeline.read_text(encoding="utf-8"))
            row["generation_seconds"] = payload.get("generation_seconds")
            row["decode_v_ms"] = _profile_total(payload, "decode_v")
            routes = payload.get("performance", {}).get("decode_routes", {})
            row["cross_kv_triton_calls"] = sum(
                int(count)
                for name, count in routes.items()
                if name.startswith("cross_kv_persistent")
            )
        if method != "bf16":
            fit_report = output / "params" / method / "fit_report.json"
            reconstruction = output / "reconstruction" / f"{method}.json"
            quality = output / "runs" / method / "quality_vs_bf16.json"
            if fit_report.is_file():
                payload = json.loads(fit_report.read_text(encoding="utf-8"))
                parameters = payload.get("parameters", {})
                row["materialized_parameter_mib"] = parameters.get("mib")
                row["logical_parameter_mib"] = parameters.get("logical_mib")
            if reconstruction.is_file():
                payload = json.loads(reconstruction.read_text(encoding="utf-8"))
                row["k_pooled_nmse"] = payload["k"]["pooled"]["nmse"]
                row["v_pooled_nmse"] = payload["v"]["pooled"]["nmse"]
            if quality.is_file():
                payload = json.loads(quality.read_text(encoding="utf-8"))
                for name in ("psnr_db", "ssim", "lpips", "fid"):
                    row[name] = payload.get(name)
            if fit_report.is_file() and reconstruction.is_file() and quality.is_file():
                row["status"] = "complete"
        rows.append(row)
    baseline = next(row for row in rows if row["method"] == "layer_shared")
    for row in rows:
        for value_key, speedup_key in (
            ("generation_seconds", "generation_speedup_vs_layer_shared"),
            ("decode_v_ms", "decode_v_speedup_vs_layer_shared"),
        ):
            before = baseline[value_key]
            after = row[value_key]
            if before is not None and after is not None and float(after) > 0.0:
                row[speedup_key] = float(before) / float(after)
    return rows


def write_comparison(output: Path) -> None:
    rows = _comparison_rows(output)
    _write_json(
        output / "comparison.json",
        {
            "schema_version": 1,
            "reference": "bf16",
            "reconstruction_metrics": ["k_pooled_nmse", "v_pooled_nmse"],
            "generation_metrics": ["psnr_db", "ssim", "lpips", "fid"],
            "parameter_metrics": [
                "materialized_parameter_mib",
                "logical_parameter_mib",
            ],
            "latency_metrics": [
                "generation_seconds",
                "decode_v_ms",
                "cross_kv_triton_calls",
                "generation_speedup_vs_layer_shared",
                "decode_v_speedup_vs_layer_shared",
            ],
            "optional_vbench_long_metrics": [
                "vbench_visual_quality",
                "vbench_temporal_quality",
                "vbench_frame_wise_quality",
                "vbench_text_alignment",
            ],
            "rows": rows,
        },
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(manifest: dict[str, Any], output: Path) -> None:
    env = os.environ.copy()
    if "WAN_MODEL_DIR" in env:
        env["FORCING_WAN_MODEL_DIR"] = env["WAN_MODEL_DIR"]
    if "FORCING_WAN_MODEL_DIR" not in env:
        raise ValueError("set WAN_MODEL_DIR or FORCING_WAN_MODEL_DIR")
    effective_wan = str(
        Path(env["FORCING_WAN_MODEL_DIR"]).expanduser().resolve()
    )
    recorded_wan = manifest["fair_generation_protocol"].get("wan_model_dir")
    if recorded_wan != effective_wan:
        raise ValueError(
            "WAN model directory differs from fair_generation_protocol: "
            f"recorded={recorded_wan!r}, current={effective_wan!r}"
        )
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
    parser.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "configs/legacy.json")
    parser.add_argument(
        "--calibration-trace",
        type=Path,
        default=PROJECT_ROOT / "data/self-forcing/traces/legacy.npz",
    )
    parser.add_argument("--held-out-trace", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/self_forcing_dmd.pt",
    )
    parser.add_argument("--prompts", type=Path, default=PROJECT_ROOT / "assets/moviegen128.txt")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/paper/cross_kv_sharing",
    )
    parser.add_argument("--frames", type=int, default=717)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--local-attn-size", type=int)
    parser.add_argument("--span-frames", type=int)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--reconstruction-device", default="cpu")
    parser.add_argument("--quality-device", default="auto")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--max-quality-frames", type=int)
    parser.add_argument("--attention-steps", type=int)
    parser.add_argument("--include-tf32-latency-candidate", action="store_true")
    parser.add_argument("--vbench-long", action="store_true")
    parser.add_argument("--vbench-root", type=Path)
    parser.add_argument("--vbench-cache-dir", type=Path)
    parser.add_argument("--vbench-devices")
    parser.add_argument("--execute", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    _validate_one_video_protocol(args)
    if args.frames < 1 or (args.frames + 3) % 12:
        raise ValueError("frames must be positive and satisfy (frames + 3) % 12 == 0")
    if args.num_gpus < 1:
        raise ValueError("num-gpus must be positive")
    if args.attention_steps is not None:
        raise ValueError(
            "--attention-steps is not supported because value refinement does "
            "not preserve shared CrossKV weights"
        )
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if args.execute:
        required = [args.calibration_trace, args.checkpoint, args.prompts]
        required.append(args.held_out_trace or args.calibration_trace)
        if args.model_config is not None:
            required.append(args.model_config)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("execute inputs do not exist: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest_path = args.output / "manifest.json"
    _write_json(manifest_path, manifest)
    write_comparison(args.output.resolve())
    if args.execute:
        execute(manifest, args.output.resolve())
        write_comparison(args.output.resolve())
    else:
        print(f"wrote {manifest_path} ({len(manifest['steps'])} planned steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
