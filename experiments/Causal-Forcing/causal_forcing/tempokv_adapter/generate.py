#!/usr/bin/env python3
"""Generate Causal Forcing videos with TempoKV or native bf16 KV caches.

Validation happens before the model is loaded: a parameter set fitted for a
different codec config, a different coordinate system, or a different layer
count is rejected up front rather than producing a plausible-looking video from
mismatched predictors.  After the run, the TempoKV report is checked for actual
codec activity so a misconfigured run cannot pass as a TempoKV result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

RUNTIME_ROOT = PROJECT_ROOT / "experiments" / "Causal-Forcing"
if RUNTIME_ROOT.is_dir() and str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

REQUIRED_RUNTIME_FILES = (
    "inference.py",
    "causal_forcing/kv_cache.py",
    "causal_forcing/pipeline/__init__.py",
    "causal_forcing/pipeline/causal_inference.py",
    "causal_forcing/utils/wan_wrapper.py",
    "causal_forcing/wan/modules/causal_model.py",
)
EXPECTED_COORDINATE_SYSTEM = "pre_rope_normalized"
VIDEO_FRAMES_PER_LATENT = 4

from tempokv.config import is_temporal_v_predictor  # noqa: E402


def latent_frames_for(video_frames: int, num_frame_per_block: int) -> int:
    """Convert a video frame count to whole latent blocks.

    Wan latents cover four video frames each, and the first latent is shared by
    the first frame only, hence the ``+3``.
    """
    if video_frames < 1:
        raise ValueError("video frame count must be positive")
    if num_frame_per_block < 1:
        raise ValueError("num-frame-per-block must be positive")
    shifted = video_frames + VIDEO_FRAMES_PER_LATENT - 1
    period = VIDEO_FRAMES_PER_LATENT * num_frame_per_block
    if shifted % period:
        raise ValueError(
            f"FRAMES must map exactly to {num_frame_per_block}-frame latent "
            f"blocks; valid values satisfy (FRAMES + 3) % {period} == 0"
        )
    return shifted // VIDEO_FRAMES_PER_LATENT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Causal Forcing videos with TempoKV or bf16 KV caches."
    )
    parser.add_argument("--mode", choices=("tempokv", "bf16", "trace"), default="tempokv")
    parser.add_argument("--config", type=Path, help="TempoKV codec config JSON")
    parser.add_argument("--k-params", type=Path)
    parser.add_argument("--v-params", type=Path)
    parser.add_argument("--k-history-length", type=int, choices=range(1, 9))
    parser.add_argument("--v-history-length", type=int, choices=range(0, 6))
    # Overrides so one config per model covers every bit width, instead of one
    # committed config per combination.
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
            "predict V from reconstructed K (cross_kv), its own learned or "
            "fixed temporal history, CrossKV plus fixed V history, innovation, "
            "or direct quantization"
        ),
    )
    parser.add_argument("--k-bits", type=int, choices=range(1, 9))
    parser.add_argument("--v-bits", type=int, choices=range(1, 9))
    parser.add_argument("--anchor-bits", type=int, choices=range(1, 17))
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Causal Forcing YAML; defaults to the bundled chunkwise config",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, help="required when --mode trace")
    parser.add_argument(
        "--frames",
        type=int,
        default=717,
        help="final video frame count; 1 latent frame = 4 video frames",
    )
    parser.add_argument(
        "--num-frame-per-block",
        type=int,
        default=3,
        help="3 for chunkwise Causal Forcing, 1 for framewise",
    )
    parser.add_argument(
        "--span-frames",
        type=int,
        help="latent frames per TempoKV span; one anchor is emitted per span",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--local-attn-size", type=int, default=-1)
    parser.add_argument("--triton-cache", type=Path)
    parser.add_argument(
        "--allow-triton-fallback",
        action="store_true",
        help="use the config's Triton flags instead of enforcing strict acceleration",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--report-timing",
        action="store_true",
        help="record the diffusion-only generation interval",
    )
    parser.add_argument(
        "--retain-final-cache",
        action="store_true",
        help="run the final clean refresh before releasing the cache",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="load generator_ema instead of generator from the checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_config(path: Path):
    from tempokv.config import config_from_mapping

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("codec config must be a JSON object")
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload["codec"], dict):
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    return config_from_mapping(payload)


def _validate_parameters(k_path: Path, v_path: Path, config) -> dict[str, object]:
    from tempokv.codec import TempoKVCodec
    from tempokv.parameters import CrossKVParameters, TemporalParameters

    temporal = TemporalParameters.load(k_path)
    # The temporal V predictor reads its own reconstructed V history, so its
    # parameters are a TemporalParameters set rather than a Cross(K) map.
    temporal_v = is_temporal_v_predictor(config.v_predictor_mode)
    cross = (
        TemporalParameters.load(v_path)
        if temporal_v
        else CrossKVParameters.load(v_path)
    )
    expected_signature = config.format_signature()
    if (
        temporal.codec_signature != expected_signature
        or cross.codec_signature != expected_signature
    ):
        raise ValueError(
            "Causal Forcing parameters were not fitted with the current codec config"
        )
    if (
        not temporal.parameter_set_id
        or not cross.parameter_set_id
        or temporal.parameter_set_id != cross.parameter_set_id
    ):
        raise ValueError(
            "Causal Forcing K/V parameters must share a current parameter_set_id"
        )
    # A shared (layer_ids=None) K predictor covers every layer; only when both
    # sides are layered must their layer sets agree.
    if (
        temporal.layer_ids is not None
        and cross.layer_ids is not None
        and temporal.layer_ids != cross.layer_ids
    ):
        raise ValueError("Causal Forcing K and V parameters cover different layers")
    if temporal.stride != config.stride or cross.stride != config.stride:
        raise ValueError("parameter stride does not match the codec config")
    if (
        temporal.coordinate_system != EXPECTED_COORDINATE_SYSTEM
        or cross.coordinate_system != EXPECTED_COORDINATE_SYSTEM
    ):
        raise ValueError(
            "Causal Forcing parameters must use pre_rope_normalized coordinates; "
            "Self-Forcing parameters are not reusable because the checkpoint and "
            "cache distribution differ"
        )
    codec = TempoKVCodec(temporal, cross, config)
    if temporal_v:
        # Legacy temporal V pools both axes away, so the head count is not part
        # of the parameter geometry.
        head_dim = int(temporal.history_weights.shape[-1])
        heads = None
        parameter_tensors = (
            temporal.history_weights,
            temporal.history_bias,
            cross.history_weights,
            cross.history_bias,
        )
    else:
        heads, head_dim = tuple(int(item) for item in cross.weight.shape[-3:-1])
        parameter_tensors = (
            temporal.history_weights,
            temporal.history_bias,
            cross.weight,
            cross.bias,
            cross.history_weights,
        )
    predictor_parameter_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in parameter_tensors
        if tensor is not None
    )
    return {
        "k_params": str(k_path),
        "v_params": str(v_path),
        "layers": None if temporal.layer_ids is None else len(temporal.layer_ids),
        "heads": heads,
        "head_dimension": head_dim,
        "k_history_length": codec.temporal.history_length,
        "v_history_length": codec.v_parameters.history_length,
        "k_bits": config.k_bits,
        "v_bits": config.v_bits,
        "anchor_bits": config.anchor_bits,
        "v_predictor_mode": config.v_predictor_mode,
        "v_history_diagonal": False if temporal_v else cross.history_diagonal,
        "predictor_parameter_bytes": predictor_parameter_bytes,
        "stride": config.stride,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }


def _generation_command(
    args: argparse.Namespace, paths: dict[str, Path], latent_frames: int
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.num_gpus}",
        str(Path(__file__).with_name("worker.py")),
        "--mode",
        args.mode,
        "--",
        "--config_path",
        str(paths["model_config"]),
        "--checkpoint_path",
        str(paths["checkpoint"]),
        "--model_root",
        str(paths["model_root"]),
        "--data_path",
        str(paths["prompts"]),
        "--output_folder",
        str(paths["videos"]),
        "--num_output_frames",
        str(latent_frames),
        "--num_samples",
        str(args.samples),
        "--seed",
        str(args.seed),
        "--start_index",
        str(args.start_index),
        "--end_index",
        str(args.end_index),
        "--save_with_index",
        "--local_attn_size",
        str(args.local_attn_size),
    ]
    if args.use_ema:
        command.append("--use_ema")
    if args.skip_existing:
        command.append("--skip_existing")
    if args.report_timing:
        command.append("--report_timing")
    if args.retain_final_cache:
        command.append("--retain_final_cache")
    return command


def _load_worker_metrics(metrics_dir: Path, expected_ranks: int) -> dict[str, object]:
    rank_metrics: list[dict[str, object]] = []
    for expected_rank in range(expected_ranks):
        path = metrics_dir / f"rank-{expected_rank}.json"
        if not path.is_file():
            raise RuntimeError(f"missing metrics for rank {expected_rank}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("rank") != expected_rank:
            raise RuntimeError(f"invalid metrics for rank {expected_rank}: {path}")
        rank_metrics.append(payload)
    decode_routes: dict[str, int] = {}
    for metric in rank_metrics:
        for name, count in (metric.get("decode_routes") or {}).items():
            decode_routes[name] = decode_routes.get(name, 0) + int(count)
    peak_allocated = max(int(m["peak_allocated_bytes"]) for m in rank_metrics)
    peak_reserved = max(int(m["peak_reserved_bytes"]) for m in rank_metrics)
    return {
        "max_worker_seconds": max(float(m["wall_seconds"]) for m in rank_metrics),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_allocated / (1024**3),
        "peak_reserved_gib": peak_reserved / (1024**3),
        "decode_routes": decode_routes,
        "ranks": rank_metrics,
    }


def _rank_summaries(videos_dir: Path) -> list[dict[str, object]]:
    summary_path = videos_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Causal Forcing did not write the current run summary")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return [rank for rank in payload.get("rank_summaries") or [] if isinstance(rank, dict)]


def _merge_tempokv_reports(ranks: list[dict[str, object]]) -> dict[str, object]:
    """Combine per-rank reports.

    Every rank holds its own KV cache on its own device, so **all cache-state
    byte figures are per-device**: they are the worst single rank, which is the
    number that has to fit on one GPU.  Mixing a summed "current" with a maxed
    "peak" in the same JSON would produce a report whose current exceeds its
    peak, and multiplying by the rank count would describe a device that does
    not exist.  Only genuinely additive quantities are summed: call counts,
    finalized frames, and the cumulative encoded volume over the whole run.
    Latency is reported as the slowest rank.
    """
    reports = [rank.get("tempokv_report") for rank in ranks]
    reports = [report for report in reports if isinstance(report, dict)]
    if not reports:
        raise RuntimeError(
            "TempoKV run produced no report; the bridge was not installed"
        )
    summed = (
        "encode_key_calls",
        "encode_value_calls",
        "decode_tensor_calls",
        "finalized_frames",
        "ephemeral_writes",
        "cache_evictions",
        "cache_resets",
        "terminal_discard_events",
        "terminal_discarded_frames",
        "terminal_encode_frames_avoided",
        "terminal_discarded_bf16_bytes",
        "terminal_discarded_compressed_bytes",
        # Cumulative volume over the run, not a state: these do add up.
        "total_encoded_cache_bytes",
        "total_uncompressed_cache_bytes",
    )
    per_device = (
        "current_compressed_cache_bytes",
        "current_uncompressed_cache_bytes",
        "peak_compressed_cache_bytes",
        "peak_uncompressed_cache_bytes",
        "resident_bf16_bytes",
        "resident_total_bytes",
        "peak_resident_total_bytes",
    )
    slowest = ("encode_ms", "decode_ms", "generation_ms")
    merged: dict[str, object] = {
        name: reports[0].get(name)
        for name in ("model", "kv_mode", "num_layers", "num_heads", "head_dim", "stride")
    }
    for name in summed:
        merged[name] = sum(int(report.get(name) or 0) for report in reports)
    for name in per_device:
        merged[name] = max(int(report.get(name) or 0) for report in reports)
    for name in slowest:
        merged[name] = max(float(report.get(name) or 0) for report in reports)
    merged["predictor_parameter_bytes"] = max(
        int(report.get("predictor_parameter_bytes") or 0) for report in reports
    )
    # Each rank's ratio compares one coherent cache state against itself, so
    # take the worst rather than dividing two independently-maxed figures that
    # may have come from different ranks.
    ratios = [
        int(report.get("current_uncompressed_cache_bytes") or 0)
        / int(report["current_compressed_cache_bytes"])
        for report in reports
        if int(report.get("current_compressed_cache_bytes") or 0)
    ]
    merged["compression_ratio"] = min(ratios) if ratios else None
    merged["byte_scope"] = "per_device"
    merged["ranks"] = len(reports)
    return merged


def _require_tempokv_activity(report: dict[str, object]) -> None:
    empty = [
        name
        for name in (
            "encode_key_calls",
            "encode_value_calls",
            "decode_tensor_calls",
            "finalized_frames",
        )
        if int(report.get(name) or 0) <= 0
    ]
    if empty:
        raise RuntimeError(
            "TempoKV generation did not exercise the codec: "
            + ", ".join(f"{name}=0" for name in empty)
        )


def _require_direct_persistent_routes(
    routes: dict[str, object],
    v_predictor_mode: str = "cross_kv",
    k_predictor_mode: str = "shared",
) -> None:
    """Reject a nominally accelerated run that missed the production kernels."""
    k_route = (
        "direct_persistent_direct"
        if k_predictor_mode == "direct"
        else "temporal_persistent_direct"
    )
    if v_predictor_mode == "direct":
        v_route = "direct_persistent_direct"
    elif v_predictor_mode in {
        "temporal", "identity", "taylor", "taylor2"
    }:
        v_route = "temporal_persistent_direct"
    elif v_predictor_mode == "innovation":
        v_route = "innovation_persistent_direct"
    else:
        v_route = "cross_kv_persistent_direct"
    required = list(dict.fromkeys((k_route, v_route)))
    missing = [name for name in required if int(routes.get(name) or 0) <= 0]
    if missing:
        raise RuntimeError(
            "TempoKV direct persistent Triton decode was not used: "
            + ", ".join(missing)
        )
    unexpected = {
        name: int(count)
        for name, count in routes.items()
        if int(count) > 0 and name not in required
    }
    if unexpected:
        raise RuntimeError(
            "TempoKV used a non-direct or non-persistent decode route: "
            + ", ".join(
                f"{name}={count}" for name, count in sorted(unexpected.items())
            )
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    model_config = args.model_config
    if model_config is None:
        model_config = RUNTIME_ROOT / "configs" / "causal_forcing_dmd_chunkwise.yaml"
    output_dir = args.output_dir.expanduser().resolve()
    paths = {
        "checkpoint": args.checkpoint.expanduser().resolve(),
        "model_root": args.model_root.expanduser().resolve(),
        "prompts": args.prompts.expanduser().resolve(),
        "model_config": model_config.expanduser().resolve(),
        "run": output_dir,
        "videos": output_dir / "videos",
        "runtime_config": output_dir / "runtime_config.json",
    }
    if args.mode == "tempokv":
        missing = [
            name
            for name, value in (
                ("--config", args.config),
                ("--k-params", args.k_params),
                ("--v-params", args.v_params),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"mode=tempokv requires {', '.join(missing)}")
        paths.update(
            {
                "config": args.config.expanduser().resolve(),
                "k_params": args.k_params.expanduser().resolve(),
                "v_params": args.v_params.expanduser().resolve(),
            }
        )
    if args.mode == "trace" and args.trace_dir is None:
        raise ValueError("mode=trace requires --trace-dir")
    if paths["run"] == Path(paths["run"].anchor):
        raise ValueError("output-dir must not be the filesystem root")

    required_files = ["checkpoint", "prompts", "model_config"]
    if args.mode == "tempokv":
        required_files.extend(("config", "k_params", "v_params"))
    for name in required_files:
        if not paths[name].is_file():
            raise ValueError(f"{name} file does not exist: {paths[name]}")
    if not paths["model_root"].is_dir():
        raise ValueError(f"model-root directory does not exist: {paths['model_root']}")
    for relative in REQUIRED_RUNTIME_FILES:
        if not (RUNTIME_ROOT / relative).is_file():
            raise ValueError(f"bundled Causal Forcing runtime is missing: {relative}")

    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.num_gpus < 1:
        raise ValueError("num-gpus must be positive")
    if args.local_attn_size == 0 or args.local_attn_size < -1:
        raise ValueError("local-attn-size must be -1 or a positive integer")
    latent_frames = latent_frames_for(args.frames, args.num_frame_per_block)
    # Keep the ordinary launcher guard for the historical one-shot/demo
    # workflow.  The explicit benchmark protocol also supports the documented
    # all-method 51-frame sliding-window ablation, so that mode may opt into a
    # shorter window through BENCHMARK=1.
    if (
        args.local_attn_size != -1
        and args.local_attn_size < latent_frames
        and os.environ.get("CAUSAL_FORCING_BENCHMARK") != "1"
    ):
        raise ValueError(
            "local-attn-size must cover the complete generated latent sequence "
            f"({latent_frames} frames) unless BENCHMARK=1 is set for the "
            "sliding-window ablation"
        )

    config = None
    parameters = None
    span_frames = args.span_frames
    if args.mode == "tempokv":
        config = _load_config(paths["config"])
        overrides = {
            name: value
            for name, value in (
                ("k_history_length", args.k_history_length),
                ("v_history_length", args.v_history_length),
                ("k_bits", args.k_bits),
                ("v_bits", args.v_bits),
                ("anchor_bits", args.anchor_bits),
                ("v_predictor_mode", args.v_predictor_mode),
            )
            if value is not None
        }
        if overrides.get("v_predictor_mode") in {
            "temporal",
            "identity",
            "taylor",
            "taylor2",
            "innovation",
            "cross_kv_identity",
            "cross_kv_taylor",
            "cross_kv_taylor2",
        }:
            overrides["cross_kv_weight_mode"] = "layer_shared"
        if overrides:
            config = replace(config, **overrides)
        if 128 % config.group_size:
            raise ValueError("Causal Forcing group_size must divide the head dimension")
        if config.reconstruction_dtype != "bfloat16":
            raise ValueError("Causal Forcing requires reconstruction_dtype=bfloat16")
        if span_frames is None:
            # One anchor for the whole rollout.  The incremental schedule packs
            # each finalized block into the open span, so a span that never
            # closes is still read back, and it spends no second anchor.
            span_frames = latent_frames
        if span_frames < 2:
            raise ValueError("span-frames must be at least 2")
        if span_frames % args.num_frame_per_block:
            raise ValueError(
                f"span-frames ({span_frames}) must be a whole number of "
                f"{args.num_frame_per_block}-frame blocks"
            )
        # The batch schedule only packs a span once it is complete, so a span
        # that closes on the very last block is packed after the last forward
        # pass and nothing ever decodes it.  The incremental schedule packs
        # each finalized block as it arrives, so its spans are read back while
        # they are still open and the span may be as long as the rollout.
        if os.environ.get("TEMPOKV_INCREMENTAL", "1") == "0":
            minimum_frames = span_frames + args.num_frame_per_block
            if latent_frames < minimum_frames:
                raise ValueError(
                    f"generating {latent_frames} latent frames never reads back "
                    f"a {span_frames}-frame TempoKV span under the batch "
                    f"schedule; at least {minimum_frames} latent frames are "
                    "needed so a packed span is decoded by a later block. "
                    "Lower --span-frames, raise --frames, or leave "
                    "TEMPOKV_INCREMENTAL unset"
                )
        parameters = _validate_parameters(paths["k_params"], paths["v_params"], config)

    generation_command = _generation_command(args, paths, latent_frames)
    report: dict[str, object] = {
        "status": "dry_run" if args.dry_run else "running",
        "mode": args.mode,
        "model": "causal_forcing",
        "generation_mode": (
            "framewise" if args.num_frame_per_block == 1 else "chunkwise"
        ),
        "num_frame_per_block": args.num_frame_per_block,
        "latent_frames": latent_frames,
        "num_blocks": latent_frames // args.num_frame_per_block,
        "pixel_frames": args.frames,
        "local_attn_size": args.local_attn_size,
        "attention_mode": (
            "full_history"
            if args.local_attn_size in (-1, latent_frames)
            else "windowed"
        ),
        "seed": int(args.seed),
        "checkpoint": str(paths["checkpoint"]),
        "prompt_file": str(paths["prompts"]),
        "report_timing": bool(args.report_timing),
        "retain_final_cache": bool(args.retain_final_cache),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_command": generation_command,
        "output_dir": str(paths["run"]),
    }
    if args.mode == "tempokv":
        report["parameters"] = parameters
        report["span_frames"] = span_frames
        report["compression_span_frames"] = span_frames
    else:
        report["compression_span_frames"] = None
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    paths["run"].mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # The incremental-schedule and kernel knobs are the caller's choice, not
    # state left over from another run, so they survive the TEMPOKV_* cleanup
    # below.  generate.sh exports the validated Cross launch size; dropping it
    # here silently ran a different decode launch configuration.
    schedule_knobs = {
        name: environment[name]
        for name in (
            "TEMPOKV_INCREMENTAL",
            "TEMPOKV_BF16_LAG_FRAMES",
            "TEMPOKV_FLUSH_TAIL",
            "TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH",
            "TEMPOKV_ROPE_BACKEND",
        )
        if name in environment
    }
    for name in tuple(environment):
        if name.startswith("TEMPOKV_"):
            environment.pop(name)
    if args.mode == "tempokv":
        runtime_payload = asdict(config)
        if not args.allow_triton_fallback:
            runtime_payload["use_triton"] = True
            runtime_payload["strict_triton"] = True
        paths["runtime_config"].write_text(
            json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment.update(
            {
                "TEMPOKV_CONFIG_PATH": str(paths["runtime_config"]),
                "TEMPOKV_K_PARAMS_PATH": str(paths["k_params"]),
                "TEMPOKV_V_PARAMS_PATH": str(paths["v_params"]),
                "TEMPOKV_SPAN_FRAMES": str(span_frames),
                "TEMPOKV_MODEL_NAME": f"causal_forcing_{report['generation_mode']}",
                "TEMPOKV_FORCE_PARTIAL_SPAN": (
                    "0"
                    if os.environ.get("CAUSAL_FORCING_BENCHMARK") == "1"
                    else "1"
                ),
            }
        )
        environment.update(schedule_knobs)
        if args.triton_cache is not None:
            environment["TEMPOKV_TRITON_CACHE"] = str(
                args.triton_cache.expanduser().resolve()
            )
    elif args.mode == "trace":
        environment["TEMPOKV_TRACE_DIR"] = str(
            args.trace_dir.expanduser().resolve()
        )

    paths["videos"].mkdir(parents=True, exist_ok=True)
    metrics_dir = paths["run"] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.num_gpus):
        (metrics_dir / f"rank-{rank}.json").unlink(missing_ok=True)
    environment["CAUSAL_FORCING_METRICS_DIR"] = str(metrics_dir)

    started = time.perf_counter()
    completed = subprocess.run(
        generation_command, cwd=RUNTIME_ROOT, env=environment, check=False
    )
    generation_seconds = time.perf_counter() - started
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, generation_command)

    performance = _load_worker_metrics(metrics_dir, args.num_gpus)
    report.update(
        {
            "status": "complete",
            "generation_seconds": generation_seconds,
            "performance": performance,
        }
    )
    if args.mode == "trace":
        report["trace_dir"] = environment["TEMPOKV_TRACE_DIR"]
    else:
        ranks = _rank_summaries(paths["videos"])
        diffusion_times = [
            float(rank["diffusion_generation_s"])
            for rank in ranks
            if rank.get("diffusion_generation_s") is not None
        ]
        report["timing"] = {
            "diffusion_generation_s": max(diffusion_times)
            if diffusion_times
            else None,
            "end_to_end_s": float(generation_seconds),
            "total_worker_s": float(performance["max_worker_seconds"]),
        }
        videos = [
            entry["path"]
            for rank in ranks
            for entry in rank.get("videos") or []
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        ]
        for path in videos:
            resolved = Path(path)
            if not resolved.is_file() or resolved.stat().st_size == 0:
                raise RuntimeError(f"Causal Forcing video is missing or empty: {path}")
        report["videos"] = videos
        if args.mode == "tempokv":
            tempokv_report = _merge_tempokv_reports(ranks)
            _require_tempokv_activity(tempokv_report)
            if not args.allow_triton_fallback:
                _require_direct_persistent_routes(
                    performance["decode_routes"],
                    config.v_predictor_mode,
                    config.k_predictor_mode,
                )
            report["tempokv"] = tempokv_report
            (paths["run"] / "tempokv_report.json").write_text(
                json.dumps(tempokv_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    (paths["run"] / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
