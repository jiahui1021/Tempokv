#!/usr/bin/env python3
"""Measure two parameter-free K cosine baselines on the 180-frame trace.

1. identity_previous_reconstructed:
   cos(K[t,s], K_hat[t-1,s]), where K_hat is the decoder-visible 4-bit
   closed-loop reconstruction under the zero-order identity predictor.
2. per_position_time_mean:
   cos(K[t,s], mean_t K[t,s]) using clean pre-RoPE normalized K.  This is a
   descriptive statistic, not a causal predictor, because the full 180-frame
   mean includes the evaluated frame.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_temporal_trace,
    build_fixed_temporal_parameters,
)

from scripts.paper_experiments.direction.collect_k_aligned_v_cosine import (  # noqa: E402
    reshape_raw_layer,
    sha256,
    split_trace_ids,
    summarize_cosine,
    temporal_predictions,
)


IDENTITY = "identity_previous_reconstructed"
TIME_MEAN = "per_position_time_mean"


def load_config(path: Path, *, stride: int, dim: int) -> CodecConfig:
    values = asdict(CodecConfig())
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update(payload.get("codec", payload))
    values.update(
        {
            "stride": stride,
            "group_size": 64 if dim % 64 == 0 else dim,
            "use_triton": False,
            "strict_triton": False,
            "k_bits": 4,
            "v_bits": 4,
            "k_history_length": 1,
            "k_predictor_mode": "identity",
            "fit_mode": "native",
            "predictor_direction": "k_to_v",
        }
    )
    return config_from_mapping(values)


def raw_layer_ids(path: Path) -> tuple[int, ...]:
    with np.load(path, allow_pickle=False) as archive:
        layers = sorted(
            int(name[1:-2])
            for name in archive.files
            if name.startswith("L") and name.endswith("_k") and name[1:-2].isdigit()
        )
    if not layers:
        raise ValueError("trace must contain layer-wise L{n}_k arrays")
    return tuple(layers)


def write_csvs(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    layer_path = output.with_name(f"{output.stem}_per_layer.csv")
    frame_path = output.with_name(f"{output.stem}_per_frame.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "mean_cosine"))
        writer.writeheader()
        for method in (IDENTITY, TIME_MEAN):
            writer.writerow(
                {"method": method, "mean_cosine": report["methods"][method]["mean_cosine"]}
            )
    with layer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("layer_id", IDENTITY, TIME_MEAN)
        )
        writer.writeheader()
        for index, layer in enumerate(report["trace"]["layer_ids"]):
            writer.writerow(
                {
                    "layer_id": layer,
                    IDENTITY: report["methods"][IDENTITY]["per_layer_cosine"][index],
                    TIME_MEAN: report["methods"][TIME_MEAN]["per_layer_cosine"][index],
                }
            )
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame", IDENTITY, TIME_MEAN))
        writer.writeheader()
        identity_frames = report["methods"][IDENTITY]["per_frame"]
        mean_frames = report["methods"][TIME_MEAN]["per_frame"]
        for identity_row, mean_row in zip(identity_frames, mean_frames, strict=True):
            if identity_row["frame"] != mean_row["frame"]:
                raise ValueError("method frame indices do not match")
            writer.writerow(
                {
                    "frame": identity_row["frame"],
                    IDENTITY: identity_row["cosine"],
                    TIME_MEAN: mean_row["cosine"],
                }
            )
    return summary_path, layer_path, frame_path


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    available_layers = raw_layer_ids(args.trace)
    layer_ids = available_layers if args.layers is None else args.layers
    missing = sorted(set(layer_ids) - set(available_layers))
    if missing:
        raise ValueError(f"trace does not contain layers {missing}")

    with np.load(args.trace, allow_pickle=False) as archive:
        first_raw = np.asarray(archive[f"L{layer_ids[0]}_k"])
        first = reshape_raw_layer(first_raw, frames=args.frames)
        sample_count, _, heads, dim = first.shape
        raw_layer_shape = list(first_raw.shape)
        expected_shape = tuple(first.shape)
        del first_raw, first
    _, eval_ids = split_trace_ids(sample_count, args.holdout_step)
    config = load_config(args.config, stride=args.stride, dim=dim)

    identity_values: list[torch.Tensor] = []
    time_mean_values: list[torch.Tensor] = []
    started = time.perf_counter()
    with np.load(args.trace, allow_pickle=False) as archive:
        for layer in layer_ids:
            clean_np = reshape_raw_layer(
                np.array(archive[f"L{layer}_k"], copy=True), frames=args.frames
            )
            if tuple(clean_np.shape) != expected_shape:
                raise ValueError(f"layer {layer} shape differs from layer {layer_ids[0]}")
            if not np.isfinite(clean_np).all():
                raise ValueError(f"layer {layer} contains NaN or Inf")
            clean = torch.from_numpy(np.ascontiguousarray(clean_np[eval_ids]))[
                None
            ].to(device)
            del clean_np

            parameters = build_fixed_temporal_parameters(
                "identity",
                history_length=1,
                dim=dim,
                stride=args.stride,
                device=device,
                signature=config.format_signature(),
                layer_ids=(layer,),
                head_count=heads,
            )
            rebuilt = _reconstruct_temporal_trace(clean, parameters, config)
            identity_prediction = temporal_predictions(clean, rebuilt, parameters)
            identity_cosine = F.cosine_similarity(
                clean[:, :, 1:].float(),
                identity_prediction[:, :, 1:].float(),
                dim=-1,
                eps=1e-8,
            ).cpu()

            clean_time_mean = clean.float().mean(dim=2, keepdim=True)
            time_mean_cosine = F.cosine_similarity(
                clean[:, :, 1:].float(),
                clean_time_mean.expand(-1, -1, args.frames - 1, -1, -1),
                dim=-1,
                eps=1e-8,
            ).cpu()
            identity_values.append(identity_cosine)
            time_mean_values.append(time_mean_cosine)
            print(
                f"[K cosine] layer {layer:02d}: identity={identity_cosine.mean().item():.8f} "
                f"time_mean={time_mean_cosine.mean().item():.8f}",
                file=sys.stderr,
                flush=True,
            )
            del clean, rebuilt, identity_prediction, clean_time_mean

    methods = {
        IDENTITY: summarize_cosine(torch.cat(identity_values, dim=0).double()),
        TIME_MEAN: summarize_cosine(torch.cat(time_mean_values, dim=0).double()),
    }
    report = {
        "experiment": "k_identity_and_per_position_time_mean_cosine",
        "status": "complete",
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": sha256(args.trace),
            "source_format": "layerwise_raw_time_major",
            "raw_layer_shape": raw_layer_shape,
            "evaluated_shape": [
                len(layer_ids),
                len(eval_ids),
                args.frames,
                heads,
                dim,
            ],
            "layer_ids": list(layer_ids),
            "eval_sample_ids": eval_ids.tolist(),
            "coordinate_system": "pre_rope_normalized",
        },
        "protocol": {
            "frames": args.frames,
            "anchor_frame_excluded": True,
            "evaluated_frames": [1, args.frames - 1],
            "identity_source": "previous decoder-visible reconstructed K",
            "identity_closed_loop": True,
            "identity_k_bits": config.k_bits,
            "identity_anchor_bits": config.anchor_bits,
            "time_mean_source": "clean K mean over all 180 frames at the same position/layer/head/channel",
            "time_mean_is_causal_predictor": False,
            "device": str(device),
        },
        "methods": methods,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_paths = write_csvs(report, args.output)
    report["csv_files"] = [str(path.resolve()) for path in csv_paths]
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be non-empty and unique")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "native.json")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--stride", type=int, default=1560)
    parser.add_argument("--holdout-step", type=int, default=5)
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frames < 3:
        raise SystemExit("--frames must be at least 3")
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads must be positive")
    report = run(args)
    for method in (IDENTITY, TIME_MEAN):
        print(f"{method}: {report['methods'][method]['mean_cosine']:.8f}")


if __name__ == "__main__":
    main()
