#!/usr/bin/env python3
"""Measure LongCat K identity and per-position time-mean cosine baselines.

Accepted inputs:

* a merged LongCat ``clean_trace.npz`` with ``keys [L,S,T,H,D]``; or
* the raw LongCat trace root containing ``event_*/layer_*.npz`` files with
  ``k_pre_rope [B,T,P,H,D]``.

The two metrics exactly match ``collect_k_identity_mean_cosine.py``:

1. ``cos(K[t,s], K_hat[t-1,s])`` under a 4-bit closed-loop identity predictor;
2. ``cos(K[t,s], mean_t K[t,s])`` using the clean full-sequence time mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.fitting import (  # noqa: E402
    _reconstruct_temporal_trace,
    build_fixed_temporal_parameters,
)

from scripts.paper_experiments.direction.collect_k_aligned_v_cosine import (  # noqa: E402
    split_trace_ids,
    summarize_cosine,
    temporal_predictions,
)
from scripts.paper_experiments.direction.collect_k_identity_mean_cosine import (  # noqa: E402
    IDENTITY,
    TIME_MEAN,
    load_config,
    write_csvs,
)


LONGCAT_LAYERS = 48
LONGCAT_HEADS = 32
LONGCAT_HEAD_DIM = 128


def scalar(archive: np.lib.npyio.NpzFile, name: str, default: Any = None) -> Any:
    return np.asarray(archive[name]).item() if name in archive else default


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_directories(root: Path) -> list[Path]:
    result = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not result:
        raise ValueError(f"no LongCat layer trace files found under {root}")
    return result


class LongCatTrace:
    """Stream LongCat K one layer at a time from merged or raw storage."""

    def __init__(self, path: Path, *, frames: int) -> None:
        self.path = path.expanduser().resolve()
        self.frames = frames
        self._merged_keys: np.ndarray | None = None
        self._event_dirs: list[Path] | None = None
        if self.path.is_file():
            self._inspect_merged()
        elif self.path.is_dir():
            self._inspect_raw()
        else:
            raise FileNotFoundError(f"trace does not exist: {self.path}")

    def _inspect_merged(self) -> None:
        with np.load(self.path, allow_pickle=False) as archive:
            if "keys" not in archive:
                raise ValueError("merged LongCat trace is missing keys")
            keys = archive["keys"]
            if keys.ndim != 5:
                raise ValueError("merged LongCat keys must use [L,S,T,H,D]")
            layers, samples, available_frames, heads, dim = map(int, keys.shape)
            if available_frames < self.frames:
                raise ValueError(
                    f"trace has {available_frames} frames; {self.frames} are required"
                )
            self.layer_ids = (
                tuple(int(value) for value in np.asarray(archive["layer_ids"]))
                if "layer_ids" in archive
                else tuple(range(layers))
            )
            self.stride = int(
                scalar(archive, "trace_stride", scalar(archive, "stride", 1560))
            )
            self.coordinate_system = str(
                scalar(archive, "coordinate_system", "pre_rope_normalized")
            )
            self.source_format = "merged_clean_trace"
            self.source_shape = list(keys.shape)
            self.samples = samples
            self.heads = heads
            self.dim = dim
            self.fingerprint = file_sha256(self.path)

    def _inspect_raw(self) -> None:
        self._event_dirs = event_directories(self.path)
        first_files = sorted(self._event_dirs[0].glob("layer_*.npz"))
        if not first_files:
            raise ValueError("first LongCat event contains no layer files")
        layer_ids: list[int] = []
        sample_count = 0
        expected_tail: tuple[int, int, int] | None = None
        stride: int | None = None
        for event_index, event_dir in enumerate(self._event_dirs):
            files = sorted(event_dir.glob("layer_*.npz"))
            current_ids: list[int] = []
            event_samples: int | None = None
            for file_index, path in enumerate(files):
                with np.load(path, allow_pickle=False) as archive:
                    if "k_pre_rope" not in archive:
                        raise ValueError(f"{path} is missing k_pre_rope")
                    key = archive["k_pre_rope"]
                    layer = int(scalar(archive, "layer_index", file_index))
                    current_ids.append(layer)
                    if key.ndim != 5:
                        raise ValueError(f"{path} must use [B,T,P,H,D]")
                    batch, available_frames, positions, heads, dim = map(int, key.shape)
                    if available_frames < self.frames:
                        raise ValueError(
                            f"{path} has {available_frames} frames; {self.frames} are required"
                        )
                    tail = (heads, dim, available_frames)
                    if expected_tail is None:
                        expected_tail = tail
                    elif tail != expected_tail:
                        raise ValueError("all raw LongCat layers must share [T,H,D]")
                    named_stride = scalar(
                        archive, "trace_stride", scalar(archive, "stride", 1560)
                    )
                    if stride is None:
                        stride = int(named_stride)
                    elif stride != int(named_stride):
                        raise ValueError("raw LongCat events use different strides")
                    coordinate = str(
                        scalar(archive, "coordinate_system", "pre_rope_normalized")
                    )
                    if coordinate != "pre_rope_normalized":
                        raise ValueError(f"unsupported coordinate system {coordinate!r}")
                    if event_samples is None:
                        event_samples = batch * positions
                if event_index == 0:
                    layer_ids.append(layer)
            if current_ids != layer_ids:
                raise ValueError("all LongCat events must contain the same ordered layers")
            assert event_samples is not None
            sample_count += event_samples
        assert expected_tail is not None and stride is not None
        self.layer_ids = tuple(layer_ids)
        self.stride = stride
        self.coordinate_system = "pre_rope_normalized"
        self.source_format = "raw_event_directory"
        self.source_shape = [
            len(layer_ids),
            sample_count,
            self.frames,
            expected_tail[0],
            expected_tail[1],
        ]
        self.samples = sample_count
        self.heads = expected_tail[0]
        self.dim = expected_tail[1]
        digest = hashlib.sha256()
        for event_dir in self._event_dirs:
            for path in sorted(event_dir.glob("layer_*.npz")):
                relative = path.relative_to(self.path)
                digest.update(str(relative).encode("utf-8"))
                digest.update(str(path.stat().st_size).encode("ascii"))
        self.fingerprint = digest.hexdigest()

    def validate_longcat_shape(self, *, allow_dynamic: bool) -> None:
        if allow_dynamic:
            return
        actual = (len(self.layer_ids), self.heads, self.dim)
        expected = (LONGCAT_LAYERS, LONGCAT_HEADS, LONGCAT_HEAD_DIM)
        if actual != expected:
            raise ValueError(
                f"LongCat shape must be L/H/D={expected}, got {actual}; "
                "use --allow-dynamic-shape only for tests"
            )

    def layers(self, selected: tuple[int, ...]) -> Iterator[tuple[int, np.ndarray]]:
        if self.path.is_file():
            if self._merged_keys is None:
                with np.load(self.path, allow_pickle=False) as archive:
                    self._merged_keys = np.array(
                        archive["keys"][:, :, : self.frames], copy=True
                    )
            for layer in selected:
                position = self.layer_ids.index(layer)
                yield layer, np.ascontiguousarray(self._merged_keys[position])
            return
        assert self._event_dirs is not None
        for layer in selected:
            records: list[np.ndarray] = []
            for event_dir in self._event_dirs:
                matches = sorted(event_dir.glob(f"layer_{layer:02d}.npz"))
                if len(matches) != 1:
                    raise ValueError(
                        f"{event_dir} must contain one layer_{layer:02d}.npz"
                    )
                with np.load(matches[0], allow_pickle=False) as archive:
                    key = np.array(
                        archive["k_pre_rope"][:, : self.frames], copy=True
                    )
                batch, frames, positions, heads, dim = key.shape
                records.append(
                    key.transpose(0, 2, 1, 3, 4).reshape(
                        batch * positions, frames, heads, dim
                    )
                )
            yield layer, np.ascontiguousarray(np.concatenate(records, axis=0))


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    trace = LongCatTrace(args.trace, frames=args.frames)
    trace.validate_longcat_shape(allow_dynamic=args.allow_dynamic_shape)
    layer_ids = trace.layer_ids if args.layers is None else args.layers
    missing = sorted(set(layer_ids) - set(trace.layer_ids))
    if missing:
        raise ValueError(f"trace does not contain layers {missing}")
    _, eval_ids = split_trace_ids(trace.samples, args.holdout_step)
    config = load_config(args.config, stride=trace.stride, dim=trace.dim)

    identity_values: list[torch.Tensor] = []
    time_mean_values: list[torch.Tensor] = []
    started = time.perf_counter()
    for layer, clean_np in trace.layers(layer_ids):
        if clean_np.shape != (
            trace.samples,
            args.frames,
            trace.heads,
            trace.dim,
        ):
            raise ValueError(f"layer {layer} has unexpected shape {clean_np.shape}")
        if not np.isfinite(clean_np).all():
            raise ValueError(f"layer {layer} contains NaN or Inf")
        clean = torch.from_numpy(np.ascontiguousarray(clean_np[eval_ids]))[
            None
        ].to(device)
        parameters = build_fixed_temporal_parameters(
            "identity",
            history_length=1,
            dim=trace.dim,
            stride=trace.stride,
            device=device,
            signature=config.format_signature(),
            layer_ids=(layer,),
            head_count=trace.heads,
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
            f"[LongCat K cosine] layer {layer:02d}: "
            f"identity={identity_cosine.mean().item():.8f} "
            f"time_mean={time_mean_cosine.mean().item():.8f}",
            file=sys.stderr,
            flush=True,
        )

    methods = {
        IDENTITY: summarize_cosine(torch.cat(identity_values, dim=0).double()),
        TIME_MEAN: summarize_cosine(torch.cat(time_mean_values, dim=0).double()),
    }
    report = {
        "experiment": "longcat_k_identity_and_per_position_time_mean_cosine",
        "status": "complete",
        "model": "longcat",
        "trace": {
            "path": str(trace.path),
            "fingerprint": trace.fingerprint,
            "fingerprint_kind": (
                "sha256" if trace.path.is_file() else "relative_path_and_size_sha256"
            ),
            "source_format": trace.source_format,
            "source_shape": trace.source_shape,
            "evaluated_shape": [
                len(layer_ids),
                len(eval_ids),
                args.frames,
                trace.heads,
                trace.dim,
            ],
            "layer_ids": list(layer_ids),
            "eval_sample_ids": eval_ids.tolist(),
            "trace_stride": trace.stride,
            "coordinate_system": trace.coordinate_system,
        },
        "protocol": {
            "frames": args.frames,
            "anchor_frame_excluded": True,
            "evaluated_frames": [1, args.frames - 1],
            "identity_source": "previous decoder-visible reconstructed K",
            "identity_closed_loop": True,
            "identity_k_bits": config.k_bits,
            "identity_anchor_bits": config.anchor_bits,
            "time_mean_source": (
                f"clean K mean over all {args.frames} frames at the same "
                "position/layer/head/channel"
            ),
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
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="LongCat clean_trace.npz or raw trace root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "native.json"
    )
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--holdout-step", type=int, default=5)
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--allow-dynamic-shape", action="store_true")
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
