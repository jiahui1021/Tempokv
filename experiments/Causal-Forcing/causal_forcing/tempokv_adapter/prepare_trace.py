#!/usr/bin/env python3
"""Convert sampled Causal Forcing layer traces into TempoKV fitting layout.

Per-layer events are stored as ``[B, F, P, H, D]``.  Fitting wants the temporal
axis in the middle and each sampled spatial position treated as an independent
sequence, so this reshapes to ``[B*P, F, H, D]`` and stacks the layers into
``[num_layers, samples, T, H, D]`` — the layout ``scripts/common/fit_params.py``
already consumes for Self-Forcing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np


REQUIRED_ARRAYS = {
    "k_pre_rope",
    "v",
    "layer_index",
    "num_layers",
    "num_heads",
    "head_dim",
    "token_indices",
    "head_indices",
    "global_start_frame",
    "global_end_frame",
    "trace_stride",
    "coordinate_system",
    "cross_source_kind",
    "generation_mode",
    "num_frame_per_block",
}


def _event_directories(root: Path) -> list[Path]:
    directories = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not directories:
        raise ValueError(f"no layer trace files found under {root}")
    indices_by_rank: dict[str, list[int]] = {}
    pattern = re.compile(r"event_(\d{6})_frames_(\d+)_(\d+)$")
    for directory in directories:
        if not re.fullmatch(r"rank_\d{3}", directory.parent.name):
            raise ValueError(f"unexpected trace rank directory: {directory.parent}")
        match = pattern.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"unexpected trace event directory: {directory}")
        event_index, start, end = (int(value) for value in match.groups())
        if start < 0 or end <= start:
            raise ValueError(f"invalid trace frame range in {directory}")
        indices_by_rank.setdefault(directory.parent.name, []).append(event_index)
    for rank, indices in indices_by_rank.items():
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise ValueError(f"{rank} trace event indices are not contiguous")
    return directories


def _event(event_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    files = sorted(event_dir.glob("layer_*.npz"))
    if not files:
        raise ValueError(f"{event_dir} contains no layer trace files")
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    seen_layers: list[int] = []
    metadata: dict[str, object] | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(REQUIRED_ARRAYS - set(data.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
            layer = int(np.asarray(data["layer_index"]).item())
            key = np.asarray(data["k_pre_rope"])
            value = np.asarray(data["v"])
            current = {
                "num_layers": int(np.asarray(data["num_layers"]).item()),
                "num_heads": int(np.asarray(data["num_heads"]).item()),
                "head_dim": int(np.asarray(data["head_dim"]).item()),
                "token_indices": tuple(
                    int(item) for item in np.asarray(data["token_indices"])
                ),
                "global_start_frame": int(
                    np.asarray(data["global_start_frame"]).item()
                ),
                "global_end_frame": int(np.asarray(data["global_end_frame"]).item()),
                "trace_stride": int(np.asarray(data["trace_stride"]).item()),
                "coordinate_system": str(np.asarray(data["coordinate_system"]).item()),
                "cross_source_kind": str(np.asarray(data["cross_source_kind"]).item()),
                "generation_mode": str(np.asarray(data["generation_mode"]).item()),
                "num_frame_per_block": int(
                    np.asarray(data["num_frame_per_block"]).item()
                ),
            }
            head_indices = np.asarray(data["head_indices"])
        if key.shape != value.shape or key.ndim != 5:
            raise ValueError(f"{path} must use [B,F,P,H,D] matching K/V arrays")
        if key.shape[-2:] != (current["num_heads"], current["head_dim"]):
            raise ValueError(
                f"{path} K/V shape does not match its recorded head layout"
            )
        if current["coordinate_system"] != "pre_rope_normalized":
            raise ValueError(
                f"{path} must be recorded in pre_rope_normalized coordinates, got "
                f"{current['coordinate_system']!r}"
            )
        token_indices = np.asarray(current["token_indices"])
        if (
            token_indices.ndim != 1
            or token_indices.shape[0] != key.shape[2]
            or np.any(token_indices < 0)
            or np.any(token_indices >= current["trace_stride"])
            or np.any(np.diff(token_indices) <= 0)
        ):
            raise ValueError(f"{path} contains invalid token_indices")
        if not np.array_equal(
            head_indices, np.arange(current["num_heads"], dtype=head_indices.dtype)
        ):
            raise ValueError(f"{path} contains invalid head_indices")
        frames = current["global_end_frame"] - current["global_start_frame"]
        if frames != key.shape[1]:
            raise ValueError(f"{path} frame range does not match its frame count")
        if metadata is None:
            metadata = current
        elif metadata != current:
            raise ValueError(f"{event_dir} layer trace metadata is inconsistent")
        # [B,F,P,H,D] -> [B*P,F,H,D], keeping F as the temporal axis.
        _, frame_count, _, heads, width = key.shape
        keys.append(key.transpose(0, 2, 1, 3, 4).reshape(-1, frame_count, heads, width))
        values.append(
            value.transpose(0, 2, 1, 3, 4).reshape(-1, frame_count, heads, width)
        )
        seen_layers.append(layer)
    assert metadata is not None
    if seen_layers != list(range(metadata["num_layers"])):
        raise ValueError(
            f"{event_dir} layer indices must be 0 through "
            f"{metadata['num_layers'] - 1}"
        )
    return np.stack(keys), np.stack(values), metadata


def convert_trace_root(root: Path, output: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {root}")
    key_events: list[np.ndarray] = []
    value_events: list[np.ndarray] = []
    expected_tail: tuple[int, ...] | None = None
    shared: dict[str, object] | None = None
    for event_dir in _event_directories(root):
        keys, values, metadata = _event(event_dir)
        tail = keys.shape[2:]
        if expected_tail is None:
            expected_tail = tail
        elif tail != expected_tail:
            raise ValueError("all trace events must use the same head layout")
        # Different events are different rollouts, so their absolute frame
        # ranges legitimately differ; everything else must match.
        comparable = {
            name: value
            for name, value in metadata.items()
            if name not in {"global_start_frame", "global_end_frame"}
        }
        if shared is None:
            shared = comparable
        elif comparable != shared:
            raise ValueError("all trace events must use matching trace metadata")
        key_events.append(keys)
        value_events.append(values)
    assert shared is not None
    keys = np.concatenate(key_events, axis=1)
    values = np.concatenate(value_events, axis=1)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        keys=keys,
        values=values,
        layer_ids=np.arange(int(shared["num_layers"]), dtype=np.int32),
        stride=np.asarray(shared["trace_stride"], dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray(shared["cross_source_kind"]),
        generation_mode=np.asarray(shared["generation_mode"]),
        num_frame_per_block=np.asarray(
            shared["num_frame_per_block"], dtype=np.int32
        ),
        num_heads=np.asarray(shared["num_heads"], dtype=np.int32),
        head_dim=np.asarray(shared["head_dim"], dtype=np.int32),
        token_indices=np.asarray(shared["token_indices"], dtype=np.int32),
        frame_seq_length=np.asarray(shared["trace_stride"], dtype=np.int32),
        trace_event_count=np.asarray(len(key_events), dtype=np.int32),
    )
    report = {
        "status": "complete",
        "source": str(root),
        "output": str(output),
        "events": len(key_events),
        "generation_mode": shared["generation_mode"],
        "num_frame_per_block": shared["num_frame_per_block"],
        "coordinate_system": "pre_rope_normalized",
        "shape": list(keys.shape),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Causal Forcing [B,F,P,H,D] traces for TempoKV fitting."
    )
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        convert_trace_root(args.trace_root, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
