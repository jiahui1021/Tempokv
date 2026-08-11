#!/usr/bin/env python3
"""Convert sampled Self-Forcing layer traces into TempoKV fitting layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np


# One fitting prompt produces two independent 24-frame records, so the
# collected trace holds ``2 * _LEGACY_PROMPT_COUNT`` events.
_LEGACY_PROMPT_COUNT = 12
_LEGACY_TOKEN_INDICES = (0, 223, 445, 668, 891, 1114, 1336, 1559)
_LEGACY_EVENT_STARTS = (0, 37440) * _LEGACY_PROMPT_COUNT
_LEGACY_EVENT_ENDS = (37440, 74880) * _LEGACY_PROMPT_COUNT
_LEGACY_TRACE_SHAPE = (30, 2 * _LEGACY_PROMPT_COUNT, 1, 8, 24, 12, 128)
_LEGACY_TRACE_PROTOCOL = f"direct_v_{_LEGACY_PROMPT_COUNT}p_51f_8t_24x2_v1"


def _event_directories(root: Path) -> list[Path]:
    directories = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not directories:
        raise ValueError(f"no layer trace files found under {root}")
    indices_by_rank: dict[str, list[int]] = {}
    pattern = re.compile(r"event_(\d{6})_tokens_(\d+)_(\d+)$")
    for directory in directories:
        if not re.fullmatch(r"rank_\d{3}", directory.parent.name):
            raise ValueError(f"unexpected trace rank directory: {directory.parent}")
        match = pattern.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"unexpected trace event directory: {directory}")
        event_index, start, end = (int(value) for value in match.groups())
        if start < 0 or end <= start:
            raise ValueError(f"invalid trace token range in {directory}")
        indices_by_rank.setdefault(directory.parent.name, []).append(event_index)
    for rank, indices in indices_by_rank.items():
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise ValueError(f"{rank} trace event indices are not contiguous")
    return directories


def _event(
    event_dir: Path,
    *,
    legacy_protocol: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    files = sorted(event_dir.glob("layer_*.npz"))
    if len(files) != 30:
        raise ValueError(f"{event_dir} must contain exactly 30 layer files")
    keys = []
    values = []
    seen_layers = []
    event_metadata: dict[str, object] | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "k_pre_rope",
                "v",
                "layer_index",
                "token_indices",
                "head_indices",
                "tokens_start",
                "tokens_end",
                "frame_seq_length",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
            layer = int(np.asarray(data["layer_index"]).item())
            key = np.asarray(data["k_pre_rope"])
            value = np.asarray(data["v"])
            token_indices = np.asarray(data["token_indices"])
            head_indices = np.asarray(data["head_indices"])
            tokens_start = int(np.asarray(data["tokens_start"]).item())
            tokens_end = int(np.asarray(data["tokens_end"]).item())
            frame_seq_length = int(
                np.asarray(data["frame_seq_length"]).item()
            )
        if key.shape != value.shape or key.ndim != 5:
            raise ValueError(f"{path} must use [B,F,P,H,D] matching K/V arrays")
        if key.shape[-2:] != (12, 128):
            raise ValueError(f"{path} must contain 12 heads with dimension 128")
        if frame_seq_length != 1560:
            raise ValueError(f"{path} frame_seq_length must be 1560")
        if (
            token_indices.ndim != 1
            or token_indices.shape[0] != key.shape[2]
            or not np.issubdtype(token_indices.dtype, np.integer)
            or np.any(token_indices < 0)
            or np.any(token_indices >= frame_seq_length)
            or np.any(np.diff(token_indices) <= 0)
        ):
            raise ValueError(f"{path} contains invalid token_indices")
        if not np.array_equal(head_indices, np.arange(12, dtype=head_indices.dtype)):
            raise ValueError(f"{path} contains invalid head_indices")
        if (
            tokens_start < 0
            or tokens_end <= tokens_start
            or tokens_end - tokens_start != key.shape[1] * frame_seq_length
        ):
            raise ValueError(f"{path} token range does not match its frame count")
        current_metadata = {
            "token_indices": tuple(int(item) for item in token_indices),
            "tokens_start": tokens_start,
            "tokens_end": tokens_end,
            "frame_seq_length": frame_seq_length,
        }
        if event_metadata is None:
            event_metadata = current_metadata
        elif event_metadata != current_metadata:
            raise ValueError(f"{event_dir} layer trace metadata is inconsistent")
        # Tempo's generic schema flattens B/P.  The legacy protocol preserves
        # every production event and the original B/P axes so fitting can
        # replay record-by-record FP64 accumulation in the original order.
        batch, frames, samples, heads, width = key.shape
        if legacy_protocol:
            keys.append(key.transpose(0, 2, 1, 3, 4))
            values.append(value.transpose(0, 2, 1, 3, 4))
        else:
            keys.append(
                key.transpose(0, 2, 1, 3, 4).reshape(
                    -1, frames, heads, width
                )
            )
            values.append(
                value.transpose(0, 2, 1, 3, 4).reshape(
                    -1, frames, heads, width
                )
            )
        seen_layers.append(layer)
    if seen_layers != list(range(30)):
        raise ValueError(f"{event_dir} layer indices must be 0 through 29")
    assert event_metadata is not None
    return np.stack(keys), np.stack(values), event_metadata


def convert_trace_root(
    root: Path,
    output: Path,
    *,
    legacy_protocol: bool = False,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {root}")
    key_events = []
    value_events = []
    expected_tail = None
    expected_protocol = None
    event_starts: list[int] = []
    event_ends: list[int] = []
    event_directories = _event_directories(root)
    if legacy_protocol and {
        directory.parent.name for directory in event_directories
    } != {"rank_000"}:
        raise ValueError("legacy trace collection requires rank_000 only")
    for event_dir in event_directories:
        keys, values, metadata = _event(event_dir, legacy_protocol=legacy_protocol)
        tail = keys.shape[1:] if legacy_protocol else keys.shape[2:]
        if expected_tail is None:
            expected_tail = tail
        elif tail != expected_tail:
            raise ValueError("all trace events must use the same frame/head layout")
        protocol = {
            "token_indices": metadata["token_indices"],
            "frame_seq_length": metadata["frame_seq_length"],
        }
        if expected_protocol is None:
            expected_protocol = protocol
        elif protocol != expected_protocol:
            raise ValueError("all trace events must use matching sampling metadata")
        event_starts.append(int(metadata["tokens_start"]))
        event_ends.append(int(metadata["tokens_end"]))
        key_events.append(keys)
        value_events.append(values)
    if legacy_protocol:
        keys = np.stack(key_events, axis=1)
        values = np.stack(value_events, axis=1)
    else:
        keys = np.concatenate(key_events, axis=1)
        values = np.concatenate(value_events, axis=1)
    assert expected_protocol is not None
    frame_counts = [
        (end - start) // int(expected_protocol["frame_seq_length"])
        for start, end in zip(event_starts, event_ends)
    ]
    if legacy_protocol and any(count != 24 for count in frame_counts):
        raise ValueError("legacy trace events must contain exactly 24 frames")
    if legacy_protocol:
        if tuple(expected_protocol["token_indices"]) != _LEGACY_TOKEN_INDICES:
            raise ValueError(
                "legacy trace token indices do not match the reference"
            )
        if tuple(event_starts) != _LEGACY_EVENT_STARTS or tuple(
            event_ends
        ) != _LEGACY_EVENT_ENDS:
            raise ValueError(
                f"legacy trace must contain {_LEGACY_PROMPT_COUNT} ordered "
                "pairs of [0,37440) and [37440,74880) events"
            )
        if tuple(keys.shape) != _LEGACY_TRACE_SHAPE:
            raise ValueError(
                "legacy trace must use shape "
                f"{_LEGACY_TRACE_SHAPE}, got {tuple(keys.shape)}"
            )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        keys=keys,
        values=values,
        layer_ids=np.arange(30, dtype=np.int32),
        stride=np.asarray(1560, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
        trace_layout=np.asarray(
            "legacy_event_bpfhd_v1" if legacy_protocol else "tempokv_flat_samples_v1"
        ),
        token_indices=np.asarray(
            expected_protocol["token_indices"], dtype=np.int32
        ),
        event_tokens_start=np.asarray(event_starts, dtype=np.int64),
        event_tokens_end=np.asarray(event_ends, dtype=np.int64),
        tokens_start=np.asarray(min(event_starts), dtype=np.int64),
        tokens_end=np.asarray(max(event_ends), dtype=np.int64),
        frame_seq_length=np.asarray(
            expected_protocol["frame_seq_length"], dtype=np.int32
        ),
        trace_event_count=np.asarray(len(key_events), dtype=np.int32),
        prompt_count=np.asarray(
            _LEGACY_PROMPT_COUNT if legacy_protocol else 0, dtype=np.int32
        ),
        frames_per_event=np.asarray(frame_counts[0], dtype=np.int32),
        tokens_per_frame=np.asarray(
            len(expected_protocol["token_indices"]), dtype=np.int32
        ),
        num_layers=np.asarray(30, dtype=np.int32),
        num_heads=np.asarray(12, dtype=np.int32),
        head_dim=np.asarray(128, dtype=np.int32),
        temporal_units=np.asarray(keys.shape[-3], dtype=np.int32),
        legacy_trace_protocol=np.asarray(
            _LEGACY_TRACE_PROTOCOL if legacy_protocol else ""
        ),
    )
    report = {
        "status": "complete",
        "source": str(root),
        "output": str(output),
        "events": len(key_events),
        "legacy_protocol": legacy_protocol,
        "shape": list(keys.shape),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Self-Forcing [B,F,P,H,D] traces for TempoKV fitting."
    )
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy-protocol", action="store_true")
    args = parser.parse_args(argv)
    try:
        convert_trace_root(
            args.trace_root,
            args.output,
            legacy_protocol=args.legacy_protocol,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
