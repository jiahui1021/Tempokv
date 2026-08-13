#!/usr/bin/env python3
"""Convert sampled Causal Forcing layer traces into TempoKV fitting layout.

Per-layer events are stored as ``[B, F, P, H, D]``.  Fitting wants the temporal
axis in the middle and each sampled spatial position treated as an independent
sequence, so the native path reshapes to ``[B*P, F, H, D]`` and stacks the layers
into ``[num_layers, samples, T, H, D]`` — the layout
``scripts/common/fit_params.py`` already consumes for Self-Forcing.

The legacy path (``--legacy-protocol``) preserves the original ``[B, P, F, H, D]``
axes and stacks events along a new event axis, producing the
``legacy_event_bpfhd_v1`` layout that the legacy fitter requires.
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
_ATTENTION_KINDS = ("q_post_rope", "k_post_rope")

# One legacy fitting prompt produces two independent 24-frame records, so the
# collected trace holds ``2 * _LEGACY_PROMPT_COUNT`` events.  Keep these in
# step with the reference fitter in scripts/common/fit_params.py.
_LEGACY_PROMPT_COUNT = 12
_LEGACY_TOKEN_INDICES = (0, 223, 445, 668, 891, 1114, 1336, 1559)
_LEGACY_EVENT_STARTS = (0, 37440) * _LEGACY_PROMPT_COUNT
_LEGACY_EVENT_ENDS = (37440, 74880) * _LEGACY_PROMPT_COUNT
_LEGACY_TRACE_SHAPE = (30, 2 * _LEGACY_PROMPT_COUNT, 1, 8, 24, 12, 128)
_LEGACY_TRACE_PROTOCOL = f"direct_v_{_LEGACY_PROMPT_COUNT}p_51f_8t_24x2_v1"


def _event_directories(
    root: Path, *, legacy_protocol: bool = False
) -> list[Path]:
    directories = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not directories:
        raise ValueError(f"no layer trace files found under {root}")
    indices_by_rank: dict[str, list[int]] = {}
    pattern = re.compile(
        r"event_(\d{6})_tokens_(\d+)_(\d+)$"
        if legacy_protocol
        else r"event_(\d{6})_frames_(\d+)_(\d+)$"
    )
    for directory in directories:
        if not re.fullmatch(r"rank_\d{3}", directory.parent.name):
            raise ValueError(f"unexpected trace rank directory: {directory.parent}")
        match = pattern.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"unexpected trace event directory: {directory}")
        event_index, start, end = (int(value) for value in match.groups())
        if start < 0 or end <= start:
            raise ValueError(f"invalid trace range in {directory}")
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
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, object],
]:
    files = sorted(event_dir.glob("layer_*.npz"))
    if not files:
        raise ValueError(f"{event_dir} contains no layer trace files")
    if legacy_protocol and len(files) != 30:
        raise ValueError(f"{event_dir} must contain exactly 30 layer files")
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rotated: dict[str, list[np.ndarray]] = {}
    attention_kinds: tuple[str, ...] | None = None
    seen_layers: list[int] = []
    metadata: dict[str, object] | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if legacy_protocol:
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
            else:
                required = REQUIRED_ARRAYS
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
            present = tuple(
                kind for kind in _ATTENTION_KINDS if kind in data.files
            )
            if attention_kinds is None:
                attention_kinds = present
            elif attention_kinds != present:
                raise ValueError(
                    f"{event_dir} layer files disagree on rotated Q/K presence"
                )
            if present and len(present) != len(_ATTENTION_KINDS):
                raise ValueError(
                    f"{path} must contain both {' and '.join(_ATTENTION_KINDS)}"
                )
            layer = int(np.asarray(data["layer_index"]).item())
            key = np.asarray(data["k_pre_rope"])
            value = np.asarray(data["v"])
            attention = {kind: np.asarray(data[kind]) for kind in present}
            if legacy_protocol:
                current: dict[str, object] = {
                    "token_indices": tuple(
                        int(item) for item in np.asarray(data["token_indices"])
                    ),
                    "tokens_start": int(np.asarray(data["tokens_start"]).item()),
                    "tokens_end": int(np.asarray(data["tokens_end"]).item()),
                    "frame_seq_length": int(
                        np.asarray(data["frame_seq_length"]).item()
                    ),
                }
            else:
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
                    "global_end_frame": int(
                        np.asarray(data["global_end_frame"]).item()
                    ),
                    "trace_stride": int(np.asarray(data["trace_stride"]).item()),
                    "coordinate_system": str(
                        np.asarray(data["coordinate_system"]).item()
                    ),
                    "cross_source_kind": str(
                        np.asarray(data["cross_source_kind"]).item()
                    ),
                    "generation_mode": str(
                        np.asarray(data["generation_mode"]).item()
                    ),
                    "num_frame_per_block": int(
                        np.asarray(data["num_frame_per_block"]).item()
                    ),
                }
            head_indices = np.asarray(data["head_indices"])
        if key.shape != value.shape or key.ndim != 5:
            raise ValueError(f"{path} must use [B,F,P,H,D] matching K/V arrays")
        if legacy_protocol:
            if key.shape[-2:] != (12, 128):
                raise ValueError(f"{path} must contain 12 heads with dimension 128")
            frame_seq_length = int(current["frame_seq_length"])
            if frame_seq_length != 1560:
                raise ValueError(f"{path} frame_seq_length must be 1560")
            token_indices = np.asarray(current["token_indices"])
            if (
                token_indices.ndim != 1
                or token_indices.shape[0] != key.shape[2]
                or np.any(token_indices < 0)
                or np.any(token_indices >= frame_seq_length)
                or np.any(np.diff(token_indices) <= 0)
            ):
                raise ValueError(f"{path} contains invalid token_indices")
            if not np.array_equal(
                head_indices, np.arange(12, dtype=head_indices.dtype)
            ):
                raise ValueError(f"{path} contains invalid head_indices")
            tokens_start = int(current["tokens_start"])
            tokens_end = int(current["tokens_end"])
            if (
                tokens_start < 0
                or tokens_end <= tokens_start
                or tokens_end - tokens_start != key.shape[1] * frame_seq_length
            ):
                raise ValueError(f"{path} token range does not match its frame count")
        else:
            if key.shape[-2:] != (current["num_heads"], current["head_dim"]):
                raise ValueError(
                    f"{path} K/V shape does not match its recorded head layout"
                )
            if current["coordinate_system"] != "pre_rope_normalized":
                raise ValueError(
                    f"{path} must be recorded in pre_rope_normalized coordinates, "
                    f"got {current['coordinate_system']!r}"
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
                head_indices,
                np.arange(current["num_heads"], dtype=head_indices.dtype),
            ):
                raise ValueError(f"{path} contains invalid head_indices")
            frames = current["global_end_frame"] - current["global_start_frame"]
            if frames != key.shape[1]:
                raise ValueError(f"{path} frame range does not match its frame count")
        if metadata is None:
            metadata = current
        elif metadata != current:
            raise ValueError(f"{event_dir} layer trace metadata is inconsistent")
        _, frame_count, _, heads, width = key.shape

        def move(array: np.ndarray, name: str) -> np.ndarray:
            if array.shape != key.shape:
                raise ValueError(
                    f"{path} {name} shape {array.shape} does not match K"
                )
            moved = array.transpose(0, 2, 1, 3, 4)
            if legacy_protocol:
                return moved
            return moved.reshape(-1, frame_count, heads, width)

        if legacy_protocol:
            # Preserve [B,P,F,H,D] so fitting can replay record-by-record.
            keys.append(move(key, "k_pre_rope"))
            values.append(move(value, "v"))
        else:
            # [B,F,P,H,D] -> [B*P,F,H,D], keeping F as the temporal axis.
            keys.append(move(key, "k_pre_rope"))
            values.append(move(value, "v"))
        for kind, array in attention.items():
            rotated.setdefault(kind, []).append(move(array, kind))
        seen_layers.append(layer)
    assert metadata is not None
    num_layers = 30 if legacy_protocol else int(metadata["num_layers"])
    if seen_layers != list(range(num_layers)):
        raise ValueError(
            f"{event_dir} layer indices must be 0 through {num_layers - 1}"
        )
    return (
        np.stack(keys),
        np.stack(values),
        {kind: np.stack(parts) for kind, parts in rotated.items()},
        metadata,
    )


def convert_trace_root(
    root: Path,
    output: Path,
    *,
    legacy_protocol: bool = False,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {root}")
    key_events: list[np.ndarray] = []
    value_events: list[np.ndarray] = []
    rotated_events: dict[str, list[np.ndarray]] = {}
    expected_attention_kinds: set[str] | None = None
    event_starts: list[int] = []
    event_ends: list[int] = []
    expected_tail: tuple[int, ...] | None = None
    shared: dict[str, object] | None = None
    for event_dir in _event_directories(root, legacy_protocol=legacy_protocol):
        keys, values, rotated, metadata = _event(
            event_dir, legacy_protocol=legacy_protocol
        )
        tail = keys.shape[1:] if legacy_protocol else keys.shape[2:]
        if expected_tail is None:
            expected_tail = tail
        elif tail != expected_tail:
            raise ValueError("all trace events must use the same head layout")
        if legacy_protocol:
            # Events legitimately differ in their absolute token range; only
            # the sampling metadata must match.
            comparable = {
                "token_indices": metadata["token_indices"],
                "frame_seq_length": metadata["frame_seq_length"],
            }
            event_starts.append(int(metadata["tokens_start"]))
            event_ends.append(int(metadata["tokens_end"]))
        else:
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
        present_attention_kinds = set(rotated)
        if expected_attention_kinds is None:
            expected_attention_kinds = present_attention_kinds
        elif expected_attention_kinds != present_attention_kinds:
            raise ValueError("trace events do not capture the same rotated Q/K arrays")
        for kind, array in rotated.items():
            rotated_events.setdefault(kind, []).append(array)
    assert shared is not None
    if legacy_protocol:
        keys = np.stack(key_events, axis=1)
        values = np.stack(value_events, axis=1)
    else:
        keys = np.concatenate(key_events, axis=1)
        values = np.concatenate(value_events, axis=1)
    merge = np.stack if legacy_protocol else np.concatenate
    rotated = {
        kind: merge(parts, axis=1) for kind, parts in rotated_events.items()
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if legacy_protocol:
        frame_counts = [
            (end - start) // int(shared["frame_seq_length"])
            for start, end in zip(event_starts, event_ends)
        ]
        if any(count != 24 for count in frame_counts):
            raise ValueError("legacy trace events must contain exactly 24 frames")
        if tuple(shared["token_indices"]) != _LEGACY_TOKEN_INDICES:
            raise ValueError("legacy trace token indices do not match the reference")
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
        # Store uncompressed: the merged trace is float activation data, so
        # deflate spends about a minute on a production trace to save under a
        # tenth of the file, and every fitting run pays the decompression again.
        np.savez(
            output,
            keys=keys,
            values=values,
            **rotated,
            layer_ids=np.arange(30, dtype=np.int32),
            stride=np.asarray(1560, dtype=np.int32),
            coordinate_system=np.asarray("pre_rope_normalized"),
            cross_source_kind=np.asarray("normalized"),
            trace_layout=np.asarray("legacy_event_bpfhd_v1"),
            token_indices=np.asarray(shared["token_indices"], dtype=np.int32),
            event_tokens_start=np.asarray(event_starts, dtype=np.int64),
            event_tokens_end=np.asarray(event_ends, dtype=np.int64),
            tokens_start=np.asarray(min(event_starts), dtype=np.int64),
            tokens_end=np.asarray(max(event_ends), dtype=np.int64),
            frame_seq_length=np.asarray(shared["frame_seq_length"], dtype=np.int32),
            trace_event_count=np.asarray(len(event_starts), dtype=np.int32),
            prompt_count=np.asarray(_LEGACY_PROMPT_COUNT, dtype=np.int32),
            frames_per_event=np.asarray(frame_counts[0], dtype=np.int32),
            tokens_per_frame=np.asarray(
                len(shared["token_indices"]), dtype=np.int32
            ),
            num_layers=np.asarray(30, dtype=np.int32),
            num_heads=np.asarray(12, dtype=np.int32),
            head_dim=np.asarray(128, dtype=np.int32),
            temporal_units=np.asarray(keys.shape[-3], dtype=np.int32),
            legacy_trace_protocol=np.asarray(_LEGACY_TRACE_PROTOCOL),
        )
    else:
        np.savez_compressed(
            output,
            keys=keys,
            values=values,
            **rotated,
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
        "legacy_protocol": legacy_protocol,
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
