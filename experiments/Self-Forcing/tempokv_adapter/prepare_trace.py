#!/usr/bin/env python3
"""Convert sampled Self-Forcing layer traces into TempoKV fitting layout.

Per-layer events are stored as ``[B, F, P, H, D]``.  The flat layout treats
each sampled position as a sequence (``[L, B*P, F, H, D]``); the legacy layout
keeps ``[B, P, F, H, D]`` per event so fitting can replay record by record.
``fit_params.py`` checks the legacy protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


_LEGACY_PROMPT_COUNT = 12
_LEGACY_TRACE_PROTOCOL = f"direct_v_{_LEGACY_PROMPT_COUNT}p_51f_8t_24x2_v1"
_ATTENTION_KINDS = ("q_post_rope", "k_post_rope")


def _event(
    event_dir: Path, *, legacy_protocol: bool
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Stack one event's layer files into ``{name: [L, ...]}`` arrays."""
    arrays: dict[str, list[np.ndarray]] = {}
    metadata: dict[str, object] = {}
    for path in sorted(event_dir.glob("layer_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            _, frames, _, heads, width = data["k_pre_rope"].shape
            for name, source in (
                ("keys", "k_pre_rope"),
                ("values", "v"),
                *((kind, kind) for kind in _ATTENTION_KINDS if kind in data.files),
            ):
                moved = np.asarray(data[source]).transpose(0, 2, 1, 3, 4)
                if not legacy_protocol:
                    moved = moved.reshape(-1, frames, heads, width)
                arrays.setdefault(name, []).append(moved)
            if not metadata:
                metadata = {
                    "token_indices": tuple(int(i) for i in data["token_indices"]),
                    **{
                        name: int(np.asarray(data[name]).item())
                        for name in ("tokens_start", "tokens_end", "frame_seq_length")
                    },
                }
    return {name: np.stack(parts) for name, parts in arrays.items()}, metadata


def convert_trace_root(
    root: Path,
    output: Path,
    *,
    legacy_protocol: bool = False,
    trace_role: str = "calibration",
) -> dict[str, object]:
    root = root.expanduser().resolve()
    event_dirs = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not event_dirs:
        raise ValueError(f"no layer trace files found under {root}")
    # Events are ordered by rank, then index.  Inference re-seeds per prompt
    # from the global prompt index, so multi-GPU collection yields the same
    # records; only their summation order depends on the rank split.
    events: dict[str, list[np.ndarray]] = {}
    event_starts: list[int] = []
    event_ends: list[int] = []
    protocol: dict[str, object] = {}
    for event_dir in event_dirs:
        arrays, metadata = _event(event_dir, legacy_protocol=legacy_protocol)
        protocol = protocol or metadata
        event_starts.append(int(metadata["tokens_start"]))
        event_ends.append(int(metadata["tokens_end"]))
        for name, array in arrays.items():
            events.setdefault(name, []).append(array)
    merge = np.stack if legacy_protocol else np.concatenate
    merged = {name: merge(parts, axis=1) for name, parts in events.items()}
    keys = merged["keys"]
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed: deflate barely shrinks float activations and every fitting
    # run would pay the decompression again.
    np.savez(
        output,
        **merged,
        layer_ids=np.arange(30, dtype=np.int32),
        stride=np.asarray(1560, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
        trace_layout=np.asarray(
            "legacy_event_bpfhd_v1" if legacy_protocol else "tempokv_flat_samples_v1"
        ),
        token_indices=np.asarray(protocol["token_indices"], dtype=np.int32),
        event_tokens_start=np.asarray(event_starts, dtype=np.int64),
        event_tokens_end=np.asarray(event_ends, dtype=np.int64),
        tokens_start=np.asarray(min(event_starts), dtype=np.int64),
        tokens_end=np.asarray(max(event_ends), dtype=np.int64),
        frame_seq_length=np.asarray(protocol["frame_seq_length"], dtype=np.int32),
        trace_event_count=np.asarray(len(event_starts), dtype=np.int32),
        prompt_count=np.asarray(
            _LEGACY_PROMPT_COUNT if legacy_protocol else 0, dtype=np.int32
        ),
        frames_per_event=np.asarray(
            (event_ends[0] - event_starts[0]) // int(protocol["frame_seq_length"]),
            dtype=np.int32,
        ),
        tokens_per_frame=np.asarray(len(protocol["token_indices"]), dtype=np.int32),
        num_layers=np.asarray(30, dtype=np.int32),
        num_heads=np.asarray(12, dtype=np.int32),
        head_dim=np.asarray(128, dtype=np.int32),
        temporal_units=np.asarray(keys.shape[-3], dtype=np.int32),
        legacy_trace_protocol=np.asarray(
            _LEGACY_TRACE_PROTOCOL if legacy_protocol else ""
        ),
        trace_role=np.asarray(trace_role),
    )
    report = {
        "status": "complete",
        "source": str(root),
        "output": str(output),
        "events": len(event_starts),
        "legacy_protocol": legacy_protocol,
        "trace_role": trace_role,
        "arrays": sorted(merged),
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
    parser.add_argument(
        "--trace-role", choices=("calibration", "diagnostic"), default="calibration"
    )
    args = parser.parse_args(argv)
    convert_trace_root(
        args.trace_root,
        args.output,
        legacy_protocol=args.legacy_protocol,
        trace_role=args.trace_role,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
