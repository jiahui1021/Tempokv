#!/usr/bin/env python3
"""Convert sampled Causal Forcing layer traces into TempoKV fitting layout.

Per-layer events are stored as ``[B, F, P, H, D]``.  The native path treats
each sampled position as a sequence, giving ``[L, B*P, F, H, D]``.  The legacy
path (``--legacy-protocol``) keeps ``[B, P, F, H, D]`` and stacks events into
the ``legacy_event_bpfhd_v1`` layout; ``fit_params.py`` checks that protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


_ATTENTION_KINDS = ("q_post_rope", "k_post_rope")
_LEGACY_PROMPT_COUNT = 12
_LEGACY_TRACE_PROTOCOL = f"direct_v_{_LEGACY_PROMPT_COUNT}p_51f_8t_24x2_v1"


def _scalar(data, name: str):
    return np.asarray(data[name]).item()


def _event(
    event_dir: Path, *, legacy_protocol: bool
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Stack one event's layer files into ``{name: [L, ...]}`` arrays."""
    arrays: dict[str, list[np.ndarray]] = {}
    metadata: dict[str, object] = {}
    for path in sorted(event_dir.glob("layer_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            key = np.asarray(data["k_pre_rope"])
            _, frames, _, heads, width = key.shape
            for name, source in (
                ("keys", "k_pre_rope"),
                ("values", "v"),
                *((kind, kind) for kind in _ATTENTION_KINDS if kind in data.files),
            ):
                moved = np.asarray(data[source]).transpose(0, 2, 1, 3, 4)
                if not legacy_protocol:
                    moved = moved.reshape(-1, frames, heads, width)
                arrays.setdefault(name, []).append(moved)
            if metadata:
                continue
            metadata["token_indices"] = tuple(int(i) for i in data["token_indices"])
            if legacy_protocol:
                for name in ("tokens_start", "tokens_end", "frame_seq_length"):
                    metadata[name] = int(_scalar(data, name))
            else:
                for name in (
                    "num_layers",
                    "num_heads",
                    "head_dim",
                    "trace_stride",
                    "num_frame_per_block",
                ):
                    metadata[name] = int(_scalar(data, name))
                for name in ("coordinate_system", "cross_source_kind", "generation_mode"):
                    metadata[name] = str(_scalar(data, name))
    return {name: np.stack(parts) for name, parts in arrays.items()}, metadata


def convert_trace_root(
    root: Path,
    output: Path,
    *,
    legacy_protocol: bool = False,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    event_dirs = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not event_dirs:
        raise ValueError(f"no layer trace files found under {root}")
    events: dict[str, list[np.ndarray]] = {}
    event_starts: list[int] = []
    event_ends: list[int] = []
    shared: dict[str, object] = {}
    for event_dir in event_dirs:
        arrays, metadata = _event(event_dir, legacy_protocol=legacy_protocol)
        shared = shared or metadata
        if legacy_protocol:
            event_starts.append(int(metadata["tokens_start"]))
            event_ends.append(int(metadata["tokens_end"]))
        for name, array in arrays.items():
            events.setdefault(name, []).append(array)
    merge = np.stack if legacy_protocol else np.concatenate
    merged = {name: merge(parts, axis=1) for name, parts in events.items()}
    keys = merged["keys"]
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if legacy_protocol:
        # Uncompressed: deflate barely shrinks float activations and every
        # fitting run would pay the decompression again.
        np.savez(
            output,
            **merged,
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
            frames_per_event=np.asarray(keys.shape[-3], dtype=np.int32),
            tokens_per_frame=np.asarray(len(shared["token_indices"]), dtype=np.int32),
            num_layers=np.asarray(30, dtype=np.int32),
            num_heads=np.asarray(12, dtype=np.int32),
            head_dim=np.asarray(128, dtype=np.int32),
            temporal_units=np.asarray(keys.shape[-3], dtype=np.int32),
            legacy_trace_protocol=np.asarray(_LEGACY_TRACE_PROTOCOL),
        )
    else:
        np.savez_compressed(
            output,
            **merged,
            layer_ids=np.arange(int(shared["num_layers"]), dtype=np.int32),
            stride=np.asarray(shared["trace_stride"], dtype=np.int32),
            coordinate_system=np.asarray(shared["coordinate_system"]),
            cross_source_kind=np.asarray(shared["cross_source_kind"]),
            generation_mode=np.asarray(shared["generation_mode"]),
            num_frame_per_block=np.asarray(
                shared["num_frame_per_block"], dtype=np.int32
            ),
            num_heads=np.asarray(shared["num_heads"], dtype=np.int32),
            head_dim=np.asarray(shared["head_dim"], dtype=np.int32),
            token_indices=np.asarray(shared["token_indices"], dtype=np.int32),
            frame_seq_length=np.asarray(shared["trace_stride"], dtype=np.int32),
            trace_event_count=np.asarray(len(event_dirs), dtype=np.int32),
        )
    report = {
        "status": "complete",
        "source": str(root),
        "output": str(output),
        "events": len(event_dirs),
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
    convert_trace_root(
        args.trace_root, args.output, legacy_protocol=args.legacy_protocol
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
