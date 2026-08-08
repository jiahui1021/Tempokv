#!/usr/bin/env python3
"""Convert LongCat trace events into TempoKV fitting layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


LONGCAT_NUM_LAYERS = 48
LONGCAT_NUM_HEADS = 32
LONGCAT_HEAD_DIM = 128
LONGCAT_STRIDE = 1560


def _scalar(data: np.lib.npyio.NpzFile, name: str, default: object = None) -> object:
    if name not in data:
        return default
    return np.asarray(data[name]).item()


def _event_directories(root: Path) -> list[Path]:
    directories = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not directories:
        raise ValueError(f"no LongCat layer trace files found under {root}")
    return directories


def _event(
    event_dir: Path,
    *,
    expected_layers: int | None,
    expected_heads: int | None,
    expected_head_dim: int | None,
    expected_stride: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    files = sorted(event_dir.glob("layer_*.npz"))
    if expected_layers is not None and len(files) != expected_layers:
        raise ValueError(
            f"{event_dir} must contain exactly {expected_layers} layer files, "
            f"found {len(files)}"
        )
    if not files:
        raise ValueError(f"{event_dir} contains no layer files")

    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    seen_layers: list[int] = []
    metadata: dict[str, int | str] | None = None
    expected_tail: tuple[int, int, int, int] | None = None

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {"k_pre_rope", "v", "layer_index"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
            layer = int(np.asarray(data["layer_index"]).item())
            key = np.asarray(data["k_pre_rope"])
            value = np.asarray(data["v"])
            stride = int(_scalar(data, "stride", expected_stride or -1))
            temporal_units = int(_scalar(data, "temporal_units", key.shape[1]))
            coordinate_system = str(
                _scalar(data, "coordinate_system", "pre_rope_normalized")
            )
            cross_source_kind = str(
                _scalar(data, "cross_source_kind", "normalized")
            )

        if key.shape != value.shape or key.ndim != 5:
            raise ValueError(f"{path} must use matching [B,F,P,H,D] K/V arrays")
        batch, frames, samples, heads, width = (int(item) for item in key.shape)
        if frames != temporal_units:
            raise ValueError(f"{path} temporal_units metadata does not match its shape")
        if expected_heads is not None and heads != expected_heads:
            raise ValueError(f"{path} must contain {expected_heads} heads, got {heads}")
        if expected_head_dim is not None and width != expected_head_dim:
            raise ValueError(
                f"{path} must contain head dimension {expected_head_dim}, got {width}"
            )
        if stride <= 0:
            raise ValueError(f"{path} contains an invalid stride {stride}")
        if expected_stride is not None and stride != expected_stride:
            raise ValueError(
                f"{path} was collected with stride {stride}, expected {expected_stride}"
            )
        if coordinate_system != "pre_rope_normalized":
            raise ValueError(
                f"{path} uses {coordinate_system!r}; LongCat requires "
                "pre_rope_normalized"
            )
        if cross_source_kind != "normalized":
            raise ValueError(f"{path} has unsupported CrossKV source metadata")

        tail = (frames, samples, heads, width)
        if expected_tail is None:
            expected_tail = tail
            metadata = {
                "temporal_units": frames,
                "stride": stride,
                "num_heads": heads,
                "head_dim": width,
                "coordinate_system": coordinate_system,
            }
        elif tail != expected_tail:
            raise ValueError(f"all layers in {event_dir} must share [F,P,H,D]")

        # [B,F,P,H,D] -> [B*P,F,H,D], retaining one spatial sequence per sample.
        keys.append(key.transpose(0, 2, 1, 3, 4).reshape(-1, frames, heads, width))
        values.append(
            value.transpose(0, 2, 1, 3, 4).reshape(-1, frames, heads, width)
        )
        seen_layers.append(layer)

    if seen_layers != list(range(len(files))):
        raise ValueError(f"{event_dir} layer indices must be 0 through {len(files) - 1}")
    assert metadata is not None
    return np.stack(keys), np.stack(values), metadata


def convert_trace_root(
    root: Path,
    output: Path,
    *,
    expected_layers: int | None = LONGCAT_NUM_LAYERS,
    expected_heads: int | None = LONGCAT_NUM_HEADS,
    expected_head_dim: int | None = LONGCAT_HEAD_DIM,
    expected_stride: int | None = LONGCAT_STRIDE,
) -> dict[str, object]:
    """Merge event directories into ``[L,S,T,H,D]`` fitting arrays."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {root}")
    key_events: list[np.ndarray] = []
    value_events: list[np.ndarray] = []
    event_metadata: dict[str, int | str] | None = None
    for event_dir in _event_directories(root):
        keys, values, metadata = _event(
            event_dir,
            expected_layers=expected_layers,
            expected_heads=expected_heads,
            expected_head_dim=expected_head_dim,
            expected_stride=expected_stride,
        )
        if event_metadata is None:
            event_metadata = metadata
        elif metadata != event_metadata:
            raise ValueError("all LongCat trace events must use the same metadata")
        key_events.append(keys)
        value_events.append(values)

    keys = np.concatenate(key_events, axis=1)
    values = np.concatenate(value_events, axis=1)
    assert event_metadata is not None
    layers, samples, frames, heads, width = (int(item) for item in keys.shape)
    if expected_layers is not None and layers != expected_layers:
        raise ValueError(f"expected {expected_layers} layers, got {layers}")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        keys=keys,
        values=values,
        layer_ids=np.arange(layers, dtype=np.int32),
        stride=np.asarray(event_metadata["stride"], dtype=np.int32),
        temporal_units=np.asarray(frames, dtype=np.int32),
        num_layers=np.asarray(layers, dtype=np.int32),
        num_heads=np.asarray(heads, dtype=np.int32),
        head_dim=np.asarray(width, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
    )
    report = {
        "status": "complete",
        "source": str(root),
        "output": str(output),
        "events": len(key_events),
        "shape": list(keys.shape),
        "stride": int(event_metadata["stride"]),
        "coordinate_system": "pre_rope_normalized",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert LongCat [B,F,P,H,D] traces into TempoKV fitting layout."
    )
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-layers", type=int, default=LONGCAT_NUM_LAYERS)
    parser.add_argument("--expected-heads", type=int, default=LONGCAT_NUM_HEADS)
    parser.add_argument("--expected-head-dim", type=int, default=LONGCAT_HEAD_DIM)
    parser.add_argument("--stride", type=int, default=LONGCAT_STRIDE)
    parser.add_argument(
        "--allow-dynamic-shape",
        action="store_true",
        help="skip the default LongCat 48/32/128 shape checks",
    )
    args = parser.parse_args(argv)
    try:
        convert_trace_root(
            args.trace_root,
            args.output,
            expected_layers=None if args.allow_dynamic_shape else args.expected_layers,
            expected_heads=None if args.allow_dynamic_shape else args.expected_heads,
            expected_head_dim=None if args.allow_dynamic_shape else args.expected_head_dim,
            expected_stride=args.stride,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
