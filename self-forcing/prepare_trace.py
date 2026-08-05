#!/usr/bin/env python3
"""Convert sampled Self-Forcing layer traces into TempoKV fitting layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _event_directories(root: Path) -> list[Path]:
    directories = sorted({path.parent for path in root.rglob("layer_*.npz")})
    if not directories:
        raise ValueError(f"no layer trace files found under {root}")
    return directories


def _event(event_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(event_dir.glob("layer_*.npz"))
    if len(files) != 30:
        raise ValueError(f"{event_dir} must contain exactly 30 layer files")
    keys = []
    values = []
    seen_layers = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {"k_pre_rope", "v", "layer_index"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
            layer = int(np.asarray(data["layer_index"]).item())
            key = np.asarray(data["k_pre_rope"])
            value = np.asarray(data["v"])
        if key.shape != value.shape or key.ndim != 5:
            raise ValueError(f"{path} must use [B,F,P,H,D] matching K/V arrays")
        if key.shape[-2:] != (12, 128):
            raise ValueError(f"{path} must contain 12 heads with dimension 128")
        # [B,F,P,H,D] -> [B*P,F,H,D], with F as the temporal axis.
        batch, frames, samples, heads, width = key.shape
        keys.append(key.transpose(0, 2, 1, 3, 4).reshape(-1, frames, heads, width))
        values.append(
            value.transpose(0, 2, 1, 3, 4).reshape(-1, frames, heads, width)
        )
        seen_layers.append(layer)
    if seen_layers != list(range(30)):
        raise ValueError(f"{event_dir} layer indices must be 0 through 29")
    return np.stack(keys), np.stack(values)


def convert_trace_root(root: Path, output: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {root}")
    key_events = []
    value_events = []
    expected_tail = None
    for event_dir in _event_directories(root):
        keys, values = _event(event_dir)
        tail = keys.shape[2:]
        if expected_tail is None:
            expected_tail = tail
        elif tail != expected_tail:
            raise ValueError("all trace events must use the same frame/head layout")
        key_events.append(keys)
        value_events.append(values)
    keys = np.concatenate(key_events, axis=1)
    values = np.concatenate(value_events, axis=1)
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
    )
    report = {
        "status": "complete",
        "source": str(root),
        "output": str(output),
        "events": len(key_events),
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
    args = parser.parse_args(argv)
    try:
        convert_trace_root(args.trace_root, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

