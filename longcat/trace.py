"""Clean LongCat KV trace collection for TempoKV parameter fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


DEFAULT_TRACE_SPATIAL_POSITIONS = 32


def _spatial_indices(num_positions: int, limit: int) -> np.ndarray:
    if num_positions <= 0:
        raise ValueError("the cache must contain at least one spatial position")
    if limit <= 0:
        raise ValueError("max_spatial_positions must be positive")
    if limit >= num_positions:
        return np.arange(num_positions, dtype=np.int64)
    # Linspace keeps the sample spread over the whole frame and is deterministic.
    return np.linspace(0, num_positions - 1, limit, dtype=np.int64)


def _as_float32_numpy(value: torch.Tensor) -> np.ndarray:
    # NumPy does not support torch.bfloat16 on all supported versions.  Fitting
    # already converts CUDA traces to float32, so use that stable interchange
    # dtype for every saved trace.
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def _reshape_sample(
    value: torch.Tensor,
    *,
    temporal_units: int,
    spatial_indices: np.ndarray,
) -> np.ndarray:
    if value.ndim != 4:
        raise ValueError(f"LongCat KV tensors must use [B,H,S,D], got {tuple(value.shape)}")
    batch, heads, sequence, width = (int(item) for item in value.shape)
    if sequence % temporal_units:
        raise ValueError(
            f"cache sequence length {sequence} is not divisible by temporal units "
            f"{temporal_units}"
        )
    spatial = sequence // temporal_units
    if spatial_indices.size and int(spatial_indices.max()) >= spatial:
        raise ValueError("spatial trace index exceeds the cache frame size")
    array = _as_float32_numpy(value).reshape(
        batch, heads, temporal_units, spatial, width
    )
    # [B,H,F,P,D] -> [B,F,P,H,D], then retain one shared P index set for all F.
    return array.transpose(0, 2, 3, 1, 4)[:, :, spatial_indices, :, :]


def _validate_cache(
    kv_cache_dict: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    temporal_units: int,
) -> tuple[list[int], int, int, int, int]:
    if temporal_units <= 0:
        raise ValueError("temporal_units must be positive")
    if not kv_cache_dict:
        raise ValueError("LongCat returned an empty KV cache")

    layer_ids = sorted(int(layer) for layer in kv_cache_dict)
    expected_layers = list(range(len(layer_ids)))
    if layer_ids != expected_layers:
        raise ValueError(
            f"LongCat layer ids must be contiguous from zero, got {layer_ids}"
        )

    first_key, first_value = kv_cache_dict[layer_ids[0]]
    if not isinstance(first_key, torch.Tensor) or not isinstance(first_value, torch.Tensor):
        raise TypeError("trace collection requires uncompressed Tensor K/V caches")
    if first_key.shape != first_value.shape or first_key.ndim != 4:
        raise ValueError("each LongCat K/V pair must have matching [B,H,S,D] shapes")
    batch, heads, sequence, width = (int(item) for item in first_key.shape)
    if sequence % temporal_units:
        raise ValueError(
            f"cache sequence length {sequence} is not divisible by temporal units "
            f"{temporal_units}"
        )
    for layer_id in layer_ids[1:]:
        key, value = kv_cache_dict[layer_id]
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("trace collection requires uncompressed Tensor K/V caches")
        if tuple(key.shape) != (batch, heads, sequence, width) or key.shape != value.shape:
            raise ValueError("all LongCat layers must share one [B,H,S,D] cache layout")

    return layer_ids, batch, heads, sequence, width


@dataclass(slots=True)
class LongCatTraceCollector:
    """Write clean, pre-RoPE LongCat cache events for the fitter.

    The collector samples the same spatial positions for K and V and preserves
    the temporal axis.  Every event contains one file per LongCat layer.
    """

    output_dir: Path
    max_spatial_positions: int = DEFAULT_TRACE_SPATIAL_POSITIONS
    coordinate_system: str = "pre_rope_normalized"
    cross_source_kind: str = "normalized"
    _event_index: int = 0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.max_spatial_positions <= 0:
            raise ValueError("max_spatial_positions must be positive")
        if self.coordinate_system != "pre_rope_normalized":
            raise ValueError(
                "LongCat TempoKV traces must use pre_rope_normalized coordinates"
            )
        if self.output_dir.is_dir():
            existing = []
            for path in self.output_dir.iterdir():
                if path.is_dir() and path.name.startswith("event_"):
                    try:
                        existing.append(int(path.name.removeprefix("event_")))
                    except ValueError:
                        continue
            if existing:
                self._event_index = max(existing) + 1

    def record(
        self,
        kv_cache_dict: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        *,
        temporal_units: int,
        event_name: str | None = None,
    ) -> dict[str, object]:
        layer_ids, batch, heads, sequence, width = _validate_cache(
            kv_cache_dict, temporal_units=temporal_units
        )
        stride = sequence // temporal_units
        indices = _spatial_indices(stride, min(self.max_spatial_positions, stride))
        if event_name is None:
            event_name = f"event_{self._event_index:05d}"
        self._event_index += 1
        if not event_name or Path(event_name).name != event_name:
            raise ValueError("event_name must be a non-empty directory name")
        event_dir = self.output_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=False)

        for layer_id in layer_ids:
            key, value = kv_cache_dict[layer_id]
            if key.shape != value.shape:
                raise ValueError(f"layer {layer_id} has mismatched K/V shapes")
            key_sample = _reshape_sample(
                key, temporal_units=temporal_units, spatial_indices=indices
            )
            value_sample = _reshape_sample(
                value, temporal_units=temporal_units, spatial_indices=indices
            )
            np.savez_compressed(
                event_dir / f"layer_{layer_id:02d}.npz",
                k_pre_rope=key_sample,
                v=value_sample,
                layer_index=np.asarray(layer_id, dtype=np.int32),
                temporal_units=np.asarray(temporal_units, dtype=np.int32),
                trace_stride=np.asarray(stride, dtype=np.int32),
                # Compatibility alias for traces written before trace/runtime
                # stride metadata was separated.
                stride=np.asarray(stride, dtype=np.int32),
                spatial_indices=indices,
                coordinate_system=np.asarray(self.coordinate_system),
                cross_source_kind=np.asarray(self.cross_source_kind),
            )

        return {
            "event": event_name,
            "directory": str(event_dir),
            "layers": len(layer_ids),
            "shape": [batch, temporal_units, int(indices.size), heads, width],
            "sequence": sequence,
            "trace_stride": stride,
            "coordinate_system": self.coordinate_system,
        }


def dump_clean_kv(
    kv_cache_dict: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    output_dir: str | Path,
    *,
    temporal_units: int,
    max_spatial_positions: int = DEFAULT_TRACE_SPATIAL_POSITIONS,
    event_name: str | None = None,
) -> dict[str, object]:
    """Write one LongCat clean-cache event and return its metadata."""
    collector = LongCatTraceCollector(
        Path(output_dir), max_spatial_positions=max_spatial_positions
    )
    return collector.record(
        kv_cache_dict, temporal_units=temporal_units, event_name=event_name
    )
