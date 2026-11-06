"""KV-owned storage accounting; aliases and views count their allocation once."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Iterator

import torch


PAYLOAD_FIELDS = (
    "anchor_quant", "anchor_scale", "residual_quant", "residual_scale",
    "residual_zero_point", "channel_mean",
)


def tensors(value: Any, seen: set[int] | None = None) -> Iterator[torch.Tensor]:
    """Walk payload containers without following a packed state's predictor parameters."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif hasattr(value, "anchor_quant") and hasattr(value, "residual_quant"):
        for name in PAYLOAD_FIELDS:
            yield from tensors(getattr(value, name, None), seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from tensors(getattr(value, field.name), seen)


def allocation_bytes(value: Any) -> int:
    allocations: dict[tuple, int] = {}
    for tensor in tensors(value):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        allocations[key] = storage.nbytes()
    return sum(allocations.values())


def logical_bytes(value: Any) -> int:
    """Count logical tensors, deduplicating identical views (not their allocation)."""
    views: dict[tuple, int] = {}
    for tensor in tensors(value):
        key = (str(tensor.device), tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
               tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
        views[key] = tensor.numel() * tensor.element_size()
    return sum(views.values())


def cache_snapshot(caches: list[dict], *, history: Any = None, parameters: Any = None) -> dict:
    packed = [span["quant_data"] for layer in caches for role in ("k", "v")
              for span in layer[role].quantized_spans]
    dense = [chunk for layer in caches for role in ("k", "v")
             for chunk in layer[role].chunks if chunk is not None]
    owned = {"packed": packed, "dense": dense, "history": history}
    return {
        "scope": "measured_tensor_allocations",
        "packed_allocation_bytes": allocation_bytes(packed),
        "packed_logical_bytes": logical_bytes(packed),
        "dense_bytes": allocation_bytes(dense),
        "history_bytes": allocation_bytes(history),
        "kv_owned_bytes": allocation_bytes(owned),
        "parameter_bytes": allocation_bytes(parameters),
        "workspace_bytes": None,
        "workspace_scope": "excluded_from_kv_budget; whole_device_peak_reported_by_worker",
    }
