"""Experiment-only incremental storage accounting with unchanged budget events.

Only the cache that changed is traversed. Global ledgers contain metadata, never
payload tensors, and deduplicate shared allocations/views across cache owners.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from tempokv.runtime.cache import ChunkedKVCache

from .accounting import allocation_bytes, tensors
from .budget_manager import BudgetManager, ReleasableKVCache


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def _cache_maps(cache):
    packed, dense, logical = {}, {}, {}
    payloads = [span["quant_data"] for span in cache.quantized_spans]
    for tensor in tensors(payloads):
        key = _storage_key(tensor)
        packed[key] = key[2]
        view = (key[0], key[1], tensor.storage_offset(), tuple(tensor.shape),
                tuple(tensor.stride()), str(tensor.dtype))
        logical[view] = tensor.numel() * tensor.element_size()
    for tensor in tensors(cache.chunks):
        key = _storage_key(tensor)
        dense[key] = key[2]
    return {"packed_allocation_bytes": packed, "packed_logical_bytes": logical,
            "dense_bytes": dense, "kv_owned_bytes": packed | dense}


class TrackedReleasableKVCache(ReleasableKVCache):
    """Notify accounting observers before the original post-write budget check."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accounting_observers: set[Callable] = set()

    def _touch(self):
        for observer in tuple(self._accounting_observers):
            observer(self)

    def write(self, *args, **kwargs):
        # ReleasableKVCache.write invokes after_write too early for a ledger.
        ChunkedKVCache.write(self, *args, **kwargs)
        self._touch()
        if self.after_write is not None:
            self.after_write()

    def store_quantized(self, *args, **kwargs):
        super().store_quantized(*args, **kwargs)
        self._touch()

    def extend_quantized(self, *args, **kwargs):
        super().extend_quantized(*args, **kwargs)
        self._touch()

    def clear(self):
        super().clear()
        self._touch()

    def release_before(self, first_visible_frame):
        removed = super().release_before(first_visible_frame)
        if removed:
            self._touch()
        return removed


class FastBudgetManager(BudgetManager):
    """Same budget policy and report schema, with per-cache allocation ledgers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_ledgers = {}
        self._refs = {name: Counter() for name in (
            "packed_allocation_bytes", "packed_logical_bytes", "dense_bytes", "kv_owned_bytes")}
        self._totals = dict.fromkeys(self._refs, 0)
        for layer in self.caches:
            for role in ("k", "v"):
                cache = layer[role]
                if not isinstance(cache, TrackedReleasableKVCache):
                    raise TypeError("FastBudgetManager requires TrackedReleasableKVCache")
        for layer in self.caches:
            for role in ("k", "v"):
                cache = layer[role]
                cache._accounting_observers.add(self._touch)
                if cache not in self._cache_ledgers:
                    self._touch(cache)

    def _touch(self, cache):
        updated = _cache_maps(cache)
        previous = self._cache_ledgers.get(cache, {})
        for name, current in updated.items():
            old = previous.get(name, {})
            refs = self._refs[name]
            for key in old.keys() - current.keys():
                refs[key] -= 1
                if refs[key] == 0:
                    del refs[key]
                    self._totals[name] -= old[key]
            for key in current.keys() - old.keys():
                if refs[key] == 0:
                    self._totals[name] += current[key]
                refs[key] += 1
        self._cache_ledgers[cache] = updated

    def snapshot(self, event: str, *, reserved_bytes: int = 0) -> dict:
        report = {
            "scope": "measured_tensor_allocations",
            **self._totals,
            "history_bytes": 0,
            # Codec parameter containers can populate lazily during generation.
            "parameter_bytes": allocation_bytes(self.parameters),
            "workspace_bytes": None,
            "workspace_scope": "excluded_from_kv_budget; whole_device_peak_reported_by_worker",
        }
        usage = report["kv_owned_bytes"] + reserved_bytes
        self.peak_bytes = max(self.peak_bytes, usage)
        report.update(event=event, end_frame=self.end_frame, reserved_incoming_bytes=reserved_bytes,
                      checked_bytes=usage, budget_bytes=self.budget_bytes,
                      peak_checked_bytes=self.peak_bytes,
                      first_visible_frame=max(0, self.end_frame - self.history_frames))
        if self.budget_bytes is not None and usage > self.budget_bytes:
            raise MemoryError(f"KV budget exceeded at {event}: {usage} > {self.budget_bytes} bytes")
        self.events.append(report)
        if self.event_sink is not None:
            self.event_sink(report)
        return report

    def close(self):
        for cache in self._cache_ledgers:
            cache._accounting_observers.discard(self._touch)
        self._cache_ledgers.clear()
        for refs in self._refs.values():
            refs.clear()
        self._totals = dict.fromkeys(self._totals, 0)
        super().close()
