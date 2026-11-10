from __future__ import annotations

from dataclasses import dataclass
import gc
from statistics import median
from time import perf_counter
import weakref

import pytest
import torch

from rebuttal.experiments.memory.accounting import cache_snapshot
from rebuttal.experiments.memory.budget_manager import BudgetManager
from rebuttal.experiments.memory.fast_accounting import FastBudgetManager, TrackedReleasableKVCache


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache(frames=20):
    return TrackedReleasableKVCache(1, 2, 2, 4, frames, torch.bfloat16, DEVICE)


def data(frames):
    return torch.ones(1, frames * 2, 2, 4, dtype=torch.bfloat16, device=DEVICE)


@dataclass
class Payload:
    anchor_quant: object
    residual_quant: object
    anchor_scale: object = None
    residual_scale: object = None
    residual_zero_point: object = None
    channel_mean: object = None
    predictor: object = None


def test_every_event_matches_original_including_aliases_views_and_mutable_parameters():
    caches = [{"k": cache(), "v": cache()} for _ in range(2)]
    base = torch.arange(100, device=DEVICE)
    # Shared allocations and distinct/identical views, both within and across caches.
    for layer in caches:
        for c in layer.values():
            c.write(0, 8, data(4))
            c.chunks[1] = c.chunks[0]
    caches[1]["k"].chunks[0] = caches[0]["k"].chunks[0]
    parameters = {"_layer_cache": {}}
    old_events, new_events = [], []
    old = BudgetManager(caches, budget_bytes=100000, history_frames=4,
                        tokens_per_frame=2, parameters=parameters, event_sink=old_events.append)
    fast = FastBudgetManager(caches, budget_bytes=100000, history_frames=4,
                             tokens_per_frame=2, parameters=parameters, event_sink=new_events.append)

    def check(event="after_write", reserved_bytes=0):
        expected = old.snapshot(event, reserved_bytes=reserved_bytes)
        actual = fast.snapshot(event, reserved_bytes=reserved_bytes)
        assert actual == expected
        return actual

    check("initial", 17)
    for layer in caches:
        for c in layer.values():
            c.after_write = check
    caches[0]["k"].write(8, 10, data(1))  # first allocation
    caches[0]["k"].write(8, 10, data(1))  # existing allocation
    predictor = torch.ones(10000, device=DEVICE)
    payload = Payload(base[:3], base[10:15], predictor=predictor)
    caches[0]["k"].store_quantized(0, 4, payload)
    check("dataclass")
    caches[1]["v"].store_quantized(0, 4, {"a": base[:3], "b": base[10:15], "alias": base[:3]})
    report = check("dict_cross_layer_alias")
    assert report["packed_allocation_bytes"] == base.untyped_storage().nbytes()
    assert report["packed_logical_bytes"] == 8 * base.element_size()
    # Packed and dense categories may also own exactly the same allocation.
    caches[1]["k"].store_quantized(0, 4, {"dense_alias": caches[0]["v"].chunks[0]})
    check("packed_dense_alias")
    parameters["_layer_cache"][0] = {"weights": base, "predictor": predictor}
    check("lazy_parameters")
    payload.residual_quant = base[20:25]
    caches[0]["k"].extend_quantized(0, 6)
    check("extended_mutable_payload")
    caches[0]["k"].store_quantized(0, 6, {"replacement": base.clone()})
    check("packed_replacement")
    caches[0]["k"].write(0, 6, data(3))
    # Manager preparation performs all releases before its unchanged reservation check.
    assert old.prepare(6, 4) == fast.prepare(6, 4)
    for layer in caches:
        for c in layer.values():
            c.clear()
            check("clear")
    assert old_events == new_events
    fast.close()
    assert all(not c._accounting_observers and c.after_write is None
               for layer in caches for c in layer.values())


def test_whole_span_noop_release_does_not_recompute_and_close_drops_observer():
    caches = [{"k": cache(), "v": cache()}]
    c = caches[0]["k"]
    c.write(0, 8, data(4))
    c.store_quantized(0, 8, {"x": torch.ones(8, device=DEVICE)})
    fast = FastBudgetManager(caches, budget_bytes=None, history_frames=4, tokens_per_frame=2)
    ledger = fast._cache_ledgers[c]
    assert c.release_before(3) == 0
    assert fast._cache_ledgers[c] is ledger
    assert c.release_before(4) == 4
    assert fast.snapshot("release")["kv_owned_bytes"] == 0
    ref = weakref.ref(fast)
    fast.close()
    del fast
    gc.collect()
    assert ref() is None


def test_over_budget_raises_at_identical_write_and_reserved_events():
    caches = [{"k": cache(), "v": cache()}]
    old = BudgetManager(caches, budget_bytes=31, history_frames=4, tokens_per_frame=2)
    fast = FastBudgetManager(caches, budget_bytes=31, history_frames=4, tokens_per_frame=2)
    for manager in (old, fast):
        with pytest.raises(MemoryError, match="before_write: 64 > 31 bytes"):
            manager.prepare(1, 0)
        assert manager.events == []
        assert manager.peak_bytes == 64
    c = caches[0]["k"]
    c.after_write = lambda: fast.snapshot("after_write")
    with pytest.raises(MemoryError, match="after_write: 32 > 31 bytes"):
        c.write(0, 2, data(1))
    with pytest.raises(MemoryError, match="after_write: 32 > 31 bytes"):
        old.snapshot("after_write")
    assert fast.peak_bytes == old.peak_bytes
    assert fast.events == old.events == []
    fast.close()


def benchmark():
    """Time storage statistics only; no model or quantization throughput claim."""
    caches = [{"k": cache(192), "v": cache(192)} for _ in range(24)]
    for layer in caches:
        for c in layer.values():
            c.chunks = [torch.empty(1, dtype=torch.bfloat16, device=DEVICE) for _ in range(192)]
    fast = FastBudgetManager(caches, budget_bytes=None, history_frames=192, tokens_per_frame=2)
    assert fast.snapshot("check")["kv_owned_bytes"] == cache_snapshot(caches)["kv_owned_bytes"]
    timings = {}
    for name, fn in (("original", lambda: cache_snapshot(caches)),
                     ("incremental_one_cache_and_snapshot",
                      lambda: (caches[0]["k"]._touch(), fast.snapshot("benchmark")))):
        samples = []
        for _ in range(12):
            start = perf_counter()
            fn()
            samples.append(perf_counter() - start)
        timings[name] = median(samples)
    fast.close()
    return {"device": str(DEVICE), "layers": 24, "chunks_per_cache": 192,
            "median_seconds": timings,
            "statistics_speedup": timings["original"] / timings["incremental_one_cache_and_snapshot"]}


if __name__ == "__main__":
    import json
    print(json.dumps(benchmark(), indent=2))


@pytest.mark.parametrize("method", ["bf16", "tempokv"])
def test_fast_worker_uses_original_pipeline_and_budget_schedule(monkeypatch, tmp_path, method):
    from rebuttal.experiments.memory import worker
    from rebuttal.tests.test_memory_budget import (
        test_actual_worker_binds_pipeline_and_enforces_budget_during_rollout,
    )
    monkeypatch.setattr(worker, "BudgetManager", FastBudgetManager)
    monkeypatch.setattr(worker, "ReleasableKVCache", TrackedReleasableKVCache)
    test_actual_worker_binds_pipeline_and_enforces_budget_during_rollout(monkeypatch, tmp_path, method)
