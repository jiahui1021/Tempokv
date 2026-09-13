"""Ensure actual fused routes and step preparation are visible in profiles."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tempokv import profiling
from tempokv.codec import TempoKVCodec
from tempokv.runtime.cache import ChunkedKVCache
from tempokv.runtime import rope


@pytest.mark.parametrize("fused_available", [False, True])
def test_install_profiles_fused_aliases_and_steps(monkeypatch, fused_available):
    # Register original values with monkeypatch so install()'s direct setattr
    # calls are undone too; do not leave global runtime instrumentation behind.
    for owner, name in ((ChunkedKVCache, "_read_layout"),
                        (ChunkedKVCache, "read_step_sizes"),
                        (TempoKVCodec, "encode_key"), (TempoKVCodec, "encode_value"),
                        (rope, "apply_rope_by_span")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    for name in ("bridge", "pipeline.causal_inference", "utils.wan_wrapper"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    class Attention:
        def forward(self):
            return 123

    model = SimpleNamespace(attention=lambda: 123, CausalWanSelfAttention=Attention)
    if fused_available:
        model.jensen_flex_attention = lambda: "flex"
        model.jensen_triton_attention = lambda: "triton"
    monkeypatch.setitem(sys.modules, "wan.modules.causal_model", model)
    profile = profiling.RuntimeProfile()
    status = profiling.install(profile)
    assert status[profiling.JENSEN_STEPS] == "patched"
    with profile.section(profiling.GENERATION):
        with profile.section(profiling.SELF_ATTENTION):
            assert model.attention() == 123
            if fused_available:
                assert model.jensen_flex_attention() == "flex"
                assert model.jensen_triton_attention() == "triton"
            # Invoke the real wrapped method on invalid input: exception-safe
            # accounting must still record the attempted preparation interval.
            with pytest.raises(AttributeError):
                ChunkedKVCache.read_step_sizes(object(), 0, 1, group_size=64)
    report = profile.report()
    totals = profiling.aggregate_by_name(report)
    assert totals[profiling.ATTENTION_KERNEL]["calls"] == (3 if fused_available else 1)
    assert totals[profiling.JENSEN_STEPS]["calls"] == 1
    for name in ("jensen_flex_attention", "jensen_triton_attention"):
        assert status[name] == "patched" if fused_available else status[name].startswith("skipped:")
        if fused_available:
            assert totals[name]["calls"] == 1
    entries = {e["path"]: e for e in report["sections"]}
    path = "generation/self_attention/attention_kernel"
    children = sum(e["total_ms"] for p, e in entries.items() if p.rpartition("/")[0] == path)
    assert entries[path]["self_ms"] == pytest.approx(entries[path]["total_ms"] - children)


def test_latency_comparison_separates_backend_and_correction():
    path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_jensen_fused.py"
    spec = importlib.util.spec_from_file_location("jensen_benchmark_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.comparison(15, 16, 10)
    assert result["backend_extra_ms"] == 5
    assert result["correction_extra_ms"] == 1
    assert result["backend_vs_fa2_pct"] == 50
    assert result["correction_vs_same_backend_pct"] == pytest.approx(100 / 15)
    assert result["total_vs_fa2_pct"] == pytest.approx(60)
