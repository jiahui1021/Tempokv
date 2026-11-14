"""Exercise the real native pipeline loop with deterministic small CPU projections."""

from __future__ import annotations

import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from tempokv.research.cache_recovery import BRANCHES, RecoveryRollout, RecoverySettings

ROOT = Path(__file__).resolve().parents[2]


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        attention = SimpleNamespace(num_heads=1, head_dim=4, sink_size=0)
        self.model = SimpleNamespace(local_attn_size=-1, blocks=[SimpleNamespace(self_attn=attention)])

    def get_scheduler(self):
        return SimpleNamespace(add_noise=lambda clean, noise, step: clean + 0.25 * noise)

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, kv_cache,
                crossattn_cache, current_start):
        x = noisy_image_or_video
        assert timestep.shape == x.shape[:2]
        layer = kv_cache[0]
        end = current_start + x.shape[1]
        value = x.reshape(1, x.shape[1], 1, 4)
        history = layer["v"][:, :current_start]
        context = history.float().mean(dim=1, keepdim=True) if current_start else 0
        generated = (value.float() * 0.9 + context * 0.6).to(x.dtype)
        layer["k"][:, current_start:end] = value * -1
        layer["v"][:, current_start:end] = generated
        layer["global_end_index"] = layer["local_end_index"] = end
        text = crossattn_cache[0]
        if not text["is_init"]:
            text["k"].fill_(1)
            text["v"].fill_(2)
            text["is_init"] = True
        return None, generated.reshape_as(x)


def native_pipeline(generator=None, frames_per_block=2):
    path = ROOT / "experiments/Self-Forcing/pipeline/causal_inference.py"
    tree = ast.parse(path.read_text())
    names = {"CausalInferencePipeline", "ready_quantization_spans"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    scope = {
        "torch": torch, "List": List, "Optional": Optional, "SimpleNamespace": SimpleNamespace,
        "os": os, "tqdm": lambda **kw: nullcontext(SimpleNamespace(update=lambda _: None)),
        "efficiency": SimpleNamespace(ResidentSampler=lambda: SimpleNamespace(observe=lambda *a: None)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    config = SimpleNamespace(
        denoising_step_list=[9, 4], warp_denoising_step=False, context_noise=0,
        num_frame_per_block=frames_per_block, independent_first_frame=False, frame_seq_length=1,
    )
    pipeline = scope["CausalInferencePipeline"](
        config, torch.device("cpu"), generator=generator if generator is not None else Generator(),
        text_encoder=lambda **kw: {}, vae=object(),
    )
    pipeline._sample_resident_kv = lambda: None
    return pipeline


@pytest.mark.parametrize("bits", [2, 4, 16])
def test_native_fork_controls_protection_and_equal_budget(bits):
    torch.manual_seed(29)
    settings = RecoverySettings(bits=bits, group_size=4, future_chunks=3)
    pipeline = native_pipeline()
    noise = torch.randn(1, 18, 4, 1, 1, dtype=torch.bfloat16)
    rollout = RecoveryRollout(pipeline, pipeline.inference, settings, noise, ["fixture"])
    assert "forward" not in vars(pipeline.generator)
    evidence, native = rollout.run()
    assert "forward" not in vars(pipeline.generator)
    assert native.shape == noise.shape
    rows = evidence["branches"]
    assert set(rows) == set(BRANCHES)
    assert rows["stop"]["future_sha256"] == rows["sham"]["future_sha256"]
    assert rows["reencode_reference"]["future_sha256"] == evidence["reference_suffix_sha256"]
    assert all(r["relative_l2"] == 0 for r in rows["reencode_reference"]["fork_cache_rows"])
    assert len({row["observable_prefix_sha256"] for row in rows.values()}) == 1
    assert evidence["warm_cost"]["rounded_through_exclusive_token"] == 6
    assert rows["continue"]["rounded_through_exclusive_token"] == 12
    expected_pairs = 3 * 2 * (14 + 16 + 18)
    assert all(row["future_query_key_pairs_all_heads_layers"] == expected_pairs for row in rows.values())
    assert all(row["future_rng_calls"] == 9 for row in rows.values())
    assert evidence["warm_prefix_relative_l2"][:3] == [0, 0, 0]
    if bits == 16:
        assert evidence["saved_unrounded_kv_bytes"] == 0
        assert all(row["future_chunk_relative_l2"] == [0, 0, 0] for row in rows.values())
    else:
        assert set(rollout.unrounded) == {0, 2, 4}
        assert evidence["saved_unrounded_kv_bytes"] == 96
        assert max(evidence["warm_prefix_relative_l2"][3:]) > 0
        assert rows["undo_rounding"]["future_sha256"] != rows["stop"]["future_sha256"]
        assert rows["reencode_own"]["cache_input_prefix_sha256"] == rows["stop"]["cache_input_prefix_sha256"]
        assert rows["reencode_reference"]["cache_input_prefix_sha256"] != rows["stop"]["cache_input_prefix_sha256"]


def test_unsupported_rolling_cache_and_no_quantization_exposure_rejected():
    pipeline = native_pipeline()
    pipeline.local_attn_size = 6
    with pytest.raises(ValueError, match="native full history"):
        RecoveryRollout(
            pipeline, pipeline.inference, RecoverySettings(),
            torch.zeros(1, 28, 4, 1, 1, dtype=torch.bfloat16), ["fixture"],
        )
    with pytest.raises(ValueError, match="rounded chunk"):
        RecoverySettings(fork_chunk=3, protect_chunks=2)


def test_delaying_rounding_matches_fork_spans_without_changing_warm_latents():
    torch.manual_seed(29)
    noise = torch.randn(1, 18, 4, 1, 1, dtype=torch.bfloat16)
    entry = torch.get_rng_state()
    cases = []
    for warm_start in (None, 6):
        torch.set_rng_state(entry)
        settings = RecoverySettings(bits=2, group_size=4, future_chunks=3, warm_start_chunk=warm_start)
        pipeline = native_pipeline()
        rollout = RecoveryRollout(pipeline, pipeline.inference, settings, noise, ["fixture"])
        evidence, _ = rollout.run()
        cases.append((evidence, rollout))
    (exposed, a), (unexposed, b) = cases
    assert exposed["warm_exposed_chunks"] == 3 and unexposed["warm_exposed_chunks"] == 0
    assert a.rounded_prefix_end == b.rounded_prefix_end == 6
    for layer_a, layer_b in zip(a.warm_cache.kv, b.warm_cache.kv, strict=True):
        for values_a, values_b in zip(layer_a, layer_b, strict=True):
            assert torch.equal(values_a[:, :6], values_b[:, :6])
    assert unexposed["warm_prefix_relative_l2"] == [0] * 6
    for branch in ("undo_rounding", "reencode_own"):
        assert unexposed["branches"][branch]["future_sha256"] == unexposed["reference_suffix_sha256"]
    assert max(unexposed["branches"]["stop"]["future_chunk_relative_l2"]) > 0
