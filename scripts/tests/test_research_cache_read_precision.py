"""Exercise the precision schedule through the native inference loop on CPU."""

import pytest
import torch

from tempokv.research.cache_read_precision import ROLE_ARMS, UNIFORM, ReadPrecisionRollout, read_schedules
from tempokv.research.cache_persistent_precision import ARMS as PERSISTENT_ARMS, PersistentReadPrecisionRollout
from tempokv.research.cache_recovery import RecoverySettings
from tempokv.research.precision import fake_quantize
from test_research_cache_recovery import Generator, native_pipeline


class TwoLayerGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.model.blocks.append(self.model.blocks[0])

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, kv_cache,
                crossattn_cache, current_start):
        x = noisy_image_or_video
        value = x.reshape(1, x.shape[1], 1, 4)
        end = current_start + x.shape[1]
        for cache, text in zip(kv_cache, crossattn_cache, strict=True):
            context = cache["v"][:, :current_start].float().mean(dim=1, keepdim=True) if current_start else 0
            cache["k"][:, current_start:end] = -value
            cache["v"][:, current_start:end] = value
            cache["global_end_index"] = cache["local_end_index"] = end
            text["is_init"] = True
            value = (0.9 * value.float() + 0.6 * context).to(x.dtype)
        return None, value.reshape_as(x)


@pytest.mark.parametrize("bits", (2, 4, 16))
def test_native_timed_precision_controls_and_budget(bits):
    torch.manual_seed(29)
    pipeline = native_pipeline(TwoLayerGenerator())
    settings = RecoverySettings(bits=bits, group_size=4, warm_start_chunk=5, future_chunks=3)
    noise = torch.randn(1, 18, 4, 1, 1, dtype=torch.bfloat16)
    evidence, ref = ReadPrecisionRollout(pipeline, pipeline.inference, settings, noise, ["test"]).run()
    branches = evidence["branches"]
    assert set(branches) == {*read_schedules(2), UNIFORM}
    assert ref.shape == noise.shape
    assert "forward" not in vars(pipeline.generator)
    assert evidence["aged_positions_per_head"] == 6
    writer = branches["rounded_write_only"]
    assert writer["prefix_sha256"] == evidence["reference_prefix_sha256"]
    assert writer["last_prefix_latent_relative_l2"] == 0
    assert writer["last_write_cache_rows"][0]["error_energy"] == 0
    assert writer["last_write_cache_rows"][1]["error_energy"] == 0
    assert branches["rounded_all"]["prefix_sha256"] == branches["protect_clean"]["prefix_sha256"]
    for name, expected in read_schedules(2).items():
        reads = branches[name]["target_reads"]
        assert [r["timestep"] for r in reads] == [9, 4, 0]
        assert tuple(r["call_index"] for r in reads if r["original_aged_history"]) == expected
    one_read = [b for b in branches.values() if b["protected_aged_history_calls"] == 1]
    assert len(one_read) == 3
    assert len({b["extra_ideal_cache_read_bytes"] for b in one_read}) == 1
    baseline = branches[UNIFORM]
    assert baseline["extra_ideal_cache_read_bytes"] == one_read[0]["extra_ideal_cache_read_bytes"]
    precision = baseline["uniform_precision"]
    positions = evidence["aged_positions_per_head"]
    high = precision["higher_positions"]
    total = (positions - high) * precision["lower_bits"] + high * precision["higher_bits"]
    extra_scale_bits_per_element = 0 if bits == 16 else 8
    assert 3 * total == (16 + 2 * bits - extra_scale_bits_per_element) * positions
    assert len({b["future_query_key_pairs_all_heads_layers"] for b in branches.values()}) == 1
    if bits == 16:
        assert all(b["future_sha256"] == evidence["reference_suffix_sha256"] for b in branches.values())
    else:
        assert writer["last_write_cache_rows"][3]["error_energy"] > 0
        assert max(writer["future_chunk_relative_l2"]) > 0
        assert branches["rounded_all"]["last_prefix_latent_relative_l2"] > 0


def test_study_rejects_more_than_one_exposed_chunk():
    pipeline = native_pipeline(TwoLayerGenerator())
    with pytest.raises(ValueError, match="exactly one exposed"):
        ReadPrecisionRollout(
            pipeline, pipeline.inference, RecoverySettings(future_chunks=3),
            torch.zeros(1, 18, 4, 1, 1, dtype=torch.bfloat16), ["test"],
        )


def test_role_swap_preserves_the_uniform_byte_budget():
    torch.manual_seed(29)
    pipeline = native_pipeline(TwoLayerGenerator())
    settings = RecoverySettings(bits=4, group_size=4, warm_start_chunk=5, future_chunks=3)
    noise = torch.randn(1, 18, 4, 1, 1, dtype=torch.bfloat16)
    rollout = ReadPrecisionRollout(pipeline, pipeline.inference, settings, noise, ["test"], role_study=True)
    evidence, _ = rollout.run()
    branches = evidence["branches"]
    assert evidence["study"] == "roles"
    assert set(branches) == {"rounded_all", UNIFORM, *ROLE_ARMS}
    assert branches["rounded_all"]["extra_ideal_cache_read_bytes"] == 0
    uniform = branches[UNIFORM]
    positions = evidence["aged_positions_per_head"]

    def coefficient_bits(plan):
        high = plan["higher_positions"]
        return (positions - high) * plan["lower_bits"] + high * plan["higher_bits"]

    for arm, role in ROLE_ARMS.items():
        row = branches[arm]
        plans = row["precision_by_role"]
        assert plans[role] == {"lower_bits": 8, "higher_bits": 8, "higher_positions": 0}
        assert sum(map(coefficient_bits, plans.values())) == 2 * coefficient_bits(uniform["uniform_precision"])
        assert row["extra_ideal_cache_read_bytes"] == uniform["extra_ideal_cache_read_bytes"]
        assert row["protected_aged_history_calls"] == 0
    assert branches["k_high_equal_bytes"]["future_sha256"] != branches["v_high_equal_bytes"]["future_sha256"]


def test_persistent_policies_match_bytes_as_native_spans_age():
    class RecordingGenerator(TwoLayerGenerator):
        def __init__(self):
            super().__init__()
            self.history = []

        def forward(self, **kwargs):
            start = kwargs["current_start"]
            self.history.append(kwargs["kv_cache"][0]["k"][:, :start].clone())
            return super().forward(**kwargs)

    torch.manual_seed(29)
    generator = RecordingGenerator()
    pipeline = native_pipeline(generator, frames_per_block=3)
    settings = RecoverySettings(bits=4, group_size=4, warm_start_chunk=6, future_chunks=3)
    noise = torch.randn(1, 27, 4, 1, 1, dtype=torch.bfloat16)
    evidence, _ = PersistentReadPrecisionRollout(pipeline, pipeline.inference, settings, noise, ["test"]).run()
    branches = evidence["branches"]
    assert set(branches) == set(PERSISTENT_ARMS)
    assert len({branches[arm]["total_ideal_aged_read_bytes"] for arm in PERSISTENT_ARMS[1:]}) == 1
    assert branches["rounded_all"]["total_ideal_aged_read_bytes"] < branches[UNIFORM]["total_ideal_aged_read_bytes"]
    for name, rows in branches.items():
        reads = rows["aged_history_reads"]
        assert [r["aged_positions_per_head"] for r in reads] == [12] * 3 + [15] * 3 + [18] * 3
        expected = {"protect_first_denoise": [0] * 3, "protect_last_denoise": [1] * 3}.get(name, [])
        assert [r["call_index"] for r in reads if r["original_aged_history"]] == expected
        assert max(rows["future_chunk_relative_l2"]) > 0
    for arm, protected in (("protect_first_denoise", 0), ("protect_last_denoise", 1)):
        run_start = (PERSISTENT_ARMS.index(arm) + 1) * 27
        for chunk in range(6, 9):
            calls = generator.history[run_start + chunk * 3:run_start + (chunk + 1) * 3]
            eligible = (chunk - 2) * 3
            original = calls[protected][:, :eligible]
            rounded = fake_quantize(original, 4, 4).value
            for index, cache in enumerate(calls):
                expected = original if index == protected else rounded
                assert torch.equal(cache[:, :eligible], expected)
