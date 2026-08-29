from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tempokv.codec import PackedKV, TempoKVCodec, decode_tensor
from tempokv.config import CodecConfig, config_from_mapping
from tempokv.fitting import fit_predictors


DIRECTION_DIR = Path(__file__).parents[1] / "paper_experiments" / "direction"
if str(DIRECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DIRECTION_DIR))


def _module(name: str):
    path = DIRECTION_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"direction_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _module("protocol")
plot_figure = _module("plot_figure")


def _base(**updates):
    values = {
        "k_bits": 2,
        "v_bits": 2,
        "stride": 4,
        "group_size": 64,
        "k_history_length": 2,
        "v_history_length": 1,
        "k_predictor_mode": "shared",
        "cross_kv_weight_mode": "layer_shared",
        "cross_matmul_precision": "ieee",
        "fit_mode": "legacy",
    }
    values.update(updates)
    return config_from_mapping(values)


def _trace(seed: int, *, layers=3, records=8, frames=5, heads=2, dim=128):
    generator = torch.Generator().manual_seed(seed)
    shape = (layers, records, heads, dim)
    keys = torch.zeros(layers, records, frames, heads, dim)
    values = torch.zeros_like(keys)
    keys[:, :, 0] = torch.randn(shape, generator=generator)
    values[:, :, 0] = torch.randn(shape, generator=generator)
    for frame in range(1, frames):
        keys[:, :, frame] = 0.9 * keys[:, :, frame - 1] + 0.1 * torch.randn(
            shape, generator=generator
        )
        values[:, :, frame] = (
            0.6 * keys[:, :, frame]
            + 0.3 * values[:, :, frame - 1]
            + 0.1 * torch.randn(shape, generator=generator)
        )
    return keys, values


# --------------------------------------------------------------------------
# The direction is a config setting with real constraints
# --------------------------------------------------------------------------


def test_the_production_direction_is_the_default() -> None:
    config = _base()
    assert config.predictor_direction == "k_to_v"
    assert (config.primary_tensor, config.secondary_tensor) == ("k", "v")
    assert config.mirrored is False


def test_the_mirror_exchanges_the_tensor_roles() -> None:
    config = _base(predictor_direction="v_to_k")
    assert (config.primary_tensor, config.secondary_tensor) == ("v", "k")
    assert config.mirrored is True


def test_the_mirror_needs_a_cross_term_to_point_backwards() -> None:
    """Every other V mode ignores the source, so the flag would do nothing."""
    with pytest.raises(ValueError, match="requires v_predictor_mode=cross_kv"):
        _base(predictor_direction="v_to_k", v_predictor_mode="temporal")


def test_the_mirror_refuses_unequal_bit_widths() -> None:
    """Exchanging roles must not also reallocate bits between K and V."""
    with pytest.raises(ValueError, match="k_bits == v_bits"):
        _base(predictor_direction="v_to_k", k_bits=2, v_bits=4)


def test_an_unknown_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="k_to_v or v_to_k"):
        _base(predictor_direction="sideways")


def test_the_production_signature_is_unchanged_by_the_new_field() -> None:
    """Existing parameter archives must keep loading."""
    signature = _base().format_signature()
    assert "predictor_direction" not in signature
    mirrored = _base(predictor_direction="v_to_k").format_signature()
    assert "predictor_direction" in mirrored
    assert mirrored != signature


# --------------------------------------------------------------------------
# The codec runs both directions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["k_to_v", "v_to_k"])
def test_the_codec_round_trips_in_either_direction(direction: str) -> None:
    keys, values = _trace(0)
    config = _base(predictor_direction=direction, use_triton=False)
    temporal, secondary = fit_predictors(
        keys,
        values,
        config=config,
        iterations=2,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_mode="diagonal",
        coordinate_system="pre_rope_normalized",
    )
    codec = TempoKVCodec(temporal, secondary, config)
    span_keys = keys[0, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
    span_values = values[0, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
    packed = codec.encode(span_keys, span_values, layer=0)

    expected_kinds = (
        ("cross_kv", "temporal") if direction == "v_to_k" else ("temporal", "cross_kv")
    )
    assert (packed.key.predictor_kind, packed.value.predictor_kind) == expected_kinds

    primary = packed.value if direction == "v_to_k" else packed.key
    secondary_state = packed.key if direction == "v_to_k" else packed.value
    rebuilt_primary = decode_tensor(primary, use_triton=False)
    rebuilt_secondary = decode_tensor(
        secondary_state, source=rebuilt_primary, use_triton=False
    )
    assert rebuilt_primary.shape == span_keys.shape
    assert rebuilt_secondary.shape == span_keys.shape
    assert torch.isfinite(rebuilt_primary).all()
    assert torch.isfinite(rebuilt_secondary).all()


def test_the_codec_addresses_the_cache_by_role() -> None:
    """The bridges rely on this to run one code path in both directions."""
    keys, values = _trace(0)
    for direction, primary in (("k_to_v", "k"), ("v_to_k", "v")):
        config = _base(predictor_direction=direction, use_triton=False)
        temporal, secondary = fit_predictors(
            keys, values, config=config, iterations=2,
            temporal_ridge=1.0, cross_ridge=0.1, v_history_mode="diagonal",
        )
        codec = TempoKVCodec(temporal, secondary, config)
        assert codec.primary_role == primary
        assert codec.secondary_role != primary


def test_packed_pairs_need_exactly_one_temporal_side() -> None:
    """Otherwise the cross projection has no defined source."""
    keys, values = _trace(0)
    config = _base(use_triton=False)
    temporal, secondary = fit_predictors(
        keys, values, config=config, iterations=2,
        temporal_ridge=1.0, cross_ridge=0.1, v_history_mode="diagonal",
    )
    codec = TempoKVCodec(temporal, secondary, config)
    span_keys = keys[0, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
    span_values = values[0, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
    packed = codec.encode(span_keys, span_values, layer=0)
    with pytest.raises(ValueError, match="temporally predicted"):
        PackedKV(packed.value, packed.value)


# --------------------------------------------------------------------------
# Fitting mirrors the pair rather than duplicating the equations
# --------------------------------------------------------------------------


def test_mirrored_fitting_matches_fitting_on_a_swapped_trace() -> None:
    """The mirror must be a role exchange, not a second set of equations."""
    keys, values = _trace(3)
    mirrored_config = _base(predictor_direction="v_to_k", use_triton=False)
    forward_config = _base(use_triton=False)
    kwargs = dict(
        iterations=2,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_mode="diagonal",
        coordinate_system="pre_rope_normalized",
    )
    mirrored_temporal, mirrored_cross = fit_predictors(
        keys, values, config=mirrored_config, **kwargs
    )
    swapped_temporal, swapped_cross = fit_predictors(
        values, keys, config=forward_config, **kwargs
    )
    torch.testing.assert_close(
        mirrored_temporal.history_weights, swapped_temporal.history_weights
    )
    torch.testing.assert_close(mirrored_cross.weight, swapped_cross.weight)


# --------------------------------------------------------------------------
# The study's arms and accounting
# --------------------------------------------------------------------------


def test_the_three_arms_isolate_the_dependency_direction() -> None:
    names = [item.name for item in protocol.ARMS]
    assert names == ["temporal_only", "k_to_v", "v_to_k"]
    modes = {item.name: item.v_predictor_mode for item in protocol.ARMS}
    assert modes["temporal_only"] == "temporal"
    # The two cross arms differ only in direction, which is the clean pair.
    assert modes["k_to_v"] == modes["v_to_k"] == "cross_kv"
    directions = {item.name: item.predictor_direction for item in protocol.ARMS}
    assert directions["k_to_v"] == "k_to_v"
    assert directions["v_to_k"] == "v_to_k"


def test_every_arm_shares_one_codec_apart_from_the_two_study_variables() -> None:
    mappings = {item.name: item.codec_mapping(1560) for item in protocol.ARMS}
    variable = {"v_predictor_mode", "predictor_direction"}
    reference = mappings["k_to_v"]
    for name, mapping in mappings.items():
        fixed = {k: v for k, v in mapping.items() if k not in variable}
        assert fixed == {
            k: v for k, v in reference.items() if k not in variable
        }, name
    assert reference["k_bits"] == reference["v_bits"] == 2


def test_an_unknown_arm_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="temporal_only"):
        protocol.arm("sideways")


def test_direction_comparison_ranks_on_total_error() -> None:
    """Attention consumes both tensors, so neither one alone decides."""
    results = [
        {"arm": "temporal_only", "k_nmse": 0.10, "v_nmse": 0.20},
        {"arm": "k_to_v", "k_nmse": 0.10, "v_nmse": 0.10},
        # Better K, much worse V: worse overall, and must be reported as such.
        {"arm": "v_to_k", "k_nmse": 0.05, "v_nmse": 0.35},
    ]
    summary = protocol.summarize(results)
    comparison = summary["direction_comparison"]
    assert comparison["k_to_v_total_nmse"] == pytest.approx(0.20)
    assert comparison["v_to_k_total_nmse"] == pytest.approx(0.40)
    assert comparison["mirror_penalty"] == pytest.approx(1.0)
    assert comparison["preferred_direction"] == "k_to_v"


def test_each_arm_is_reported_against_the_no_dependency_baseline() -> None:
    results = [
        {"arm": "temporal_only", "k_nmse": 0.10, "v_nmse": 0.20},
        {"arm": "k_to_v", "k_nmse": 0.10, "v_nmse": 0.15},
    ]
    rows = {row["arm"]: row for row in protocol.summarize(results)["per_arm"]}
    assert rows["k_to_v"]["v_nmse_vs_temporal_only"] == pytest.approx(-0.25)
    assert rows["temporal_only"]["k_nmse_vs_temporal_only"] == pytest.approx(0.0)


def test_reconstruct_returns_k_and_v_in_their_own_order() -> None:
    """A mirrored run must not report V's error under K's name."""
    keys, values = _trace(5)
    for direction in ("k_to_v", "v_to_k"):
        arm_spec = protocol.arm(direction)
        config = arm_spec.codec(4)
        temporal, secondary = fit_predictors(
            keys, values, config=config, iterations=2,
            temporal_ridge=1.0, cross_ridge=0.1, v_history_mode="diagonal",
        )
        rebuilt_keys, rebuilt_values = protocol.reconstruct(
            arm_spec, keys, values, temporal, secondary, config
        )
        assert rebuilt_keys.shape == keys.shape
        assert rebuilt_values.shape == values.shape
        # The rebuilt tensors must track their own reference, not each other's.
        def error(rebuilt, reference):
            return float((rebuilt.double() - reference.double()).square().sum())
        assert error(rebuilt_keys, keys) < error(rebuilt_keys, values)
        assert error(rebuilt_values, values) < error(rebuilt_values, keys)


# --------------------------------------------------------------------------
# Figure inputs
# --------------------------------------------------------------------------


def test_quality_rows_report_a_missing_evaluation_rather_than_a_number() -> None:
    rows = plot_figure.quality_rows(
        {
            "arms": [
                {
                    "arm": "v_to_k",
                    "predictor_direction": "v_to_k",
                    "quality_status": "not_run",
                    "quality_metrics": None,
                }
            ]
        }
    )
    assert rows[0]["quality_status"] == "not_run"
    assert rows[0]["psnr"] is None


def test_reconstruction_rows_carry_the_total_used_for_ranking() -> None:
    report = {
        "arms": [
            {
                "arm": "k_to_v",
                "predictor_direction": "k_to_v",
                "v_predictor_mode": "cross_kv",
                "k_nmse": 0.1,
                "v_nmse": 0.2,
            }
        ],
        "per_arm": [{"arm": "k_to_v"}],
    }
    row = plot_figure.reconstruction_rows(report)[0]
    assert row["total_nmse"] == pytest.approx(0.3)
