"""Regressions for the defects found in the repository review.

Each test names the failure it prevents rather than the code it touches, so a
future change that reintroduces one of these fails with the reason attached.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from tempokv.attention_objective import temporal_rollout
from tempokv.config import CodecConfig, config_from_mapping
from tempokv.quantization import quantize_asymmetric, quantize_symmetric


# --------------------------------------------------------------------------
# Integral codec settings
# --------------------------------------------------------------------------


def test_whole_floats_are_coerced_so_signatures_cannot_diverge() -> None:
    """``4`` and ``4.0`` described the same codec but signed differently."""
    integral = config_from_mapping(
        {"k_bits": 4, "v_bits": 4, "group_size": 64, "stride": 1560}
    )
    floating = config_from_mapping(
        {"k_bits": 4.0, "v_bits": 4, "group_size": 64.0, "stride": 1560.0}
    )
    assert isinstance(floating.k_bits, int)
    assert isinstance(floating.group_size, int)
    assert isinstance(floating.stride, int)
    assert floating.format_signature() == integral.format_signature()


@pytest.mark.parametrize(
    "payload",
    [
        {"k_bits": 4.5},
        {"group_size": 64.5},
        {"stride": 1560.5},
    ],
)
def test_non_integral_settings_are_refused_at_construction(payload: dict) -> None:
    """Failing here beats failing later inside a reshape or a slice."""
    with pytest.raises(ValueError, match="whole number"):
        config_from_mapping({"k_bits": 4, "v_bits": 4, **payload})


def test_booleans_are_not_accepted_as_bit_widths() -> None:
    with pytest.raises(ValueError, match="whole number"):
        config_from_mapping({"k_bits": True, "v_bits": 4})


# --------------------------------------------------------------------------
# Scale storage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("quantize", [quantize_symmetric, quantize_asymmetric])
def test_a_scale_that_overflows_its_storage_dtype_is_refused(quantize) -> None:
    """It used to store ``inf`` and dequantize every value to NaN."""
    value = torch.full((1, 1, 4, 64), 1e7)
    with pytest.raises(ValueError, match="overflows"):
        quantize(value, bits=4, group_size=64, scale_dtype=torch.float16)


@pytest.mark.parametrize("quantize", [quantize_symmetric, quantize_asymmetric])
def test_ordinary_inputs_still_quantize_into_float16_scales(quantize) -> None:
    """The overflow guard must not reject the range the codec actually uses."""
    torch.manual_seed(0)
    result = quantize(
        torch.randn(1, 1, 4, 64), bits=4, group_size=64, scale_dtype=torch.float16
    )
    scale = result[1]
    assert torch.isfinite(scale).all()


# --------------------------------------------------------------------------
# Refinement rollout precision
# --------------------------------------------------------------------------


def _rollout_config(reconstruction_dtype: str) -> CodecConfig:
    return CodecConfig(
        k_bits=4,
        v_bits=4,
        anchor_bits=4,
        group_size=4,
        stride=2,
        scale_dtype="float32",
        reconstruction_dtype=reconstruction_dtype,
        k_history_length=1,
        v_history_length=0,
        use_triton=False,
    )


def test_the_rollout_follows_the_configured_reconstruction_dtype() -> None:
    """It hardcoded bfloat16, so a float32 codec was refined against another."""
    torch.manual_seed(0)
    target = torch.randn(1, 2, 12, 4)
    weight = torch.full((1, 1, 4), 0.8)
    bias = torch.zeros(1, 4)
    outputs = {
        name: temporal_rollout(
            target, weight, bias, bits=4, config=_rollout_config(name)
        )
        for name in ("float32", "bfloat16")
    }
    assert not torch.equal(outputs["float32"], outputs["bfloat16"])


# --------------------------------------------------------------------------
# Profiling attribution
# --------------------------------------------------------------------------


PROFILING = Path(__file__).parents[2] / "src" / "tempokv" / "profiling.py"


def test_decode_sections_are_attributed_by_cache_role_not_predictor_kind() -> None:
    """A temporal V, and a mirrored K, both defeat the predictor-kind shortcut."""
    source = PROFILING.read_text(encoding="utf-8")
    assert "_ACTIVE_CACHE_ROLE" in source
    # The old shortcut decided K purely from the predictor kind.
    assert 'DECODE_K if kind == "temporal" else DECODE_V' not in source
