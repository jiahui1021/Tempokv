from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton validation", allow_module_level=True)

from tempokv import (  # noqa: E402
    CodecConfig,
    CrossKVParameters,
    TemporalParameters,
    decode_tensor,
    encode_cross_kv,
    encode_temporal,
)
from tempokv.triton_ops import reset_route_report, route_report  # noqa: E402


def settings() -> CodecConfig:
    return CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=4,
        group_size=64,
        stride=16,
        residual_mode="asymmetric",
        scale_dtype="bfloat16",
        use_triton=True,
        strict_triton=True,
    )


def test_temporal_persistent_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(4)
    key = torch.randn(1, 2, 50, 128, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    weights = torch.zeros(3, 3, 128)
    weights[0, 0] = 0.8
    weights[1, 0] = 0.6
    weights[1, 1] = 0.2
    weights[2, 0] = 0.5
    weights[2, 1] = 0.25
    weights[2, 2] = 0.1
    parameters = TemporalParameters(weights, torch.zeros(3, 128), 16)
    packed, _ = encode_temporal(key, parameters, settings())
    reference = decode_tensor(
        packed, output_dtype=torch.bfloat16, use_triton=False
    )
    reset_route_report()
    accelerated = decode_tensor(
        packed,
        output_dtype=torch.bfloat16,
        use_triton=True,
        strict_triton=True,
    )
    torch.testing.assert_close(accelerated, reference, rtol=0, atol=0)
    assert route_report() == {"temporal_persistent": 1}


def test_cross_kv_persistent_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(8)
    key = torch.randn(1, 2, 50, 128, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    value = torch.randn_like(key)
    weight = torch.randn(
        2, 128, 128, generator=generator, device="cuda", dtype=torch.float32
    ) * 0.02
    bias = torch.randn(
        2, 128, generator=generator, device="cuda", dtype=torch.float32
    ) * 0.01
    parameters = CrossKVParameters(weight, bias, 16)
    packed, _ = encode_cross_kv(value, key, parameters, settings())
    reference = decode_tensor(
        packed,
        source=key,
        output_dtype=torch.bfloat16,
        use_triton=False,
    )
    reset_route_report()
    accelerated = decode_tensor(
        packed,
        source=key,
        output_dtype=torch.bfloat16,
        use_triton=True,
        strict_triton=True,
    )
    torch.testing.assert_close(accelerated, reference, rtol=2e-2, atol=2e-2)
    assert route_report() == {"cross_kv_persistent": 1}


@pytest.mark.parametrize(
    ("bits", "mode"),
    [(2, "symmetric"), (4, "asymmetric"), (8, "symmetric")],
)
def test_general_kernels_match_reference(bits: int, mode: str) -> None:
    generator = torch.Generator(device="cuda").manual_seed(40 + bits)
    key = torch.randn(1, 2, 37, 32, generator=generator, device="cuda")
    value = torch.randn_like(key)
    config = CodecConfig(
        k_bits=bits,
        v_bits=bits,
        anchor_bits=bits,
        group_size=16,
        stride=16,
        residual_mode=mode,
        scale_dtype="float32",
        reconstruction_dtype="float32",
        use_triton=True,
        strict_triton=True,
    )
    temporal = TemporalParameters(
        torch.tensor([[[0.8] * 32]]), torch.zeros(1, 32), 16
    )
    cross = CrossKVParameters(
        torch.eye(32, device="cuda").repeat(2, 1, 1),
        torch.zeros(2, 32, device="cuda"),
        16,
    )
    key_packed, key_rebuilt = encode_temporal(key, temporal, config)
    value_packed, _ = encode_cross_kv(value, key_rebuilt, cross, config)
    key_reference = decode_tensor(
        key_packed, output_dtype=torch.float32, use_triton=False
    )
    value_reference = decode_tensor(
        value_packed,
        source=key_reference,
        output_dtype=torch.float32,
        use_triton=False,
    )
    key_accelerated = decode_tensor(
        key_packed,
        output_dtype=torch.float32,
        use_triton=True,
        strict_triton=True,
    )
    value_accelerated = decode_tensor(
        value_packed,
        source=key_accelerated,
        output_dtype=torch.float32,
        use_triton=True,
        strict_triton=True,
    )
    torch.testing.assert_close(key_accelerated, key_reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        value_accelerated, value_reference, rtol=1e-5, atol=1e-5
    )
