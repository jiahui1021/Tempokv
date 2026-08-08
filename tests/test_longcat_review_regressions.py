from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from longcat.bridge import (
    LongCatRuntimeSpec,
    LongCatTempoKVBridge,
    decode_kv_cache,
    encode_kv_cache,
    validate_cache_layout,
)
from tempokv import (
    CodecConfig,
    CrossKVParameters,
    TemporalParameters,
    TempoKVCodec,
    encode_cross_kv,
)
from tempokv.parameters import load_parameters
from scripts.fit_params import build_parser as build_fit_parser
from scripts.fit_params import run as run_fit


def _codec() -> TempoKVCodec:
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=4,
        group_size=2,
        stride=2,
        residual_mode="asymmetric",
        scale_dtype="float32",
        reconstruction_dtype="float32",
        use_triton=False,
        v_history_length=0,
    )
    temporal = TemporalParameters(
        torch.full((2, 1, 1, 1, 4), 0.8),
        torch.zeros(2, 1, 1, 4),
        stride=2,
        layer_ids=(0, 1),
        coordinate_system="pre_rope_normalized",
    )
    cross = CrossKVParameters(
        torch.eye(4).reshape(1, 4, 4).repeat(2, 1, 1, 1),
        torch.zeros(2, 1, 4),
        stride=2,
        layer_ids=(0, 1),
        coordinate_system="pre_rope_normalized",
    )
    return TempoKVCodec(temporal, cross, config)


def _cache() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    key = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4) / 10
    value = key * 0.7 + 0.25
    return {
        0: (key.clone(), value.clone()),
        1: (key.clone() + 1, value.clone() - 1),
    }


def _spec() -> LongCatRuntimeSpec:
    return LongCatRuntimeSpec(num_layers=2, num_heads=1, head_dim=4, stride=2)


class _RecordingCodec:
    def __init__(self, codec: TempoKVCodec) -> None:
        self._codec = codec
        self.config = codec.config
        self.temporal = codec.temporal
        self.cross_kv = codec.cross_kv
        self.reconstructed_keys: dict[int, torch.Tensor] = {}
        self.value_sources: dict[int, torch.Tensor] = {}
        self.reconstructed_values: dict[int, torch.Tensor] = {}

    def encode_key(
        self,
        key: torch.Tensor,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ):
        packed, reconstructed = self._codec.encode_key(
            key,
            layer=layer,
            reuse_input=reuse_input,
        )
        self.reconstructed_keys[layer] = reconstructed.clone()
        return packed, reconstructed

    def encode_value(
        self,
        value: torch.Tensor,
        reconstructed_key: torch.Tensor,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ):
        self.value_sources[layer] = reconstructed_key.clone()
        parameters = self._codec._cross_for_layer(value.device, layer)
        packed, reconstructed = encode_cross_kv(
            value,
            reconstructed_key,
            parameters,
            self.config,
            reuse_input_for_reconstruction=reuse_input,
        )
        self.reconstructed_values[layer] = reconstructed.clone()
        return packed


def test_longcat_closed_loop_and_full_cache_bytes() -> None:
    cache = _cache()
    actual_bytes = sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in cache.values()
    )
    codec = _RecordingCodec(_codec())

    encoded, event = encode_kv_cache(
        cache,
        codec=codec,
        temporal_units=3,
        spec=_spec(),
    )

    assert event["uncompressed_cache_bytes"] == actual_bytes
    for layer, (packed_key, packed_value) in encoded.items():
        decoded_key, decoded_value = decode_kv_cache(
            packed_key,
            packed_value,
            codec,
        )
        torch.testing.assert_close(
            decoded_key,
            codec.reconstructed_keys[layer],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            decoded_key,
            codec.value_sources[layer],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            decoded_value,
            codec.reconstructed_values[layer],
            rtol=0,
            atol=0,
        )


def test_default_53_frame_condition_has_stride_1560() -> None:
    tensor = torch.empty((1, 32, 21_840, 128), dtype=torch.bfloat16, device="meta")
    cache = {layer: (tensor, tensor) for layer in range(48)}
    codec = SimpleNamespace(
        config=SimpleNamespace(stride=1560),
        temporal=SimpleNamespace(coordinate_system="pre_rope_normalized"),
        cross_kv=SimpleNamespace(coordinate_system="pre_rope_normalized"),
    )

    layout = validate_cache_layout(
        cache,
        temporal_units=14,
        codec=codec,
        spec=LongCatRuntimeSpec(),
    )

    assert layout["sequence"] == 21_840
    assert layout["stride"] == 1560


def test_report_separates_current_peak_and_total_cache_bytes() -> None:
    bridge = LongCatTempoKVBridge(_codec(), spec=_spec())
    bridge.quantization_events.extend(
        (
            {"storage_bytes": 100, "uncompressed_cache_bytes": 400},
            {"storage_bytes": 300, "uncompressed_cache_bytes": 1000},
            {"storage_bytes": 200, "uncompressed_cache_bytes": 800},
        )
    )

    report = bridge.report()

    assert report["current_compressed_cache_bytes"] == 200
    assert report["peak_compressed_cache_bytes"] == 300
    assert report["total_encoded_cache_bytes"] == 600
    assert report["current_uncompressed_cache_bytes"] == 800
    assert report["peak_uncompressed_cache_bytes"] == 1000
    assert report["total_uncompressed_cache_bytes"] == 2200
    assert report["compressed_cache_bytes"] == 300
    assert report["uncompressed_cache_bytes"] == 1000
    assert report["compression_ratio"] == pytest.approx(1000 / 300)


def test_bridge_report_proves_tempokv_was_triggered() -> None:
    bridge = LongCatTempoKVBridge(_codec(), spec=_spec())
    dit = SimpleNamespace(parameters=lambda: iter(()), _tempokv_num_units=3)

    encoded = bridge.quantize(dit, _cache())
    bridge.decode(*encoded[0])
    report = bridge.report()

    assert report["encode_key_calls"] > 0
    assert report["encode_value_calls"] > 0
    assert report["decode_tensor_calls"] > 0


def test_trace_mode_bypasses_original_longcat_quantizer() -> None:
    attention_module = SimpleNamespace()

    class FakeAttention:
        num_heads = 1
        head_dim = 4

        def forward_with_kv_cache(self, *args, **kwargs):
            del args, kwargs
            return None

    attention_module.Attention = FakeAttention
    attention_module.uncompress_kv_cache = lambda key, value: (key, value)

    class FakeDIT:
        def __init__(self) -> None:
            self.blocks = [SimpleNamespace(attn=FakeAttention()) for _ in range(2)]
            self.cp_split_hw = (1, 1)
            self.original_quantize_calls = 0

        def quantize_kv_cache(self, kv_cache_dict, offload_kv_cache=False):
            del kv_cache_dict, offload_kv_cache
            self.original_quantize_calls += 1
            return "original"

    dit_module = SimpleNamespace(
        LongCatVideoTransformer3DModel=FakeDIT,
        _onload_kv_cache=lambda pair, device: pair,
    )

    class FakePipeline:
        def _cache_clean_latents(self, cond_latents, *args, **kwargs):
            del cond_latents, args, kwargs
            return None

    bridge = LongCatTempoKVBridge(None, mode="trace", spec=_spec())
    bridge.install(
        dit_module=dit_module,
        attention_module=attention_module,
        pipeline_module=SimpleNamespace(LongCatVideoPipeline=FakePipeline),
    )
    dit = FakeDIT()
    bridge.attach(dit)
    clean_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    returned = dit.quantize_kv_cache(clean_cache, offload_kv_cache=True)

    assert returned is clean_cache
    assert dit.original_quantize_calls == 0


def test_context_parallel_trace_budget_is_global(tmp_path: Path) -> None:
    rank_zero = LongCatTempoKVBridge(
        None,
        mode="trace",
        trace_dir=tmp_path / "rank0",
        max_trace_spatial_positions=34,
        context_parallel_size=4,
        context_parallel_rank=0,
        spec=_spec(),
    )
    rank_three = LongCatTempoKVBridge(
        None,
        mode="trace",
        trace_dir=tmp_path / "rank3",
        max_trace_spatial_positions=34,
        context_parallel_size=4,
        context_parallel_rank=3,
        spec=_spec(),
    )

    assert rank_zero.local_trace_spatial_positions == 9
    assert rank_three.local_trace_spatial_positions == 8
    assert rank_zero.trace_collector is not None
    assert rank_three.trace_collector is not None
    assert rank_zero.trace_collector.max_spatial_positions == 9
    assert rank_three.trace_collector.max_spatial_positions == 8


def test_trace_stride_is_independent_of_runtime_stride(tmp_path: Path) -> None:
    generator = np.random.default_rng(8)
    keys = generator.normal(size=(1, 12, 4, 1, 4)).astype(np.float32)
    values = (keys * 0.7 + 0.2).astype(np.float32)
    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        keys=keys,
        values=values,
        trace_stride=np.asarray(5, dtype=np.int32),
        stride=np.asarray(5, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "k_bits": 4,
                "v_bits": 4,
                "anchor_bits": 4,
                "group_size": 2,
                "stride": 7,
                "residual_mode": "asymmetric",
                "scale_dtype": "float32",
                "reconstruction_dtype": "float32",
                "use_triton": False,
                "strict_triton": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "fit"
    args = build_fit_parser().parse_args(
        [
            str(trace),
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--history-length",
            "2",
            "--iterations",
            "1",
        ]
    )

    report = run_fit(args)
    temporal = load_parameters(output / "k_temporal.npz")
    cross = load_parameters(output / "v_cross_kv.npz")

    assert report["input"]["trace_stride"] == 5
    assert report["input"]["runtime_stride"] == 7
    assert temporal.stride == 7
    assert cross.stride == 7
