"""Install TempoKV into the upstream Rolling Forcing runtime.

The upstream project stays external and pinned by ``UPSTREAM.md``.  At import
time this bridge replaces only the attention function and pipeline cache
factory, keeping model weights, denoising windows, and VAE behavior owned by
Rolling Forcing itself.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch

from tempokv.codec import PackedTensor, TempoKVCodec, decode_tensor
from tempokv.integration import load_codec_from_env
from tempokv.runtime.cache import set_quantized_decoder
from tempokv.runtime.metrics import TempoKVRuntimeReport, bf16_bytes

from .cache import RollingChunkedKVCache
from .rollingforcing import rollingforcing


ROLLING_NUM_LAYERS = 30
ROLLING_NUM_HEADS = 12
ROLLING_HEAD_DIM = 128
ROLLING_DEFAULT_CACHE_FRAMES = 24
ROLLING_DEFAULT_SINK_FRAMES = 3
ROLLING_COORDINATE_SYSTEM = "pre_rope_normalized"
_STATE_KEY = "_tempokv_packed"


def wrap_cache_state(state: PackedTensor) -> dict[str, Any]:
    if not isinstance(state, PackedTensor):
        raise TypeError(f"expected PackedTensor, got {type(state).__name__}")
    return {_STATE_KEY: state, "method": "tempokv", "predictor_kind": state.predictor_kind}


def unwrap_cache_state(state: Any) -> PackedTensor | None:
    if isinstance(state, PackedTensor):
        return state
    if isinstance(state, dict):
        packed = state.get(_STATE_KEY)
        return packed if isinstance(packed, PackedTensor) else None
    return None


class RollingTempoKVEncoder:
    """Encode a refreshed Rolling block without assuming contiguous addresses."""

    def __init__(self, codec: TempoKVCodec, report: TempoKVRuntimeReport) -> None:
        self.codec = codec
        self.report = report
        self.blocks_encoded = 0
        self._encoded_starts: set[int] = set()

    def reset_after_eviction(self) -> None:
        # Rolling compaction may split a packed span. The next clean block is
        # therefore a fresh anchor; no predictor state is carried across it.
        self.blocks_encoded = 0

    def drop_frames(self, starts: list[int]) -> None:
        """Remove report entries for spans no longer resident after compaction."""
        for start in starts:
            for layer_index in range(self.report.num_layers):
                self.report.drop_span(("k", layer_index, int(start)))
                self.report.drop_span(("v", layer_index, int(start)))

    def encode_block(
        self,
        layers: list[dict[str, Any]],
        *,
        local_start: int,
        local_end: int,
        absolute_start_frame: int,
    ) -> None:
        if local_end <= local_start:
            raise ValueError("cannot encode an empty Rolling Forcing block")
        for layer_index, layer in enumerate(layers):
            cache_k = layer["k"]
            cache_v = layer["v"]
            keys = cache_k.read_bhsd(local_start, local_end)
            packed_key, reconstructed_key = self.codec.encode_key(
                keys, layer=layer_index, reuse_input=True
            )
            cache_k.store_quantized(local_start, local_end, wrap_cache_state(packed_key))
            self.report.encode_key_calls += 1
            self.report.record_span(
                ("k", layer_index, int(absolute_start_frame)),
                compressed_bytes=packed_key.storage_bytes(),
                uncompressed_bytes=bf16_bytes(tuple(packed_key.shape)),
            )
            values = cache_v.read_bhsd(local_start, local_end)
            packed_value = self.codec.encode_value(
                values, reconstructed_key, layer=layer_index, reuse_input=True
            )
            cache_v.store_quantized(local_start, local_end, wrap_cache_state(packed_value))
            self.report.encode_value_calls += 1
            self.report.record_span(
                ("v", layer_index, int(absolute_start_frame)),
                compressed_bytes=packed_value.storage_bytes(),
                uncompressed_bytes=bf16_bytes(tuple(packed_value.shape)),
            )
            del keys, values, packed_key, packed_value, reconstructed_key
        if int(absolute_start_frame) not in self._encoded_starts:
            self.report.finalized_frames += int(
                (local_end - local_start) // layers[0]["k"].frame_seq_length
            )
            self._encoded_starts.add(int(absolute_start_frame))
        self.blocks_encoded += 1


class RollingTraceCollector:
    """Collect clean first-block K/V as the fitter's ``[L,S,T,H,D]`` schema."""

    def __init__(self, output: str | Path, *, tokens_per_frame: int = 8) -> None:
        if tokens_per_frame < 1:
            raise ValueError("trace tokens_per_frame must be positive")
        self.output = Path(output).expanduser().resolve()
        self.tokens_per_frame = int(tokens_per_frame)
        self._seen_starts: set[tuple[int, int]] = set()
        self._keys: dict[int, list[np.ndarray]] = {}
        self._values: dict[int, list[np.ndarray]] = {}
        self.frame_stride: int | None = None
        self.num_heads: int | None = None
        self.head_dim: int | None = None
        self.temporal_units: int | None = None

    def capture(
        self,
        layer_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        frame_stride: int,
        absolute_start_frame: int,
        num_frames: int,
    ) -> None:
        key = (int(layer_index), int(absolute_start_frame))
        if key in self._seen_starts:
            return
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("Rolling trace K/V must use matching [B,S,H,D] tensors")
        batch, sequence, heads, head_dim = (int(item) for item in keys.shape)
        if sequence != num_frames * frame_stride:
            raise ValueError("Rolling trace block length does not match frame stride")
        if self.tokens_per_frame > frame_stride:
            raise ValueError("trace tokens_per_frame exceeds frame stride")
        if self.frame_stride is None:
            self.frame_stride = frame_stride
            self.num_heads = heads
            self.head_dim = head_dim
            self.temporal_units = num_frames
        elif (self.frame_stride, self.num_heads, self.head_dim, self.temporal_units) != (
            frame_stride,
            heads,
            head_dim,
            num_frames,
        ):
            raise ValueError("Rolling trace blocks changed shape or frame stride")
        token_indices = torch.linspace(
            0, frame_stride - 1, steps=self.tokens_per_frame, device=keys.device
        ).round().long().unique()
        def sample(tensor: torch.Tensor) -> np.ndarray:
            return (
                tensor.detach()
                .reshape(batch, num_frames, frame_stride, heads, head_dim)
                .index_select(2, token_indices)
                .to(device="cpu", dtype=torch.float32)
                .numpy()
                .transpose(0, 2, 1, 3, 4)
                .reshape(batch * int(token_indices.numel()), num_frames, heads, head_dim)
            )
        self._keys.setdefault(int(layer_index), []).append(sample(keys))
        self._values.setdefault(int(layer_index), []).append(sample(values))
        self._seen_starts.add(key)

    def start_run(self) -> None:
        """Begin a new prompt rollout while retaining prior prompt samples."""
        self._seen_starts.clear()
        self._keys.clear()
        self._values.clear()

    def write(self, *, num_layers: int) -> Path:
        if self.frame_stride is None or self.temporal_units is None:
            raise RuntimeError("Rolling inference produced no clean trace blocks")
        if sorted(self._keys) != list(range(num_layers)):
            raise RuntimeError("Rolling trace did not capture every transformer layer")
        keys = np.stack([np.concatenate(self._keys[i], axis=0) for i in range(num_layers)])
        values = np.stack([np.concatenate(self._values[i], axis=0) for i in range(num_layers)])
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            with np.load(self.output, allow_pickle=False) as previous:
                keys = np.concatenate((previous["keys"], keys), axis=1)
                values = np.concatenate((previous["values"], values), axis=1)
        np.savez(
            self.output,
            keys=keys,
            values=values,
            layer_ids=np.arange(num_layers, dtype=np.int32),
            num_layers=np.asarray(num_layers, dtype=np.int32),
            num_heads=np.asarray(self.num_heads, dtype=np.int32),
            head_dim=np.asarray(self.head_dim, dtype=np.int32),
            temporal_units=np.asarray(self.temporal_units, dtype=np.int32),
            trace_stride=np.asarray(self.frame_stride, dtype=np.int32),
            stride=np.asarray(self.frame_stride, dtype=np.int32),
            tokens_per_frame=np.asarray(self.tokens_per_frame, dtype=np.int32),
            trace_layout=np.asarray("tempokv_flat_samples_v1"),
            coordinate_system=np.asarray(ROLLING_COORDINATE_SYSTEM),
            cross_source_kind=np.asarray("normalized"),
            trace_role=np.asarray("calibration"),
        )
        self._seen_starts.clear()
        self._keys.clear()
        self._values.clear()
        return self.output


def _decode_state(
    state: Any,
    *,
    source_tensor: torch.Tensor | None,
    output_tensor: torch.Tensor | None,
    output_layout: str,
    codec: TempoKVCodec,
    report: TempoKVRuntimeReport,
) -> torch.Tensor:
    packed = unwrap_cache_state(state)
    if packed is None:
        raise TypeError(f"unsupported Rolling Forcing cache state: {type(state).__name__}")
    report.decode_tensor_calls += 1
    kwargs: dict[str, Any] = {
        "source": source_tensor,
        "output_tensor": output_tensor,
        "output_layout": output_layout,
        "use_triton": codec.config.use_triton,
        "strict_triton": codec.config.strict_triton,
    }
    return decode_tensor(packed, **kwargs)


def _provide_legacy_trq_import_shim() -> None:
    """Allow old Rolling snapshots to import without their retired TRQ module.

    The upstream rolling attention module imports ``trq.backends`` at module
    import time even though the TempoKV bridge replaces the function before
    the first forward. A tiny fail-closed shim keeps that import boundary
    explicit; none of its placeholders may be called on the TempoKV path.
    """
    try:
        import trq.backends.rolling_forcing  # type: ignore[import-not-found]

        return
    except ModuleNotFoundError:
        pass
    trq = types.ModuleType("trq")
    backends = types.ModuleType("trq.backends")
    rolling = types.ModuleType("trq.backends.rolling_forcing")

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("legacy TRQ cache was called; TempoKV bridge installation failed")

    rolling.create_cache = unavailable  # type: ignore[attr-defined]
    rolling.cache_size = unavailable  # type: ignore[attr-defined]
    rolling.evict_cache_pair = unavailable  # type: ignore[attr-defined]
    rolling.quantize_cache_pair = unavailable  # type: ignore[attr-defined]
    rolling.read_cache = unavailable  # type: ignore[attr-defined]
    rolling.write_cache = unavailable  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "trq": trq,
            "trq.backends": backends,
            "trq.backends.rolling_forcing": rolling,
        }
    )


def install(runtime_root: str | Path, *, mode: str = "tempokv") -> type:
    """Patch an external Rolling Forcing checkout and return its pipeline class."""
    runtime_root = Path(runtime_root).expanduser().resolve()
    if not (runtime_root / "inference.py").is_file():
        raise FileNotFoundError(f"Rolling Forcing inference.py is missing: {runtime_root}")
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))

    _provide_legacy_trq_import_shim()
    pipeline_package = importlib.import_module("pipeline")
    pipeline_module = importlib.import_module("pipeline.rolling_forcing_inference")
    causal_module = importlib.import_module("wan.modules.causal_model")
    base = pipeline_module.CausalInferencePipeline
    if mode not in {"tempokv", "trace"}:
        raise ValueError(f"unsupported Rolling Forcing bridge mode: {mode}")
    codec = load_codec_from_env() if mode == "tempokv" else None
    report = TempoKVRuntimeReport(
        model="rolling_forcing",
        kv_mode=mode,
        num_layers=ROLLING_NUM_LAYERS,
        num_heads=ROLLING_NUM_HEADS,
        head_dim=ROLLING_HEAD_DIM,
        stride=codec.config.stride if codec is not None else 1,
    )

    def decode_with_codec(
        state: Any,
        source_tensor: torch.Tensor | None = None,
        output_tensor: torch.Tensor | None = None,
        output_layout: str = "BSHD",
    ) -> torch.Tensor:
        return _decode_state(
            state,
            source_tensor=source_tensor,
            output_tensor=output_tensor,
            output_layout=output_layout,
            codec=codec,
            report=report,
        )

    if codec is not None:
        set_quantized_decoder(decode_with_codec)
    causal_module.rollingforcing = rollingforcing

    class TempoKVRollingInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.tempokv_codec = codec
            self.tempokv_report = report
            self.tempokv_encoder = (
                RollingTempoKVEncoder(codec, report) if codec is not None else None
            )
            trace_output = os.environ.get("TEMPOKV_TRACE_OUTPUT")
            self.tempokv_trace_collector = (
                RollingTraceCollector(
                    trace_output,
                    tokens_per_frame=int(
                        os.environ.get("TEMPOKV_TRACE_TOKENS_PER_FRAME", "8")
                    ),
                )
                if mode == "trace" and trace_output
                else None
            )
            if mode == "trace" and self.tempokv_trace_collector is None:
                raise ValueError("TEMPOKV_TRACE_OUTPUT must be set for trace mode")
            self.num_transformer_blocks = len(self.generator.model.blocks)
            report.num_layers = self.num_transformer_blocks
            if codec is not None and codec.config.stride <= 0:
                raise ValueError("TempoKV stride must be positive")
            self._tempokv_runtime_ready = False

        def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
            frame_stride = int(self.frame_seq_length)
            if codec is not None and codec.config.stride != frame_stride:
                raise ValueError(
                    f"Rolling Forcing TempoKV stride {codec.config.stride} does not "
                    f"match runtime frame stride {frame_stride}"
                )
            blocks = self.generator.model.blocks
            num_layers = len(blocks)
            num_heads = int(blocks[0].self_attn.num_heads)
            head_dim = int(blocks[0].self_attn.head_dim)
            cache_frames = int(
                os.environ.get("TEMPOKV_ROLLING_CACHE_FRAMES", str(ROLLING_DEFAULT_CACHE_FRAMES))
            )
            if cache_frames < int(self.num_frame_per_block) * 2:
                raise ValueError("TEMPOKV_ROLLING_CACHE_FRAMES must fit two blocks")
            if cache_frames % int(self.num_frame_per_block):
                raise ValueError("Rolling Forcing cache frames must be block aligned")
            layers: list[dict[str, Any]] = []
            for layer_index in range(num_layers):
                k_cache = RollingChunkedKVCache(
                    batch_size,
                    frame_stride,
                    num_heads,
                    head_dim,
                    cache_frames,
                    dtype,
                    device,
                    layout="BSHD",
                    cache_role=f"rolling.k.{layer_index}",
                )
                v_cache = RollingChunkedKVCache(
                    batch_size,
                    frame_stride,
                    num_heads,
                    head_dim,
                    cache_frames,
                    dtype,
                    device,
                    layout="BSHD",
                    cache_role=f"rolling.v.{layer_index}",
                )
                layers.append(
                    {
                        "k": k_cache,
                        "v": v_cache,
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "cache_backend": "tempokv",
                        "_tempokv_encoder": self.tempokv_encoder,
                        "_tempokv_trace": self.tempokv_trace_collector,
                        "_tempokv_layers": layers,
                        "_tempokv_report": report,
                        "_tempokv_num_layers": num_layers,
                    }
                )
            # The list is complete only after construction; point every layer at
            # the final list so the last attention block can encode all layers.
            for layer in layers:
                layer["_tempokv_layers"] = layers
            self.kv_cache_clean = layers
            self.frame_seq_length = frame_stride
            report.num_layers = num_layers
            report.num_heads = num_heads
            report.head_dim = head_dim
            self._tempokv_runtime_ready = True

        def inference_rolling_forcing(self, *args: Any, **kwargs: Any):
            if self.tempokv_trace_collector is not None:
                self.tempokv_trace_collector.start_run()
            result = super().inference_rolling_forcing(*args, **kwargs)
            if self.tempokv_trace_collector is not None:
                self.tempokv_trace_collector.write(
                    num_layers=self.num_transformer_blocks
                )
            else:
                report.require_tempokv_activity()
                report_path = os.environ.get("TEMPOKV_ROLLING_REPORT")
                if report_path:
                    path = Path(report_path).expanduser().resolve()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(report.to_dict(), indent=2) + "\n",
                        encoding="utf-8",
                    )
            return result

    # ``inference.py`` imports the class from the package, while tests and
    # direct callers often import the concrete module. Patch both names.
    pipeline_module.CausalInferencePipeline = TempoKVRollingInferencePipeline
    pipeline_package.CausalInferencePipeline = TempoKVRollingInferencePipeline
    return TempoKVRollingInferencePipeline


__all__ = [
    "ROLLING_DEFAULT_CACHE_FRAMES",
    "ROLLING_DEFAULT_SINK_FRAMES",
    "RollingTempoKVEncoder",
    "install",
    "wrap_cache_state",
]
