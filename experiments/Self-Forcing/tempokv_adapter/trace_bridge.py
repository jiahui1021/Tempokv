from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


_FRAME_SEQ_LENGTH = 1560
_HEADS = 12
_HEAD_DIM = 128
_LAYERS = 30


def _positive_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error
    if parsed < 1:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _sample_projection(
    tensor: torch.Tensor,
    *,
    tokens_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tensor.ndim != 3:
        raise ValueError(
            "Self-Forcing projection must use [B,S,C], got "
            f"{tuple(tensor.shape)}"
        )
    batch, sequence, channels = tensor.shape
    if channels != _HEADS * _HEAD_DIM:
        raise ValueError(
            "Self-Forcing trace requires 12 heads with dimension 128, got "
            f"projection width {channels}"
        )
    if sequence % _FRAME_SEQ_LENGTH:
        raise ValueError(
            f"projection sequence {sequence} is not divisible by frame size "
            f"{_FRAME_SEQ_LENGTH}"
        )
    if tokens_per_frame > _FRAME_SEQ_LENGTH:
        raise ValueError(
            "TEMPOKV_TRACE_TOKENS_PER_FRAME cannot exceed the frame size "
            f"{_FRAME_SEQ_LENGTH}"
        )

    frame_count = sequence // _FRAME_SEQ_LENGTH
    token_indices = torch.div(
        torch.arange(tokens_per_frame, device=tensor.device) * _FRAME_SEQ_LENGTH,
        tokens_per_frame,
        rounding_mode="floor",
    )
    sampled = (
        tensor.detach()
        .reshape(
            batch,
            frame_count,
            _FRAME_SEQ_LENGTH,
            _HEADS,
            _HEAD_DIM,
        )
        .index_select(2, token_indices)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    return sampled, token_indices.to(device="cpu", dtype=torch.int32).numpy()


def _argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    index: int,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def install(runtime_root: str | Path) -> type:
    """Install clean pre-RoPE K/V capture into the bundled Self-Forcing runtime."""
    runtime_root = Path(runtime_root).resolve()
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))

    output_value = os.environ.get("TEMPOKV_TRACE_DIR")
    if not output_value:
        raise ValueError("TEMPOKV_TRACE_DIR must be set for trace collection")
    output_root = Path(output_value).expanduser().resolve()
    tokens_per_frame = _positive_env("TEMPOKV_TRACE_TOKENS_PER_FRAME", 8)

    pipeline_package = importlib.import_module("pipeline")
    pipeline_module = importlib.import_module("pipeline.causal_inference")
    base = pipeline_module.CausalInferencePipeline

    class TraceVAEWrapper(torch.nn.Module):
        """Avoid loading and running the VAE when only attention traces are needed."""

        def __init__(self) -> None:
            super().__init__()
            self.model = _TraceVAEModel()

        def clear_cache(self) -> None:
            return None

        def decode_to_pixel(
            self, latent: torch.Tensor, use_cache: bool = False
        ) -> torch.Tensor:
            del use_cache
            return torch.zeros(
                (latent.shape[0], latent.shape[1], 3, 1, 1),
                device=latent.device,
                dtype=torch.float32,
            )

    class _TraceVAEModel:
        def clear_cache(self) -> None:
            return None

    pipeline_module.WanVAEWrapper = TraceVAEWrapper

    class TraceCausalInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            frame_seq_length = int(getattr(self, "frame_seq_length", 0))
            if frame_seq_length != _FRAME_SEQ_LENGTH:
                raise ValueError(
                    "TempoKV Self-Forcing trace collection requires frame size "
                    f"{_FRAME_SEQ_LENGTH}, got {frame_seq_length}"
                )

            blocks = getattr(getattr(self.generator, "model", None), "blocks", None)
            if blocks is None or len(blocks) != _LAYERS:
                actual = "missing" if blocks is None else str(len(blocks))
                raise ValueError(
                    f"Self-Forcing trace requires exactly {_LAYERS} layers, got "
                    f"{actual}"
                )

            output_root.mkdir(parents=True, exist_ok=True)
            self._trace_enabled = False
            self._trace_start = 0
            self._trace_event_index = 0
            self._trace_layers: dict[int, dict[str, np.ndarray]] = {}
            self._trace_token_indices: np.ndarray | None = None
            self._trace_sequence_layers: dict[
                int, dict[str, list[np.ndarray]]
            ] = {}
            self._trace_sequence_token_indices: np.ndarray | None = None
            self._trace_sequence_start: int | None = None
            self._trace_sequence_end: int | None = None

            self.generator.register_forward_pre_hook(
                self._trace_generator_start,
                with_kwargs=True,
            )
            self.generator.register_forward_hook(
                self._trace_generator_end,
                with_kwargs=True,
                always_call=True,
            )
            for layer_index, block in enumerate(blocks):
                block.self_attn.norm_k.register_forward_hook(
                    self._trace_projection_hook(layer_index, "k_pre_rope")
                )
                block.self_attn.v.register_forward_hook(
                    self._trace_projection_hook(layer_index, "v")
                )

        def inference(self, *args: Any, **kwargs: Any) -> Any:
            self._reset_trace_sequence()
            try:
                result = super().inference(*args, **kwargs)
                self._write_trace_event()
                return result
            except BaseException:
                self._reset_trace_sequence()
                raise

        def _trace_generator_start(
            self,
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            del module
            timestep = _argument(args, kwargs, "timestep", 2)
            kv_cache = _argument(args, kwargs, "kv_cache", 3)
            current_start = _argument(args, kwargs, "current_start", 5)
            context_noise = int(getattr(self.args, "context_noise", 0))
            is_clean = (
                isinstance(timestep, torch.Tensor)
                and timestep.numel() > 0
                and bool(torch.all(timestep == context_noise).item())
            )
            self._trace_enabled = kv_cache is not None and is_clean
            self._trace_layers = {}
            self._trace_token_indices = None
            if self._trace_enabled:
                if current_start is None:
                    raise ValueError("clean Self-Forcing call is missing current_start")
                self._trace_start = int(current_start)

        def _trace_projection_hook(self, layer_index: int, kind: str):
            def capture(
                module: torch.nn.Module,
                args: tuple[Any, ...],
                output: torch.Tensor,
            ) -> None:
                del module, args
                if not self._trace_enabled:
                    return
                sampled, token_indices = _sample_projection(
                    output,
                    tokens_per_frame=tokens_per_frame,
                )
                if self._trace_token_indices is None:
                    self._trace_token_indices = token_indices
                elif not np.array_equal(self._trace_token_indices, token_indices):
                    raise RuntimeError("Self-Forcing trace token samples do not match")
                self._trace_layers.setdefault(layer_index, {})[kind] = sampled

            return capture

        def _trace_generator_end(
            self,
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs
            if not self._trace_enabled:
                return
            if output is None:
                self._trace_enabled = False
                self._trace_layers = {}
                self._trace_token_indices = None
                return
            try:
                self._append_trace_block()
            finally:
                self._trace_enabled = False
                self._trace_layers = {}
                self._trace_token_indices = None

        def _append_trace_block(self) -> None:
            if sorted(self._trace_layers) != list(range(_LAYERS)):
                raise RuntimeError(
                    "clean Self-Forcing pass did not capture all 30 attention layers"
                )
            token_indices = self._trace_token_indices
            if token_indices is None:
                raise RuntimeError("clean Self-Forcing pass did not capture token indices")

            first = self._trace_layers[0]
            if set(first) != {"k_pre_rope", "v"}:
                raise RuntimeError("clean Self-Forcing pass did not capture both K and V")
            frames = first["k_pre_rope"].shape[1]
            block_end = self._trace_start + frames * _FRAME_SEQ_LENGTH
            if self._trace_sequence_start is None:
                self._trace_sequence_start = self._trace_start
                self._trace_sequence_end = self._trace_start
                self._trace_sequence_token_indices = token_indices
            if self._trace_start != self._trace_sequence_end:
                raise RuntimeError(
                    "clean Self-Forcing trace blocks are not temporally contiguous"
                )
            if not np.array_equal(
                self._trace_sequence_token_indices, token_indices
            ):
                raise RuntimeError(
                    "clean Self-Forcing trace blocks use different token samples"
                )

            for layer_index in range(_LAYERS):
                arrays = self._trace_layers[layer_index]
                if set(arrays) != {"k_pre_rope", "v"}:
                    raise RuntimeError(
                        f"layer {layer_index} did not capture both clean K and V"
                    )
                keys = arrays["k_pre_rope"]
                values = arrays["v"]
                if keys.shape != values.shape:
                    raise ValueError(
                        f"layer {layer_index} K/V shapes differ: "
                        f"{keys.shape} vs {values.shape}"
                    )
                sequence = self._trace_sequence_layers.setdefault(
                    layer_index,
                    {"k_pre_rope": [], "v": []},
                )
                sequence["k_pre_rope"].append(keys)
                sequence["v"].append(values)
            self._trace_sequence_end = block_end

        def _write_trace_event(self) -> None:
            if self._trace_sequence_start is None or self._trace_sequence_end is None:
                raise RuntimeError("Self-Forcing inference produced no clean K/V trace")
            if sorted(self._trace_sequence_layers) != list(range(_LAYERS)):
                raise RuntimeError(
                    "Self-Forcing inference did not capture all 30 attention layers"
                )
            token_indices = self._trace_sequence_token_indices
            if token_indices is None:
                raise RuntimeError("Self-Forcing inference did not capture token indices")

            tokens_start = self._trace_sequence_start
            tokens_end = self._trace_sequence_end
            rank = int(os.environ.get("RANK", "0"))
            rank_dir = output_root / f"rank_{rank:03d}"
            event_name = (
                f"event_{self._trace_event_index:06d}_tokens_"
                f"{tokens_start}_{tokens_end}"
            )
            event_dir = rank_dir / event_name
            temporary_dir = rank_dir / f".{event_name}.tmp-{os.getpid()}"
            if event_dir.exists():
                raise FileExistsError(f"trace event already exists: {event_dir}")
            temporary_dir.mkdir(parents=True, exist_ok=False)

            try:
                for layer_index in range(_LAYERS):
                    arrays = self._trace_sequence_layers[layer_index]
                    keys = np.concatenate(arrays["k_pre_rope"], axis=1)
                    values = np.concatenate(arrays["v"], axis=1)
                    if keys.shape != values.shape:
                        raise ValueError(
                            f"layer {layer_index} K/V shapes differ: "
                            f"{keys.shape} vs {values.shape}"
                        )
                    np.savez_compressed(
                        temporary_dir / f"layer_{layer_index:02d}.npz",
                        k_pre_rope=keys,
                        v=values,
                        layer_index=np.asarray(layer_index, dtype=np.int32),
                        token_indices=token_indices,
                        head_indices=np.arange(_HEADS, dtype=np.int32),
                        tokens_start=np.asarray(tokens_start, dtype=np.int64),
                        tokens_end=np.asarray(tokens_end, dtype=np.int64),
                        frame_seq_length=np.asarray(
                            _FRAME_SEQ_LENGTH, dtype=np.int32
                        ),
                    )
                temporary_dir.replace(event_dir)
                self._trace_event_index += 1
                self._reset_trace_sequence()
            except BaseException:
                shutil.rmtree(temporary_dir, ignore_errors=True)
                raise

        def _reset_trace_sequence(self) -> None:
            self._trace_sequence_layers = {}
            self._trace_sequence_token_indices = None
            self._trace_sequence_start = None
            self._trace_sequence_end = None

    TraceCausalInferencePipeline.__name__ = "TraceCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TraceCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TraceCausalInferencePipeline
    return TraceCausalInferencePipeline
