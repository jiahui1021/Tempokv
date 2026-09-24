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


# Pre-RoPE K and V feed the predictors; rotated Q and K feed the attention
# objective.  All four are sampled at the same token positions.
_TRACE_KINDS = ("k_pre_rope", "v", "q_post_rope", "k_post_rope")


def _sample_projection(
    tensor: torch.Tensor,
    *,
    tokens_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch, sequence, _ = tensor.shape
    frame_count = sequence // _FRAME_SEQ_LENGTH
    # Match the legacy deterministic sampler: rounded linspace over both ends.
    token_indices = torch.linspace(
        0,
        _FRAME_SEQ_LENGTH - 1,
        steps=tokens_per_frame,
        device=tensor.device,
    ).round().long().unique()
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
    tokens_per_frame = int(os.environ.get("TEMPOKV_TRACE_TOKENS_PER_FRAME", "8"))
    event_frames = int(os.environ.get("TEMPOKV_TRACE_EVENT_FRAMES", "0"))
    rope_precision = os.environ.get("TEMPOKV_TRACE_ROPE_PRECISION", "native").strip()

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
            # Trace collection reads clean pre-RoPE K/V out of the cache, so it
            # needs the frame-addressable cache for the same reason TempoKV
            # does, not the dense post-RoPE BF16 one.
            self._uses_pre_rope_cache = True
            self._trace_rope_precision = rope_precision
            self._apply_trace_rope_precision()
            blocks = self.generator.model.blocks

            output_root.mkdir(parents=True, exist_ok=True)
            self._trace_enabled = False
            self._clean_refresh_announced = False
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
                block.self_attn.tempokv_trace_recorder = self._trace_rope_recorder(
                    layer_index
                )

        def _apply_trace_rope_precision(self) -> None:
            if self.kv_cache1 is None:
                return
            for layer in self.kv_cache1:
                layer["k"].rope_precision = self._trace_rope_precision

        def _initialize_kv_cache(self, *args: Any, **kwargs: Any) -> Any:
            result = super()._initialize_kv_cache(*args, **kwargs)
            self._apply_trace_rope_precision()
            return result

        def note_clean_refresh(self) -> None:
            self._clean_refresh_announced = True

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
            kv_cache = _argument(args, kwargs, "kv_cache", 3)
            current_start = _argument(args, kwargs, "current_start", 5)
            # The pipeline announced this pass with `note_clean_refresh`; the
            # timestep is not consulted, so a changed denoising schedule cannot
            # cause noisy K/V to be traced as if it were clean.
            self._trace_enabled = kv_cache is not None and self._clean_refresh_announced
            self._clean_refresh_announced = False
            self._trace_layers = {}
            self._trace_token_indices = None
            if self._trace_enabled:
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
                sampled, self._trace_token_indices = _sample_projection(
                    output,
                    tokens_per_frame=tokens_per_frame,
                )
                self._trace_layers.setdefault(layer_index, {})[kind] = sampled

            return capture

        def _trace_rope_recorder(self, layer_index: int):
            def record(
                query_post_rope: torch.Tensor,
                key_post_rope: torch.Tensor,
            ) -> None:
                if not self._trace_enabled:
                    return
                for kind, tensor in (
                    ("q_post_rope", query_post_rope),
                    ("k_post_rope", key_post_rope),
                ):
                    if tensor.ndim == 4:
                        tensor = tensor.reshape(
                            tensor.shape[0], tensor.shape[1], -1
                        )
                    sampled, _ = _sample_projection(
                        tensor, tokens_per_frame=tokens_per_frame
                    )
                    self._trace_layers.setdefault(layer_index, {})[kind] = sampled

            return record

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
            try:
                if output is not None:
                    self._append_trace_block()
            finally:
                self._trace_enabled = False
                self._trace_layers = {}
                self._trace_token_indices = None

        def _append_trace_block(self) -> None:
            frames = self._trace_layers[0]["k_pre_rope"].shape[1]
            if self._trace_sequence_start is None:
                self._trace_sequence_start = self._trace_start
                self._trace_sequence_end = self._trace_start
                self._trace_sequence_token_indices = self._trace_token_indices
            # A gap would splice two separate moments into one trace sequence.
            if self._trace_start != self._trace_sequence_end:
                raise RuntimeError(
                    "clean Self-Forcing trace blocks are not temporally contiguous"
                )
            for layer_index in range(_LAYERS):
                sequence = self._trace_sequence_layers.setdefault(
                    layer_index, {kind: [] for kind in _TRACE_KINDS}
                )
                for kind in _TRACE_KINDS:
                    sequence[kind].append(self._trace_layers[layer_index][kind])
            self._trace_sequence_end = (
                self._trace_start + frames * _FRAME_SEQ_LENGTH
            )

        def _write_trace_event(self) -> None:
            if self._trace_sequence_start is None or self._trace_sequence_end is None:
                raise RuntimeError("Self-Forcing inference produced no clean K/V trace")
            token_indices = self._trace_sequence_token_indices

            sequence_start = self._trace_sequence_start
            sequence_end = self._trace_sequence_end
            rank = int(os.environ.get("RANK", "0"))
            rank_dir = output_root / f"rank_{rank:03d}"
            combined = {
                layer_index: {
                    kind: np.concatenate(arrays[kind], axis=1)
                    for kind in _TRACE_KINDS
                }
                for layer_index, arrays in self._trace_sequence_layers.items()
            }
            total_frames = (sequence_end - sequence_start) // _FRAME_SEQ_LENGTH
            frames_per_event = event_frames or total_frames
            complete_events = total_frames // frames_per_event
            if complete_events < 1:
                raise ValueError(
                    "trace did not reach one complete production event of "
                    f"{frames_per_event} frames"
                )
            for event_offset in range(complete_events):
                frame_start = event_offset * frames_per_event
                frame_end = frame_start + frames_per_event
                tokens_start = sequence_start + frame_start * _FRAME_SEQ_LENGTH
                tokens_end = sequence_start + frame_end * _FRAME_SEQ_LENGTH
                event_name = (
                    f"event_{self._trace_event_index:06d}_tokens_"
                    f"{tokens_start}_{tokens_end}"
                )
                event_dir = rank_dir / event_name
                temporary_dir = rank_dir / f".{event_name}.tmp-{os.getpid()}"
                temporary_dir.mkdir(parents=True, exist_ok=False)
                try:
                    for layer_index in range(_LAYERS):
                        merged = combined[layer_index]
                        np.savez(
                            temporary_dir / f"layer_{layer_index:02d}.npz",
                            **{
                                kind: merged[kind][:, frame_start:frame_end]
                                for kind in _TRACE_KINDS
                            },
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
                except BaseException:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                    raise
            self._reset_trace_sequence()

        def _reset_trace_sequence(self) -> None:
            self._clean_refresh_announced = False
            self._trace_sequence_layers = {}
            self._trace_sequence_token_indices = None
            self._trace_sequence_start = None
            self._trace_sequence_end = None

    TraceCausalInferencePipeline.__name__ = "TraceCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TraceCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TraceCausalInferencePipeline
    return TraceCausalInferencePipeline
