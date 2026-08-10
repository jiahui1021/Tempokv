"""Collect Causal Forcing clean K/V traces for TempoKV parameter fitting.

Only the state TempoKV will actually compress is sampled: the normalized
**pre-RoPE** K and the raw V left behind by the clean/context refresh pass.
The four noisy denoising passes per block are never traced — fitting on them
would match the predictor to a distribution the long-term cache never holds.

Which pass counts as clean is not inferred from a timestep: the pipeline
announces it with ``note_clean_refresh`` before the forward, and confirms the
frame range with ``finalize_kv_cache`` afterwards.  Sampling only that pass also
avoids four fifths of the device-to-host traffic, since the four noisy
denoising passes per block would otherwise be copied out and thrown away.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


DEFAULT_TOKENS_PER_FRAME = 8


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
    frame_seq_length: int,
    num_heads: int,
    head_dim: int,
    tokens_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Take the same spatial positions from every frame of one projection.

    Sampling different positions per frame would destroy the temporal
    correlation the predictor is fitted on, so the token indices are a fixed
    stride and are recorded alongside the samples.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"Causal Forcing projection must use [B,S,C], got {tuple(tensor.shape)}"
        )
    batch, sequence, channels = tensor.shape
    if channels != num_heads * head_dim:
        raise ValueError(
            f"Causal Forcing trace expects {num_heads} heads of dimension "
            f"{head_dim}, got projection width {channels}"
        )
    if sequence % frame_seq_length:
        raise ValueError(
            f"projection sequence {sequence} is not divisible by frame size "
            f"{frame_seq_length}"
        )
    if tokens_per_frame > frame_seq_length:
        raise ValueError(
            "TEMPOKV_TRACE_TOKENS_PER_FRAME cannot exceed the frame size "
            f"{frame_seq_length}"
        )
    frame_count = sequence // frame_seq_length
    token_indices = torch.div(
        torch.arange(tokens_per_frame, device=tensor.device) * frame_seq_length,
        tokens_per_frame,
        rounding_mode="floor",
    )
    sampled = (
        tensor.detach()
        .reshape(batch, frame_count, frame_seq_length, num_heads, head_dim)
        .index_select(2, token_indices)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    return sampled, token_indices.to(device="cpu", dtype=torch.int32).numpy()


def install(runtime_root: str | Path | None = None) -> type:
    """Install clean pre-RoPE K/V capture into the bundled Causal Forcing runtime."""
    if runtime_root is not None:
        runtime_root = Path(runtime_root).resolve()
        if str(runtime_root) not in sys.path:
            sys.path.insert(0, str(runtime_root))

    output_value = os.environ.get("TEMPOKV_TRACE_DIR")
    if not output_value:
        raise ValueError("TEMPOKV_TRACE_DIR must be set for trace collection")
    output_root = Path(output_value).expanduser().resolve()
    tokens_per_frame = _positive_env(
        "TEMPOKV_TRACE_TOKENS_PER_FRAME", DEFAULT_TOKENS_PER_FRAME
    )

    pipeline_package = importlib.import_module("causal_forcing.pipeline")
    pipeline_module = importlib.import_module(
        "causal_forcing.pipeline.causal_inference"
    )
    base = pipeline_module.CausalInferencePipeline

    class _TraceVAEModel:
        def clear_cache(self) -> None:
            return None

    class TraceVAEWrapper(torch.nn.Module):
        """Skip the VAE entirely: a trace run never needs decoded pixels."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
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

    pipeline_module.WanVAEWrapper = TraceVAEWrapper

    class TraceCausalInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            blocks = self.generator.model.blocks
            output_root.mkdir(parents=True, exist_ok=True)
            self._trace_event_index = 0
            self._capturing = False
            self._latest: dict[int, dict[str, np.ndarray]] = {}
            self._latest_token_indices: np.ndarray | None = None
            self._sequence: dict[int, dict[str, list[np.ndarray]]] = {}
            self._sequence_token_indices: np.ndarray | None = None
            self._sequence_start: int | None = None
            self._sequence_end: int | None = None
            for layer_index, block in enumerate(blocks):
                block.self_attn.norm_k.register_forward_hook(
                    self._capture(layer_index, "k_pre_rope")
                )
                block.self_attn.v.register_forward_hook(
                    self._capture(layer_index, "v")
                )

        # -- capture ------------------------------------------------------

        def _capture(self, layer_index: int, kind: str):
            def hook(
                module: torch.nn.Module,
                args: tuple[Any, ...],
                output: torch.Tensor,
            ) -> None:
                del module, args
                if not self._capturing:
                    return
                sampled, token_indices = _sample_projection(
                    output,
                    frame_seq_length=self.frame_seq_length,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    tokens_per_frame=tokens_per_frame,
                )
                if self._latest_token_indices is None:
                    self._latest_token_indices = token_indices
                elif not np.array_equal(self._latest_token_indices, token_indices):
                    raise RuntimeError("Causal Forcing trace token samples do not match")
                self._latest.setdefault(layer_index, {})[kind] = sampled

            return hook

        # -- write-mode hooks ---------------------------------------------

        def reset_kv_cache_state(self) -> None:
            self._reset_sequence()

        def note_clean_refresh(self) -> None:
            self._capturing = True

        def finalize_kv_cache(self, start_frame: int, num_frames: int) -> None:
            super().finalize_kv_cache(start_frame, num_frames)
            if not self._capturing:
                raise RuntimeError(
                    "Causal Forcing finalized a block without announcing a "
                    "clean refresh first; the trace would be empty"
                )
            self._capturing = False
            self._commit(start_frame, num_frames)

        def inference(self, *args: Any, **kwargs: Any) -> Any:
            self._reset_sequence()
            try:
                result = super().inference(*args, **kwargs)
            except BaseException:
                self._reset_sequence()
                raise
            self._write_event()
            return result

        # -- accumulation --------------------------------------------------

        def _commit(self, start_frame: int, num_frames: int) -> None:
            layers = self.num_transformer_blocks
            if sorted(self._latest) != list(range(layers)):
                raise RuntimeError(
                    f"clean Causal Forcing pass did not capture all {layers} "
                    "attention layers"
                )
            token_indices = self._latest_token_indices
            if token_indices is None:
                raise RuntimeError("clean Causal Forcing pass captured no token indices")
            if self._sequence_start is None:
                self._sequence_start = start_frame
                self._sequence_end = start_frame
                self._sequence_token_indices = token_indices
            if start_frame != self._sequence_end:
                raise RuntimeError(
                    "clean Causal Forcing trace blocks are not temporally contiguous"
                )
            if not np.array_equal(self._sequence_token_indices, token_indices):
                raise RuntimeError(
                    "clean Causal Forcing trace blocks use different token samples"
                )
            for layer_index in range(layers):
                arrays = self._latest[layer_index]
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
                if keys.shape[1] != num_frames:
                    raise RuntimeError(
                        f"layer {layer_index} captured {keys.shape[1]} frames but "
                        f"the pipeline finalized {num_frames}"
                    )
                bucket = self._sequence.setdefault(
                    layer_index, {"k_pre_rope": [], "v": []}
                )
                bucket["k_pre_rope"].append(keys)
                bucket["v"].append(values)
            self._sequence_end = start_frame + num_frames
            self._latest = {}
            self._latest_token_indices = None

        def _write_event(self) -> None:
            if self._sequence_start is None or self._sequence_end is None:
                raise RuntimeError("Causal Forcing inference produced no clean K/V trace")
            layers = self.num_transformer_blocks
            if sorted(self._sequence) != list(range(layers)):
                raise RuntimeError(
                    f"Causal Forcing inference did not capture all {layers} layers"
                )
            token_indices = self._sequence_token_indices
            if token_indices is None:
                raise RuntimeError("Causal Forcing inference captured no token indices")

            generation_mode = (
                "framewise" if self.num_frame_per_block == 1 else "chunkwise"
            )
            rank = int(os.environ.get("RANK", "0"))
            rank_dir = output_root / f"rank_{rank:03d}"
            event_name = (
                f"event_{self._trace_event_index:06d}_frames_"
                f"{self._sequence_start}_{self._sequence_end}"
            )
            event_dir = rank_dir / event_name
            temporary_dir = rank_dir / f".{event_name}.tmp-{os.getpid()}"
            if event_dir.exists():
                raise FileExistsError(f"trace event already exists: {event_dir}")
            temporary_dir.mkdir(parents=True, exist_ok=False)

            try:
                for layer_index in range(layers):
                    arrays = self._sequence[layer_index]
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
                        num_layers=np.asarray(layers, dtype=np.int32),
                        num_heads=np.asarray(self.num_heads, dtype=np.int32),
                        head_dim=np.asarray(self.head_dim, dtype=np.int32),
                        token_indices=token_indices,
                        head_indices=np.arange(self.num_heads, dtype=np.int32),
                        global_start_frame=np.asarray(
                            self._sequence_start, dtype=np.int64
                        ),
                        global_end_frame=np.asarray(
                            self._sequence_end, dtype=np.int64
                        ),
                        trace_stride=np.asarray(self.frame_seq_length, dtype=np.int32),
                        frame_seq_length=np.asarray(
                            self.frame_seq_length, dtype=np.int32
                        ),
                        coordinate_system=np.asarray("pre_rope_normalized"),
                        cross_source_kind=np.asarray("normalized"),
                        generation_mode=np.asarray(generation_mode),
                        num_frame_per_block=np.asarray(
                            self.num_frame_per_block, dtype=np.int32
                        ),
                        is_clean_refresh=np.asarray(True),
                    )
                temporary_dir.replace(event_dir)
                self._trace_event_index += 1
                self._reset_sequence()
            except BaseException:
                shutil.rmtree(temporary_dir, ignore_errors=True)
                raise

        def _reset_sequence(self) -> None:
            self._capturing = False
            self._latest = {}
            self._latest_token_indices = None
            self._sequence = {}
            self._sequence_token_indices = None
            self._sequence_start = None
            self._sequence_end = None

    TraceCausalInferencePipeline.__name__ = "TraceCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TraceCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TraceCausalInferencePipeline
    return TraceCausalInferencePipeline


__all__ = ["DEFAULT_TOKENS_PER_FRAME", "install"]
