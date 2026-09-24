"""Collect Causal Forcing clean traces for TempoKV parameter fitting.

Only the clean refresh pass is sampled (pre-RoPE K, V, and the post-RoPE Q/K
for the attention objective); noisy denoising passes never enter the cache.
The pipeline announces that pass with ``note_clean_refresh``.
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
_TRACE_KINDS = ("k_pre_rope", "v", "q_post_rope", "k_post_rope")


def _sample_projection(
    tensor: torch.Tensor,
    *,
    frame_seq_length: int,
    num_heads: int,
    head_dim: int,
    tokens_per_frame: int,
    legacy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the same spatial positions from every frame."""
    batch, sequence, _ = tensor.shape
    frame_count = sequence // frame_seq_length
    if legacy:
        # Rounded linspace over both frame ends, as the legacy protocol expects.
        token_indices = torch.linspace(
            0,
            frame_seq_length - 1,
            steps=tokens_per_frame,
            device=tensor.device,
        ).round().long().unique()
    else:
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
    """Install clean K/V and attention-objective trace capture."""
    if runtime_root is not None:
        runtime_root = Path(runtime_root).resolve()
        if str(runtime_root) not in sys.path:
            sys.path.insert(0, str(runtime_root))

    output_value = os.environ.get("TEMPOKV_TRACE_DIR")
    if not output_value:
        raise ValueError("TEMPOKV_TRACE_DIR must be set for trace collection")
    output_root = Path(output_value).expanduser().resolve()
    tokens_per_frame = int(
        os.environ.get("TEMPOKV_TRACE_TOKENS_PER_FRAME", DEFAULT_TOKENS_PER_FRAME)
    )
    event_frames = int(os.environ.get("TEMPOKV_TRACE_EVENT_FRAMES", "0"))

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
            # Traces sample the pre-RoPE chunked cache TempoKV compresses.
            self._uses_pre_rope_cache = True
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
                block.self_attn.tempokv_trace_recorder = self._rope_recorder(
                    layer_index
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
                sampled, self._latest_token_indices = _sample_projection(
                    output,
                    frame_seq_length=self.frame_seq_length,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    tokens_per_frame=tokens_per_frame,
                    legacy=event_frames > 0,
                )
                self._latest.setdefault(layer_index, {})[kind] = sampled

            return hook

        def _rope_recorder(self, layer_index: int):
            def record(
                query_post_rope: torch.Tensor,
                key_post_rope: torch.Tensor,
            ) -> None:
                if not self._capturing:
                    return
                for kind, tensor in (
                    ("q_post_rope", query_post_rope),
                    ("k_post_rope", key_post_rope),
                ):
                    if tensor.ndim == 4:
                        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
                    sampled, _ = _sample_projection(
                        tensor,
                        frame_seq_length=self.frame_seq_length,
                        num_heads=self.num_heads,
                        head_dim=self.head_dim,
                        tokens_per_frame=tokens_per_frame,
                        legacy=event_frames > 0,
                    )
                    self._latest.setdefault(layer_index, {})[kind] = sampled

            return record

        # -- write-mode hooks ---------------------------------------------

        def reset_kv_cache_state(self) -> None:
            self._reset_sequence()

        def note_clean_refresh(self) -> None:
            self._capturing = True

        def finalize_kv_cache(self, start_frame: int, num_frames: int) -> None:
            super().finalize_kv_cache(start_frame, num_frames)
            self._capturing = False
            self._commit(start_frame, num_frames)

        def inference(self, *args: Any, **kwargs: Any) -> Any:
            self._reset_sequence()
            # The terminal clean refresh is trace data too.
            kwargs["retain_final_cache"] = True
            try:
                result = super().inference(*args, **kwargs)
            except BaseException:
                self._reset_sequence()
                raise
            self._write_event()
            return result

        # -- accumulation --------------------------------------------------

        def _commit(self, start_frame: int, num_frames: int) -> None:
            if self._sequence_start is None:
                self._sequence_start = start_frame
                self._sequence_end = start_frame
                self._sequence_token_indices = self._latest_token_indices
            # A gap would splice two separate moments into one trace sequence.
            if start_frame != self._sequence_end:
                raise RuntimeError(
                    "clean Causal Forcing trace blocks are not temporally contiguous"
                )
            for layer_index in range(self.num_transformer_blocks):
                bucket = self._sequence.setdefault(
                    layer_index, {kind: [] for kind in _TRACE_KINDS}
                )
                for kind in _TRACE_KINDS:
                    bucket[kind].append(self._latest[layer_index][kind])
            self._sequence_end = start_frame + num_frames
            self._latest = {}
            self._latest_token_indices = None

        def _write_event(self) -> None:
            if self._sequence_start is None or self._sequence_end is None:
                raise RuntimeError("Causal Forcing inference produced no clean K/V trace")
            layers = self.num_transformer_blocks
            token_indices = self._sequence_token_indices

            generation_mode = (
                "framewise" if self.num_frame_per_block == 1 else "chunkwise"
            )
            rank = int(os.environ.get("RANK", "0"))
            rank_dir = output_root / f"rank_{rank:03d}"

            # Native collection writes the rollout as one event; legacy
            # (TEMPOKV_TRACE_EVENT_FRAMES=24) cuts it into 24-frame records.
            merged = {
                layer_index: {
                    kind: np.concatenate(arrays[kind], axis=1)
                    for kind in _TRACE_KINDS
                }
                for layer_index, arrays in self._sequence.items()
            }
            total_frames = merged[0]["k_pre_rope"].shape[1]
            frames_per_event = event_frames or total_frames
            complete_events = total_frames // frames_per_event

            for event_offset in range(complete_events):
                frame_start = event_offset * frames_per_event
                frame_end = frame_start + frames_per_event
                global_start_frame = self._sequence_start + frame_start
                global_end_frame = self._sequence_start + frame_end
                if event_frames:
                    tokens_start = global_start_frame * self.frame_seq_length
                    tokens_end = global_end_frame * self.frame_seq_length
                    event_name = (
                        f"event_{self._trace_event_index:06d}_tokens_"
                        f"{tokens_start}_{tokens_end}"
                    )
                else:
                    event_name = (
                        f"event_{self._trace_event_index:06d}_frames_"
                        f"{self._sequence_start}_{self._sequence_end}"
                    )
                event_dir = rank_dir / event_name
                temporary_dir = rank_dir / f".{event_name}.tmp-{os.getpid()}"
                temporary_dir.mkdir(parents=True, exist_ok=False)

                try:
                    for layer_index in range(layers):
                        arrays = merged[layer_index]
                        np.savez(
                            temporary_dir / f"layer_{layer_index:02d}.npz",
                            **{
                                kind: arrays[kind][:, frame_start:frame_end]
                                for kind in _TRACE_KINDS
                            },
                            layer_index=np.asarray(layer_index, dtype=np.int32),
                            num_layers=np.asarray(layers, dtype=np.int32),
                            num_heads=np.asarray(self.num_heads, dtype=np.int32),
                            head_dim=np.asarray(self.head_dim, dtype=np.int32),
                            token_indices=token_indices,
                            head_indices=np.arange(self.num_heads, dtype=np.int32),
                            global_start_frame=np.asarray(
                                global_start_frame, dtype=np.int64
                            ),
                            global_end_frame=np.asarray(
                                global_end_frame, dtype=np.int64
                            ),
                            tokens_start=np.asarray(
                                global_start_frame * self.frame_seq_length,
                                dtype=np.int64,
                            ),
                            tokens_end=np.asarray(
                                global_end_frame * self.frame_seq_length,
                                dtype=np.int64,
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
                except BaseException:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                    raise
            self._reset_sequence()

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
