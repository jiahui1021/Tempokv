"""Optional sampled online reconstruction observer without changing codec math."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def wan_pixel_interval(latent_frame: int, pixel_frames: int) -> tuple[int, int]:
    if latent_frame < 0:
        raise ValueError("latent_frame must be non-negative")
    start, stop = (0, 1) if latent_frame == 0 else (4*(latent_frame-1)+1, 4*latent_frame+1)
    if start >= pixel_frames:
        raise ValueError("latent frame is outside the decoded video")
    return start, min(stop, pixel_frames)


class ReconstructionObserver:
    def __init__(self, path: str | Path, *, model: str, frames: int, fps: float = 16,
                 tokens_per_frame: int = 8, prompt_hashes: dict[int, str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.model, self.frames, self.fps = model, frames, fps
        self.tokens_per_frame = int(tokens_per_frame)
        if self.tokens_per_frame <= 0 or fps <= 0:
            raise ValueError("tokens_per_frame and fps must be positive")
        self.prompt_hashes = prompt_hashes or {}
        self.context: dict[str, Any] = {}
        self.offsets: dict[tuple[int, str], int] = {}

    def set_context(self, *, prompt_id: int, seed: int) -> None:
        self.context = {"prompt_id": prompt_id, "prompt_sha256": self.prompt_hashes.get(prompt_id), "seed": seed}
        self.offsets.clear()

    def sample(self, tensor: torch.Tensor, stride: int):
        parts = []
        positions = []
        for begin in range(0, tensor.shape[2], stride):
            width = min(stride, tensor.shape[2]-begin)
            indices = torch.linspace(0, width-1, min(width, self.tokens_per_frame), device=tensor.device).round().long().unique()
            parts.append(tensor[:, :, begin:begin+width].index_select(2, indices).detach().float().clone())
            positions.append(indices.cpu().tolist())
        return parts, positions

    def observe(self, sample, reconstruction: torch.Tensor, *, role: str, layer: int,
                stride: int, is_new_span: bool) -> None:
        originals, positions = sample
        start = self.offsets.get((layer, role), 0)
        for index, (target, indices) in enumerate(zip(originals, positions, strict=True)):
            selected = torch.tensor(indices, device=reconstruction.device)
            result = reconstruction[:, :, index*stride:index*stride+min(stride, reconstruction.shape[2]-index*stride)].index_select(2, selected).float()
            pixel_start, pixel_end = wan_pixel_interval(start+index, self.frames)
            for batch in range(target.shape[0]):
                signal = float(target[batch].double().square().sum())
                error = float((target[batch]-result[batch]).double().square().sum())
                row = {**self.context, "model": self.model, "sample_id": batch, "role": role,
                       "layer": layer, "head": "all", "absolute_latent_frame": start+index,
                       "pixel_start": pixel_start, "pixel_end": pixel_end,
                       "start_seconds": pixel_start/self.fps, "end_seconds": pixel_end/self.fps,
                       "is_anchor": is_new_span and index == 0,
                       "metric_scope": "online_local_reconstruction", "token_indices": indices,
                       "sampling": "fixed_uniform_spatial_positions_all_channels_and_heads",
                       "numel": target[batch].numel(), "signal_sse": signal, "error_sse": error,
                       "prediction_sse": None, "quant_error_sse": None,
                       "reconstruction_nmse": error/signal if signal > 1e-20 else None}
                self.handle.write(json.dumps(row, allow_nan=False)+"\n")
        self.handle.flush()
        self.offsets[layer, role] = start+len(originals)

    def close(self):
        self.handle.close()


def _decode(packed, source=None):
    if callable(getattr(packed, "decode", None)):
        return packed.decode(source=source, output_layout="BHSD")
    from tempokv.codec import decode_tensor
    return decode_tensor(packed, source=source, output_layout="BHSD", use_triton=False)


class ObservedEncoder:
    def __init__(self, encoder, observer, role, layer):
        self.encoder, self.observer, self.role, self.layer = encoder, observer, role, layer

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def append(self, tensor, **kwargs):
        sample = self.observer.sample(tensor, self.encoder.packed.stride)
        result = self.encoder.append(tensor, **kwargs)
        self.observer.observe(sample, result, role=self.role, layer=self.layer,
                              stride=self.encoder.packed.stride, is_new_span=False)
        return result


class ObservedCodec:
    def __init__(self, codec, observer: ReconstructionObserver):
        self.codec, self.observer = codec, observer

    def __getattr__(self, name):
        return getattr(self.codec, name)

    def encode_primary(self, tensor, *, layer=0, reuse_input=False):
        sample = self.observer.sample(tensor, self.config.stride)
        packed, result = self.codec.encode_primary(tensor, layer=layer, reuse_input=reuse_input)
        self.observer.observe(sample, result, role=self.primary_role, layer=layer,
                              stride=self.config.stride, is_new_span=True)
        return packed, result

    def encode_secondary(self, tensor, reconstructed_primary, *, layer=0, reuse_input=False):
        sample = self.observer.sample(tensor, self.config.stride)
        packed = self.codec.encode_secondary(tensor, reconstructed_primary, layer=layer, reuse_input=reuse_input)
        result = tensor if reuse_input else _decode(packed, reconstructed_primary)
        self.observer.observe(sample, result, role=self.secondary_role, layer=layer,
                              stride=self.config.stride, is_new_span=True)
        return packed

    def begin_incremental_primary(self, tensor, *, capacity_units, layer=0, reuse_input=False):
        sample = self.observer.sample(tensor, self.config.stride)
        encoder, result = self.codec.begin_incremental_primary(tensor, capacity_units=capacity_units, layer=layer, reuse_input=reuse_input)
        self.observer.observe(sample, result, role=self.primary_role, layer=layer,
                              stride=self.config.stride, is_new_span=True)
        return ObservedEncoder(encoder, self.observer, self.primary_role, layer), result

    def begin_incremental_secondary(self, tensor, reconstructed_primary, *, capacity_units, layer=0, reuse_input=False):
        sample = self.observer.sample(tensor, self.config.stride)
        encoder = self.codec.begin_incremental_secondary(tensor, reconstructed_primary, capacity_units=capacity_units,
                                                         layer=layer, reuse_input=reuse_input)
        result = tensor if reuse_input else _decode(encoder.packed, reconstructed_primary)
        self.observer.observe(sample, result, role=self.secondary_role, layer=layer,
                              stride=self.config.stride, is_new_span=True)
        return ObservedEncoder(encoder, self.observer, self.secondary_role, layer)
