"""Explicit position and protection contracts for full-window research inputs."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import json
import torch
from tempokv.spectral.sensitivity import effective_positions

@dataclass(frozen=True)
class ProbeContext:
    prompt_index: int | str
    seed: int
    sample_id: int
    layer_id: int
    head_id: int
    chunk_id: int
    phase: str
    denoise_step_index: int | None
    denoise_timestep: int | None
    current_chunk_start: int
    grid_t: int
    grid_h: int
    grid_w: int
    query_abs_positions: torch.Tensor
    key_abs_positions: torch.Tensor
    sink_frames: int = 0
    protect_recent_frames: int = 0
    source: str = "bf16"

    def __post_init__(self):
        if self.phase not in {"denoise", "clean_refresh"} or min(self.grid_t, self.grid_h, self.grid_w) < 1:
            raise ValueError("invalid phase/grid")
        for name in ("query_abs_positions", "key_abs_positions"):
            x = getattr(self, name)
            if x.ndim != 1 or x.dtype != torch.int64 or torch.any(x < 0) or (x.numel() > 1 and torch.any(x[1:] <= x[:-1])):
                raise ValueError(f"{name} needs unique increasing int64 absolute positions")
        if torch.any(self.query_abs_positions < self.current_chunk_start): raise ValueError("queries must belong to the current chunk")
        if self.current_chunk_start % self.frame_tokens: raise ValueError("chunk start must be frame aligned")

    @property
    def frame_tokens(self): return self.grid_h * self.grid_w

    @property
    def protected_mask(self):
        return ((self.key_abs_positions >= self.current_chunk_start - self.protect_recent_frames * self.frame_tokens)
                | (self.key_abs_positions < self.sink_frames * self.frame_tokens))

    @property
    def eligible_mask(self): return (~self.protected_mask) & (self.key_abs_positions < self.current_chunk_start)

    def row(self):
        return {"prompt_index": self.prompt_index, "seed": self.seed, "sample_id": self.sample_id,
                "layer": self.layer_id, "head": self.head_id, "chunk_index": self.chunk_id, "phase": self.phase,
                "denoise_step_index": self.denoise_step_index, "denoise_timestep": self.denoise_timestep,
                "current_chunk_start": self.current_chunk_start, "grid_t": self.grid_t, "grid_h": self.grid_h, "grid_w": self.grid_w,
                "query_samples": self.query_abs_positions.numel(), "history_token_count": int((self.key_abs_positions < self.current_chunk_start).sum()),
                "protected_token_count": int(self.protected_mask.sum()), "protected_fraction": float(self.protected_mask.float().mean()),
                "key_token_count": self.key_abs_positions.numel(), "source": self.source}

    @property
    def identity(self):
        row = self.row(); row["query_positions"] = self.query_abs_positions.tolist(); row["key_positions"] = self.key_abs_positions.tolist()
        return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]


def native_positions(metadata, tokens):
    return torch.from_numpy(effective_positions(tokens=tokens, current_start=int(metadata["current_start"]),
        local_start_index=int(metadata["local_start_index"]), attention_start=int(metadata["attention_start"]),
        frame_seqlen=int(metadata["frame_seqlen"]), sink_frames=int(metadata.get("sink_frames", 0)),
        contiguous=metadata.get("contiguous", True), absolute_positions=metadata.get("absolute_positions"))).long()

@dataclass(frozen=True)
class FrozenAttention:
    context: ProbeContext
    q_post: torch.Tensor
    k_post: torch.Tensor
    v: torch.Tensor
    attention_fn: object
    k_pre: torch.Tensor | None = None
    rotate_pre: object | None = None
    pre_valid_mask: torch.Tensor | None = None

    def __post_init__(self):
        q, k, v = self.q_post, self.k_post, self.v
        if q.ndim != 4 or k.shape != v.shape or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError("frozen attention requires aligned [B,Seq,Heads,D] tensors")
        if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)): raise ValueError("Q/K/V dtype/device differ")
        if q.shape[1] != self.context.query_abs_positions.numel() or k.shape[1] != self.context.key_abs_positions.numel():
            raise ValueError("position mapping length disagrees with Q/K/V")
        if self.k_pre is not None and self.k_pre.shape != k.shape: raise ValueError("pre-RoPE K shape differs")
        if self.pre_valid_mask is not None and (self.pre_valid_mask.shape != (k.shape[1],) or self.pre_valid_mask.dtype != torch.bool):
            raise ValueError("pre_valid_mask must identify exact clean pre-RoPE positions")
