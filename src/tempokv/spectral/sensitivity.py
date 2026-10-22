"""E2 exact leave-one-block-out attention on sampled queries and the full KV window."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F

from .metrics import relative_error

Attention = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Block:
    history_frame: int
    spatial_block_id: int
    indices: tuple[int, ...]
    freq_k_high_ratio: float = float("nan")
    freq_v_high_ratio: float = float("nan")
    temporal_similarity: float = float("nan")


def effective_positions(*, tokens: int, current_start: int, local_start_index: int,
                        attention_start: int, frame_seqlen: int, sink_frames: int = 0,
                        contiguous: bool = True, absolute_positions=None) -> np.ndarray:
    if min(tokens, frame_seqlen) < 1 or current_start < 0 or current_start % frame_seqlen:
        raise ValueError("invalid token count, frame size or frame-aligned current_start")
    if absolute_positions is not None:
        raw = np.asarray(absolute_positions)
        if raw.shape != (tokens,) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("explicit absolute positions must be integer [effective_k_tokens]")
        positions = raw.astype(np.int64, copy=True)
        if np.any(positions < 0) or np.any(np.diff(positions) <= 0):
            raise ValueError("explicit absolute positions must be unique and increasing")
        return positions
    if sink_frames or not contiguous:
        raise ValueError("sink/gapped windows require an explicit absolute position mapping")
    offset = local_start_index - attention_start
    start = current_start - offset
    if offset < 0 or offset >= tokens or start < 0:
        raise ValueError("inconsistent contiguous window metadata")
    return np.arange(start, start + tokens, dtype=np.int64)


def reference_attention(q, k, v):
    """CPU/offline reference. Runtime probes inject the repository attention function."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2)).transpose(1, 2).contiguous()


def validate_qkv(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("expected matching [B,S,heads,D] K/V")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:] or min(q.shape[1], k.shape[1]) < 1:
        raise ValueError("Q/K/V batch, selected heads and head dimension must agree")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V dtype and device must agree")


def remove_attention(q, k, v, remove=(), *, attention_fn: Attention = reference_attention):
    validate_qkv(q, k, v)
    if len(remove) == 0:
        return attention_fn(q, k, v)
    indices = torch.as_tensor(remove, dtype=torch.long, device=k.device)
    if torch.any(indices < 0) or torch.any(indices >= k.shape[1]):
        raise ValueError("removed token is outside the attention window")
    keep = torch.ones(k.shape[1], dtype=torch.bool, device=k.device)
    keep[indices] = False
    if not keep.any():
        raise ValueError("cannot remove every attention token")
    return attention_fn(q, k[:, keep], v[:, keep])


def attention_mass(q, k, blocks: list[Block], *, chunk_size: int = 1024) -> list[float]:
    """Two-pass FP32 QK, O(B*heads*Q*chunk_size) memory, full-window denominator."""
    scale = q.shape[-1] ** -0.5
    query = q.float().transpose(1, 2)
    log_z = None
    for start in range(0, k.shape[1], chunk_size):
        scores = (query @ k[:, start:start + chunk_size].float().permute(0, 2, 3, 1)) * scale
        chunk_z = torch.logsumexp(scores, dim=-1)
        log_z = chunk_z if log_z is None else torch.logaddexp(log_z, chunk_z)
    result = []
    for block in blocks:
        indices = torch.tensor(block.indices, dtype=torch.long, device=k.device)
        scores = (query @ k.index_select(1, indices).float().permute(0, 2, 3, 1)) * scale
        result.append(float(torch.exp(scores - log_z[..., None]).sum(dim=-1).mean()))
    return result


@torch.no_grad()
def leave_one_block_out(q, k, v, blocks: list[Block], *, attention_fn: Attention = reference_attention,
                        chunk_size: int = 1024, eps: float = 1e-12):
    validate_qkv(q, k, v)
    reference = attention_fn(q, k, v)
    masses = attention_mass(q, k, blocks, chunk_size=chunk_size)
    rows = []
    for block, mass in zip(blocks, masses):
        actual = remove_attention(q, k, v, block.indices, attention_fn=attention_fn)
        rows.append({"attention_mass": mass, "oracle_error": relative_error(reference, actual, eps)})
    return reference, rows
