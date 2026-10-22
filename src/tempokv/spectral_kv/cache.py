"""Frame-addressed experimental spectral cache; all positions remain absolute."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .attention import AttentionGroup, KVView
from .model_spec import ModelSpec
from .policy import FrameMeta, FullPolicy, HistoryPlan, Mode
from .projection import cell_centers, compress_grid, make_projection
from .rope3d import full_positions, rope_apply, rope_invert


@dataclass
class FrameEntry:
    frame_idx: int
    grid: tuple[int, int]
    count: float
    k_pre: torch.Tensor
    v: torch.Tensor
    pos_hw: tuple[torch.Tensor, torch.Tensor]
    original_grid: tuple[int, int]


def entry_positions(entry: FrameEntry):
    h, w = entry.grid
    ph, pw = entry.pos_hw
    return (torch.full((h * w,), float(entry.frame_idx), device=entry.k_pre.device),
            ph[:, None].expand(h, w).flatten(), pw[None, :].expand(h, w).flatten())


class SpectralKVCache:
    def __init__(self, spec: ModelSpec, layer: int, policy=None, proj_kind="dct", keep_originals=False,
                 *, freqs=None, taper=None, sink_frames=1, precision="float64"):
        self.spec, self.layer, self.policy = spec, layer, policy or FullPolicy()
        self.proj_kind, self.keep_originals, self.taper = proj_kind, keep_originals, taper
        self.freqs, self.sink_frames, self.precision = freqs, sink_frames, precision
        self.entries: dict[int, FrameEntry] = {}
        self.originals: dict[int, FrameEntry] = {}
        self.current_frames: set[int] = set()
        self.represented_frames: set[int] = set()

    def write_chunk(self, k, v, frame_idx0: int, k_is_roped=False):
        if k.ndim != 4 or k.shape != v.shape or k.shape[2:] != (self.spec.heads, self.spec.head_dim):
            raise ValueError("cache expects matching [B,S,heads,D] in the runtime model shape")
        if k.dtype != v.dtype or k.device != v.device:
            raise ValueError("cache K/V dtype and device must agree")
        h, w = self.spec.frame_hw
        if frame_idx0 < 0 or k.shape[1] == 0 or k.shape[1] % (h * w):
            raise ValueError("write_chunk must contain complete frames at a nonnegative absolute frame")
        frames = k.shape[1] // (h * w)
        if k_is_roped:
            positions = full_positions(frame_idx0, frames, (h, w), device=k.device)
            k = rope_invert(k, *positions, self.spec, freqs=self.freqs, precision=self.precision)
        self.current_frames = set(range(frame_idx0, frame_idx0 + frames))
        for offset in range(frames):
            frame = frame_idx0 + offset
            sl = slice(offset * h * w, (offset + 1) * h * w)
            # Clone only this cache's owned current frame; caller tensors are never mutated.
            entry = FrameEntry(frame, (h, w), 1.0, k[:, sl].detach().clone(), v[:, sl].detach().clone(),
                               (torch.arange(h, device=k.device).float(), torch.arange(w, device=k.device).float()), (h, w))
            self.entries[frame] = entry
            self.represented_frames.add(frame)
            if self.keep_originals:
                self.originals[frame] = entry

    def evict_outside(self, frames):
        frames = set(int(f) for f in frames)
        for mapping in (self.entries, self.originals):
            for frame in list(mapping):
                if frame not in frames:
                    del mapping[frame]
        self.represented_frames.intersection_update(frames)

    def _compress(self, entry: FrameEntry, target):
        th, tw = target
        h, w = entry.grid
        if th > h or tw > w:
            raise ValueError("cannot re-expand a summary without keep_originals=true")
        batch, _, heads, dim = entry.k_pre.shape
        ph, pw = make_projection(self.proj_kind, h, th, self.taper), make_projection(self.proj_kind, w, tw, self.taper)
        values = []
        for tensor in (entry.k_pre, entry.v):
            grid = tensor.reshape(batch, h, w, heads, dim)
            values.append(compress_grid(grid, ph, pw).reshape(batch, th * tw, heads, dim))
        oh, ow = entry.original_grid
        return FrameEntry(entry.frame_idx, (th, tw), oh * ow / (th * tw), *values,
                          (cell_centers(oh, th).to(entry.k_pre.device), cell_centers(ow, tw).to(entry.k_pre.device)),
                          entry.original_grid)

    def age(self, chunk_idx: int, *, q_probe=None, current_frames=None):
        current = self.current_frames if current_frames is None else set(current_frames)
        meta = [FrameMeta(frame, self.entries[frame].grid if frame in self.entries else self.spec.frame_hw,
                          self.spec.frame_hw, frame in current, frame < self.sink_frames)
                for frame in sorted(self.represented_frames)]
        plan = self.policy.plan(self.layer, chunk_idx, meta, q_probe=q_probe)
        if not isinstance(plan, HistoryPlan) or set(plan.entries) != self.represented_frames:
            raise ValueError("policy must provide one plan for every represented frame")
        for frame, (mode, target) in plan.entries.items():
            protected = frame in current or frame < self.sink_frames
            if protected:
                mode, target = Mode.FULL, None
            if mode == Mode.DROP:
                self.entries.pop(frame, None)
                continue
            source = self.originals.get(frame) if self.keep_originals else self.entries.get(frame)
            if source is None:
                raise ValueError("policy requested a dropped frame; keep_originals=true is required for re-selection")
            if mode == Mode.FULL:
                if source.grid != source.original_grid:
                    raise ValueError("policy requested FULL from a summary without an original")
                self.entries[frame] = source
            elif mode == Mode.SUMMARY:
                if target is None or len(target) != 2 or min(target) < 1:
                    raise ValueError("SUMMARY requires a positive target grid")
                if source.grid == tuple(target) and not (self.keep_originals and source is self.originals.get(frame)):
                    continue
                self.entries[frame] = self._compress(source, tuple(target))
            else:
                raise ValueError("invalid frame mode")
        return plan

    def materialize(self, chunk_idx=0, *, bias=True, frames=None):
        selected = set(self.entries) if frames is None else set(frames)
        groups = []
        for frame, entry in sorted(self.entries.items()):
            if frame not in selected:
                continue
            k = rope_apply(entry.k_pre, *entry_positions(entry), self.spec, freqs=self.freqs, precision=self.precision)
            groups.append(AttentionGroup(k, entry.v, math.log(entry.count) if bias else 0.0))
        if not groups:
            raise ValueError("cannot materialize an empty attention cache")
        return KVView(groups)

    @property
    def resident_bytes(self):
        tensors = [tensor for entry in self.entries.values() for tensor in (entry.k_pre, entry.v, *entry.pos_hw)]
        if self.keep_originals:
            tensors += [tensor for entry in self.originals.values() for tensor in (entry.k_pre, entry.v, *entry.pos_hw)]
        seen = set()
        total = 0
        for tensor in tensors:
            pointer = (tensor.device, tensor.untyped_storage().data_ptr())
            if pointer not in seen:
                seen.add(pointer)
                total += tensor.untyped_storage().nbytes()
        return total
