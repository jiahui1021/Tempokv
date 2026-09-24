"""Simulated KV-token pruning for the pruning pilot.

Pruned tokens are dropped from the attention key set only; the cache and its
predictor history are untouched.  Scores come from the open-loop K residual,
adjacent-frame cosine and received attention mass.

Environment: TEMPOKV_PRUNE_CRITERION (none|random|attention|similarity|
residual|attn_residual), TEMPOKV_PRUNE_RATIO, TEMPOKV_PRUNE_PROTECT (6),
TEMPOKV_PRUNE_LAMBDA (1), TEMPOKV_PRUNE_QSTRIDE (8), TEMPOKV_PRUNE_STATS_DIR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from tempokv.codec import predict_temporal


CRITERIA = (
    "none",
    "random",
    "attention",
    "similarity",
    "residual",
    "attn_residual",
)
_QUERY_CHUNK = 128


def _rank01(values: torch.Tensor) -> torch.Tensor:
    """Within-frame percentile in [0, 1): scale-free, so A and E add."""
    return values.argsort().argsort().float() / values.numel()


class LayerPruner:
    def __init__(
        self,
        layer: int,
        codec,
        *,
        criterion: str,
        ratio: float,
        protect_frames: int,
        lam: float,
        query_stride: int,
        frame_seqlen: int,
        max_frames: int,
        seed: int,
    ) -> None:
        self.layer = layer
        self.codec = codec
        self.criterion = criterion
        self.ratio = ratio
        self.protect_frames = protect_frames
        self.lam = lam
        self.query_stride = query_stride
        self.frame_seqlen = frame_seqlen
        self.max_frames = max_frames
        self.seed = seed
        self.legacy = codec.config.fit_mode == "legacy"
        self.reset()

    def reset(self) -> None:
        self._last_start: int | None = None
        self._pending: tuple[int, torch.Tensor] | None = None
        self._history: list[torch.Tensor] = []
        self._residual: dict[int, torch.Tensor] = {}
        self._similarity: dict[int, torch.Tensor] = {}
        self._attention: torch.Tensor | None = None
        self._pruned: torch.Tensor | None = None
        self._observations = [0] * self.max_frames
        self._decided = 0
        self.rows: list[dict[str, np.ndarray]] = []

    def _buffers(self, device: torch.device) -> None:
        if self._pruned is None:
            tokens = self.max_frames * self.frame_seqlen
            self._pruned = torch.zeros(tokens, dtype=torch.bool, device=device)
            self._attention = torch.zeros(tokens, dtype=torch.float32, device=device)

    def __call__(self, query, new_key, attn_k, attn_v, positions, block_start):
        """Filter one attention call's K/V; ``positions`` index ``attn_k``."""
        self._buffers(attn_k.device)
        first_step = block_start != self._last_start
        if first_step:
            if self._pending is not None:
                self._register(*self._pending)
            self._last_start = block_start
        # The last call of a block is the clean refresh, so whatever is pending
        # when the next block starts is the clean K that went into the cache.
        self._pending = (block_start // self.frame_seqlen, new_key)

        keep = ~self._pruned[positions]
        if not bool(keep.all()):
            index = keep.nonzero().squeeze(1)
            attn_k = attn_k.index_select(1, index)
            attn_v = attn_v.index_select(1, index)
            positions = positions[index]
        if first_step:
            self._observe(query, attn_k, positions, block_start)
            self._decide(block_start // self.frame_seqlen)
        return attn_k, attn_v

    @torch.no_grad()
    def _register(self, first_frame: int, key: torch.Tensor) -> None:
        temporal = self.codec._temporal_for_layer(key.device, self.layer)
        order = int(temporal.history_length)
        size = self.frame_seqlen
        for offset in range(key.shape[1] // size):
            # Codec layout is [B, H, S, D]; the attention layer hands [B, S, H, D].
            current = key[:1, offset * size:(offset + 1) * size].transpose(1, 2)
            frame = first_frame + offset
            if self._history and order:
                prediction = predict_temporal(
                    self._history, temporal, legacy=self.legacy
                )
                residual = current.float() - prediction
                self._residual[frame] = residual.square().sum((1, 3)).sqrt()[0]
                previous = self._history[0].float().transpose(1, 2).flatten(2)
                now = current.float().transpose(1, 2).flatten(2)
                self._similarity[frame] = torch.nn.functional.cosine_similarity(
                    now, previous, dim=-1
                )[0]
            self._history = [current.contiguous(), *self._history][: max(order, 1)]

    @torch.no_grad()
    def _observe(self, query, attn_k, positions, block_start) -> None:
        queries = query[0, :: self.query_stride].transpose(0, 1)  # [H, Q, D]
        keys = attn_k[0, :, :, : queries.shape[-1]].transpose(0, 1)  # [H, K, D]
        scale = queries.shape[-1] ** -0.5
        received = torch.zeros(keys.shape[1], dtype=torch.float32, device=keys.device)
        for start in range(0, queries.shape[1], _QUERY_CHUNK):
            chunk = queries[:, start:start + _QUERY_CHUNK]
            scores = torch.matmul(chunk, keys.transpose(1, 2)).float() * scale
            received += scores.softmax(-1).sum((0, 1))
        cached = positions < block_start
        self._attention.index_add_(0, positions[cached], received[cached])
        for frame in range(block_start // self.frame_seqlen):
            self._observations[frame] += 1

    @torch.no_grad()
    def _decide(self, block_frame: int) -> None:
        size = self.frame_seqlen
        while self._decided < block_frame - self.protect_frames:
            frame = self._decided
            self._decided += 1
            tokens = slice(frame * size, (frame + 1) * size)
            attention = self._attention[tokens] / max(self._observations[frame], 1)
            residual = self._residual.pop(frame, None)
            similarity = self._similarity.pop(frame, None)
            # Frame 0 has no temporal history, so no residual or similarity; it
            # is never pruned under any criterion to keep the budget identical.
            prune_count = round(self.ratio * size) if residual is not None else 0
            if self.criterion != "none" and prune_count:
                if self.criterion == "random":
                    generator = torch.Generator().manual_seed(
                        self.seed * 100003 + self.layer * 1009 + frame
                    )
                    order = torch.randperm(size, generator=generator).to(attention.device)
                else:
                    score = {
                        "attention": lambda: attention,
                        "similarity": lambda: -similarity,
                        "residual": lambda: residual,
                        "attn_residual": lambda: _rank01(attention)
                        + self.lam * _rank01(residual),
                    }[self.criterion]()
                    order = score.argsort()
                self._pruned[frame * size + order[:prune_count]] = True
            nan = torch.full_like(attention, float("nan"))
            self.rows.append(
                {
                    "frame": frame,
                    "attention": attention.cpu().numpy(),
                    "residual": (nan if residual is None else residual).cpu().numpy(),
                    "similarity": (nan if similarity is None else similarity)
                    .cpu()
                    .numpy(),
                    "pruned": self._pruned[tokens].cpu().numpy(),
                }
            )


class PruningController:
    def __init__(self, layers: list[LayerPruner], stats_dir: Path | None) -> None:
        self.layers = layers
        self.stats_dir = stats_dir
        self._videos = 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def finish(self, total_frames: int) -> None:
        """Write this video's per-token statistics and its retention."""
        if self.stats_dir is None:
            return
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        stem = self.stats_dir / f"rank{rank}-video{self._videos:04d}"
        self._videos += 1
        frames = np.array([row["frame"] for row in self.layers[0].rows], dtype=np.int32)
        arrays = {"frames": frames}
        for key in ("attention", "residual", "similarity", "pruned"):
            arrays[key] = np.stack(
                [np.stack([row[key] for row in layer.rows]) for layer in self.layers]
            )  # [L, frames, tokens]
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        pruned = int(arrays["pruned"].sum())
        total = len(self.layers) * total_frames * self.layers[0].frame_seqlen
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "criterion": self.layers[0].criterion,
                    "ratio": self.layers[0].ratio,
                    "protect_frames": self.layers[0].protect_frames,
                    "pruned_tokens": pruned,
                    "total_tokens": total,
                    "retention": 1.0 - pruned / total,
                },
                indent=2,
            )
            + "\n"
        )


def from_env(codec, *, num_layers: int, frame_seqlen: int, max_frames: int):
    criterion = os.environ.get("TEMPOKV_PRUNE_CRITERION")
    if not criterion:
        return None
    if criterion not in CRITERIA:
        raise ValueError(f"TEMPOKV_PRUNE_CRITERION must be one of {CRITERIA}")
    if getattr(codec.config, "predictor_direction", "k_to_v") != "k_to_v":
        raise ValueError("pruning residuals need the K temporal predictor (k_to_v)")
    if getattr(codec.config, "jensen_correction", False):
        raise ValueError("pruning does not support the Jensen correction")
    ratio = float(os.environ.get("TEMPOKV_PRUNE_RATIO", "0"))
    if not 0.0 <= ratio < 1.0:
        raise ValueError("TEMPOKV_PRUNE_RATIO must be in [0, 1)")
    stats_dir = os.environ.get("TEMPOKV_PRUNE_STATS_DIR")
    layers = [
        LayerPruner(
            layer,
            codec,
            criterion=criterion,
            ratio=ratio,
            protect_frames=int(os.environ.get("TEMPOKV_PRUNE_PROTECT", "6")),
            lam=float(os.environ.get("TEMPOKV_PRUNE_LAMBDA", "1")),
            query_stride=int(os.environ.get("TEMPOKV_PRUNE_QSTRIDE", "8")),
            frame_seqlen=frame_seqlen,
            max_frames=max_frames,
            seed=int(os.environ.get("TEMPOKV_PRUNE_SEED", "0")),
        )
        for layer in range(num_layers)
    ]
    return PruningController(layers, Path(stats_dir) if stats_dir else None)
