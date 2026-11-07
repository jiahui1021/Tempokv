"""Head-aligned block diagonal plus one global cross-head low-rank matrix.

Row-vector convention: y = block(x, D) + (x @ A) @ B. Unlike per-head
diagonal+SVD, A/B mix *all* heads. Three alternating ridge / reduced-rank
updates follow CORAL-KV Algorithm 1. This module is a matrix-structure study,
not a reproduction of CORAL's cross-layer inputs or its residual quantizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockDPLR:
    blocks: torch.Tensor  # [H,D,D]
    a: torch.Tensor       # [H*D,r]
    b: torch.Tensor       # [r,H*D]

    def __post_init__(self):
        if self.blocks.ndim != 3 or self.blocks.shape[1] != self.blocks.shape[2]:
            raise ValueError("DPLR blocks must be [heads,dim,dim]")
        channels = self.blocks.shape[0]*self.blocks.shape[1]
        if self.a.shape != (channels, self.b.shape[0]) or self.b.shape[1] != channels:
            raise ValueError("DPLR low-rank factors must span all heads")

    @property
    def rank(self):
        return self.a.shape[1]

    def __call__(self, x):
        if x.shape[-2:] != self.blocks.shape[:2]:
            raise ValueError("DPLR input head geometry mismatch")
        output = torch.einsum("...hi,hij->...hj", x, self.blocks)
        if self.rank:
            output = output + ((x.flatten(-2) @ self.a) @ self.b).reshape_as(output)
        return output

    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.blocks, self.a, self.b))

    def state(self):
        return {"blocks": self.blocks.cpu(), "a": self.a.cpu(), "b": self.b.cpu()}

    @classmethod
    def from_state(cls, state, device="cpu"):
        return cls(*(state[k].to(device) for k in ("blocks", "a", "b")))


class RegressionStatistics:
    """Sufficient statistics from calibration tokens only; exact reduced-rank solve."""
    def __init__(self, x, target, *, ridge=1e-3):
        if x.shape != target.shape or x.ndim != 3 or ridge <= 0:
            raise ValueError("fit needs matching [tokens,heads,dim] and positive ridge")
        self.heads, self.dim = x.shape[-2:]
        self.channels = self.heads*self.dim
        x, target = x.double().flatten(1), target.double().flatten(1)
        if not torch.isfinite(x).all() or not torch.isfinite(target).all():
            raise ValueError("non-finite DPLR calibration data")
        self.gram, self.xy = x.T @ x, x.T @ target
        self.yy = target.square().sum()
        self.block_gram = torch.stack([self.gram[s:s+self.dim, s:s+self.dim]
                                      for s in range(0, self.channels, self.dim)])
        self.block_xy = torch.stack([self.xy[s:s+self.dim, s:s+self.dim]
                                    for s in range(0, self.channels, self.dim)])
        self.block_penalty = ridge*self.block_gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)
        self.penalty = ridge*self.gram.diagonal().mean().clamp_min(1e-20)
        self.block_chol = torch.linalg.cholesky(self.block_gram + self.block_penalty[:, None, None]*torch.eye(self.dim, device=x.device))
        self.chol = torch.linalg.cholesky(self.gram+self.penalty*torch.eye(self.channels, device=x.device))
        self.initial_blocks = self.fit_blocks(x.new_empty(self.channels, 0), x.new_empty(0, self.channels))
        # All ranks share the first block update and its exact decomposition.
        self.initial_svd = None

    def fit_blocks(self, a, b):
        rhs = self.block_xy
        if a.shape[-1]:
            ga = (self.gram @ a).reshape(self.heads, self.dim, -1)
            bh = b.reshape(b.shape[0], self.heads, self.dim).permute(1, 0, 2)
            rhs = rhs - ga @ bh
        return torch.cholesky_solve(rhs, self.block_chol)

    def reduced_rank_svd(self, blocks):
        # G @ block_diag(D), without constructing the global dense matrix.
        gd = torch.einsum("ihd,hdj->ihj", self.gram.reshape(self.channels, self.heads, self.dim), blocks).reshape(self.channels, self.channels)
        remaining = self.xy-gd
        whitened = torch.linalg.solve_triangular(self.chol, remaining, upper=False)
        return torch.linalg.svd(whitened, full_matrices=False)

    def factors(self, decomposition, rank):
        u, singular, vh = decomposition
        a = torch.linalg.solve_triangular(self.chol.T, u[:, :rank]*singular[:rank], upper=True)
        return a, vh[:rank]

    def objective(self, blocks, a, b):
        value = self.yy + (blocks*(self.block_gram @ blocks)).sum()-2*(blocks*self.block_xy).sum()
        value += (self.block_penalty[:, None, None]*blocks.square()).sum()
        if a.shape[-1]:
            ga = self.gram @ a
            bh = b.reshape(b.shape[0], self.heads, self.dim).permute(1, 0, 2)
            cross = ga.reshape(self.heads, self.dim, -1) @ bh
            value += 2*(blocks*cross).sum() + (b*((a.T@ga)@b)).sum()-2*(a*(self.xy@b.T)).sum()
            value += self.penalty*((a.T@a)*(b@b.T).T).sum()
        return float(value)

    @torch.no_grad()
    def fit(self, rank, *, rounds=3, dtype=torch.float32):
        if not 0 <= rank <= self.channels or rounds < 1:
            raise ValueError("invalid global DPLR rank or alternating rounds")
        blocks = self.initial_blocks
        a, b = self.gram.new_empty(self.channels, 0), self.gram.new_empty(0, self.channels)
        logs = [{"round": 0, "objective": self.objective(blocks, a, b)}]
        if rank:
            if self.initial_svd is None:
                self.initial_svd = self.reduced_rank_svd(blocks)
            a, b = self.factors(self.initial_svd, rank)
            logs.append({"round": 1, "objective": self.objective(blocks, a, b)})
            for step in range(1, rounds):
                blocks = self.fit_blocks(a, b)
                a, b = self.factors(self.reduced_rank_svd(blocks), rank)
                logs.append({"round": step+1, "objective": self.objective(blocks, a, b)})
        return BlockDPLR(blocks.to(dtype), a.to(dtype), b.to(dtype)), logs


@torch.no_grad()
def replay_layer(reconstructed_keys, target, model, bias, history_weights, config, *, return_reconstruction=False):
    """Production anchor/residual/BF16 recurrence, changing only the K→V map."""
    from tempokv.fitting import _anchor_roundtrip, _quantization_roundtrip
    if reconstructed_keys.shape != target.shape or target.ndim != 4:
        raise ValueError("layer replay needs matching [records,frames,heads,dim]")
    if config.fit_mode != "native":
        raise ValueError("block-DPLR experiment currently matches native reference arithmetic")
    dtype = getattr(torch, config.reconstruction_dtype)
    rebuilt = torch.empty_like(target, dtype=dtype)
    rebuilt[:, 0] = _anchor_roundtrip(target[:, 0], config).to(dtype)
    signal = error = prediction_error = 0.0
    for frame in range(1, target.shape[1]):
        prediction = model(reconstructed_keys[:, frame].float())
        prediction.add_(bias)
        for lag in range(1, min(frame, history_weights.shape[0])+1):
            prediction.add_(history_weights[lag-1]*rebuilt[:, frame-lag].float())
        residual = target[:, frame].float()-prediction
        quantized = _quantization_roundtrip(residual, bits=config.v_bits, config=config,
                                            symmetric=config.residual_mode == "symmetric")
        rebuilt[:, frame] = (prediction+quantized).to(dtype)
        signal += target[:, frame].double().square().sum().item()
        error += (target[:, frame].double()-rebuilt[:, frame].double()).square().sum().item()
        prediction_error += residual.double().square().sum().item()
    result = {"signal_sse": signal, "error_sse": error, "prediction_sse": prediction_error,
              "numel": target[:, 1:].numel()}
    return (result, rebuilt) if return_reconstruction else result
