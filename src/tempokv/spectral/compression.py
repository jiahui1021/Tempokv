"""E3 independent offline simulations; selection never mutates the model cache."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from .config import Compression
from .metrics import nmse, relative_error, standardize
from .sensitivity import Block, reference_attention, remove_attention


def frequency_scores(blocks: list[Block]) -> np.ndarray:
    return np.asarray([(b.freq_k_high_ratio + b.freq_v_high_ratio) / 2 for b in blocks])


def policy_scores(policy: str, blocks: list[Block], rows: list[dict], alpha: float = 0.5):
    if policy == "frequency":
        return frequency_scores(blocks)
    if policy == "similarity":
        # Redundant frames receive a lower keep priority.
        return -np.asarray([b.temporal_similarity for b in blocks])
    if policy == "attention":
        return np.asarray([r["attention_mass"] for r in rows])
    if policy == "oracle_greedy":
        return np.asarray([r["oracle_error"] for r in rows])
    if policy == "frequency_attention":
        if alpha == 0:
            return standardize([r["attention_mass"] for r in rows])
        if alpha == 1:
            return standardize(frequency_scores(blocks))
        return alpha * standardize(frequency_scores(blocks)) + (1 - alpha) * standardize(
            [r["attention_mass"] for r in rows]
        )
    raise ValueError(f"no scalar score for {policy}")


def _spread_order(n: int) -> list[int]:
    """Coarse-to-fine uniform priorities in frame/block order."""
    if n == 0:
        return []
    result = [0] if n == 1 else [0, n - 1]
    intervals = [(0, n - 1)]
    while intervals:
        left, right = intervals.pop(0)
        middle = (left + right) // 2
        if middle > left:
            result.append(middle)
            intervals.extend(((left, middle), (middle, right)))
    return result


def select_tokens(blocks: list[Block], budget: int, *, policy: str, rows: list[dict],
                  seed: int = 0, alpha: float = 0.5) -> tuple[list[int], int]:
    """Exact token budget. The last ranked block can be partially retained.

    All policies use the same rule for that boundary block, including unequal
    edge blocks. Noncandidate tokens are handled separately and always retained.
    """
    tokens = [token for b in blocks for token in b.indices]
    if len(tokens) != len(set(tokens)) or not 0 <= budget <= len(tokens):
        raise ValueError("candidate blocks must be disjoint and budget feasible")
    if policy == "random":
        order = np.random.default_rng(seed).permutation(len(blocks)).tolist()
    elif policy == "uniform":
        chronological = sorted(range(len(blocks)), key=lambda i: (blocks[i].history_frame, blocks[i].spatial_block_id))
        order = [chronological[i] for i in _spread_order(len(blocks))]
    else:
        scores = policy_scores(policy, blocks, rows, alpha)
        if not np.isfinite(scores).any():
            raise ValueError("not measured: no finite policy scores")
        order = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf), kind="stable").tolist()
    kept, partial = [], 0
    for index in order:
        remaining = budget - len(kept)
        if remaining == 0:
            break
        block_tokens = blocks[index].indices
        take = min(remaining, len(block_tokens))
        kept.extend(block_tokens[:take])
        if take < len(block_tokens):
            partial = take
    return sorted(kept), partial


@torch.no_grad()
def evaluate_budgets(q, k, v, blocks: list[Block], rows: list[dict], config: Compression,
                     *, attention_fn=reference_attention, reference=None, seed: int = 0,
                     protected_indices=(), eps: float = 1e-12) -> list[dict]:
    if len(rows) != len(blocks):
        raise ValueError("one score row is required for every candidate block")
    candidate = {token for block in blocks for token in block.indices}
    if len(candidate) != sum(len(b.indices) for b in blocks):
        raise ValueError("candidate blocks overlap")
    if any(token < 0 or token >= k.shape[1] for token in candidate):
        raise ValueError("candidate outside KV window")
    if candidate.intersection(protected_indices):
        raise ValueError("current chunk/recent protected tokens cannot be candidates")
    if reference is None:
        reference = attention_fn(q, k, v)
    results = []
    for policy in config.policies:
        seeds = config.random_seeds if policy == "random" else (0,)
        alphas = config.combination_alphas if policy == "frequency_attention" else (0.5,)
        for random_seed in seeds:
            for alpha in alphas:
                for ratio in config.retention_ratios:
                    budget = int(len(candidate) * ratio + 0.5)
                    base = {
                        "policy": policy, "retention_ratio": ratio,
                        "random_seed": seed + random_seed if policy == "random" else "",
                        "combination_alpha": alpha if policy == "frequency_attention" else "",
                        "candidate_tokens": len(candidate), "target_kept_candidate_tokens": budget,
                        "untouched_tokens": k.shape[1] - len(candidate),
                        "candidate_scope": "sampled_candidate_set", "budget_granularity": "tokens_partial_boundary_block",
                    }
                    try:
                        kept, partial = select_tokens(blocks, budget, policy=policy, rows=rows,
                                                      seed=seed + random_seed, alpha=alpha)
                    except ValueError as exc:
                        if not str(exc).startswith("not measured:"):
                            raise
                        results.append({**base, "status": str(exc), "kept_candidate_tokens": "",
                                        "actual_retained_tokens": "", "partial_block_tokens": "",
                                        "relative_attention_output_error": float("nan"), "nmse": float("nan")})
                        continue
                    remove = sorted(candidate - set(kept))
                    actual = remove_attention(q, k, v, remove, attention_fn=attention_fn)
                    results.append({**base, "status": "measured", "kept_candidate_tokens": len(kept),
                                    "actual_retained_tokens": k.shape[1] - len(remove),
                                    "partial_block_tokens": partial,
                                    "relative_attention_output_error": relative_error(reference, actual, eps),
                                    "nmse": nmse(reference, actual, eps)})
    return results


def save_snapshot(path: Path, q, k, v, blocks: list[Block], rows: list[dict], metadata: dict) -> None:
    """Opt-in, one sample/head; FP32 serialization preserves every BF16 value."""
    payload = {"blocks": [asdict(b) for b in blocks], "scores": rows,
               "metadata": metadata, "dtype": str(q.dtype).split(".")[-1]}
    with path.open("xb") as handle:
        np.savez_compressed(handle, q=q.detach().float().cpu().numpy(),
                            k=k.detach().float().cpu().numpy(), v=v.detach().float().cpu().numpy(),
                            description=np.asarray(json.dumps(payload)))


def load_snapshot(path: Path, device="cpu"):
    with np.load(path, allow_pickle=False) as data:
        payload = json.loads(str(data["description"]))
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[payload["dtype"]]
        q, k, v = (torch.from_numpy(data[name].copy()).to(device=device, dtype=dtype) for name in ("q", "k", "v"))
    blocks = [Block(**{**b, "indices": tuple(b["indices"])}) for b in payload["blocks"]]
    return q, k, v, blocks, payload["scores"], payload["metadata"]
