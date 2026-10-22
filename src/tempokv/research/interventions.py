"""Pure tensor interventions on one frozen full attention window.

These are research approximations, not cache codecs. In particular, merged and
relocated keys are formed before RoPE and rotated at their new coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import torch

from .context import FrozenAttention


@dataclass(frozen=True)
class KVVariant:
    k_post: torch.Tensor
    v: torch.Tensor
    coordinates: torch.Tensor
    log_count: torch.Tensor
    provenance: tuple[tuple[int, ...], ...]
    details: dict

    @property
    def token_count(self) -> int:
        return self.k_post.shape[1]


def _validate(window: FrozenAttention) -> None:
    k, v, q = window.k_post, window.v, window.q_post
    if (k.ndim != 4 or k.shape != v.shape or q.ndim != 4
            or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]
            or any(d < 1 for d in k.shape) or any(d < 1 for d in q.shape)):
        raise ValueError("requires nonempty aligned [B,Tokens,Heads,D] Q/K/V")
    if k.shape[1] != window.context.key_abs_positions.numel():
        raise ValueError("key positions disagree with the tensor length")
    if q.shape[1] != window.context.query_abs_positions.numel():
        raise ValueError("query positions disagree with the tensor length")
    for x in (q, k, v):
        if not x.is_floating_point() or x.dtype != k.dtype or x.device != k.device:
            raise ValueError("Q/K/V require matching floating dtype/device")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Q/K/V must be finite")


def _indices(indices, count: int, *, name: str = "indices", unique: bool = True) -> list[int]:
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 1 or indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"{name} must be a one-dimensional integer sequence")
        values = indices.detach().cpu().tolist()
    else:
        try:
            values = list(indices)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer sequence") from exc
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
        raise ValueError(f"{name} must contain integers")
    result = [int(i) for i in values]
    if any(i < 0 or i >= count for i in result):
        raise ValueError(f"{name} contains an out-of-range index")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    return result


def _eligible(window: FrozenAttention, indices: list[int]) -> None:
    mask = window.context.eligible_mask.detach().cpu()
    if any(not bool(mask[i]) for i in indices):
        raise ValueError("targets must be unprotected historical tokens")


def _coordinates(window: FrozenAttention) -> torch.Tensor:
    pos = window.context.key_abs_positions.to(device=window.k_post.device)
    h, w = window.context.grid_h, window.context.grid_w
    return torch.stack((pos // (h * w), (pos // w) % h, pos % w), dim=1).float()


def _require_pre(window: FrozenAttention, indices: list[int]) -> None:
    k = window.k_pre
    if k is None or not callable(window.rotate_pre):
        raise ValueError("this intervention requires pre-RoPE K and rotate_pre")
    if k.shape != window.k_post.shape or k.dtype != window.k_post.dtype or k.device != window.k_post.device:
        raise ValueError("pre-RoPE K must match post-RoPE K shape/dtype/device")
    if not bool(torch.isfinite(k[:, indices]).all()):
        raise ValueError("selected pre-RoPE keys must be finite")
    valid = getattr(window, "pre_valid_mask", None)
    if valid is not None:
        if valid.ndim != 1 or valid.dtype != torch.bool or valid.numel() != k.shape[1]:
            raise ValueError("pre_valid_mask requires one boolean per key")
        if not bool(valid.detach().cpu()[indices].all()):
            raise ValueError("selected pre-RoPE keys are unavailable")


def _rotate(window: FrozenAttention, pre: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    result = window.rotate_pre(pre, coordinates)
    if (not isinstance(result, torch.Tensor) or result.shape != pre.shape
            or result.dtype != pre.dtype or result.device != pre.device
            or not bool(torch.isfinite(result).all())):
        raise ValueError("rotate_pre must return finite K with unchanged shape/dtype/device")
    return result


@torch.no_grad()
def identity(window: FrozenAttention) -> KVVariant:
    """Preserve the complete window, including the exact stored post-RoPE K."""
    _validate(window)
    n = window.k_post.shape[1]
    return KVVariant(window.k_post, window.v, _coordinates(window),
                     torch.zeros(n, dtype=torch.float32, device=window.k_post.device),
                     tuple((i,) for i in range(n)), {"intervention": "identity", "approximation": "exact"})


@torch.no_grad()
def drop(window: FrozenAttention, indices: Sequence[int]) -> KVVariant:
    """Remove selected historical K/V pairs together."""
    result = identity(window)
    selected = _indices(indices, result.token_count)
    _eligible(window, selected)
    if not selected:
        return result
    if len(selected) == result.token_count:
        raise ValueError("cannot drop every key")
    removed = set(selected)
    keep = [i for i in range(result.token_count) if i not in removed]
    return KVVariant(result.k_post[:, keep], result.v[:, keep], result.coordinates[keep],
                     result.log_count[keep], tuple(result.provenance[i] for i in keep),
                     {"intervention": "drop", "approximation": "approximate", "indices": selected})


@torch.no_grad()
def replace(window: FrozenAttention, target_indices: Sequence[int], donor_indices: Sequence[int]) -> KVVariant:
    """Copy historical donor pre-K/V, rotating K at each target position."""
    result = identity(window)
    targets = _indices(target_indices, result.token_count, name="target_indices")
    donors = _indices(donor_indices, result.token_count, name="donor_indices", unique=False)
    if len(targets) != len(donors):
        raise ValueError("targets and donors must have equal lengths")
    _eligible(window, targets)
    historical = window.context.key_abs_positions.detach().cpu() < window.context.current_chunk_start
    if any(not bool(historical[i]) for i in donors):
        raise ValueError("donors must be historical; current noisy keys are forbidden")
    if not targets:
        return result
    _require_pre(window, donors)
    k, v = result.k_post.clone(), result.v.clone()
    k[:, targets] = _rotate(window, window.k_pre[:, donors].clone(), result.coordinates[targets])
    v[:, targets] = window.v[:, donors]
    provenance = list(result.provenance)
    for target, donor in zip(targets, donors):
        provenance[target] = (donor,)
    return KVVariant(k, v, result.coordinates, result.log_count, tuple(provenance),
                     {"intervention": "replace", "approximation": "approximate",
                      "target_indices": targets, "donor_indices": donors})


def _group_weights(groups: list[list[int]], weights) -> list[torch.Tensor]:
    if weights is None:
        return [torch.ones(len(group), dtype=torch.float64) for group in groups]
    try:
        rows = list(weights)
    except TypeError as exc:
        raise ValueError("weights must contain a weight sequence per group") from exc
    if len(groups) == 1 and (not rows or isinstance(rows[0], (int, float))
                             or isinstance(rows[0], torch.Tensor) and rows[0].ndim == 0):
        rows = [rows]
    if len(rows) != len(groups):
        raise ValueError("weights must contain one sequence per group")
    result = []
    for group, row in zip(groups, rows):
        try:
            values = torch.as_tensor(row, dtype=torch.float64, device="cpu")
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid group weights") from exc
        if values.ndim != 1 or values.numel() != len(group) or not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
            raise ValueError("group weights must be finite, positive, and match the group length")
        result.append(values)
    return result


@torch.no_grad()
def merge(window: FrozenAttention, groups: Sequence[Sequence[int]], *, multiplicity: bool = True,
          weights=None) -> KVVariant:
    """Average pre-K/V and absolute coordinates, then apply RoPE once.

    Weights affect pre-K, V, and representative coordinates identically. The
    optional log-count is log(group size), independent of averaging weights.
    Nonidentical keys generally remain an approximation with either bias choice.
    """
    result = identity(window)
    if not isinstance(multiplicity, bool):
        raise ValueError("multiplicity must be boolean")
    group_list = [_indices(group, result.token_count, name="group") for group in groups]
    if any(not group for group in group_list):
        raise ValueError("merge groups cannot be empty")
    flat = [i for group in group_list for i in group]
    if len(set(flat)) != len(flat):
        raise ValueError("merge groups must be disjoint")
    _eligible(window, flat)
    group_weights = _group_weights(group_list, weights)
    if not group_list:
        return result
    _require_pre(window, flat)
    # Compute every representative before modifying or selecting output tensors.
    representatives = {}
    duplicate_exact = bool(multiplicity)
    for group, weight in zip(group_list, group_weights):
        work_dtype = torch.float64 if window.k_pre.dtype == torch.float64 else torch.float32
        scaled_weight = weight / weight.max()
        alpha = (scaled_weight / scaled_weight.sum()).to(device=result.k_post.device, dtype=work_dtype)
        tensor_alpha = alpha.view(1, -1, 1, 1)
        pre_group, value_group = window.k_pre[:, group].to(work_dtype), window.v[:, group].to(work_dtype)
        # Centering preserves exact duplicate keys/values, including groups of
        # three whose FP32 uniform weights cannot sum to an exact rational mean.
        pre = (pre_group[:, :1] + ((pre_group - pre_group[:, :1]) * tensor_alpha).sum(1, keepdim=True)).to(window.k_pre.dtype)
        value = (value_group[:, :1] + ((value_group - value_group[:, :1]) * tensor_alpha).sum(1, keepdim=True)).to(window.v.dtype)
        coordinate = (result.coordinates[group] * alpha.float()[:, None]).sum(0, keepdim=True)
        key = _rotate(window, pre, coordinate)
        if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(coordinate).all()):
            raise ValueError("merged representative must be finite")
        duplicate_exact = duplicate_exact and all(
            torch.equal(key[:, 0], window.k_post[:, i]) and torch.equal(value[:, 0], window.v[:, i]) for i in group)
        representatives[min(group)] = (key, value, coordinate, math.log(len(group)) if multiplicity else 0., tuple(group))
    omitted = set(flat) - set(representatives)
    keys, values, coordinates, counts, provenance = [], [], [], [], []
    for i in range(result.token_count):
        if i in omitted:
            continue
        if i in representatives:
            key, value, coordinate, count, source = representatives[i]
        else:
            key, value, coordinate, count, source = result.k_post[:, i:i+1], result.v[:, i:i+1], result.coordinates[i:i+1], 0., (i,)
        keys.append(key); values.append(value); coordinates.append(coordinate); counts.append(count); provenance.append(source)
    return KVVariant(torch.cat(keys, 1), torch.cat(values, 1), torch.cat(coordinates, 0),
                     torch.tensor(counts, dtype=torch.float32, device=result.k_post.device), tuple(provenance),
                     {"intervention": "merge", "approximation": "exact_duplicate_key_value" if duplicate_exact else "approximate",
                      "groups": group_list, "weights": [w.tolist() for w in group_weights], "multiplicity": multiplicity,
                      "representative_counts": [len(source) for source in provenance]})


def _noise_direction(window, source, indices, kind, generator, rank):
    b, n, h, d = source.shape
    if kind == "gaussian":
        noise = torch.randn((b, n, h, d), generator=generator, dtype=torch.float32)
    elif kind == "low_rank":
        left = torch.randn((b, n, rank), generator=generator, dtype=torch.float32)
        right = torch.randn((b, rank, h * d), generator=generator, dtype=torch.float32)
        noise = torch.bmm(left, right).reshape(b, n, h, d)
    elif kind == "temporal_noise":
        pattern = torch.randn((b, window.context.frame_tokens, h, d), generator=generator, dtype=torch.float32)
        spatial = window.context.key_abs_positions.detach().cpu()[indices] % window.context.frame_tokens
        noise = pattern[:, spatial]
    else:  # Synthetic error direction only; no quantizer or production codec.
        noise = torch.rand((b, n, h, d), generator=generator, dtype=torch.float32) - .5
    return noise.to(device=source.device, dtype=torch.float64 if source.dtype == torch.float64 else torch.float32)


def _perturbed_role(window, source, indices, kind, target_nmse, generator, rank):
    selected = source[:, indices]
    energy = selected.double().square().sum()
    if kind == "zero":
        changed = torch.zeros_like(selected)
        nmse = float((changed.double() - selected.double()).square().sum() / energy) if float(energy) > 0 else None
        return changed, nmse, "zero_ablation" if nmse is not None else "zero_energy", None if nmse is not None else "relative NMSE is undefined for zero-energy input"
    if float(energy) == 0:
        return selected.clone(), None, "zero_energy_unattainable", "nonzero relative NMSE cannot be normalized against zero input energy"
    direction = _noise_direction(window, selected, indices, kind, generator, rank)
    direction_energy = direction.double().square().sum()
    if float(direction_energy) == 0:
        return selected.clone(), 0., "zero_direction_unattainable", "sampled noise direction has zero energy"
    scale = math.sqrt(target_nmse * float(energy) / float(direction_energy))
    changed = (selected.to(direction.dtype) + direction * scale).to(source.dtype)
    if not bool(torch.isfinite(changed).all()):
        raise ValueError("perturbation overflowed the input dtype")
    actual = float((changed.double() - selected.double()).square().sum() / energy)
    return changed, actual, "measured", None


@torch.no_grad()
def perturb(window: FrozenAttention, indices: Sequence[int], *, kind: str = "gaussian", role: str = "kv",
            target_nmse: float, seed: int, rank: int = 1) -> KVVariant:
    """Apply a seeded direction, independently normalized for selected K and V.

    K noise is applied to stored post-RoPE K at unchanged coordinates. Reported
    NMSE measures the returned dtype-rounded tensors, not the requested strength.
    A zero ablation deliberately reports its actual error without normalization.
    """
    result = identity(window)
    selected = _indices(indices, result.token_count)
    _eligible(window, selected)
    if kind not in {"gaussian", "low_rank", "temporal_noise", "quantization_like", "zero"}:
        raise ValueError("unknown perturbation kind")
    if role not in {"k", "v", "kv"}:
        raise ValueError("role must be k, v, or kv")
    if isinstance(target_nmse, bool) or not math.isfinite(float(target_nmse)) or target_nmse < 0:
        raise ValueError("target_nmse must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("rank must be a positive integer")
    details = {"intervention": "perturb", "approximation": "approximate", "kind": kind, "role": role,
               "target_nmse": float(target_nmse), "indices": selected, "seed": int(seed), "rank": int(rank),
               "nmse_k": 0., "nmse_v": 0., "status_k": "not_targeted", "status_v": "not_targeted"}
    if kind == "quantization_like":
        details["noise_model"] = "synthetic uniform quantization-error direction; not a production codec"
    if not selected or target_nmse == 0:
        for name in ("k", "v"):
            if name in role:
                details[f"status_{name}"] = "no_targets" if not selected else "exact_null"
        details["approximation"] = "exact"
        return KVVariant(result.k_post, result.v, result.coordinates, result.log_count, result.provenance, details)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    output = {"k": result.k_post, "v": result.v}
    for name in ("k", "v"):
        if name not in role:
            continue
        source = output[name]
        changed, actual, status, reason = _perturbed_role(window, source, selected, kind, float(target_nmse), generator, int(rank))
        output[name] = source.clone()
        output[name][:, selected] = changed
        details[f"nmse_{name}"], details[f"status_{name}"] = actual, status
        if reason is not None:
            details[f"reason_{name}"] = reason
    return KVVariant(output["k"], output["v"], result.coordinates, result.log_count, result.provenance, details)
