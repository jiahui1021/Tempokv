"""Shared efficiency/horizon record contract for the Self-Forcing experiments.

CANONICAL COPY: Tempokv/experiments/Self-Forcing/tempokv_adapter/efficiency_record.py

This file is vendored byte-identically into:
  - qvg/Quant-VideoGen/experiments/Self-Forcing/efficiency_record.py
  - kv-quant-4-videogen/Self-Forcing/kv_quant/efficiency_record.py

The three repositories run in separate conda environments and cannot import
each other, so the contract is duplicated rather than shared.  Any edit must be
applied to all three copies; ``tests/test_efficiency_record_vendoring.py``
fails if they drift.

``resident_analytic.v1`` -- the byte definition every copy implements:

    Resident KV bytes at a sampling point = summed over all 30 layers, over K
    and V:

    (a) every tensor actually held in the packed state of each resident
        quantized span -- payload, scales, zero points, codebooks, indices, and
        any BF16 residual buffer -- counted as ``numel x element_size``; plus

    (b) every dense chunk still allocated and not covered by a packed span,
        counted the same way.

    It excludes predictor parameters (shared across spans, reported separately)
    and cross-attention caches (identical in every method, never quantized).

    ``bf16_equivalent_bytes`` at the same point, over the same spans and
    chunks, is ``batch x tokens x heads x head_dim x 2 bytes``, counted once
    for K and once for V.

Bytes must come from real tensors.  A figure derived from a configuration
formula is not a substitute and must never be reported through this module.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import torch


SCHEMA_VERSION = "efficiency.v1"
HORIZON_SCHEMA_VERSION = "horizon.v1"
RESIDENT_DEFINITION = "resident_analytic.v1"

BF16_BYTES_PER_VALUE = 2

# Every reported horizon lands on this grid.  Self-Forcing's VAE maps latent
# length L to 4L-3 pixel frames and the model finalizes three latent frames per
# block, so only multiples of twelve (offset by three) correspond to a whole
# number of completed blocks.
FRAME_GRID = 12


def tensor_bytes(obj: Any, *, skip_fields: frozenset[str] = frozenset()) -> int:
    """Bytes of every tensor reachable from ``obj``, counting storage once.

    Walks dicts, lists, tuples, sets, and dataclasses.  ``skip_fields`` names
    dict keys and dataclass fields to leave out -- used to exclude predictor
    parameters, which are shared across spans and reported separately.

    Tensors that share storage (a view and its base, or the same tensor stored
    under two keys) are counted once: the figure is device bytes held, not
    references taken.
    """
    seen: set[int] = set()

    def walk(node: Any) -> int:
        if isinstance(node, torch.Tensor):
            pointer = node.data_ptr()
            if pointer in seen:
                return 0
            seen.add(pointer)
            return node.numel() * node.element_size()
        if isinstance(node, dict):
            return sum(
                walk(value)
                for key, value in node.items()
                if key not in skip_fields
            )
        if isinstance(node, (list, tuple, set)):
            return sum(walk(item) for item in node)
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            return sum(
                walk(getattr(node, field.name))
                for field in dataclasses.fields(node)
                if field.name not in skip_fields
            )
        return 0

    return walk(obj)


def bf16_equivalent_bytes(
    batch: int, tokens: int, heads: int, head_dim: int
) -> int:
    """BF16 cost of ``tokens`` cache positions, counting K and V separately."""
    if min(batch, tokens, heads, head_dim) < 0:
        raise ValueError("cache dimensions must not be negative")
    return batch * tokens * heads * head_dim * BF16_BYTES_PER_VALUE * 2


class ResidentSampler:
    """Peak and final resident bytes over a run's sampling points.

    Sampling points are block boundaries, not pack calls: the last incomplete
    span never packs, so a pack-triggered sampler never sees the state in which
    every token is resident.
    """

    def __init__(self) -> None:
        self._peak_resident = 0
        self._peak_bf16 = 0
        self._final_resident = 0
        self._observations: list[tuple[int, int]] = []

    def observe(self, resident_bytes: int, bf16_equivalent_bytes: int) -> None:
        if resident_bytes < 0 or bf16_equivalent_bytes < 0:
            raise ValueError("resident byte counts must not be negative")
        resident_bytes = int(resident_bytes)
        equivalent = int(bf16_equivalent_bytes)
        self._peak_resident = max(self._peak_resident, resident_bytes)
        self._peak_bf16 = max(self._peak_bf16, equivalent)
        self._final_resident = resident_bytes
        self._observations.append((resident_bytes, equivalent))

    @property
    def samples(self) -> list[tuple[int, int]]:
        """Every observed ``(resident, bf16_equivalent)`` pair, in order.

        The runner replays these into a fresh sampler when it assembles the
        record, so the per-block detail survives the trip through the run
        summary instead of being reduced to a peak too early.
        """
        return list(self._observations)

    @property
    def peak_resident_bytes(self) -> int:
        return self._peak_resident

    @property
    def peak_bf16_equivalent_bytes(self) -> int:
        return self._peak_bf16

    @property
    def final_resident_bytes(self) -> int:
        return self._final_resident

    @property
    def num_samples(self) -> int:
        return len(self._observations)


def largest_grid_horizon(completed_frames: int) -> int:
    """Largest horizon at or below ``completed_frames`` on the frame grid.

    A run that died mid-block completed a frame count that is usually off the
    ``(frames + 3) % 12 == 0`` grid.  Rounding down is the honest answer: the
    partially generated block was never finished, so it is not a horizon the
    method reached.
    """
    if completed_frames < 9:
        return 0
    return completed_frames - ((completed_frames + 3) % FRAME_GRID)


def write_record(path: str | Path, payload: dict) -> None:
    """Write ``payload`` as JSON, replacing ``path`` atomically."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _measured(
    per_prompt_seconds: Sequence[float], prompt_indices: Sequence[int]
) -> tuple[list[int], float | None]:
    """Drop the warm-up prompt and take the median of what is left.

    Prompt 0 absorbs Triton compilation and allocator warm-up, so it is never
    part of a reported figure.  ``measured_prompt_indices`` is derived, never
    hard-coded, so a manifest that runs a different prompt range stays honest.
    """
    measured_indices = list(prompt_indices[1:])
    measured_seconds = [
        float(value) for value in per_prompt_seconds[1 : len(prompt_indices)]
    ]
    return measured_indices, (median(measured_seconds) if measured_seconds else None)


def build_efficiency_payload(
    *,
    repo: str,
    method: str,
    bits: int | None,
    run: dict,
    sampler: ResidentSampler,
    cache_shape: dict,
    fake_quant: bool,
    predictor_parameter_bytes: int,
    gpu_memory: dict,
    per_prompt_generation_seconds: Sequence[float],
    prompt_indices: Sequence[int],
    segments_ms: dict,
    instrumentation: dict,
) -> dict:
    """Assemble one ``efficiency.v1`` record.

    ``profile_available`` is true only when every one of the four measured
    segments carries a number.  ``denoising_other`` is a residual the collector
    computes, so a single missing segment makes the whole decomposition
    meaningless -- saying so here keeps the collector from inventing one.
    """
    measured_indices, median_seconds = _measured(
        per_prompt_generation_seconds, prompt_indices
    )
    segments = {
        name: (None if segments_ms.get(name) is None else float(segments_ms[name]))
        for name in ("e2e", "cache_encode", "cache_decode", "vae_decode")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "method": method,
        "bits": bits,
        "run": dict(run),
        "kv_cache": {
            "definition": RESIDENT_DEFINITION,
            "peak_resident_bytes": sampler.peak_resident_bytes,
            "peak_bf16_equivalent_bytes": sampler.peak_bf16_equivalent_bytes,
            "final_resident_bytes": sampler.final_resident_bytes,
            "num_samples": sampler.num_samples,
            "fake_quant": bool(fake_quant),
            **{key: int(value) for key, value in cache_shape.items()},
        },
        "predictor_parameter_bytes": int(predictor_parameter_bytes),
        "gpu_memory": {
            key: (value if isinstance(value, bool) else int(value))
            for key, value in gpu_memory.items()
        },
        "timing": {
            "per_prompt_generation_seconds": [
                float(value) for value in per_prompt_generation_seconds
            ],
            "measured_prompt_indices": measured_indices,
            "generation_seconds_median": median_seconds,
            "profile_available": all(value is not None for value in segments.values()),
            "segments_ms": segments,
            "instrumentation": dict(instrumentation),
        },
    }


def build_horizon_payload(
    *,
    repo: str,
    method: str,
    bits: int | None,
    run: dict,
    budget: dict,
    probe_mode: str,
    max_feasible_frames: int,
    oom_reached: bool,
    peak_allocated_bytes_at_max: int,
    kv_resident_bytes_at_max: int,
    bisect_probes: Iterable[dict],
) -> dict:
    """Assemble one ``horizon.v1`` record.

    ``oom_reached`` false means the probe horizon was reached without running
    out of memory, so ``max_feasible_frames`` is a lower bound, not the answer.
    """
    if probe_mode not in {"run_to_oom", "bisect_on_init"}:
        raise ValueError(f"unknown probe mode: {probe_mode!r}")
    if max_feasible_frames < 0 or (
        max_feasible_frames and (max_feasible_frames + 3) % FRAME_GRID
    ):
        raise ValueError(
            "max_feasible_frames must satisfy (frames + 3) % 12 == 0, got "
            f"{max_feasible_frames}"
        )
    return {
        "schema_version": HORIZON_SCHEMA_VERSION,
        "repo": repo,
        "method": method,
        "bits": bits,
        "run": dict(run),
        "budget": dict(budget),
        "probe_mode": probe_mode,
        "max_feasible_frames": int(max_feasible_frames),
        "max_feasible_latent_frames": (int(max_feasible_frames) + 3) // 4,
        "oom_reached": bool(oom_reached),
        "peak_allocated_bytes_at_max": int(peak_allocated_bytes_at_max),
        "kv_resident_bytes_at_max": int(kv_resident_bytes_at_max),
        "bisect_probes": [dict(probe) for probe in bisect_probes],
    }


__all__ = [
    "BF16_BYTES_PER_VALUE",
    "FRAME_GRID",
    "HORIZON_SCHEMA_VERSION",
    "RESIDENT_DEFINITION",
    "SCHEMA_VERSION",
    "ResidentSampler",
    "bf16_equivalent_bytes",
    "build_efficiency_payload",
    "build_horizon_payload",
    "largest_grid_horizon",
    "tensor_bytes",
    "write_record",
]
