"""Pure frame-selection policies for independent Spectral KV experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral, Real
from typing import Callable, Mapping, Protocol


class Mode(IntEnum):
    FULL = 0
    SUMMARY = 1
    DROP = 2


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _grid(value, name="grid") -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in value
    ):
        raise ValueError(f"{name} must contain positive integer height and width")
    return tuple(int(v) for v in value)


@dataclass(frozen=True)
class FrameMeta:
    frame_idx: int
    grid: tuple[int, int]
    original_grid: tuple[int, int]
    is_current: bool = False
    is_sink: bool = False

    def __post_init__(self):
        object.__setattr__(self, "frame_idx", _nonnegative(self.frame_idx, "frame_idx"))
        object.__setattr__(self, "grid", _grid(self.grid))
        object.__setattr__(self, "original_grid", _grid(self.original_grid, "original_grid"))
        if any(a > b for a, b in zip(self.grid, self.original_grid)):
            raise ValueError("grid cannot exceed original_grid")
        if not isinstance(self.is_current, bool) or not isinstance(self.is_sink, bool):
            raise ValueError("is_current and is_sink must be boolean")


@dataclass(frozen=True)
class HistoryPlan:
    entries: Mapping[int, tuple[Mode, tuple[int, int] | None]]


class HistoryPolicy(Protocol):
    def plan(self, layer: int, chunk_idx: int, meta: list[FrameMeta], q_probe=None) -> HistoryPlan: ...


def target_grid(hw: tuple[int, int], ratio: int | float) -> tuple[int, int]:
    """Per-axis rounded grid for a requested token-reduction ratio."""
    hw = _grid(hw)
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio) or ratio < 1:
        raise ValueError("ratio must be finite and at least one")
    return tuple(max(1, min(axis, round(axis / math.sqrt(ratio)))) for axis in hw)


def _metadata(meta) -> list[FrameMeta]:
    meta = list(meta)
    if any(not isinstance(frame, FrameMeta) for frame in meta):
        raise ValueError("meta must contain FrameMeta entries")
    if len({frame.frame_idx for frame in meta}) != len(meta):
        raise ValueError("meta must not contain duplicate frame indices")
    return meta


def _protected(meta: list[FrameMeta], recent: int, sink: int) -> set[int]:
    current = [frame.frame_idx for frame in meta if frame.is_current]
    current_start = min(current) if current else max((frame.frame_idx for frame in meta), default=-1) + 1
    return {
        frame.frame_idx for frame in meta
        if frame.is_current or frame.is_sink or frame.frame_idx < sink or frame.frame_idx >= current_start - recent
    }


def _summary(frame: FrameMeta, desired: tuple[int, int]) -> tuple[Mode, tuple[int, int] | None]:
    # Express the policy's target in original coordinates.  The cache controls
    # whether that target can be restored from retained originals or is bounded
    # by irreversible earlier compression.
    return (Mode.FULL, None) if desired == frame.original_grid else (Mode.SUMMARY, desired)


class FullPolicy:
    def plan(self, layer, chunk_idx, meta, q_probe=None) -> HistoryPlan:
        return HistoryPlan({frame.frame_idx: (Mode.FULL, None) for frame in _metadata(meta)})


class WindowSinkPolicy:
    """Keep the newest ``window`` historical frames, current chunk and sinks."""

    def __init__(self, window: int, sink: int = 1):
        self.window = _nonnegative(window, "window")
        self.sink = _nonnegative(sink, "sink")

    def plan(self, layer, chunk_idx, meta, q_probe=None) -> HistoryPlan:
        meta = _metadata(meta)
        current = [frame.frame_idx for frame in meta if frame.is_current]
        current_start = min(current) if current else max((frame.frame_idx for frame in meta), default=-1) + 1
        newest = sorted((frame.frame_idx for frame in meta if frame.frame_idx < current_start), reverse=True)[:self.window]
        keep = set(newest) | _protected(meta, 0, self.sink)
        return HistoryPlan({frame.frame_idx: (Mode.FULL if frame.frame_idx in keep else Mode.DROP, None) for frame in meta})


class OracleTopKPolicy:
    """Select up to k unprotected frames by externally measured frame scores.

    Scores are one shared value per frame, independent of heads.  The current
    frames, explicit sinks, frame indices below ``sink``, and frames with
    the ``recent`` historical frame positions before the current chunk are
    retained in addition to that k.  Without a current chunk, the newest
    ``recent`` positions are protected.  Thus changing current chunk length does
    not change the protected history length.
    """

    def __init__(self, k: int, others: str = "summary", ratio: int = 4, recent: int = 6, sink: int = 1):
        self.k = _nonnegative(k, "k")
        if others not in ("summary", "drop"):
            raise ValueError("others must be 'summary' or 'drop'")
        target_grid((1, 1), ratio)
        self.others, self.ratio = others, ratio
        self.recent, self.sink = _nonnegative(recent, "recent"), _nonnegative(sink, "sink")

    def plan(self, layer, chunk_idx, meta, q_probe=None) -> HistoryPlan:
        meta = _metadata(meta)
        protected = _protected(meta, self.recent, self.sink)
        candidates = [frame for frame in meta if frame.frame_idx not in protected]
        if candidates and self.k:
            if not isinstance(q_probe, Mapping):
                raise ValueError("oracle q_probe must map candidate frame indices to finite scores")
            scores = {}
            for frame in candidates:
                score = q_probe.get(frame.frame_idx)
                if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                    raise ValueError("oracle q_probe requires one finite real score per candidate frame")
                scores[frame.frame_idx] = float(score)
            # Equal scores prefer the newer frame; input ordering has no effect.
            ranked = sorted(candidates, key=lambda frame: (scores[frame.frame_idx], frame.frame_idx), reverse=True)
            protected |= {frame.frame_idx for frame in ranked[:self.k]}
        entries = {}
        for frame in meta:
            if frame.frame_idx in protected:
                entries[frame.frame_idx] = (Mode.FULL, None)
            elif self.others == "drop":
                entries[frame.frame_idx] = (Mode.DROP, None)
            else:
                entries[frame.frame_idx] = _summary(frame, target_grid(frame.original_grid, self.ratio))
        return HistoryPlan(entries)


class DistanceSpectralPolicy:
    """Retain each axis at ceil(size * clip(r0 * distance**-b, r_min, 1))."""

    def __init__(self, b: float, r0: float = 1, r_min: float = 0.125, recent: int = 6, sink: int = 1, kind: str = "dct"):
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in (b, r0, r_min)):
            raise ValueError("b, r0 and r_min must be finite real values")
        if b < 0 or r0 <= 0 or not 0 < r_min <= 1:
            raise ValueError("b must be nonnegative, r0 positive and r_min in (0,1]")
        if kind not in ("dct", "avg"):
            raise ValueError("kind must be 'dct' or 'avg'")
        self.b, self.r0, self.r_min, self.kind = float(b), float(r0), float(r_min), kind
        self.recent, self.sink = _nonnegative(recent, "recent"), _nonnegative(sink, "sink")

    def plan(self, layer, chunk_idx, meta, q_probe=None) -> HistoryPlan:
        meta = _metadata(meta)
        protected = _protected(meta, self.recent, self.sink)
        current = [frame.frame_idx for frame in meta if frame.is_current]
        current_start = min(current) if current else max((frame.frame_idx for frame in meta), default=-1) + 1
        entries = {}
        for frame in meta:
            if frame.frame_idx in protected:
                entries[frame.frame_idx] = (Mode.FULL, None)
                continue
            distance = max(1, current_start - frame.frame_idx)
            fraction = min(1.0, max(self.r_min, self.r0 * distance ** -self.b))
            desired = tuple(max(1, min(axis, math.ceil(axis * fraction))) for axis in frame.original_grid)
            entries[frame.frame_idx] = _summary(frame, desired)
        return HistoryPlan(entries)


class ExternalSelectorPolicy:
    """Validate an external selector; current and explicit sink frames survive."""

    def __init__(self, fn: Callable):
        if not callable(fn):
            raise ValueError("fn must be callable")
        self.fn = fn

    def plan(self, layer, chunk_idx, meta, q_probe=None) -> HistoryPlan:
        meta = _metadata(meta)
        result = self.fn(layer, chunk_idx, meta, q_probe)
        entries = result.entries if isinstance(result, HistoryPlan) else result
        if not isinstance(entries, Mapping) or set(entries) != {frame.frame_idx for frame in meta}:
            raise ValueError("external selector must return one entry for every frame")
        normalized = {}
        for frame in meta:
            entry = entries[frame.frame_idx]
            if not isinstance(entry, (tuple, list)) or len(entry) != 2 or isinstance(entry[0], bool):
                raise ValueError("external entries must be (Mode, target_grid_or_None)")
            try:
                mode = Mode(entry[0])
            except (TypeError, ValueError) as error:
                raise ValueError("external selector returned an invalid mode") from error
            grid = entry[1]
            if mode == Mode.SUMMARY:
                grid = _grid(grid, "summary target grid")
                if any(a > b for a, b in zip(grid, frame.original_grid)):
                    raise ValueError("external summary target cannot exceed the original grid")
            elif grid is not None:
                raise ValueError("FULL and DROP entries must not specify a target grid")
            if frame.is_current or frame.is_sink:
                if mode != Mode.FULL:
                    raise ValueError("external selector cannot remove or summarize protected frames")
            normalized[frame.frame_idx] = (mode, grid)
        return HistoryPlan(normalized)
