"""Bounded paired replay with frozen targets and complete RNG auditing.

The callback owns a fresh replay of its model, noise and cache for every call.
This engine never feeds reference latents or reference caches into a branch.
Only selected CPU step snapshots and one protected CPU reference trajectory are
retained; branch traces are released after their rows have been consumed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import math
import random
from typing import Any, Callable, Iterable

import numpy as np
import torch


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class FrozenEvent:
    layer: int
    head: int
    chunk: int
    step: int
    absolute_tokens: tuple[int, ...]
    query_scope: str = "full_current_chunk"

    def __post_init__(self):
        for name in ("layer", "head", "chunk", "step"):
            _nonnegative(getattr(self, name), name)
        if not isinstance(self.absolute_tokens, tuple):
            raise ValueError("absolute_tokens must be a frozen tuple")
        for token in self.absolute_tokens:
            _nonnegative(token, "absolute token")
        if len(set(self.absolute_tokens)) != len(self.absolute_tokens):
            raise ValueError("absolute_tokens must be unique")
        if not isinstance(self.query_scope, str) or not self.query_scope:
            raise ValueError("query_scope must be explicit")


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    events: tuple[FrozenEvent, ...]
    kind: str = "drop"
    role: str = "kv"
    strength: float = 0.
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a nonempty string")
        if not isinstance(self.events, tuple) or not all(isinstance(event, FrozenEvent) for event in self.events):
            raise ValueError("events must be a tuple of FrozenEvent targets")
        if len(set(self.events)) != len(self.events):
            raise ValueError("a replay case cannot repeat a target event")
        if self.role not in {"k", "v", "kv"}:
            raise ValueError("role must be k, v or kv")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")


@dataclass
class RolloutOutput:
    latents: torch.Tensor
    payload: Any = None


@dataclass(frozen=True)
class RNGState:
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]
    python: tuple
    numpy: tuple
    cuda_initialized: bool


def capture_rng_state() -> RNGState:
    """Capture global RNG states in memory without initializing CUDA."""
    numpy = np.random.get_state()
    initialized = torch.cuda.is_initialized()
    return RNGState(torch.get_rng_state().clone(),
                    tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if initialized else (),
                    random.getstate(), (numpy[0], numpy[1].copy(), *numpy[2:]), initialized)


def restore_rng_state(state: RNGState) -> None:
    """Restore a captured state; no serialized state or pickle is involved."""
    if not isinstance(state, RNGState):
        raise TypeError("state must be an in-memory RNGState")
    torch.set_rng_state(state.torch_cpu)
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    if state.cuda_initialized:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))


def _fingerprint(state: RNGState) -> str:
    digest = hashlib.sha256()
    def add(label, value):
        digest.update(label.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes())
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")
    add("torch_cpu", state.torch_cpu)
    add("cuda_initialized", state.cuda_initialized)
    for index, value in enumerate(state.torch_cuda):
        add(f"torch_cuda_{index}", value)
    add("python", state.python)
    for index, value in enumerate(state.numpy):
        add(f"numpy_{index}", value)
    return digest.hexdigest()


def rng_fingerprint() -> str:
    return _fingerprint(capture_rng_state())


class ReplayRNGError(RuntimeError):
    """The branch consumed a different random stream from the reference."""


class ReplayTargetError(RuntimeError):
    """A frozen intervention did not fire exactly once at its declared event."""


class ReplayMemoryError(MemoryError):
    """The requested snapshots and active trajectories exceed the capture cap."""


class _Budget:
    def __init__(self, max_capture_mb):
        if not math.isfinite(max_capture_mb) or max_capture_mb <= 0:
            raise ValueError("max_capture_mb must be finite and positive")
        self.limit = int(max_capture_mb * 1024 ** 2)
        self.used = 0
        self.peak = 0

    def check(self, extra=0):
        if self.used + extra > self.limit:
            raise ReplayMemoryError(f"capture requires {self.used + extra} bytes; cap is {self.limit} bytes")
        self.peak = max(self.peak, self.used + extra)

    def reserve(self, amount):
        self.check(amount)
        self.used += amount

    def release(self, amount):
        self.used -= amount


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


class TraceRecorder:
    """Fingerprint telemetry plus bounded, explicitly selectable CPU outputs."""
    def __init__(self, max_capture_mb=256, record_steps: set[tuple[int, int]] | None = None, *, _budget=None):
        self._budget = _Budget(max_capture_mb) if _budget is None else _budget
        self.record_steps = None if record_steps is None else frozenset(record_steps)
        if self.record_steps is not None:
            for key in self.record_steps:
                self._step_key(*key)
        self.rng_trace: list[dict] = []
        self.outputs: dict[tuple[int, int], torch.Tensor] = {}
        self.chunks: dict[int, tuple[int, int]] = {}
        self.injections: Counter[FrozenEvent] = Counter()
        self.capture_bytes = 0

    @staticmethod
    def _step_key(chunk, step):
        _nonnegative(chunk, "chunk"); _nonnegative(step, "step")
        return chunk, step

    def record_rng(self, stage, event_key):
        if not isinstance(stage, str) or not stage:
            raise ValueError("RNG stage must be a nonempty string")
        # Keys are scalar metadata; keeping live tensors here could retain caches.
        def freeze(value):
            if isinstance(value, FrozenEvent): return freeze(asdict(value))
            if isinstance(value, dict): return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
            if isinstance(value, (list, tuple)): return tuple(freeze(item) for item in value)
            if value is None or isinstance(value, (str, int, float, bool)): return value
            raise TypeError("RNG event keys must contain scalar metadata, not live tensors")
        self.rng_trace.append({"stage": stage, "event_key": freeze(event_key), "fingerprint": rng_fingerprint()})

    @torch.no_grad()
    def record_output(self, chunk, step, tensor):
        key = self._step_key(chunk, step)
        if self.record_steps is not None and key not in self.record_steps:
            return False
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("step output must be a tensor")
        if key in self.outputs:
            raise ValueError(f"step output {key} was recorded twice")
        amount = _bytes(tensor)
        self._budget.reserve(amount)
        try:
            self.outputs[key] = tensor.detach().to(device="cpu", copy=True)
        except BaseException:
            self._budget.release(amount)
            raise
        self.capture_bytes += amount
        return True

    def record_chunk(self, chunk, frame0, frames):
        _nonnegative(chunk, "chunk"); _nonnegative(frame0, "frame0")
        if type(frames) is not int or frames < 1:
            raise ValueError("frames must be a positive integer")
        span = (frame0, frames)
        if chunk in self.chunks:
            if self.chunks[chunk] != span: raise ValueError("chunk frame mapping changed during replay")
            return
        if any(frame0 < start + length and start < frame0 + frames for start, length in self.chunks.values()):
            raise ValueError("chunk frame mappings cannot overlap")
        self.chunks[chunk] = span

    def record_injection(self, event: FrozenEvent):
        if not isinstance(event, FrozenEvent):
            raise TypeError("injection must identify a FrozenEvent")
        self.injections[event] += 1

    def _release_outputs(self):
        self.outputs.clear()
        self._budget.release(self.capture_bytes)
        self.capture_bytes = 0


@dataclass
class PairedReplayResult:
    reference_trace: TraceRecorder
    reference_latents: torch.Tensor
    rows: list[dict]
    last_output: RolloutOutput
    branch_summaries: list[dict]


def _validate_output(output, trace):
    if not isinstance(output, RolloutOutput) or not isinstance(output.latents, torch.Tensor):
        raise TypeError("run_callback must return RolloutOutput with tensor latents")
    if output.latents.ndim < 2 or min(output.latents.shape[:2]) < 1:
        raise ValueError("latents must have nonempty [B,T,...] dimensions")
    for frame0, frames in trace.chunks.values():
        if frame0 + frames > output.latents.shape[1]:
            raise ValueError("chunk mapping extends past the returned latent trajectory")


def _relative_l2(reference, actual, budget, transient_bytes):
    if reference.shape != actual.shape:
        raise ValueError("paired output shapes differ")
    # Bounded CPU FP32 scratch; do not copy the full active branch to CPU.
    available = budget.limit - budget.used - transient_bytes
    elements = min(65536, available // 16, max(1, reference.numel()))
    if elements < 1:
        raise ReplayMemoryError("capture cap leaves no space for bounded metric scratch")
    budget.check(transient_bytes + elements * 16)
    numerator = denominator = 0.
    def pieces(a, b):
        if a.is_contiguous() and b.is_contiguous():
            yield a.reshape(-1), b.reshape(-1)
        elif a.ndim:
            for left, right in zip(a.unbind(0), b.unbind(0)):
                yield from pieces(left, right)
        else:
            yield a.reshape(-1), b.reshape(-1)
    for left, right in pieces(reference.detach(), actual.detach()):
        for start in range(0, left.numel(), elements):
            a = left[start:start + elements].to(device="cpu", dtype=torch.float32, copy=True)
            b = right[start:start + elements].to(device="cpu", dtype=torch.float32, copy=True)
            numerator += float((a - b).square().sum(dtype=torch.float64))
            denominator += float(a.square().sum(dtype=torch.float64))
    return math.sqrt(numerator) / (math.sqrt(denominator) + 1e-12)


def _base(case, target_chunk):
    def common(name):
        values = {getattr(event, name) for event in case.events}
        return next(iter(values)) if len(values) == 1 else None
    return {**case.metadata, "case_id": case.case_id, "intervention": case.kind, "role": case.role,
            "strength": case.strength, "target_layer": common("layer"), "target_head": common("head"),
            "target_chunk": common("chunk") if case.events else target_chunk, "target_step": common("step"),
            "target_events": [asdict(event) for event in case.events], "paired_rng": True}


def _case_rows(case, reference_trace, trace, reference_latents, output, horizons, target_chunk, budget):
    base = _base(case, target_chunk)
    result = []
    temporary = _bytes(output.latents)
    for chunk, step in sorted(set(reference_trace.outputs) | set(trace.outputs)):
        row = {**base, "chunk_index": chunk, "denoise_step_index": step, "horizon_chunks": chunk - target_chunk,
               "metric_scope": "D_step", "model_output_relative_error": None, "clean_latent_relative_error": None}
        key = chunk, step
        if key not in reference_trace.outputs or key not in trace.outputs:
            row.update(status="not_measured", reason="paired_step_output_not_recorded")
        else:
            row.update(status="measured", model_output_relative_error=_relative_l2(
                reference_trace.outputs[key], trace.outputs[key], budget, temporary))
        result.append(row)
    for horizon in sorted({0, *horizons}):
        chunk = target_chunk + horizon
        row = {**base, "chunk_index": chunk, "horizon_chunks": horizon,
               "denoise_step_index": None, "model_output_relative_error": None, "clean_latent_relative_error": None,
               "metric_scope": "D_clean_current" if horizon == 0 else "D_clean_horizon"}
        if chunk not in reference_trace.chunks or chunk not in trace.chunks:
            row.update(status="not_measured", reason="chunk_horizon_not_available")
        else:
            frame0, frames = reference_trace.chunks[chunk]
            row.update(status="measured", frame0=frame0, frames=frames, clean_latent_relative_error=_relative_l2(
                reference_latents[:, frame0:frame0 + frames], output.latents[:, frame0:frame0 + frames], budget, temporary))
        result.append(row)
    return result


@torch.no_grad()
def paired_replay(run_callback: Callable, cases_or_factory: Iterable[ReplayCase] | Callable, *,
                  max_capture_mb=256, horizons=(0, 1, 2, 4, 8), target_chunk=0,
                  record_steps=None, on_branch=None) -> PairedReplayResult:
    """Replay a baseline and independent branches from an identical entry RNG.

    A cases factory runs after the reference and receives its recorder. Every
    target must fire exactly once. Callback RNG sequences and final states must
    match exactly. On success or failure, the caller resumes at reference-end
    RNG state. The on_branch hook consumes ephemeral branch traces before their
    CPU outputs are released; only the final callback output is returned.
    """
    _nonnegative(target_chunk, "target_chunk")
    horizons = tuple(horizons)
    for horizon in horizons: _nonnegative(horizon, "horizon")
    budget = _Budget(max_capture_mb)
    entry = capture_rng_state()
    reference_end = entry
    reference_trace = TraceRecorder(max_capture_mb, record_steps, _budget=budget)
    try:
        reference_output = run_callback(None, reference_trace)
        reference_end = capture_rng_state()
        _validate_output(reference_output, reference_trace)
        amount = _bytes(reference_output.latents)
        # Include the source trajectory during its protected CPU copy.
        budget.check(2 * amount)
        budget.reserve(amount)
        reference_latents = reference_output.latents.detach().to(device="cpu", copy=True)
        cases = list(cases_or_factory(reference_trace) if callable(cases_or_factory) else cases_or_factory)
        if not all(isinstance(case, ReplayCase) for case in cases):
            raise TypeError("cases must contain ReplayCase instances")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("case_id values must be unique")
        rows, summaries = [], []
        last_output = reference_output
        if cases:
            if entry.cuda_initialized != reference_end.cuda_initialized:
                raise ReplayRNGError("CUDA must be initialized before paired replay, so entry RNG can be restored")
            last_output = None
            del reference_output
        for case in cases:
            # Release the previous full branch before replaying another branch.
            last_output = None
            trace = TraceRecorder(max_capture_mb, record_steps, _budget=budget)
            restore_rng_state(entry)
            output = None
            try:
                output = run_callback(case, trace)
                branch_end = capture_rng_state()
                if trace.rng_trace != reference_trace.rng_trace or _fingerprint(branch_end) != _fingerprint(reference_end):
                    raise ReplayRNGError(f"case {case.case_id}: per-call RNG fingerprints or final RNG differ from reference")
                _validate_output(output, trace)
                budget.check(_bytes(output.latents))
                if output.latents.shape != reference_latents.shape:
                    raise ValueError("branch latent trajectory shape differs from reference")
                if trace.chunks != reference_trace.chunks:
                    raise ValueError("branch chunk frame mapping differs from reference")
                wrong = [event for event in case.events if trace.injections[event] != 1]
                extra = [event for event in trace.injections if event not in case.events]
                if wrong or extra:
                    raise ReplayTargetError(f"case {case.case_id}: every target must be hit exactly once; missing/repeated={wrong}, unexpected={extra}")
                branch_rows = _case_rows(case, reference_trace, trace, reference_latents, output,
                                         horizons, target_chunk, budget)
                rows.extend(branch_rows)
                summaries.append({"case_id": case.case_id, "paired_rng": True, "rng_calls": len(trace.rng_trace),
                                  "final_rng_fingerprint": _fingerprint(branch_end),
                                  "hits": [{**asdict(event), "hits": trace.injections[event]} for event in case.events],
                                  "measured_rows": sum(row["status"] == "measured" for row in branch_rows),
                                  "peak_capture_bytes": budget.peak})
                if on_branch is not None:
                    on_branch(case, trace, output, branch_rows)
                last_output = output
            finally:
                trace._release_outputs()
                output = None
        return PairedReplayResult(reference_trace, reference_latents, rows, last_output, summaries)
    finally:
        restore_rng_state(reference_end)
