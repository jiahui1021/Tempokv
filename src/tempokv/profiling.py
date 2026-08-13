"""CUDA-event runtime breakdown for the Self-Forcing KV-cache hot path.

Stage 0 of the acceleration plan refuses to rewrite a kernel before the profile
says that kernel owns the wall clock.  ``torch.profiler`` answers that question
for individual CUDA kernels, but not for the *sections* the plan is written in
terms of — "V cache read", "read-time RoPE", "attention" — because each of those
is many kernels plus the Python, allocator and launch work between them.

This module brackets those sections with CUDA events instead.  Recording an
event does not synchronize, so an instrumented run keeps the launch pattern of an
uninstrumented one; the accumulated intervals are read back only when the event
pool fills or when the report is built.

Sections are keyed by their **call path** rather than by name, so a cache read
performed inside a quantization event is not confused with one performed inside
attention, and every interval is attributed to exactly one parent.  That makes
each section's *self* time exact: total minus the sum of its direct children.

Nothing here is imported by the codec, the cache or the kernels.  Instrumentation
is installed by :func:`install`, which the Self-Forcing worker calls only when a
profile output path is requested, and which reports what it could not patch so a
missing section is never mistaken for a fast one.

One caveat: an event pair measures the stream it was recorded on.  Every section
on the hot path — attention, cache read, RoPE, decode, encode — runs on the
current stream, but the TensorRT VAE decoder creates its own, so ``vae_decode``
is only trustworthy for the default VAE path.
"""

from __future__ import annotations

from contextlib import contextmanager
import functools
from statistics import median
import sys
import time
from typing import Any, Callable, Iterator

import torch


# Sections, in the order the report presents them.  The names are the plan's
# vocabulary; the tree they form at runtime comes from the call paths, not from
# this list.
GENERATION = "generation"
SELF_ATTENTION = "self_attention"
CACHE_READ_K = "cache_read_k"
CACHE_READ_V = "cache_read_v"
DECODE_K = "decode_k"
DECODE_V = "decode_v"
ROPE = "rope"
ATTENTION_KERNEL = "attention_kernel"
QUANTIZE_EVENT = "quantize_event"
ENCODE_K = "encode_k"
ENCODE_V = "encode_v"
VAE_DECODE = "vae_decode"

# One drain costs one ``torch.cuda.synchronize``.  A Self-Forcing block issues
# roughly 4000 sections (30 layers x 5 generator calls x ~7 sections plus a
# quantization event), so this bound drains about once every two blocks: rare
# enough that the added synchronization is far below the noise of a 717-frame
# generation, while keeping the number of live CUDA events bounded.
DEFAULT_POOL_SIZE = 8192


class RuntimeProfile:
    """Accumulate GPU time per section call path.

    On CUDA the interval between two recorded events is measured on the device,
    so the figure includes the gaps where the GPU waited for the CPU to queue
    more work — which is exactly what a launch-bound section looks like.  Off
    CUDA the class falls back to a host clock so the aggregation and reporting
    logic stays exercisable without a device.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be positive")
        self.device = device
        self._use_events = device is not None and device.type == "cuda"
        self._pool_size = pool_size
        self._free: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending: list[tuple[str, Any, Any]] = []
        self._durations: dict[str, list[float]] = {}
        self._stack: list[str] = []
        self._drains = 0

    # -- measurement ---------------------------------------------------------

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Measure one call of ``name`` under the currently open sections."""
        self._stack.append(name)
        path = "/".join(self._stack)
        if self._use_events:
            start, end = self._acquire()
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending.append((path, start, end))
                self._stack.pop()
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self._durations.setdefault(path, []).append(elapsed)
            self._stack.pop()

    def _acquire(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._free:
            return self._free.pop()
        if len(self._pending) >= self._pool_size:
            self.drain()
            if self._free:
                return self._free.pop()
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def drain(self) -> None:
        """Read back every finished interval, synchronizing once."""
        if not self._pending:
            return
        if self._use_events:
            torch.cuda.synchronize(self.device)
            self._drains += 1
            for path, start, end in self._pending:
                self._durations.setdefault(path, []).append(
                    start.elapsed_time(end)
                )
                self._free.append((start, end))
        self._pending.clear()

    # -- reporting -----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Build the section tree with total, self and share-of-generation time."""
        self.drain()
        totals = {
            path: _ordered_sum(durations)
            for path, durations in self._durations.items()
        }
        child_totals: dict[str, float] = {}
        for path, total in totals.items():
            parent = path.rpartition("/")[0]
            if parent:
                child_totals[parent] = child_totals.get(parent, 0.0) + total
        generation_ms = totals.get(GENERATION)
        sections = []
        for path in sorted(totals):
            durations = self._durations[path]
            total = totals[path]
            self_ms = total - child_totals.get(path, 0.0)
            sections.append(
                {
                    "path": path,
                    "name": path.rpartition("/")[2],
                    "depth": path.count("/"),
                    "calls": len(durations),
                    "total_ms": total,
                    "self_ms": self_ms,
                    "mean_ms": total / len(durations),
                    "median_ms": median(durations),
                    "p90_ms": _percentile(durations, 0.90),
                    "share_of_generation": (
                        total / generation_ms if generation_ms else None
                    ),
                    "self_share_of_generation": (
                        self_ms / generation_ms if generation_ms else None
                    ),
                }
            )
        return {
            "backend": "cuda_events" if self._use_events else "host_clock",
            "device": str(self.device) if self.device is not None else None,
            "generation_ms": generation_ms,
            "drains": self._drains,
            "sections": sections,
        }


def aggregate_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collapse the call tree onto section names.

    The tree separates a cache read performed during attention from one
    performed during a quantization event, which is what makes self time exact.
    The Stage 0 decision gate asks a coarser question — what fraction of the
    generation does *decode* own, wherever it ran — so the shares it compares
    have to be summed back over every path that ends in the same name.
    """
    generation_ms = report.get("generation_ms")
    collapsed: dict[str, dict[str, Any]] = {}
    for entry in report["sections"]:
        bucket = collapsed.setdefault(
            entry["name"],
            {"name": entry["name"], "calls": 0, "total_ms": 0.0, "self_ms": 0.0},
        )
        bucket["calls"] += entry["calls"]
        bucket["total_ms"] += entry["total_ms"]
        bucket["self_ms"] += entry["self_ms"]
    for bucket in collapsed.values():
        bucket["share_of_generation"] = (
            bucket["total_ms"] / generation_ms if generation_ms else None
        )
        bucket["self_share_of_generation"] = (
            bucket["self_ms"] / generation_ms if generation_ms else None
        )
    return collapsed


def _ordered_sum(values: list[float]) -> float:
    """Sum in call order so a report is reproducible for a fixed run."""
    total = 0.0
    for value in values:
        total += value
    return total


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty section")
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[index]


# -- instrumentation ---------------------------------------------------------


def _defining_class(cls: type, name: str) -> type | None:
    """Find the class in ``cls``'s MRO that actually defines ``name``.

    The TempoKV bridge replaces ``CausalInferencePipeline`` with a subclass that
    overrides some methods and inherits others, so patching a fixed class would
    silently miss whichever half the bridge did not define.
    """
    for candidate in cls.__mro__:
        if name in vars(candidate):
            return candidate
    return None


def _wrap_function(module: Any, name: str, profile: RuntimeProfile, section: str) -> None:
    original = getattr(module, name)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with profile.section(section):
            return original(*args, **kwargs)

    setattr(module, name, wrapper)


def _wrap_method(
    cls: type,
    name: str,
    profile: RuntimeProfile,
    section: str | Callable[[Any], str],
) -> None:
    owner = _defining_class(cls, name)
    if owner is None:
        raise AttributeError(f"{cls.__name__} has no method {name}")
    original = vars(owner)[name]

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        resolved = section(self) if callable(section) else section
        with profile.section(resolved):
            return original(self, *args, **kwargs)

    setattr(owner, name, wrapper)


def install(profile: RuntimeProfile) -> dict[str, str]:
    """Bracket the Self-Forcing hot path with profile sections.

    Returns one entry per target: ``"patched"``, or the reason it was skipped.
    A BF16 baseline run has no codec and no bridge, so its decode and encode
    targets are legitimately absent; every other absence is a bug in this
    installer and has to be visible in the report rather than read as zero time.
    """
    status: dict[str, str] = {}

    def attempt(target: str, action: Callable[[], None]) -> None:
        try:
            action()
        except (AttributeError, ImportError, KeyError) as error:
            status[target] = f"skipped: {type(error).__name__}: {error}"
        else:
            status[target] = "patched"

    def patch_cache_read() -> None:
        from .runtime.cache import ChunkedKVCache

        _wrap_method(
            ChunkedKVCache,
            "_read_layout",
            profile,
            lambda cache: (
                CACHE_READ_V if cache.cache_role == "V" else CACHE_READ_K
            ),
        )

    def patch_rope() -> None:
        from .runtime import rope

        # ``causal_rope_apply_long_input`` delegates here, so one patch covers
        # both the contiguous and the span-aware entry point.
        _wrap_function(rope, "apply_rope_by_span", profile, ROPE)

    def patch_decode() -> None:
        bridge = sys.modules["bridge"]
        # The bridge imported ``decode_tensor`` by value, so the codec module is
        # the wrong place to patch: only the bridge's own binding is reached by
        # the cache's decoder callback.
        original = bridge.decode_cache_state

        @functools.wraps(original)
        def wrapper(state: Any, **kwargs: Any) -> Any:
            kind = getattr(state, "predictor_kind", None)
            if kind is None and isinstance(state, dict):
                kind = state.get("predictor_kind")
            section = DECODE_K if kind == "temporal" else DECODE_V
            with profile.section(section):
                return original(state, **kwargs)

        bridge.decode_cache_state = wrapper

    def patch_encode() -> None:
        from .codec import TempoKVCodec

        _wrap_method(TempoKVCodec, "encode_key", profile, ENCODE_K)
        _wrap_method(TempoKVCodec, "encode_value", profile, ENCODE_V)

    def patch_attention() -> None:
        causal_model = sys.modules["wan.modules.causal_model"]
        _wrap_function(causal_model, "attention", profile, ATTENTION_KERNEL)
        _wrap_method(
            causal_model.CausalWanSelfAttention,
            "forward",
            profile,
            SELF_ATTENTION,
        )

    def patch_pipeline() -> None:
        pipeline_module = sys.modules["pipeline.causal_inference"]
        pipeline_class = pipeline_module.CausalInferencePipeline
        _wrap_method(pipeline_class, "inference", profile, GENERATION)
        _wrap_method(pipeline_class, "quantize_kv_cache", profile, QUANTIZE_EVENT)

    def patch_vae() -> None:
        wrapper_module = sys.modules["utils.wan_wrapper"]
        _wrap_method(
            wrapper_module.WanVAEWrapper, "decode_to_pixel", profile, VAE_DECODE
        )

    attempt("cache_read", patch_cache_read)
    attempt("rope", patch_rope)
    attempt("decode", patch_decode)
    attempt("encode", patch_encode)
    attempt("attention", patch_attention)
    attempt("pipeline", patch_pipeline)
    attempt("vae", patch_vae)
    return status


# -- presentation -----------------------------------------------------------


def format_report(report: dict[str, Any]) -> str:
    """Render a report as an indented table ordered by the call tree."""
    lines = [
        f"backend={report['backend']} device={report['device']} "
        f"drains={report['drains']}"
    ]
    generation_ms = report.get("generation_ms")
    if generation_ms:
        lines.append(f"generation: {generation_ms / 1000.0:.3f} s")
    header = (
        f"{'section':<40}{'calls':>9}{'total s':>11}{'self s':>11}"
        f"{'share':>8}{'self':>8}{'median ms':>14}{'p90 ms':>14}"
    )
    lines.extend(("", header, "-" * len(header)))
    for entry in report["sections"]:
        indent = "  " * entry["depth"]
        share = entry["share_of_generation"]
        self_share = entry["self_share_of_generation"]
        lines.append(
            f"{indent + entry['name']:<40}"
            f"{entry['calls']:>9}"
            f"{entry['total_ms'] / 1000.0:>11.3f}"
            f"{entry['self_ms'] / 1000.0:>11.3f}"
            f"{_format_share(share):>8}"
            f"{_format_share(self_share):>8}"
            f"{entry['median_ms']:>14.4f}"
            f"{entry['p90_ms']:>14.4f}"
        )
    skipped = {
        target: reason
        for target, reason in (report.get("instrumentation") or {}).items()
        if reason != "patched"
    }
    if skipped:
        lines.append("")
        lines.append("not instrumented:")
        for target, reason in sorted(skipped.items()):
            lines.append(f"  {target}: {reason}")
    return "\n".join(lines)


def _format_share(share: float | None) -> str:
    return "-" if share is None else f"{share * 100:.1f}%"


__all__ = [
    "ATTENTION_KERNEL",
    "CACHE_READ_K",
    "CACHE_READ_V",
    "DECODE_K",
    "DECODE_V",
    "ENCODE_K",
    "ENCODE_V",
    "GENERATION",
    "QUANTIZE_EVENT",
    "ROPE",
    "SELF_ATTENTION",
    "VAE_DECODE",
    "RuntimeProfile",
    "aggregate_by_name",
    "format_report",
    "install",
]
