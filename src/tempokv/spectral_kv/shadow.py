"""Suite B E3-off: candidate outputs never feed back into the BF16 trajectory."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import torch

from tempokv.spectral.metrics import relative_error
from tempokv.spectral.sensitivity import Block, attention_mass
from .attention import spectral_attention_fast, spectral_attention_ref
from .cache import SpectralKVCache
from .method_spec import MethodSpec, parse_method
from .policy import DistanceSpectralPolicy, FullPolicy, OracleTopKPolicy, WindowSinkPolicy
from .rope3d import full_positions, rope_invert


def make_policy(method: MethodSpec, *, recent=6, sink=1):
    if method.family in {"full", "kvskip"}:
        return FullPolicy()
    if method.family == "drop":
        return WindowSinkPolicy(method.window, method.sink)
    if method.family == "pyramid":
        return DistanceSpectralPolicy(method.exponent, recent=recent, sink=sink)
    return OracleTopKPolicy(method.topk or 0, others="summary", ratio=method.ratio,
                           recent=method.window if method.window is not None else recent, sink=sink)


def frame_mass(q, k_post, positions, *, chunk_size=1024):
    positions = np.asarray(positions, dtype=np.int64)
    # Caller supplies frame numbers (one for every effective KV token).
    frames = sorted(set(positions.tolist()))
    blocks = [Block(frame, 0, tuple(np.flatnonzero(positions == frame).tolist())) for frame in frames]
    values = attention_mass(q, k_post, blocks, chunk_size=chunk_size)
    return dict(zip(frames, values))


def cache_from_window(spec, layer, k_post, v, absolute_positions, method, *, freqs=None,
                      recent=6, sink=1, precision="float64", current_frames=(), keep_originals=True):
    positions = np.asarray(absolute_positions, dtype=np.int64)
    h, w = spec.frame_hw
    frame_tokens = h * w
    if positions.shape != (k_post.shape[1],) or len(set(positions.tolist())) != len(positions):
        raise ValueError("window positions must uniquely map every KV token")
    cache = SpectralKVCache(spec, layer, make_policy(method, recent=recent, sink=sink), method.kind,
                            keep_originals, freqs=freqs, taper=method.taper, sink_frames=sink, precision=precision)
    for frame in sorted(set((positions // frame_tokens).tolist())):
        indices = np.flatnonzero(positions // frame_tokens == frame)
        expected = np.arange(frame * frame_tokens, (frame + 1) * frame_tokens)
        if not np.array_equal(positions[indices], expected):
            raise ValueError("spectral summaries require complete spatial frames in the actual window")
        selected = torch.as_tensor(indices, device=k_post.device)
        cache.write_chunk(k_post.index_select(1, selected), v.index_select(1, selected), frame, k_is_roped=True)
    cache.current_frames = set(current_frames)
    return cache


class ShadowAttention:
    def __init__(self, methods, *, recent=6, sink=1, backend="reference", eps=1e-12):
        self.methods = tuple(parse_method(name) for name in methods)
        self.recent, self.sink, self.backend, self.eps = recent, sink, backend, eps

    @torch.no_grad()
    def evaluate(self, q, k_post, v, spec, *, layer=0, absolute_positions, current_frames,
                 chunk_idx=0, freqs=None, precision="float64", attention_fn=None):
        from tempokv.spectral.sensitivity import reference_attention
        attention_fn = attention_fn or reference_attention
        reference = attention_fn(q, k_post, v)
        masses = frame_mass(q, k_post, np.asarray(absolute_positions) // (spec.frame_hw[0] * spec.frame_hw[1]))
        baseline_bytes = (k_post.numel() + v.numel()) * k_post.element_size()
        rows = []
        for method in self.methods:
            if method.family == "full":
                actual, view, cache = reference, None, None
                tokens = k_post.shape[1]
            else:
                cache = cache_from_window(spec, layer, k_post, v, absolute_positions, method, freqs=freqs,
                                          recent=self.recent, sink=self.sink, precision=precision,
                                          current_frames=current_frames)
                cache.age(chunk_idx, q_probe=masses)
                view = cache.materialize(chunk_idx, bias=method.bias)
                function = spectral_attention_fast if self.backend == "group_lse" else spectral_attention_ref
                actual = function(q, view)
                tokens = view.token_count
            rows.append({"method_spec": method.name, "projection": method.kind if method.family == "summary" else method.family,
                         "requested_ratio": method.ratio, "log_count_bias": method.bias,
                         "taper": method.taper, "relative_attention_output_error": relative_error(reference, actual, self.eps),
                         "tokens_attended": tokens, "full_tokens": k_post.shape[1],
                         "actual_retention": tokens / k_post.shape[1],
                         "view_resident_bytes": baseline_bytes if view is None else sum(
                             (g.k.numel() + g.v.numel()) * g.k.element_size() for g in view.groups),
                         "oracle_storage_bytes": baseline_bytes if cache is None else cache.resident_bytes,
                         "baseline_native_cache_bytes": baseline_bytes, "status": "ok",
                         "trajectory": "BF16_readonly_shadow", "reference_norm": float(torch.linalg.vector_norm(reference.float())),
                         "query_count": q.shape[1], "head_count": q.shape[2],
                         "topk_scope": "sampled_queries_per_head" if method.topk else None})
        return rows
