"""Paired cache recovery after one common low-bit Self-Forcing prefix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time

import torch

from .precision import fake_quantize
from .rollout import capture_rng_state, restore_rng_state, rng_fingerprint

SCHEMA = "research.cache_recovery.v1"
BRANCHES = ("continue", "stop", "undo_rounding", "reencode_own", "sham", "reencode_reference")
ROLES = ("k", "v")
REENCODE = {"reencode_own", "sham", "reencode_reference"}
RESTORE = {"continue", "stop", "undo_rounding", "sham"}


@dataclass(frozen=True)
class RecoverySettings:
    bits: int = 4
    group_size: int = 64
    fork_chunk: int = 6
    future_chunks: int = 8
    protect_chunks: int = 2
    warm_start_chunk: int | None = None

    def __post_init__(self):
        fields = asdict(self)
        warm_start = fields.pop("warm_start_chunk")
        if any(type(value) is not int for value in fields.values()) or (
            warm_start is not None and type(warm_start) is not int
        ):
            raise TypeError("recovery settings must be integers")
        if self.bits not in (2, 4, 16) or self.group_size < 1:
            raise ValueError("use 2 or 4 bits, or 16 for identity, and a positive group size")
        if self.protect_chunks < 1 or self.fork_chunk <= self.protect_chunks + 1:
            raise ValueError("the prefix must contain a rounded chunk and protected recent history")
        if self.future_chunks < 1:
            raise ValueError("at least one future chunk is required")
        if not self.protect_chunks < self.first_round_chunk <= self.fork_chunk:
            raise ValueError("warm start must be after protection and at or before the fork")

    @property
    def first_round_chunk(self):
        return self.protect_chunks + 1 if self.warm_start_chunk is None else self.warm_start_chunk


def _validate_cache(cache, end):
    if not cache:
        raise ValueError("a native dense BF16 cache is required")
    for layer in cache:
        if layer.get("cache_backend") != "bf16" or (
            layer["global_end_index"] != end or layer["local_end_index"] != end
        ):
            raise ValueError("expected a full, contiguous native cache at the absolute write position")
        if any(layer[role].dtype != torch.bfloat16 or layer[role].shape[1] < end for role in ROLES):
            raise ValueError("native cache dtype or capacity changed")


def _copy_kv(cache, start, stop):
    return tuple(
        tuple(layer[role][:, start:stop].to(device="cpu", copy=True) for role in ROLES)
        for layer in cache
    )


def _restore_kv(cache, values, start):
    for layer, pair in zip(cache, values, strict=True):
        for role, value in zip(ROLES, pair):
            destination = layer[role][:, start:start + value.shape[1]]
            if destination.shape != value.shape or destination.dtype != value.dtype:
                raise ValueError("cache shape or dtype changed")
            destination.copy_(value)


def _kv_bytes(values):
    return sum(t.numel() * t.element_size() for layer in values for t in layer)


@dataclass
class ForkCache:
    end: int
    kv: tuple
    text_kv: tuple

    @classmethod
    def capture(cls, pipeline, end):
        _validate_cache(pipeline.kv_cache1, end)
        text_cache = pipeline.crossattn_cache
        if not all(layer["is_init"] for layer in text_cache):
            raise ValueError("prefix text cache was not initialized")
        return cls(end, _copy_kv(pipeline.kv_cache1, 0, end), _copy_kv(text_cache, 0, None))

    def restore(self, pipeline, prefill_end):
        _validate_cache(pipeline.kv_cache1, prefill_end)
        _restore_kv(pipeline.kv_cache1, self.kv, 0)
        _restore_kv(pipeline.crossattn_cache, self.text_kv, 0)
        for layer in pipeline.kv_cache1:
            layer["global_end_index"] = layer["local_end_index"] = self.end
        for layer in pipeline.crossattn_cache:
            layer["is_init"] = True


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def _latent_errors(candidate, reference, chunk_frames):
    actual = candidate.detach().cpu().float().split(chunk_frames, dim=1)
    expected = reference.detach().cpu().float().split(chunk_frames, dim=1)
    return [
        float((a - b).norm() / b.norm().clamp_min(1e-12))
        for a, b in zip(actual, expected, strict=True)
    ]


def _cache_errors(cache, reference, chunk_tokens):
    """Pool squared errors across layers and heads before taking a norm."""
    chunks = reference.end // chunk_tokens
    device = cache[0]["k"].device
    totals = torch.zeros((2, 2, chunks), device=device, dtype=torch.float64)
    for layer, ref_pair in zip(cache, reference.kv, strict=True):
        for i, (role, ref) in enumerate(zip(ROLES, ref_pair)):
            native = ref.to(device).float().reshape(1, chunks, chunk_tokens, -1)
            actual = layer[role][:, :reference.end].float().reshape_as(native)
            totals[i, 0] += (actual - native).square().sum(dim=(0, 2, 3), dtype=torch.float64)
            totals[i, 1] += native.square().sum(dim=(0, 2, 3), dtype=torch.float64)
    return [
        {"role": role, "prefix_chunk": chunk, "error_energy": num,
         "reference_energy": den, "relative_l2": math.sqrt(num / max(den, 1e-24))}
        for role, (error, energy) in zip(ROLES, totals.tolist())
        for chunk, (num, den) in enumerate(zip(error, energy))
    ]


class RecoveryRollout:
    def __init__(self, pipeline, infer, settings: RecoverySettings, noise, prompts):
        self.pipeline = pipeline
        self.infer = infer
        self.settings = settings
        self.noise = noise
        self.prompts = prompts
        self.frames = pipeline.num_frame_per_block
        self.tokens = self.frames * pipeline.frame_seq_length
        self.fork_tokens = settings.fork_chunk * self.tokens
        self.fork_frames = settings.fork_chunk * self.frames
        self.calls_per_chunk = len(pipeline.denoising_step_list) + 1
        expected_frames = (settings.fork_chunk + settings.future_chunks) * self.frames
        if (
            noise.ndim != 5 or noise.shape[:2] != (1, expected_frames) or noise.dtype != torch.bfloat16
            or len(prompts) != 1 or pipeline.local_attn_size != -1 or pipeline.independent_first_frame
            or pipeline.args.context_noise != 0 or pipeline._uses_pre_rope_cache
            or any(block.self_attn.sink_size for block in pipeline.generator.model.blocks)
        ):
            raise ValueError("recovery needs one BF16 T2V sample, native full history, no sink, and clean timestep 0")
        self.fork_rng = None
        self.fork_rng_fingerprint = None
        self.reference_cache = None
        self.warm_cache = None
        self.unrounded = {}
        self.rounded_prefix_end = 0

    def _round_through(self, cache, case, old_end, cutoff):
        if cutoff > old_end and self.settings.bits != 16:
            for offset in range(old_end, cutoff, self.tokens):
                stop = offset + self.tokens
                if case == "warm":
                    self.unrounded[offset] = _copy_kv(cache, offset, stop)
                for layer in cache:
                    for role in ROLES:
                        span = layer[role][:, offset:stop]
                        span.copy_(fake_quantize(span, self.settings.bits, self.settings.group_size).value)
        return max(cutoff, old_end)

    def _run(self, case, noise, prefix=None):
        print(f"[cache-recovery] {case}", flush=True)
        pipeline = self.pipeline
        device = noise.device
        generator = pipeline.generator
        original_forward = generator.forward
        had_local_forward = "forward" in vars(generator)
        rebuild = case in REENCODE
        calls = []
        quantized_until = self.rounded_prefix_end if case == "continue" else 0
        written_prefix = 0
        started_prefill = None
        prefill_seconds = 0.0
        restore_seconds = 0.0
        cache_rows = []
        started_future = None

        def forward(**kwargs):
            nonlocal quantized_until, written_prefix, started_prefill
            nonlocal prefill_seconds, restore_seconds, cache_rows, started_future
            start = int(kwargs["current_start"])
            cache = kwargs["kv_cache"]
            if prefix is not None and start < self.fork_tokens:
                if start != written_prefix:
                    raise ValueError("native prefix replay changed its absolute chunk order")
                written_prefix += self.tokens
                if not rebuild:
                    # The native prefix loop ignores these forward results.
                    return None
                if started_prefill is None:
                    _sync(device)
                    started_prefill = time.perf_counter()
                # Match the native clean-refresh shape at timestep zero.
                kwargs["timestep"] = kwargs["timestep"].expand(-1, self.frames).contiguous()
                _validate_cache(cache, start)
                result = original_forward(**kwargs)
                _validate_cache(cache, start + self.tokens)
                return result

            if start == self.fork_tokens and started_future is None:
                _sync(device)
                if case == "reference":
                    self.fork_rng = capture_rng_state()
                    self.fork_rng_fingerprint = rng_fingerprint()
                    self.reference_cache = ForkCache.capture(pipeline, self.fork_tokens)
                else:
                    if written_prefix != self.fork_tokens:
                        raise ValueError("incomplete native prefix replay")
                    if started_prefill is not None:
                        prefill_seconds = time.perf_counter() - started_prefill
                    if case in RESTORE:
                        restore_start = time.perf_counter()
                        self.warm_cache.restore(pipeline, self.fork_tokens if rebuild else 0)
                        if case == "undo_rounding":
                            for offset, values in self.unrounded.items():
                                _restore_kv(cache, values, offset)
                        _sync(device)
                        restore_seconds = time.perf_counter() - restore_start
                    restore_rng_state(self.fork_rng)
                    _validate_cache(cache, self.fork_tokens)
                    cache_rows = _cache_errors(cache, self.reference_cache, self.tokens)
                    _sync(device)
                started_future = time.perf_counter()

            at_chunk_start = len(calls) % self.calls_per_chunk == 0
            _validate_cache(cache, start if at_chunk_start else start + self.tokens)
            if at_chunk_start and (
                case == "continue"
                or (case == "warm" and start >= self.settings.first_round_chunk * self.tokens)
            ):
                cutoff = max(0, start - self.settings.protect_chunks * self.tokens)
                quantized_until = self._round_through(cache, case, quantized_until, cutoff)
            before = rng_fingerprint()
            result = original_forward(**kwargs)
            _validate_cache(cache, start + self.tokens)
            calls.append((start, before, rng_fingerprint()))
            return result

        _sync(device)
        started = time.perf_counter()
        generator.forward = forward
        try:
            _, latents = self.infer(
                noise=noise, text_prompts=self.prompts, initial_latent=prefix,
                return_latents=True, decode_output=False,
            )
            _sync(device)
        finally:
            if had_local_forward:
                generator.forward = original_forward
            else:
                del generator.forward
        finished = time.perf_counter()
        end_tokens = latents.shape[1] // self.frames * self.tokens
        start_tokens = 0 if prefix is None else self.fork_tokens
        if [start for start, _, _ in calls] != [
            start for start in range(start_tokens, end_tokens, self.tokens)
            for _ in range(self.calls_per_chunk)
        ]:
            raise ValueError("native denoise and clean-refresh call schedule changed")
        if case == "warm":
            _validate_cache(pipeline.kv_cache1, self.fork_tokens)
            # Even when no warm forward reads low-bit history, round the
            # identical old spans before forking. Keep the last warm write
            # and all of its protected predecessors in native precision.
            cutoff = self.fork_tokens - (self.settings.protect_chunks + 1) * self.tokens
            self.rounded_prefix_end = self._round_through(
                pipeline.kv_cache1, case, quantized_until, cutoff,
            )
            self.warm_cache = ForkCache.capture(pipeline, self.fork_tokens)
            quantized_until = self.rounded_prefix_end
        accesses = sum((start + self.tokens) * self.tokens for start, _, _ in calls if start >= self.fork_tokens)
        return latents, calls, rng_fingerprint(), {
            "run_seconds": finished - started,
            "prefix_reencode_seconds": prefill_seconds,
            "cache_restore_seconds": restore_seconds,
            "future_seconds": None if started_future is None else finished - started_future,
            "future_query_key_pairs_all_heads_layers": accesses * pipeline.num_heads * len(pipeline.kv_cache1),
            "rounded_through_exclusive_token": quantized_until if case in ("warm", "continue") else None,
            "fork_cache_rows": cache_rows,
        }

    @torch.no_grad()
    def run(self):
        entry = capture_rng_state()
        reference_end = entry
        try:
            reference, reference_calls, end_hash, reference_cost = self._run("reference", self.noise)
            reference_end = capture_rng_state()
            restore_rng_state(entry)
            warm, prefix_calls, warm_end, warm_cost = self._run("warm", self.noise[:, :self.fork_frames])
            prefix_call_count = self.settings.fork_chunk * self.calls_per_chunk
            if prefix_calls != reference_calls[:prefix_call_count] or warm_end != self.fork_rng_fingerprint:
                raise ValueError("warm prefix RNG differs from the native reference")
            reference_prefix = reference[:, :self.fork_frames]
            reference_suffix = reference[:, self.fork_frames:]
            prefix_hash = _digest(warm)
            reference_hash = _digest(reference_suffix)
            rows = {}
            for case in BRANCHES:
                # Only this oracle consumes reference-generated latents.
                source = reference_prefix if case == "reencode_reference" else warm
                latents, calls, end, cost = self._run(case, self.noise[:, self.fork_frames:], source)
                if calls != reference_calls[prefix_call_count:] or end != end_hash:
                    raise ValueError(f"{case}: paired future RNG differs from the reference")
                if cost["future_query_key_pairs_all_heads_layers"] != reference_cost["future_query_key_pairs_all_heads_layers"]:
                    raise ValueError(f"{case}: changed future attention budget")
                if not torch.equal(latents[:, :self.fork_frames], source):
                    raise ValueError(f"{case}: native prefix input was modified")
                suffix = latents[:, self.fork_frames:]
                rows[case] = {
                    **cost, "observable_prefix_sha256": prefix_hash, "cache_input_prefix_sha256": _digest(source),
                    "future_sha256": _digest(suffix), "future_rng_calls": len(calls),
                    "future_chunk_relative_l2": _latent_errors(suffix, reference_suffix, self.frames),
                }

            if rows["stop"]["future_sha256"] != rows["sham"]["future_sha256"]:
                raise ValueError("rebuild-and-discard sham changed the stopped trajectory")
            if rows["reencode_reference"]["future_sha256"] != reference_hash:
                raise ValueError("native reference re-encode did not reproduce the reference continuation")
            if self.settings.bits == 16 and any(row["future_sha256"] != reference_hash for row in rows.values()):
                raise ValueError("native-precision identity control changed a future trajectory")
            warm_exposure = self.settings.fork_chunk - self.settings.first_round_chunk
            if warm_exposure == 0:
                if not torch.equal(warm, reference_prefix):
                    raise ValueError("no-exposure control changed the prefix before rounding")
                if any(rows[name]["future_sha256"] != reference_hash for name in ("undo_rounding", "reencode_own")):
                    raise ValueError("no-exposure control did not recover the native future")

            return {
                "schema_version": SCHEMA, "settings": asdict(self.settings),
                "warm_first_round_chunk": self.settings.first_round_chunk,
                "warm_exposed_chunks": warm_exposure,
                "frame_tokens": self.pipeline.frame_seq_length, "frames_per_chunk": self.frames,
                "layers": len(self.pipeline.kv_cache1), "heads": self.pipeline.num_heads,
                "denoise_steps": self.calls_per_chunk - 1, "fork_tokens": self.fork_tokens,
                "key_coordinate": "native_post_rope", "value_coordinate": "raw",
                "attention_window": "full_no_sink", "future_rng_fingerprint": end_hash,
                "reference_suffix_sha256": reference_hash, "reference_cost": reference_cost,
                "warm_cost": warm_cost, "warm_prefix_relative_l2": _latent_errors(warm, reference_prefix, self.frames),
                "warm_prefix_latent_bytes": warm.numel() * warm.element_size(),
                "saved_warm_kv_bytes": _kv_bytes(self.warm_cache.kv),
                "saved_unrounded_kv_bytes": sum(_kv_bytes(kv) for kv in self.unrounded.values()),
                "saved_text_cache_bytes_per_snapshot": _kv_bytes(self.warm_cache.text_kv),
                "branches": rows,
            }, reference
        finally:
            restore_rng_state(reference_end)
