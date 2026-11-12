"""Validate read precision policies under repeated autoregressive exposure."""

from __future__ import annotations

from dataclasses import asdict
import math

import torch

from .cache_read_precision import UNIFORM, mixed_history
from .cache_recovery import ROLES, RecoveryRollout, _digest, _latent_errors, _restore_kv
from .precision import fake_quantize
from .rollout import capture_rng_state, restore_rng_state

SCHEMA = "research.cache_persistent_precision.v1"
ARMS = ("rounded_all", "protect_first_denoise", "protect_last_denoise", UNIFORM)


class PersistentReadPrecisionRollout(RecoveryRollout):
    def __init__(self, pipeline, infer, settings, noise, prompts):
        super().__init__(pipeline, infer, settings, noise, prompts)
        if settings.bits != 4 or settings.first_round_chunk != settings.fork_chunk:
            raise ValueError("persistent precision uses a clean prefix and a 4-bit base after the fork")
        if self.calls_per_chunk < 3:
            raise ValueError("persistent precision needs at least two denoising calls")

    def _run_policy(self, name, protected):
        pending = []
        previous_end = 0
        using_native = False
        read_rows = []
        width = self.pipeline.head_dim
        scales = math.ceil(width / self.settings.group_size)

        def intervene(start, index, kwargs):
            nonlocal previous_end, using_native
            if start < self.fork_tokens:
                return
            cache = kwargs["kv_cache"]
            eligible = start - self.settings.protect_chunks * self.tokens
            use_native = index == protected
            if index == 0:
                # Each new span was just written in native precision by this
                # branch. Previously aged entries already have both versions
                # when a policy needs to read their original values again.
                native = tuple(
                    tuple(layer[role][:, previous_end:eligible].clone() for role in ROLES)
                    for layer in cache
                )
                if name == UNIFORM:
                    rounded, _ = mixed_history(native, self.settings, self.calls_per_chunk)
                else:
                    rounded = tuple(tuple(
                        fake_quantize(value, self.settings.bits, self.settings.group_size).value
                        for value in pair
                    ) for pair in native)
                if protected is not None:
                    pending.append((previous_end, native, rounded))
                if not use_native:
                    _restore_kv(cache, rounded, previous_end)
                previous_end = eligible

            if protected is not None and use_native != using_native:
                for offset, native, rounded in pending:
                    _restore_kv(cache, native if use_native else rounded, offset)
                using_native = use_native

            rows = 2 * len(cache) * self.pipeline.num_heads * eligible
            native_bytes = rows * width * 2
            packed_bytes = (rows * width * self.settings.bits + 7) // 8 + rows * scales * 4
            read_bytes = native_bytes if use_native else packed_bytes
            if name == UNIFORM:
                read_bytes, remainder = divmod(
                    native_bytes + (self.calls_per_chunk - 1) * packed_bytes, self.calls_per_chunk,
                )
                if remainder:
                    raise ValueError("uniform read volume is not integral")
            read_rows.append({
                "absolute_chunk": start // self.tokens,
                "call_index": index,
                "aged_positions_per_head": eligible,
                "original_aged_history": use_native,
                "ideal_aged_read_bytes": read_bytes,
            })

        latents, calls, end_hash, cost = self._run(name, self.noise, before_forward=intervene)
        expected = [
            (chunk, index, (chunk - self.settings.protect_chunks) * self.tokens)
            for chunk in range(self.settings.fork_chunk, self.settings.fork_chunk + self.settings.future_chunks)
            for index in range(self.calls_per_chunk)
        ]
        if [(r["absolute_chunk"], r["call_index"], r["aged_positions_per_head"]) for r in read_rows] != expected:
            raise ValueError("the actual repeated read scope or phase count changed")
        return latents, calls, end_hash, {
            "aged_history_reads": read_rows,
            "total_ideal_aged_read_bytes": sum(r["ideal_aged_read_bytes"] for r in read_rows),
            "future_query_key_pairs_all_heads_layers": cost["future_query_key_pairs_all_heads_layers"],
        }

    @torch.no_grad()
    def run(self):
        entry = capture_rng_state()
        reference_end = entry
        try:
            ref, ref_calls, ref_end, cost = self._run("reference", self.noise)
            reference_end = capture_rng_state()
            # Only trajectory and RNG anchors are needed in this experiment.
            self.reference_cache = None
            prefix = ref[:, :self.fork_frames]
            future = ref[:, self.fork_frames:]
            rows = {}
            for name, call_index in zip(ARMS, (None, 0, self.calls_per_chunk - 2, None), strict=True):
                restore_rng_state(entry)
                candidate, calls, end_hash, row = self._run_policy(name, call_index)
                if calls != ref_calls or end_hash != ref_end or not torch.equal(candidate[:, :self.fork_frames], prefix):
                    raise ValueError(f"{name}: unpaired RNG, call schedule or unexposed prefix")
                if row["future_query_key_pairs_all_heads_layers"] != cost["future_query_key_pairs_all_heads_layers"]:
                    raise ValueError(f"{name}: future attention scope changed")
                suffix = candidate[:, self.fork_frames:]
                errors = _latent_errors(suffix, future, self.frames)
                if not all(math.isfinite(error) for error in errors):
                    raise ValueError(f"{name}: non-finite latent comparison")
                row.update(future_chunk_relative_l2=errors, future_sha256=_digest(suffix))
                rows[name] = row
            if len({rows[name]["total_ideal_aged_read_bytes"] for name in ARMS[1:]}) != 1:
                raise ValueError("the protected-read and uniform arms differ in read budget")
            return {
                "schema_version": SCHEMA,
                "settings": asdict(self.settings),
                "key_coordinate": "native_post_rope",
                "value_coordinate": "raw",
                "denoise_steps": self.calls_per_chunk - 1,
                "reference_prefix_sha256": _digest(prefix),
                "reference_suffix_sha256": _digest(future),
                "future_rng_fingerprint": ref_end,
                "branches": rows,
            }, ref
        finally:
            restore_rng_state(reference_end)
