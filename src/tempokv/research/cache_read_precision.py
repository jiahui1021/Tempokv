"""Causal placement of a fixed high-precision read budget on native Self-Forcing."""

from __future__ import annotations

from dataclasses import asdict
import math

import torch

from .cache_recovery import ROLES, RecoveryRollout, _digest, _kv_bytes, _latent_errors, _restore_kv
from .precision import fake_quantize
from .rollout import capture_rng_state, restore_rng_state

SCHEMA = "research.cache_read_precision.v1"
UNIFORM = "uniform_equal_payload"


def read_schedules(denoise_steps):
    """Each tuple contains the calls allowed to read the original BF16 history."""
    return {
        "rounded_all": (),
        "rounded_write_only": tuple(range(denoise_steps)),
        "protect_clean": (denoise_steps,),
        **{f"protect_denoise_{i + 1}": (i,) for i in range(denoise_steps)},
    }


def _assert_history(cache, native, stop):
    for layer, pair in zip(cache, native, strict=True):
        for role, expected in zip(ROLES, pair, strict=True):
            if not torch.equal(layer[role][:, :stop], expected):
                raise ValueError(f"history before the last prefix chunk differs for {role}")


def _last_write_errors(cache, reference, start):
    rows = []
    for i, (layer, pair) in enumerate(zip(cache, reference, strict=True)):
        for role, expected in zip(ROLES, pair, strict=True):
            actual = layer[role][:, start:start + expected.shape[1]].float()
            expected = expected.float()
            error = float((actual - expected).square().sum(dtype=torch.float64))
            energy = float(expected.square().sum(dtype=torch.float64))
            if not math.isfinite(error) or not math.isfinite(energy):
                raise ValueError("non-finite clean-write cache comparison")
            rows.append({
                "layer": i, "role": role, "error_energy": error, "reference_energy": energy,
                "relative_l2": math.sqrt(error / max(energy, 1e-24)),
            })
    return rows


def _trajectory_contrast(left, right, reference, frames):
    error = (left - right).split(frames, dim=1)
    base = reference.split(frames, dim=1)
    return [float(a.norm() / b.norm().clamp_min(1e-12)) for a, b in zip(error, base, strict=True)]


class ReadPrecisionRollout(RecoveryRollout):
    def __init__(self, pipeline, infer, settings, noise, prompts):
        super().__init__(pipeline, infer, settings, noise, prompts)
        if settings.first_round_chunk != settings.fork_chunk - 1:
            raise ValueError("read precision study requires exactly one exposed prefix chunk")
        self.target_start = self.fork_tokens - self.tokens
        self.aged_end = self.target_start - settings.protect_chunks * self.tokens

    def _uniform_history(self, aged):
        if self.settings.bits == 16:
            return aged, {"lower_bits": 16, "higher_bits": 16, "higher_positions": 0}
        # The uniform arm reads one additional set of FP32 scales compared to
        # a single BF16 pass. Deduct that set from the coefficient budget.
        width = aged[0][0].shape[-1]
        scales = math.ceil(width / self.settings.group_size)
        total_bits = width * (16 + (self.calls_per_chunk - 1) * self.settings.bits) - 32 * scales
        denominator = self.calls_per_chunk * width
        low, remainder = divmod(total_bits, denominator)
        high = low + int(remainder > 0)
        count, leftover = divmod(self.aged_end * remainder, denominator)
        if leftover or not 2 <= low <= high <= 8:
            raise ValueError("this rollout cannot match the uniform packed payload exactly")
        position = torch.arange(self.aged_end, device=self.noise.device)
        chosen = position[
            (position + 1) * count // self.aged_end > position * count // self.aged_end
        ]
        uniform = []
        for pair in aged:
            rebuilt = []
            for value in pair:
                result = fake_quantize(value, low, self.settings.group_size).value
                if count:
                    original = value.index_select(1, chosen)
                    result.index_copy_(1, chosen, fake_quantize(original, high, self.settings.group_size).value)
                rebuilt.append(result)
            uniform.append(tuple(rebuilt))
        return tuple(uniform), {"lower_bits": low, "higher_bits": high, "higher_positions": count}

    def _history_states(self):
        device = self.noise.device
        native = tuple(
            tuple(t[:, :self.target_start].to(device) for t in pair)
            for pair in self.reference_cache.kv
        )
        aged = tuple(tuple(t[:, :self.aged_end] for t in pair) for pair in native)
        rounded = tuple(
            tuple(fake_quantize(t, self.settings.bits, self.settings.group_size).value for t in pair)
            for pair in aged
        )
        last_write = tuple(
            tuple(t[:, self.target_start:self.fork_tokens].to(device) for t in pair)
            for pair in self.reference_cache.kv
        )
        return native, aged, rounded, last_write

    def _scheduled_run(self, name, protected, history, uniform=None):
        native, aged, rounded, last_write = history
        if uniform is not None:
            rounded, uniform_precision = uniform
        read_rows = []
        write_rows = []
        is_native = True

        def intervene(start, index, kwargs):
            nonlocal is_native
            cache = kwargs["kv_cache"]
            if start == self.target_start:
                if index == 0:
                    _assert_history(cache, native, self.target_start)
                use_native = index in protected
                if use_native != is_native:
                    _restore_kv(cache, aged if use_native else rounded, 0)
                    is_native = use_native
                timestep = float(kwargs["timestep"].flatten()[0])
                read_rows.append({
                    "call_index": index, "timestep": timestep, "original_aged_history": use_native,
                })
            if start == self.fork_tokens and index == 0:
                if not is_native:
                    _restore_kv(cache, aged, 0)
                _assert_history(cache, native, self.target_start)
                write_rows.extend(_last_write_errors(cache, last_write, self.target_start))

        latents, calls, end_hash, cost = self._run(name, self.noise, before_forward=intervene)
        steps = [float(t) for t in self.pipeline.denoising_step_list] + [0.0]
        if [r["timestep"] for r in read_rows] != steps:
            raise ValueError("unexpected denoising or clean-write call schedule")
        if tuple(r["call_index"] for r in read_rows if r["original_aged_history"]) != protected:
            raise ValueError("protected call schedule was not applied")
        if len(write_rows) != 2 * len(native):
            raise ValueError("missing restored fork cache comparison")
        native_bytes = _kv_bytes(aged)
        elements = native_bytes // 2
        width = aged[0][0].shape[-1]
        scale_bytes = 0 if self.settings.bits == 16 else (
            elements // width * math.ceil(width / self.settings.group_size) * 4
        )
        rounded_bytes = (elements * self.settings.bits + 7) // 8 + scale_bytes
        extra_reads = 1 if uniform is not None else len(protected)
        row = {
            "protected_aged_history_calls": len(protected),
            "extra_ideal_cache_read_bytes": extra_reads * (native_bytes - rounded_bytes),
            "future_query_key_pairs_all_heads_layers": cost["future_query_key_pairs_all_heads_layers"],
            "target_reads": read_rows,
            "last_write_cache_rows": write_rows,
        }
        if uniform is not None:
            row["uniform_precision"] = uniform_precision
        return latents, calls, end_hash, row

    @torch.no_grad()
    def run(self):
        entry = capture_rng_state()
        reference_end = entry
        try:
            ref, ref_calls, ref_end, cost = self._run("reference", self.noise)
            reference_end = capture_rng_state()
            history = self._history_states()
            uniform = self._uniform_history(history[1])
            ref_future = ref[:, self.fork_frames:].cpu().float()
            ref_prefix = ref[:, :self.fork_frames]
            ref_future_hash = _digest(ref[:, self.fork_frames:])
            ref_prefix_hash = _digest(ref_prefix)
            rows, futures = {}, {}
            schedules = read_schedules(self.calls_per_chunk - 1)
            schedules[UNIFORM] = ()
            for name, protected in schedules.items():
                restore_rng_state(entry)
                latents, calls, end, row = self._scheduled_run(
                    name, protected, history, uniform if name == UNIFORM else None,
                )
                if calls != ref_calls or end != ref_end:
                    raise ValueError(f"{name}: unpaired RNG or absolute forward schedule")
                if row["future_query_key_pairs_all_heads_layers"] != cost["future_query_key_pairs_all_heads_layers"]:
                    raise ValueError(f"{name}: future attention budget changed")
                prefix = latents[:, :self.fork_frames]
                if not torch.equal(prefix[:, :-self.frames], ref_prefix[:, :-self.frames]):
                    raise ValueError(f"{name}: exposure changed a preceding prefix chunk")
                suffix = latents[:, self.fork_frames:]
                futures[name] = suffix.cpu().float()
                row.update(
                    prefix_sha256=_digest(prefix),
                    future_sha256=_digest(suffix),
                    last_prefix_latent_relative_l2=_latent_errors(
                        prefix[:, -self.frames:], ref_prefix[:, -self.frames:], self.frames,
                    )[0],
                    future_chunk_relative_l2=_latent_errors(futures[name], ref_future, self.frames),
                )
                if not all(math.isfinite(e) for e in row["future_chunk_relative_l2"]):
                    raise ValueError(f"{name}: non-finite latent comparison")
                rows[name] = row

            writer = rows["rounded_write_only"]
            if writer["prefix_sha256"] != ref_prefix_hash or any(
                r["error_energy"] != 0 for r in writer["last_write_cache_rows"] if r["layer"] == 0
            ):
                raise ValueError("writer-only exposure changed its input latent or first-layer K/V")
            if rows["rounded_all"]["prefix_sha256"] != rows["protect_clean"]["prefix_sha256"]:
                raise ValueError("protecting only the clean write changed the denoised prefix")
            if self.settings.bits == 16 and any(
                r["future_sha256"] != ref_future_hash or r["prefix_sha256"] != ref_prefix_hash for r in rows.values()
            ):
                raise ValueError("native-precision identity control changed a trajectory")

            both = futures["rounded_all"]
            write = futures["rounded_write_only"]
            denoise = futures["protect_clean"]
            return {
                "schema_version": SCHEMA,
                "settings": asdict(self.settings),
                "key_coordinate": "native_post_rope",
                "value_coordinate": "raw",
                "exposed_chunk": self.settings.fork_chunk - 1,
                "aged_positions_per_head": self.aged_end,
                "active_positions_at_fork": self.fork_tokens,
                "denoise_steps": self.calls_per_chunk - 1,
                "reference_prefix_sha256": ref_prefix_hash,
                "reference_suffix_sha256": ref_future_hash,
                "future_rng_fingerprint": ref_end,
                "branches": rows,
                "factorial_trajectory_contrasts": {
                    "writer_given_native_denoise": _trajectory_contrast(write, ref_future, ref_future, self.frames),
                    "writer_given_rounded_denoise": _trajectory_contrast(both, denoise, ref_future, self.frames),
                    "denoise_given_native_write": _trajectory_contrast(denoise, ref_future, ref_future, self.frames),
                    "denoise_given_rounded_write": _trajectory_contrast(both, write, ref_future, self.frames),
                },
            }, ref
        finally:
            restore_rng_state(reference_end)
