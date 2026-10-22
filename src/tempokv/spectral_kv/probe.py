"""Suite B observer and explicitly enabled experimental trajectory interventions."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from tempokv.spectral.sensitivity import effective_positions
from tempokv.spectral.transforms import grid_shape, spatial_layout
from .attention import spectral_attention_fast, spectral_attention_ref
from .cache import SpectralKVCache
from .calibration import CalibrationMap, LinearCalibration
from .config import SpectralKVConfig
from .context import RunContext
from .method_spec import parse_method
from .model_spec import ModelSpec
from .reporting import ReportSink
from .rope3d import full_positions, rope_apply, rope_invert
from .shadow import ShadowAttention, frame_mass, make_policy
from .stats import ChunkAxisStats, DistanceDecay, KVRefreshDiff, OnlineSpectrum
from .temporal import OnlineTemporalSpectrum


class SpectralKVProbe:
    def __init__(self, config: SpectralKVConfig, output_dir, *, rank=0, layers=1, metadata=None):
        self.config, self.output_dir, self.rank, self.layers = config, output_dir, rank, layers
        self.metadata = dict(metadata or {})
        self.method = parse_method(config.method)
        self.calibration = CalibrationMap.load(config.calibration_file) if config.calibration_file else None
        self.context = RunContext()
        self.phase, self.active = "denoise", False

    def reset(self, *, prompt_id=0, batch_size=1, metadata=None):
        if self.active:
            raise RuntimeError("flush suite B before resetting")
        self.sink = ReportSink(self.output_dir, self.config, rank=self.rank, prompt_id=prompt_id,
                               metadata={**self.metadata, **(metadata or {})})
        self.batch_size, self.prompt_id = batch_size, prompt_id
        c = self.config
        self.spectrum = OnlineSpectrum(c.spatial_bins, c.eps, max_rows=c.max_stat_rows)
        self.temporal = OnlineTemporalSpectrum(c.temporal_window, c.spatial_bins, c.temporal_bins,
                                               c.feature_channels, max_rows=c.max_stat_rows, eps=c.eps)
        self.distance = DistanceDecay(c.max_lag, c.spatial_bins, c.correlation_threshold,
                                      c.feature_channels, c.eps, max_rows=c.max_stat_rows)
        self.chunk = ChunkAxisStats(c.spatial_bins, c.temporal_bins, c.feature_channels, c.eps, max_rows=c.max_stat_rows)
        self.refresh = KVRefreshDiff(c.spatial_bins, c.eps, max_rows=c.max_stat_rows)
        self.linear = LinearCalibration(c.calibration_ridge)
        self.shadow = ShadowAttention(c.shadow_methods, recent=c.keep_recent_frames, sink=c.sink_frames,
                                      backend=c.attention_backend, eps=c.eps)
        self.specs, self.freqs, self.precisions, self.caches, self.grids = {}, {}, {}, {}, {}
        self.last_calibration = {}
        self.calls, self.tokens_attended = Counter(), Counter()
        self.context, self.phase, self.active = RunContext(), "denoise", True
        return self.sink.root

    def set_call(self, context: RunContext):
        self.context, self.phase = context, context.phase

    def _selected(self, layer):
        return not self.config.layers or layer in self.config.layers

    def _heads(self, count):
        heads = self.config.heads or tuple(range(count))
        if max(heads) >= count:
            raise ValueError("suite B configured head is outside runtime shape")
        return heads

    def _step_selected(self):
        steps = self.config.denoise_steps
        return not steps or self.context.step_idx in steps or (-1 in steps and self.context.is_last_denoise)

    def _base(self):
        return {"chunk_idx": self.context.chunk_idx, "step_idx": self.context.step_idx,
                "phase": self.phase, "frame_idx0": self.context.frame_idx0}

    def _emit(self, experiment, rows, heads=None, **extra):
        annotated = []
        for row in rows:
            row = {**row, **extra}
            if heads is not None and "head" in row:
                row["head"] = int(heads[int(row["head"])])
            annotated.append(row)
        self.sink.write(experiment, annotated)

    def _spec(self, layer, k, grids):
        f, h, w = grid_shape(k, grids)
        channels = k.shape[-1] // 2
        split = (2 * (channels - 2 * (channels // 3)), 2 * (channels // 3), 2 * (channels // 3))
        spec = ModelSpec(self.layers, k.shape[2], k.shape[3], (h, w), f, split)
        if layer in self.specs and self.specs[layer].frame_hw != spec.frame_hw:
            raise ValueError("suite B grid changed during inference")
        self.specs[layer], self.grids[layer] = spec, grids
        return spec

    @torch.no_grad()
    def on_projection(self, layer_idx, *, k_pre, v, grid_sizes, current_start, phase=None,
                      q_pre=None, freqs=None):
        if not self.active:
            return
        spec = self._spec(layer_idx, k_pre, grid_sizes)
        self.freqs[layer_idx] = freqs
        if k_pre.shape != v.shape or k_pre.shape[0] != self.batch_size:
            raise ValueError("suite B K/V shape mismatch")
        fs = spec.frame_hw[0] * spec.frame_hw[1]
        if int(current_start) % fs:
            raise ValueError("current_start must align to runtime spatial grid")
        self.context = replace(self.context, frame_idx0=int(current_start) // fs)
        if self.config.online and self.method.family not in {"full", "kvskip"}:
            if layer_idx not in self.caches:
                self.caches[layer_idx] = SpectralKVCache(
                    spec, layer_idx, make_policy(self.method, recent=self.config.keep_recent_frames, sink=self.config.sink_frames),
                    self.method.kind, self.config.keep_originals, freqs=freqs, taper=self.method.taper,
                    sink_frames=self.config.sink_frames)
            self.caches[layer_idx].write_chunk(k_pre, v, self.context.frame_idx0)
        if not self._selected(layer_idx) or self.context.chunk_idx >= self.config.max_chunks:
            return
        need_clean = self.phase == "clean" and self.context.frame_idx0 < self.config.max_clean_frames
        need_last = self.context.is_last_denoise and ("B_E2B_REFRESH" in self.config.experiments or self.config.collect_calibration)
        if not need_clean and not need_last:
            return
        heads = self._heads(spec.heads)
        with self.sink.time("analysis", k_pre.device):
            data = {role: spatial_layout(tensor, grid_sizes)[..., list(heads), :].detach().float().cpu()
                    for role, tensor in (("k", k_pre), ("v", v))}
            keep = min(data["k"].shape[1], max(0, self.config.max_clean_frames - self.context.frame_idx0))
            for sample in range(self.batch_size):
                kk, vv = data["k"][sample], data["v"][sample]
                if need_last:
                    self.refresh.stash(layer_idx, kk, vv, frame_idx0=self.context.frame_idx0, sample_id=sample, **self._base_without_frame())
                    if self.config.collect_calibration:
                        self.last_calibration[layer_idx, sample] = (self.context.frame_idx0, kk, vv, heads)
                if need_clean:
                    for role, values in (("k", kk[:keep]), ("v", vv[:keep])):
                        if "B_E1_SPECTRUM" in self.config.experiments:
                            self._emit("B_E1_SPECTRUM", self.spectrum.update(
                                layer_idx, values, frame_idx0=self.context.frame_idx0, role=role,
                                coordinate="pre" if role == "k" else "raw", sample_id=sample,
                                **self._base_without_frame()), heads, record_kind="spatial")
                            self._emit("B_E1_SPECTRUM", self.temporal.update(
                                layer_idx, values, frame_idx0=self.context.frame_idx0, role=role, sample_id=sample,
                                is_clean_pass=True, **self._base_without_frame()), heads, record_kind="temporal")
                        if "B_E4_DISTANCE" in self.config.experiments:
                            self._emit("B_E4_DISTANCE", self.distance.update(
                                layer_idx, values, frame_idx0=self.context.frame_idx0, role=role,
                                sample_id=sample, **self._base_without_frame()), heads)
                    if "B_E2B_REFRESH" in self.config.experiments:
                        self._emit("B_E2B_REFRESH", self.refresh.update(layer_idx, kk, vv,
                                   frame_idx0=self.context.frame_idx0, sample_id=sample, **self._base_without_frame()), heads)
                    if self.config.collect_calibration:
                        previous = self.last_calibration.pop((layer_idx, sample), None)
                        if previous and previous[0] == self.context.frame_idx0 and previous[1].shape == kk.shape:
                            n = min(self.config.calibration_samples_per_chunk, kk.numel() // (len(heads) * spec.head_dim))
                            indices = np.linspace(0, kk.numel() // (len(heads) * spec.head_dim) - 1, n).round().astype(int)
                            for role, last, clean in (("k", previous[1], kk), ("v", previous[2], vv)):
                                self.linear.update(layer_idx, role, last.reshape(-1, len(heads), spec.head_dim)[indices],
                                                   clean.reshape(-1, len(heads), spec.head_dim)[indices], head_ids=heads)

    def _base_without_frame(self):
        return {key: value for key, value in self._base().items() if key != "frame_idx0"}

    @torch.no_grad()
    def on_block(self, layer_idx, output):
        if (not self.active or "B_E2_CHUNK_AXIS" not in self.config.experiments or not self._selected(layer_idx)
                or self.phase != "denoise" or not self._step_selected() or self.context.chunk_idx >= self.config.max_chunks):
            return
        spec = self.specs[layer_idx]
        if output.ndim != 3 or output.shape[-1] != spec.heads * spec.head_dim:
            raise ValueError("block feature shape disagrees with runtime heads/dim")
        heads = self._heads(spec.heads)
        with self.sink.time("analysis", output.device):
            values = spatial_layout(output.reshape(*output.shape[:2], spec.heads, spec.head_dim), self.grids[layer_idx])
            values = values[..., list(heads), :].detach().float().cpu()
            for sample in range(output.shape[0]):
                self._emit("B_E2_CHUNK_AXIS", self.chunk.update(layer_idx, values[sample],
                           chunk_idx=self.context.chunk_idx, step_idx=self.context.step_idx, sample_id=sample,
                           phase=self.phase, feature_source="transformer_block_output"), heads)

    def _positions(self, q, k, metadata):
        return effective_positions(tokens=k.shape[1], current_start=int(metadata["current_start"]),
                                   local_start_index=int(metadata["local_start_index"]),
                                   attention_start=int(metadata["attention_start"]), frame_seqlen=int(metadata["frame_seqlen"]),
                                   sink_frames=int(metadata.get("sink_frames", 0)),
                                   contiguous=metadata.get("contiguous", True), absolute_positions=metadata.get("absolute_positions"))

    @torch.no_grad()
    def on_attention(self, layer_idx, *, q_post, k_post, v, metadata, attention_fn):
        if not self.active:
            return attention_fn(q_post, k_post, v)
        spec = self.specs[layer_idx]
        positions = self._positions(q_post, k_post, metadata)
        runtime_precision = metadata.get("rope_precision", "float64")
        precision = ("float32" if runtime_precision in {"complex64", "float32"} else "float64") if self.config.rope_precision == "runtime" else self.config.rope_precision
        self.precisions[layer_idx] = precision
        fs = spec.frame_hw[0] * spec.frame_hw[1]
        current_frames = set(range(self.context.frame_idx0, self.context.frame_idx0 + q_post.shape[1] // fs))
        if not np.isin(np.arange(metadata["current_start"], metadata["current_start"] + q_post.shape[1]), positions).all():
            raise ValueError("spectral KV requires the complete current chunk in the effective window")
        if self.config.online and layer_idx in self.caches:
            cache = self.caches[layer_idx]
            cache.precision = precision
            active_frames = set((positions // fs).tolist())
            cache.evict_outside(active_frames)
            masses = None
            if self.method.topk is not None:
                indices = torch.linspace(0, q_post.shape[1] - 1, min(self.config.query_tokens, q_post.shape[1]), device=q_post.device).round().long()
                masses = frame_mass(q_post.index_select(1, indices), k_post, positions // fs)
            with self.sink.time("cache_transform", q_post.device):
                cache.age(self.context.chunk_idx, q_probe=masses, current_frames=current_frames)
                view = cache.materialize(self.context.chunk_idx, bias=self.method.bias)
            with self.sink.time("attention", q_post.device):
                actual = (spectral_attention_fast if self.config.attention_backend == "group_lse" else spectral_attention_ref)(q_post, view)
            if "B_E3_ON_ROLLOUT" in self.config.experiments:
                native_bytes = metadata.get("native_cache_allocated_bytes", (k_post.numel() + v.numel()) * k_post.element_size())
                self.sink.write("B_E3_ON_ROLLOUT", [{**self._base(), "layer": layer_idx, "sample_id": "batch",
                    "method_spec": self.method.name, "tokens_attended": view.token_count, "full_tokens": k_post.shape[1],
                    "actual_retention": view.token_count / k_post.shape[1], "view_resident_bytes": cache.resident_bytes,
                    "baseline_native_cache_bytes": native_bytes, "resident_bytes": native_bytes + cache.resident_bytes,
                    "compression_vs_full_bf16": native_bytes / (native_bytes + cache.resident_bytes),
                    "relative_attention_output_error": None, "status": "ok", "trajectory": "experimental_online",
                    "attention_backend": self.config.attention_backend}])
        else:
            with self.sink.time("attention", q_post.device):
                actual = attention_fn(q_post, k_post, v)
            if "B_E3_ON_ROLLOUT" in self.config.experiments:
                native_bytes = metadata.get("native_cache_allocated_bytes", (k_post.numel() + v.numel()) * k_post.element_size())
                self.sink.write("B_E3_ON_ROLLOUT", [{**self._base(), "layer": layer_idx, "sample_id": "batch",
                    "method_spec": self.method.name, "tokens_attended": k_post.shape[1], "full_tokens": k_post.shape[1],
                    "actual_retention": 1.0, "view_resident_bytes": 0, "baseline_native_cache_bytes": native_bytes,
                    "resident_bytes": native_bytes, "compression_vs_full_bf16": 1.0,
                    "relative_attention_output_error": 0.0 if self.method.family == "full" else None,
                    "status": "ok", "trajectory": "native_full_control" if self.method.family == "full" else "experimental_kvskip"}])
        if not self._selected(layer_idx):
            return actual
        heads = self._heads(spec.heads)
        if self.phase == "clean" and "B_E1_SPECTRUM" in self.config.experiments and self.context.frame_idx0 < self.config.max_clean_frames:
            offset = int(metadata["local_start_index"]) - int(metadata["attention_start"])
            with self.sink.time("analysis", k_post.device):
                current_k = k_post[:, offset:offset + q_post.shape[1]]
                data = spatial_layout(current_k, self.grids[layer_idx])[..., list(heads), :].detach().float().cpu()
                keep = min(data.shape[1], self.config.max_clean_frames - self.context.frame_idx0)
                for sample in range(data.shape[0]):
                    self._emit("B_E1_SPECTRUM", self.spectrum.update(layer_idx, data[sample, :keep],
                               frame_idx0=self.context.frame_idx0, role="k", coordinate="post", sample_id=sample,
                               **self._base_without_frame()), heads, record_kind="spatial")
        eligible = (self.phase == "denoise" and self._step_selected()
                    and (not self.config.chunks or self.context.chunk_idx in self.config.chunks)
                    and self.calls[layer_idx] < self.config.max_analysis_calls_per_layer)
        if eligible and "B_E3_OFF_SHADOW" in self.config.experiments:
            with self.sink.time("analysis", q_post.device):
                for sample in range(q_post.shape[0]):
                    for head in heads:
                        seed = int.from_bytes(hashlib.blake2b(json.dumps([self.config.seed, str(self.prompt_id), sample,
                                               layer_idx, head, self.context.chunk_idx, self.context.step_idx]).encode(), digest_size=4).digest(), "little")
                        qi = np.sort(np.random.default_rng(seed).choice(q_post.shape[1], min(self.config.query_tokens, q_post.shape[1]), replace=False))
                        q = q_post[sample:sample + 1, :, head:head + 1].index_select(1, torch.as_tensor(qi, device=q_post.device))
                        rows = self.shadow.evaluate(q, k_post[sample:sample + 1, :, head:head + 1], v[sample:sample + 1, :, head:head + 1],
                                   replace(spec, heads=1), layer=layer_idx, absolute_positions=positions, current_frames=current_frames,
                                   chunk_idx=self.context.chunk_idx, freqs=self.freqs[layer_idx], precision=precision, attention_fn=attention_fn)
                        self._emit("B_E3_OFF_SHADOW", [{**row, **self._base(), "layer": layer_idx, "head": head,
                                  "sample_id": sample, "query_indices": qi.tolist()} for row in rows])
            self.calls[layer_idx] += 1
        return actual

    @torch.no_grad()
    def skip_clean(self, kv_cache):
        if self.method.skip == "lin":
            if self.calibration is None:
                raise ValueError("KVSKIP_lin has no calibration map")
            for layer, cache in enumerate(kv_cache):
                spec = self.specs[layer]
                heads = tuple(range(spec.heads))
                required = {(layer, role, head) for role in ("k", "v") for head in heads}
                if not required.issubset(set(self.calibration.keys)):
                    raise ValueError(f"KVSKIP_lin calibration lacks runtime coverage for layer {layer}")
                end = int(cache["local_end_index"])
                tokens = spec.chunk_frames * spec.frame_hw[0] * spec.frame_hw[1]
                positions = full_positions(self.context.frame_idx0, spec.chunk_frames, spec.frame_hw, device=cache["k"].device)
                precision = self.precisions.get(layer, "float64")
                with self.sink.time("cache_transform", cache["k"].device):
                    k = rope_invert(cache["k"][:, end - tokens:end], *positions, spec, freqs=self.freqs[layer], precision=precision)
                    k = self.calibration.apply(layer, "k", k)
                    value = self.calibration.apply(layer, "v", cache["v"][:, end - tokens:end])
                    cache["k"][:, end - tokens:end] = rope_apply(k, *positions, spec, freqs=self.freqs[layer], precision=precision)
                    cache["v"][:, end - tokens:end] = value
        self.sink.skipped["clean_passes_skipped"] += 1
        self.last_calibration.clear()
        self.refresh._stashes.clear()
        self.sink.write("B_E2B_REFRESH", [{**self._base(), "method_spec": self.method.name,
                        "status": "not measured", "reason": "clean pass skipped; evaluate paired rollout drift separately"}])

    def flush(self, *, status="completed", error=None):
        if not self.active:
            return
        try:
            for exp, stats, name in (("B_E1_SPECTRUM", self.spectrum, "spatial_stats.npz"),
                                     ("B_E1_SPECTRUM", self.temporal, "temporal_stats.npz"),
                                     ("B_E2_CHUNK_AXIS", self.chunk, "stats.npz"),
                                     ("B_E2B_REFRESH", self.refresh, "stats.npz"),
                                     ("B_E4_DISTANCE", self.distance, "stats.npz")):
                self.sink.save_state(exp, name, stats.state_dict())
            if self.config.collect_calibration:
                with (self.sink.root / "calibration_state.npz").open("xb") as handle:
                    np.savez_compressed(handle, **self.linear.state_dict())
            grids = {layer: spec.frame_hw for layer, spec in self.specs.items()}
            return self.sink.close(status=status, error=error, extra={"effective_grids": grids,
                                   "method_spec": self.method.to_dict(), "calibration_collect": self.config.collect_calibration,
                                   "coordinates_contract": "statistics pre/post explicitly labeled; all summaries in pre-RoPE coordinates"})
        finally:
            self.caches.clear()
            self.last_calibration.clear()
            self.active = False
