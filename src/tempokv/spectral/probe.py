"""Bounded online clean-feature statistics and read-only attention diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from .compression import evaluate_budgets, policy_scores, save_snapshot
from .config import SpectralConfig
from .metrics import score_metrics
from .sensitivity import Block, effective_positions, leave_one_block_out, reference_attention
from .spectrum import cosine_similarity, spatial_energy, temporal_energy
from .transforms import grid_shape, spatial_blocks, spatial_layout

COMMON = ("prompt_id", "sample_id", "rank", "inference_id", "layer", "head", "call_id",
          "denoise_call", "query_frame", "phase", "grid_h", "grid_w", "frame_seqlen",
          "effective_k_tokens", "current_start", "local_start_index", "attention_start")
SPATIAL = ("prompt_id", "sample_id", "rank", "layer", "head", "frame", "role", "cutoff",
           "ratio", "total_energy", "near_zero_energy", "grid_h", "grid_w", "phase")
SCORES = (*COMMON, "history_frame", "spatial_block_id", "absolute_token_range", "absolute_token_indices",
          "block_tokens", "freq_k_high_ratio", "freq_v_high_ratio", "temporal_similarity",
          "attention_mass", "oracle_error", "frame_age", "score", "query_token_indices",
          "candidate_scope", "candidate_count", "eligible_candidate_count", "reference_norm")
CORRELATION = (*COMMON, "score_name", "combination_alpha", "spearman", "topk_recall_sampled",
               "finite_candidates", "sampled_candidates", "excluded_missing_candidates", "topk_fraction", "candidate_scope")
BUDGET = (*COMMON, "policy", "retention_ratio", "random_seed", "combination_alpha", "candidate_tokens",
          "target_kept_candidate_tokens", "untouched_tokens", "kept_candidate_tokens", "actual_retained_tokens",
          "partial_block_tokens", "relative_attention_output_error", "nmse", "candidate_scope",
          "budget_granularity", "status", "total_history_tokens", "actual_retained_history_tokens",
          "actual_history_retention", "protected_tokens", "query_token_indices")


class SpectralProbe:
    def __init__(self, config: SpectralConfig, output_dir: str | Path, *, rank: int = 0,
                 metadata: dict | None = None) -> None:
        self.config, self.output_dir, self.rank = config, Path(output_dir), rank
        self.initial_metadata = dict(metadata or {})
        self.active = False
        self.phase, self.denoise_call = "denoise", 0
        self._sequence = 0

    def reset(self, *, prompt_id: int | str = 0, batch_size: int = 1, metadata: dict | None = None):
        if self.active:
            raise RuntimeError("flush the previous inference before resetting the spectral probe")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._sequence += 1
        self.phase, self.denoise_call = "denoise", 0
        self.inference_id = f"{self._sequence:04d}-{uuid.uuid4().hex[:12]}"
        self.prompt_id = prompt_id
        self.run_dir = self.output_dir / f"rank-{self.rank}" / f"inference-{self.inference_id}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.sample_dirs, self.writers, self.handles = {}, {}, []
        self.extra_metadata = {**self.initial_metadata, **(metadata or {})}
        self.analysis_seconds = 0.0
        self.peak_gpu_allocated_bytes = 0
        self.counts = Counter()
        self.sample_counts = defaultdict(Counter)
        self.windows = []
        self.grids = {}
        self._features = {}
        self._previous = {}
        self._temporal = defaultdict(list)
        self._clean_frames = defaultdict(set)
        self._calls = Counter()
        self._snapshots = 0
        for sample in range(batch_size):
            path = self.run_dir / f"sample-{sample}"
            path.mkdir()
            self.sample_dirs[sample] = path
            for name, columns in (("spectrum/spatial_energy.csv", SPATIAL),
                                  ("sensitivity/block_scores.csv", SCORES),
                                  ("sensitivity/correlation.csv", CORRELATION),
                                  ("compression/error_budget.csv", BUDGET)):
                target = path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = target.open("x", newline="", encoding="utf-8")
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                self.handles.append(handle)
                self.writers[sample, name] = writer
        self.active = True
        return self.run_dir

    def set_call(self, *, phase: str, denoise_call: int = 0):
        if phase not in {"clean", "denoise"}:
            raise ValueError("phase must be explicitly clean or denoise")
        self.phase, self.denoise_call = phase, denoise_call

    @contextmanager
    def _timed(self, device):
        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if is_cuda:
                torch.cuda.synchronize(device)
                self.peak_gpu_allocated_bytes = max(self.peak_gpu_allocated_bytes,
                                                   torch.cuda.max_memory_allocated(device))
            self.analysis_seconds += time.perf_counter() - started

    def _seed(self, *parts) -> int:
        value = json.dumps([self.config.seed, str(self.prompt_id), *parts]).encode()
        return int.from_bytes(hashlib.blake2b(value, digest_size=4).digest(), "little")

    def _write(self, sample, name, row):
        self.writers[sample, name].writerow(row)
        self.counts[name] += 1
        self.sample_counts[sample][name] += 1
        if name == "compression/error_budget.csv" and row.get("status") == "measured":
            self.sample_counts[sample]["compression_measured_rows"] += 1

    @torch.no_grad()
    def on_projection(self, layer_idx: int, *, k_pre, v, grid_sizes, current_start: int,
                      phase: str | None = None, q_pre=None, freqs=None):
        phase = self.phase if phase is None else phase
        if not self.active or layer_idx not in self.config.sampling.layers or phase != "clean":
            return
        with self._timed(k_pre.device):
            self._projection(layer_idx, k_pre, v, grid_sizes, int(current_start))

    def _projection(self, layer, k, v, grids, current_start):
        if k.shape != v.shape or k.shape[0] != len(self.sample_dirs):
            raise ValueError("clean K/V shape or batch changed during inference")
        f, h, w = grid_shape(k, grids)
        frame_tokens = h * w
        if current_start % frame_tokens:
            raise ValueError("clean current_start must be frame aligned")
        heads = self.config.sampling.heads
        if max(heads) >= k.shape[2]:
            raise ValueError("configured spectral head is outside the runtime head count")
        shape = (h, w, k.shape[2], k.shape[3])
        if layer in self.grids and self.grids[layer] != shape:
            raise ValueError("spatial grid/head shape changed within a clean trajectory")
        self.grids[layer] = shape
        layouts = {"k": spatial_layout(k, grids), "v": spatial_layout(v, grids)}
        positions = np.linspace(0, frame_tokens - 1,
                                min(frame_tokens, self.config.frequency.temporal_positions_per_frame)).round().astype(int)
        for fi in range(f):
            frame = current_start // frame_tokens + fi
            if frame in self._clean_frames[layer]:
                raise ValueError("duplicate clean frame; refusing to mix temporal trajectories")
            if len(self._clean_frames[layer]) >= self.config.frequency.max_clean_frames:
                self.counts["clean_frames_skipped_cap"] += 1
                continue
            self._clean_frames[layer].add(frame)
            for sample in self.sample_dirs:
                dense = {role: value[sample:sample + 1, fi:fi + 1, :, :, heads, :]
                         .detach().to(device="cpu", dtype=torch.float32) for role, value in layouts.items()}
                # These dense tensors contain the entire spatial grid, only sampled heads.
                previous = {}
                for role, value in dense.items():
                    prev = self._previous.get((sample, layer, role))
                    previous[role] = prev[1] if prev is not None and prev[0] == frame - 1 else None
                    current = value[0, 0]
                    if "spectrum" in self.config.experiments:
                        ratios, total, zero = spatial_energy(value, self.config.frequency.cutoffs, self.config.eps)
                        for hi, head in enumerate(heads):
                            for cutoff, ratio in ratios.items():
                                self._write(sample, "spectrum/spatial_energy.csv", {
                                    "prompt_id": self.prompt_id, "sample_id": sample, "rank": self.rank,
                                    "layer": layer, "head": head, "frame": frame, "role": role,
                                    "cutoff": cutoff, "ratio": float(ratio[0, 0, hi]),
                                    "total_energy": float(total[0, 0, hi]), "near_zero_energy": bool(zero[0, 0, hi]),
                                    "grid_h": h, "grid_w": w, "phase": "clean"})
                        trace = current.reshape(frame_tokens, len(heads), -1)[positions].clone()
                        self._temporal[sample, layer, role].append((frame, trace, positions))
                    self._previous[sample, layer, role] = (frame, current)
                for bid, rows, cols, _ in spatial_blocks(h, w, self.config.frequency.spatial_block_size):
                    high, similarities = {}, []
                    for role, value in dense.items():
                        patch = value[:, :, rows, cols]
                        ratios, _, _ = spatial_energy(patch, (self.config.frequency.score_cutoff,), self.config.eps)
                        high[role] = (1 - ratios[self.config.frequency.score_cutoff][0, 0]).numpy().copy()
                        prev = previous[role]
                        similarities.append(cosine_similarity(None if prev is None else prev[rows, cols],
                                                              patch[0, 0], self.config.eps))
                    sim = ((similarities[0] + similarities[1]) / 2).numpy().copy()
                    self._features[sample, layer, frame, bid] = (high["k"], high["v"], sim)

    @torch.no_grad()
    def on_attention(self, layer_idx: int, *, q_post, k_post, v, metadata: dict,
                     attention_fn=reference_attention):
        s = self.config.sampling
        phase = metadata.get("phase", self.phase)
        if (not self.active or layer_idx not in s.layers or phase != s.only_phase
                or not set(self.config.experiments).intersection({"sensitivity", "compression"})):
            return
        if s.denoise_calls and self.denoise_call not in s.denoise_calls:
            return
        f, h, w = grid_shape(q_post, metadata["grid_sizes"])
        start = int(metadata["current_start"])
        query_frames = [frame for frame in range(start // (h * w), start // (h * w) + f)
                        if not s.latent_frames or frame in s.latent_frames]
        if not query_frames:
            return
        if self._calls[layer_idx] >= s.max_oracle_calls_per_layer:
            self.counts["oracle_calls_skipped_cap"] += 1
            return
        if k_post.shape != v.shape or k_post.shape[0] != q_post.shape[0] or k_post.shape[2:] != q_post.shape[2:]:
            raise ValueError("effective attention Q/K/V shape mismatch")
        if q_post.shape[0] != len(self.sample_dirs) or max(s.heads) >= q_post.shape[2]:
            raise ValueError("attention batch/head selection disagrees with probe configuration")
        with self._timed(q_post.device):
            self._attention(layer_idx, q_post, k_post, v, metadata, query_frames, h, w, attention_fn)
        self._calls[layer_idx] += 1

    def _attention(self, layer, q, k, v, meta, query_frames, h, w, attention_fn):
        frame_tokens = h * w
        if int(meta["frame_seqlen"]) != frame_tokens:
            raise ValueError("frame_seqlen disagrees with runtime H*W")
        start = int(meta["current_start"])
        if "local_end_index" in meta and int(meta["local_end_index"]) - int(meta["attention_start"]) != k.shape[1]:
            raise ValueError("effective KV length disagrees with local cache window")
        positions = effective_positions(tokens=k.shape[1], current_start=start,
                                        local_start_index=int(meta["local_start_index"]),
                                        attention_start=int(meta["attention_start"]), frame_seqlen=frame_tokens,
                                        sink_frames=int(meta.get("sink_frames", 0)),
                                        contiguous=meta.get("contiguous", True),
                                        absolute_positions=meta.get("absolute_positions"))
        if not np.isin(np.arange(start, start + q.shape[1]), positions).all():
            raise ValueError("effective window does not include the complete current chunk")
        lookup = {int(token): index for index, token in enumerate(positions)}
        history_limit = start - self.config.oracle.protect_recent_frames * frame_tokens
        protected = np.flatnonzero(positions >= history_limit).tolist()
        geometry = []
        for frame in sorted(set((positions[positions < history_limit] // frame_tokens).tolist())):
            for bid, _, _, local_tokens in spatial_blocks(h, w, self.config.frequency.spatial_block_size):
                absolute = tuple(frame * frame_tokens + t for t in local_tokens)
                if all(token in lookup for token in absolute):
                    geometry.append((frame, bid, tuple(lookup[t] for t in absolute)))
        self.windows.append({"layer": layer, "call_id": self._calls[layer], "phase": "denoise",
                             "denoise_call": self.denoise_call, "current_start": start,
                             "local_start_index": int(meta["local_start_index"]),
                             "attention_start": int(meta["attention_start"]),
                             "effective_k_tokens": k.shape[1], "absolute_start": int(positions[0]),
                             "absolute_end_exclusive": int(positions[-1]) + 1,
                             "grid_sizes": [q.shape[1] // frame_tokens, h, w], "dtype": str(q.dtype),
                             "attention_backend": getattr(attention_fn, "__module__", "unknown"),
                             "softmax_scale": q.shape[-1] ** -0.5, "mask": "none (runtime sliced window)"})
        for sample in self.sample_dirs:
            for hi, head in enumerate(self.config.sampling.heads):
                eligible = []
                for frame, bid, indices in geometry:
                    feature = self._features.get((sample, layer, frame, bid))
                    if feature is not None:
                        eligible.append(Block(frame, bid, indices, *(float(x[hi]) for x in feature)))
                if not eligible:
                    self.counts["oracle_calls_without_clean_candidates"] += 1
                    continue
                for query_frame in query_frames:
                    seed = self._seed(sample, layer, head, query_frame, self.denoise_call)
                    rng = np.random.default_rng(seed)
                    choice = sorted(rng.choice(len(eligible), min(len(eligible), self.config.sampling.max_candidate_blocks), replace=False))
                    blocks = [eligible[int(i)] for i in choice]
                    query_offset = query_frame * frame_tokens - start
                    qi = np.sort(rng.choice(frame_tokens, min(frame_tokens, self.config.sampling.query_tokens_per_head), replace=False)) + query_offset
                    query = q[sample:sample + 1, :, head:head + 1].index_select(1, torch.as_tensor(qi, device=q.device))
                    key, value = k[sample:sample + 1, :, head:head + 1], v[sample:sample + 1, :, head:head + 1]
                    reference, scores = leave_one_block_out(query, key, value, blocks, attention_fn=attention_fn,
                                                           chunk_size=self.config.oracle.key_chunk_size, eps=self.config.eps)
                    base = {"prompt_id": self.prompt_id, "sample_id": sample, "rank": self.rank,
                            "inference_id": self.inference_id, "layer": layer, "head": head,
                            "call_id": self._calls[layer], "denoise_call": self.denoise_call,
                            "query_frame": query_frame, "phase": "denoise", "grid_h": h, "grid_w": w,
                            "frame_seqlen": frame_tokens, "effective_k_tokens": k.shape[1], "current_start": start,
                            "local_start_index": int(meta["local_start_index"]), "attention_start": int(meta["attention_start"])}
                    query_indices = json.dumps((qi + start).tolist())
                    reference_norm = float(torch.linalg.vector_norm(reference.float()))
                    for block, score in zip(blocks, scores):
                        absolute = positions[list(block.indices)].tolist()
                        self._write(sample, "sensitivity/block_scores.csv", {
                            **base, **score, "history_frame": block.history_frame, "spatial_block_id": block.spatial_block_id,
                            "absolute_token_range": f"[{min(absolute)},{max(absolute) + 1})",
                            "absolute_token_indices": json.dumps(absolute), "block_tokens": len(block.indices),
                            "freq_k_high_ratio": block.freq_k_high_ratio, "freq_v_high_ratio": block.freq_v_high_ratio,
                            "temporal_similarity": block.temporal_similarity, "frame_age": query_frame - block.history_frame,
                            "score": (block.freq_k_high_ratio + block.freq_v_high_ratio) / 2,
                            "query_token_indices": query_indices, "candidate_scope": "sampled_candidate_set",
                            "candidate_count": len(blocks), "eligible_candidate_count": len(eligible),
                            "reference_norm": reference_norm})
                    error = [row["oracle_error"] for row in scores]
                    score_values = {
                        "freq_k_high_ratio": [b.freq_k_high_ratio for b in blocks],
                        "freq_v_high_ratio": [b.freq_v_high_ratio for b in blocks],
                        "temporal_similarity": [b.temporal_similarity for b in blocks],
                        "frame_age": [query_frame - b.history_frame for b in blocks],
                        **{policy: policy_scores(policy, blocks, scores) for policy in ("frequency", "similarity", "attention")}}
                    # Compare every score on the same finite subset. Otherwise
                    # missing first-frame similarity would change the ranking
                    # population only for the similarity baseline.
                    common = np.isfinite(np.asarray(error))
                    for values in score_values.values():
                        common &= np.isfinite(np.asarray(values))
                    metric_context = {"sampled_candidates": len(blocks),
                                      "excluded_missing_candidates": int((~common).sum())}
                    for name, values in score_values.items():
                        self._write(sample, "sensitivity/correlation.csv", {
                            **base, "score_name": name, "combination_alpha": "",
                            **score_metrics(np.where(common, values, np.nan), error, self.config.oracle.topk_fraction),
                            **metric_context})
                    for alpha in self.config.compression.combination_alphas:
                        self._write(sample, "sensitivity/correlation.csv", {
                            **base, "score_name": "frequency_attention", "combination_alpha": alpha,
                            **score_metrics(np.where(common, policy_scores("frequency_attention", blocks, scores, alpha), np.nan),
                                            error, self.config.oracle.topk_fraction), **metric_context})
                    if "compression" in self.config.experiments:
                        budgets = evaluate_budgets(query, key, value, blocks, scores, self.config.compression,
                                                   attention_fn=attention_fn, reference=reference, seed=seed,
                                                   protected_indices=protected, eps=self.config.eps)
                        history = int((positions < start).sum())
                        current = int((positions >= start).sum())
                        for budget in budgets:
                            kept_history = budget["actual_retained_tokens"] - current if budget["status"] == "measured" else ""
                            self._write(sample, "compression/error_budget.csv", {
                                **base, **budget, "total_history_tokens": history,
                                "actual_retained_history_tokens": kept_history,
                                "actual_history_retention": kept_history / history if history and kept_history != "" else float("nan"),
                                "protected_tokens": len(protected), "query_token_indices": query_indices})
                    if self._snapshots < self.config.compression.snapshot_calls:
                        if key.shape[1] <= self.config.compression.max_snapshot_tokens:
                            target = self.sample_dirs[sample] / "compression" / f"window-{self._snapshots}.npz"
                            save_snapshot(target, query, key, value, blocks, scores,
                                          {**base, "selection_seed": seed, "protected_indices": protected,
                                           "query_token_indices": (qi + start).tolist(),
                                           "absolute_positions": positions.tolist(), "config": self.config.to_dict()})
                            self._snapshots += 1
                        else:
                            self.counts["snapshots_skipped_token_cap"] += 1

    def flush(self, *, status: str = "completed", error: str | None = None):
        if not self.active:
            return
        flush_started = time.perf_counter()
        temporal_counts = {}
        try:
            for sample, path in self.sample_dirs.items():
                arrays = {"heads": np.asarray(self.config.sampling.heads),
                          "cutoffs": np.asarray(self.config.frequency.cutoffs)}
                measured = 0
                for (sid, layer, role), trace in self._temporal.items():
                    if sid != sample:
                        continue
                    frames = np.asarray([x[0] for x in trace])
                    # Never FFT over a gap or concatenate different spatial positions.
                    segments = np.split(np.arange(len(trace)), np.flatnonzero(np.diff(frames) != 1) + 1)
                    for segment_index, segment in enumerate(segments):
                        prefix = f"layer_{layer}_{role}_segment_{segment_index}"
                        values = torch.stack([trace[int(i)][1] for i in segment])
                        arrays[prefix + "_frames"] = frames[segment]
                        arrays[prefix + "_positions"] = trace[int(segment[0])][2]
                        arrays[prefix + "_samples"] = values.numpy()
                        if len(segment) >= 2:
                            freq, power, ratios, zero = temporal_energy(values, self.config.frequency.cutoffs, self.config.eps)
                            for key, value in (("frequency", freq), ("energy", power), ("low_ratio", ratios), ("near_zero", zero)):
                                arrays[prefix + "_" + key] = value.numpy()
                            measured += 1
                with (path / "spectrum" / "temporal_energy.npz").open("xb") as handle:
                    np.savez_compressed(handle, **arrays)
                temporal_counts[sample] = measured
            self.analysis_seconds += time.perf_counter() - flush_started
            for sample, path in self.sample_dirs.items():
                metadata = {**self.extra_metadata, "suite_id": "A_FREQUENCY_ORACLE",
                            "experiment_ids": {"spectrum": "A_E1_SPECTRUM", "sensitivity": "A_E2_SENSITIVITY", "compression": "A_E3_SELECTION"}, "status": status, "error": error,
                            "prompt_id": self.prompt_id, "sample_id": sample, "rank": self.rank,
                            "inference_id": self.inference_id, "seed": self.config.seed,
                            "config": self.config.to_dict(), "effective_grids": self.grids,
                            "effective_windows": self.windows, "row_counts_all_samples": dict(self.counts),
                            "row_counts_sample": dict(self.sample_counts[sample]),
                            "analysis_seconds_all_samples": self.analysis_seconds,
                            "peak_gpu_allocated_bytes_process": self.peak_gpu_allocated_bytes,
                            "measurement": {"spatial": "measured" if self.sample_counts[sample]["spectrum/spatial_energy.csv"] else "not measured",
                                            "temporal_segments": temporal_counts[sample],
                                            "oracle": "measured" if self.sample_counts[sample]["sensitivity/block_scores.csv"] else "not measured",
                                            "compression": "measured" if self.sample_counts[sample]["compression_measured_rows"] else "not measured"},
                            "frequency_unit": "radial cycles/token (spatial); cycles/frame (temporal)",
                            "candidate_scope": "sampled_candidate_set; unsampled history is retained",
                            "coordinates": {"spectrum": "clean pre-RoPE K, raw V", "oracle": "runtime post-RoPE Q/K, raw V"}}
                (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        finally:
            for handle in self.handles:
                handle.close()
            self.active = False
            self._features.clear()
            self._previous.clear()
            self._temporal.clear()
        print(f"[spectral] analysis={self.analysis_seconds:.3f}s; process peak GPU allocation="
              f"{self.peak_gpu_allocated_bytes / 1024**3:.3f} GiB; results={self.run_dir}", flush=True)
