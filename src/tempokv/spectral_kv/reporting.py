"""Versioned local outputs and bounded timings for suite B only."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from .config import SUITE_ID
from .method_spec import parse_method


def json_safe(value):
    import math
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ReportSink:
    def __init__(self, output_dir, config, *, rank=0, prompt_id=0, metadata=None):
        self.config, self.rank, self.prompt_id = config, rank, prompt_id
        self.metadata = dict(metadata or {})
        runtime = str(self.metadata.get("runtime", "cpu")).replace("/", "_")
        window = self.metadata.get("local_attn_size", "toy")
        window_label = "full_history" if window == -1 else f"window-{window}"
        window_label += f"-sink-{self.metadata.get('sink_size', 0)}"
        self.root = Path(output_dir) / config.profile / runtime / window_label / f"rank-{rank}" / f"inference-{uuid.uuid4().hex[:16]}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.handles = {}
        self.counts, self.measured_counts, self.skipped = Counter(), Counter(), Counter()
        self.cpu_ms, self.gpu_events = defaultdict(float), defaultdict(list)
        self.closed = False
        self.peak_gpu_allocated_bytes = 0
        for experiment in config.experiments:
            (self.root / experiment).mkdir()

    def write(self, experiment, rows):
        if experiment not in self.config.experiments:
            return
        if experiment not in self.handles:
            self.handles[experiment] = (self.root / experiment / "records.jsonl").open("x", encoding="utf-8")
        for row in rows:
            payload = json_safe({**row, "suite_id": SUITE_ID, "experiment_id": experiment,
                                 "prompt_id": self.prompt_id, "rank": self.rank})
            self.handles[experiment].write(json.dumps(payload, allow_nan=False) + "\n")
            self.counts[experiment] += 1
            if row.get("status", "ok") in {"ok", "measured"}:
                self.measured_counts[experiment] += 1

    @contextmanager
    def time(self, section, device=None):
        device = torch.device(device or "cpu")
        started = time.perf_counter()
        events = None
        if device.type == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        try:
            yield
        finally:
            self.cpu_ms[section] += (time.perf_counter() - started) * 1000
            if events is not None:
                events[1].record()
                self.gpu_events[section].append(events)
                self.peak_gpu_allocated_bytes = max(self.peak_gpu_allocated_bytes, torch.cuda.max_memory_allocated(device))

    def save_state(self, experiment, name, arrays):
        if experiment in self.config.experiments:
            with (self.root / experiment / name).open("xb") as handle:
                np.savez_compressed(handle, **arrays)

    def close(self, *, status="completed", error=None, extra=None):
        if self.closed:
            return self.root
        for handle in self.handles.values():
            handle.close()
        gpu_ms = {}
        if self.gpu_events:
            torch.cuda.synchronize()
            gpu_ms = {section: sum(a.elapsed_time(b) for a, b in events)
                      for section, events in self.gpu_events.items()}
        timing = gpu_ms if gpu_ms else dict(self.cpu_ms)
        generator = timing.get("generator_denoise", 0) + timing.get("generator_clean", 0)
        native_attn = timing.get("attention", 0)
        payload = {
            **self.metadata, **(extra or {}), "suite_id": SUITE_ID, "profile": self.config.profile,
            "config": self.config.to_dict(), "prompt_id": self.prompt_id, "rank": self.rank,
            "status": status, "error": error,
            "trajectory_changed": self.config.online and parse_method(self.config.method).family != "full",
            "row_counts": dict(self.counts), "skipped": dict(self.skipped),
            "measurement": {exp: "measured" if self.measured_counts[exp] else "not measured" for exp in self.config.experiments},
            "host_section_ms": dict(self.cpu_ms), "gpu_section_ms": gpu_ms,
            "latency_breakdown_ms": {"attn": native_attn if generator else None,
                                     "non_attn": max(0, generator - native_attn - timing.get("analysis", 0) - timing.get("cache_transform", 0)) if generator else None,
                                     "cache_pass": timing.get("generator_clean"), "vae": timing.get("vae"),
                                     "analysis": timing.get("analysis"), "cache_transform": timing.get("cache_transform")},
            "latency_contract": "instrumented sections; cache_pass overlaps generator total; analysis overhead is separate; no speedup inferred",
            "peak_gpu_allocated_bytes_process": self.peak_gpu_allocated_bytes,
            "video_metrics": {key: "not measured" for key in ("VBench", "PSNR", "LPIPS", "FID")},
            "memory_contract": "native BF16 backing is retained by the prototype; spectral view/cache bytes are additional, not a claimed memory saving",
        }
        with (self.root / "metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
        self.closed = True
        print(f"[spectral_kv][{SUITE_ID}] profile={self.config.profile}; analysis_host="
              f"{self.cpu_ms.get('analysis', 0):.1f}ms; process_peak_GPU="
              f"{self.peak_gpu_allocated_bytes / 1024**3:.3f}GiB; results={self.root}", flush=True)
        return self.root
