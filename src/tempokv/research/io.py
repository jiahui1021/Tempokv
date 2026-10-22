"""Versioned CSV parts, atomic manifests and deterministic, non-destructive resume."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import uuid
import numpy as np
import torch
from .config import EXPERIMENTS

SCHEMA = (
    "schema_version", "experiment_id", "event_id", "data_origin", "prompt_index", "seed", "sample_id", "layer", "head", "chunk_index",
    "phase", "denoise_step_index", "denoise_timestep", "current_chunk_start", "grid_t", "grid_h", "grid_w", "query_samples",
    "key_token_count", "history_token_count", "protected_token_count", "protected_fraction", "block_id", "score_type", "score",
    "local_attention_relative_error", "model_output_relative_error", "clean_latent_relative_error", "horizon_chunks", "nmse_k", "nmse_v",
    "strategy", "intervention", "role", "strength", "requested_strength", "calibration_protocol", "retention", "actual_retained_tokens",
    "actual_retention", "k_bits", "v_bits", "payload_bytes", "scale_bytes", "metadata_bytes", "logical_bytes", "physical_kv_bytes",
    "budget_cohort", "conditioning_set", "paired_run_id", "analysis_ms", "compression_ms", "scoring_ms", "gather_ms", "attention_ms",
    "status", "reason", "details_json")


def safe(value):
    import math
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [safe(v) for v in value]
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(safe(data), handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)  # Fails on an existing destination, including concurrent creation.
    finally: temporary.unlink(missing_ok=True)


def environment():
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError): sha,dirty = "unknown",None
    sources=[]
    for package in ("research","spectral","spectral_kv"):
        sources.extend((root/"src/tempokv"/package).glob("*.py"))
    sources.extend(root/path for path in ("experiments/Self-Forcing/wan/modules/causal_model.py",
        "experiments/Self-Forcing/pipeline/causal_inference.py","experiments/Self-Forcing/tempokv_adapter/research_bridge.py"))
    digest=hashlib.sha256()
    for path in sorted(sources):
        if path.is_file():digest.update(str(path.relative_to(root)).encode()+b"\0"+path.read_bytes())
    return {"commit_sha": sha, "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "working_tree_dirty":dirty,"research_source_tree_sha256":digest.hexdigest(),
            "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "not available"}


def is_real_origin(origin):
    return origin in {"runtime_bf16","runtime_tempokv","runtime_bf16_reference","runtime_bf16_counterfactual",
                      "runtime_tempokv_reference","runtime_tempokv_counterfactual","offline_frozen_real"}


class RunStore:
    def __init__(self, config, *, rank=0, prompt_index=0, seed=None, origin="runtime_bf16", metadata=None, resume=None):
        self.config, self.origin = config, origin
        self.nonce = uuid.uuid4().hex[:12]
        self.root = Path(resume) if resume else Path(config.output.root) / f"rank-{rank}" / f"prompt-{prompt_index}-seed-{config.seed if seed is None else seed}" / f"run-{self.nonce}"
        self.root.mkdir(parents=True, exist_ok=resume is not None)
        self.seen = set(); self.counts = Counter(); self.measured = Counter(); self.handles = {}; self.writers = {}; self.parts = {}
        if resume:
            manifest = json.loads((self.root / "manifest.json").read_text())
            if manifest["config_signature"] != config.signature or manifest["data_origin"] != origin:
                raise ValueError("resume configuration or data origin differs")
            for path in self.root.glob("*/events-*.csv"):
                with path.open(newline="") as handle: self.seen.update(r["event_id"] for r in csv.DictReader(handle))
        self.metadata = {"schema_version": "research.v1", "config_signature": config.signature, "config": config.to_dict(),
                         "data_origin": origin, "rank": rank, "prompt_index": prompt_index, "seed": seed if seed is not None else config.seed,
                         **environment(), **(metadata or {})}
        atomic_json(self.root / (f"resume-{self.nonce}.json" if resume else "manifest.json"), self.metadata)
        self.closed = False

    def write(self, experiment, row):
        if experiment not in self.config.experiments: return
        row = safe({"schema_version": "research.v1", "experiment_id": experiment, "data_origin": self.origin,
                    "status": "measured", **row})
        details = {k: v for k, v in row.items() if k not in SCHEMA}
        if row.get("details_json"): details.update(json.loads(row["details_json"]))
        row["details_json"] = json.dumps(safe(details), sort_keys=True, allow_nan=False)
        if not row.get("event_id"):
            labels=("experiment_id","prompt_index","seed","sample_id","layer","head","chunk_index","phase",
                    "denoise_step_index","current_chunk_start","grid_t","grid_h","grid_w","block_id","score_type",
                    "strategy","intervention","role","requested_strength","strength","k_bits","v_bits","retention",
                    "conditioning_set","horizon_chunks","budget_cohort","paired_run_id")
            identity={k:row.get(k) for k in labels}
            identity["config_signature"]=self.config.signature
            identity["details_labels"]={k:details.get(k) for k in ("frame","history_frame","cutoff","temporal_window","temporal_control",
                "statistic","comparison_reference","noise_seed","query_indices","scenario_id","policy_id")}
            row["event_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
        if row["event_id"] in self.seen: return
        self.seen.add(row["event_id"])
        if experiment not in self.writers:
            folder = self.root / experiment; folder.mkdir(exist_ok=True)
            temporary = folder / f"events-{self.nonce}.csv.tmp"
            handle = temporary.open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=SCHEMA); writer.writeheader()
            self.handles[experiment], self.writers[experiment], self.parts[experiment] = handle, writer, temporary
        self.writers[experiment].writerow({k: row.get(k) for k in SCHEMA})
        self.counts[experiment] += 1
        if row["status"] == "measured": self.measured[experiment] += 1

    def close(self, *, status="completed", extra=None):
        if self.closed: return self.root
        for exp, handle in self.handles.items():
            handle.flush(); os.fsync(handle.fileno()); handle.close()
            path = self.parts[exp]; path.rename(path.with_suffix(""))
        measured = sorted(self.measured)
        summary = {"schema_version": "research.v1", "data_origin": self.origin, "status": status, "row_counts": dict(self.counts),
                   "completed_experiments": measured if is_real_origin(self.origin) else [],
                   "synthetic_completed_experiments": measured if self.origin.startswith("synthetic") else [],
                   "unrun_experiments": [x for x in EXPERIMENTS if x not in measured or not is_real_origin(self.origin)],
                   "quality_metrics": {x: "not_run" for x in ("PSNR", "SSIM", "LPIPS", "FID", "VBench")}, **(extra or {})}
        atomic_json(self.root / f"summary-{self.nonce}.json", summary)
        if not (self.root / "summary.json").exists(): atomic_json(self.root / "summary.json", summary)
        self.closed = True
        return self.root
