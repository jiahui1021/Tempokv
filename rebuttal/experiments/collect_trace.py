"""Identified BF16 clean trace capture that also generates real video."""
from __future__ import annotations

from pathlib import Path


def run(config: dict, stage: str, output_dir: Path) -> dict:
    from rebuttal.adapters.generation import run as generate
    from rebuttal.common.trace import convert_identified_events
    if stage in {"plan", "generate"}:
        return generate({**config, "generation_methods": ["bf16_trace"]}, stage, output_dir)
    if stage in {"convert", "evaluate"}:
        raw = config.get("raw_trace") or output_dir / "runs/bf16_trace/raw_trace"
        return convert_identified_events(raw, config.get("trace_output_dir") or output_dir / "traces")
    raise ValueError("collect_trace supports plan/generate/convert")
