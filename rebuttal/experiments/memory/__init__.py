"""Fixed storage capacity and matched-budget generation experiment."""
from __future__ import annotations

import json
from pathlib import Path


def run(config: dict, stage: str, output_dir: Path) -> dict:
    from .capacity import capacity_report
    from .generation import build_generation_jobs, execute_jobs
    from .report import summarize_generation, write_capacity
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    has_videos = any(config.get(key) for key in ("pairs", "pairs_manifest", "bf16_reference_dir", "reference_dir"))
    if stage in ("capacity", "replay") or (stage == "evaluate" and not has_videos):
        report = capacity_report(config)
        write_capacity(report, output_dir)
        return report
    if stage in ("plan", "generate"):
        plan = build_generation_jobs(config, output_dir)
        (output_dir / "generation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        for job in plan["jobs"]:
            path = Path(job["config_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(job["worker_config"], indent=2) + "\n")
        write_capacity(plan["capacity"], output_dir)
        if stage == "generate":
            try:
                plan = execute_jobs(plan, resume=bool(config.get("_resume", config.get("resume", False))))
            finally:
                (output_dir / "generation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        return plan
    if stage == "evaluate":
        from .evaluation import evaluate
        return evaluate(config, output_dir)
    if stage == "summarize":
        return summarize_generation(output_dir)
    raise ValueError(f"fixed_memory does not support stage {stage!r}")
