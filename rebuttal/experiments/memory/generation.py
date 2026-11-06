"""Concrete same-generator jobs for matched-context and matched-memory runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .capacity import capacity_report

ROOT = Path(__file__).resolve().parents[3]


def build_generation_jobs(config: dict, output_dir: Path) -> dict:
    if config.get("model", "self_forcing") not in ("self_forcing", "Self-Forcing"):
        raise ValueError("fixed-memory generation currently supports the bundled Self-Forcing generator")
    frames = int(config.get("frames", 1401))
    if (frames + 3) % 12:
        raise ValueError("Self-Forcing frames must satisfy (frames+3)%12==0")
    report = capacity_report(config)
    by_budget = {}
    for row in report["rows"]:
        by_budget.setdefault(row["budget_bytes"], {})[row["method"]] = row
    shared_h = int(config.get("matched_context_frames", 96))
    profiles = {(r["method"], r["history_frames"]): r for r in report["profile_rows"]}
    specs, exclusions = [], []
    for budget, methods in by_budget.items():
        for method, row in methods.items():
            if row["history_frames"] is not None:
                specs.append(("matched_memory", method, row["history_frames"], budget))
        bf = methods.get("bf16", {})
        short_h = bf.get("history_frames")
        if short_h is not None and ("tempokv", short_h) in profiles:
            peak = profiles[("tempokv", short_h)]["live_peak_bytes"]
            if peak <= budget:
                specs.append(("tempokv_short", "tempokv", short_h, budget))
        for method in methods:
            profile = profiles.get((method, shared_h))
            if profile and profile["live_peak_bytes"] <= budget:
                specs.append(("matched_context", method, shared_h, budget))
            else:
                exclusions.append({"comparison": "matched_context", "method": method,
                                   "budget_bytes": budget, "history_frames": shared_h,
                                   "reason": "common context does not fit this budget or schedule"})
    jobs = []
    prompts = config.get("prompts")
    prompt_lines = ([line.strip() for line in Path(prompts).read_text().splitlines() if line.strip()]
                    if prompts and Path(prompts).is_file() else None)
    input_digests = {}
    for key in ("prompts", "runtime_config", "codec_config", "k_params", "v_params"):
        value = config.get(key)
        if value and Path(value).is_file():
            input_digests[key] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    runtime_cfg = config.get("runtime_config") or str(ROOT / "experiments/Self-Forcing/configs/self_forcing_dmd.yaml")
    for scenario, method, history, budget in specs:
        for seed in config.get("seeds", [config.get("seed", 0)]):
            name = f"{scenario}/{method}/b{budget}_h{history}/seed_{seed}"
            folder = output_dir / "generation" / name
            worker_config = {key: config[key] for key in (
                "codec_config", "codec_overrides", "k_params", "v_params", "qvg_root", "qvg", "model_root"
            ) if key in config}
            worker_config.update(method=method, history_frames=history, budget_bytes=budget,
                span_frames=report["span_frames"], bf16_lag_frames=report["bf16_lag_frames"],
                rope_precision=config.get("rope_precision", "complex64"),
                memory_events=str(folder / "memory_events.jsonl"),
                memory_report=str(folder / "memory_report.json"))
            cfg_path = folder / "worker_config.json"
            command = [str(config.get("python", sys.executable)), str(ROOT / "rebuttal/experiments/memory/worker.py"),
                "--config", str(cfg_path), "--", "--config_path", str(runtime_cfg),
                "--checkpoint_path", str(config.get("checkpoint") or ""),
                "--data_path", str(prompts or ""), "--output_folder", str(folder / "videos"),
                "--num_output_frames", str((frames + 3) // 4), "--local_attn_size", str(history),
                "--sink_size", "0", "--seed", str(seed), "--num_samples", "1", "--save_with_index"]
            if config.get("use_ema", False):
                command.append("--use_ema")
            identity = hashlib.sha256(json.dumps({"worker": worker_config, "command": command,
                                                   "input_digests": input_digests},
                                                   sort_keys=True).encode()).hexdigest()
            jobs.append({"name": name, "comparison": scenario, "method": method,
                         "history_frames": history, "budget_bytes": budget, "seed": seed,
                         "config_path": str(cfg_path), "worker_config": worker_config,
                         "command": command, "identity": identity, "folder": str(folder),
                         "expected_video_count": len(prompt_lines) if prompt_lines is not None else None,
                         "input_digests": input_digests,
                         "status": "planned"})
    missing = [key for key in ("checkpoint", "model_root", "prompts") if not config.get(key)]
    if any(j["method"] == "tempokv" for j in jobs):
        missing += [key for key in ("codec_config", "k_params", "v_params") if not config.get(key)]
    if any(j["method"] == "qvg" for j in jobs) and not config.get("qvg_root"):
        missing.append("qvg_root")
    return {"schema_version": "tempokv.rebuttal.memory.generation.v1", "status": "planned",
            "generator": "bundled_self_forcing_same_pre_rope_cache", "frames": frames,
            "capacity": report, "jobs": jobs, "exclusions": exclusions,
            "missing_configuration": sorted(set(missing)),
            "schedule": "budget_specific_batch_independent_spans"}


def execute_jobs(plan: dict, *, resume: bool = False) -> dict:
    if plan["missing_configuration"]:
        raise ValueError("fixed-memory generation requires: " + ", ".join(plan["missing_configuration"]))
    for job in plan["jobs"]:
        config = job["worker_config"]
        cmd = job["command"]
        for argument in ("--checkpoint_path", "--data_path", "--config_path"):
            path = Path(cmd[cmd.index(argument) + 1])
            if not path.is_file():
                raise FileNotFoundError(f"generation input is missing: {path}")
        if not Path(config["model_root"]).is_dir():
            raise FileNotFoundError(f"model_root is missing: {config['model_root']}")
        folder = Path(job["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        status_file = folder / "job.json"
        if resume and status_file.is_file():
            previous = json.loads(status_file.read_text())
            if previous.get("identity") == job["identity"] and previous.get("status") == "complete":
                expected = {f"{i}_0.mp4" for i in range(job["expected_video_count"] or 0)}
                actual = {p.name for p in (folder / "videos").glob("*.mp4")}
                if expected and actual == expected and (folder / "memory_report.json").is_file():
                    job["status"] = "reused"
                    continue
        Path(job["config_path"]).write_text(json.dumps(config, indent=2) + "\n")
        with (folder / "run.log").open("w") as log:
            completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        job["returncode"] = completed.returncode
        job["status"] = "complete" if completed.returncode == 0 else "failed"
        if job["status"] == "complete":
            expected = {f"{i}_0.mp4" for i in range(job["expected_video_count"] or 0)}
            actual = {p.name for p in (folder / "videos").glob("*.mp4")}
            if not expected or actual != expected or not (folder / "memory_report.json").is_file():
                job["status"] = "failed"
                job["failure_reason"] = "missing_or_unexpected_videos_or_memory_report"
        status_file.write_text(json.dumps(job, indent=2) + "\n")
        if job["status"] == "failed":
            raise RuntimeError(f"fixed-memory generation failed for {job['name']}; see {folder / 'run.log'}")
    plan["status"] = "generation_complete"
    return plan
