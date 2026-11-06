#!/usr/bin/env python3
"""Unified local entry point for all six TempoKV supplementary experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rebuttal.common.manifest import digest, finish_run, prepare_run, read_json, resolve_paths, write_json

EXPERIMENTS = {
    "previous_frame": "rebuttal.experiments.previous_frame",
    "temporal_window": "rebuttal.experiments.temporal_window",
    "motion": "rebuttal.experiments.motion",
    "architecture": "rebuttal.experiments.architecture",
    "quant_calibration": "rebuttal.experiments.quant_calibration",
    "fixed_memory": "rebuttal.experiments.memory",
    "quant_aware_fit": "rebuttal.experiments.quant_aware_fit",
    "collect_trace": "rebuttal.experiments.collect_trace",
}
DEFAULT_CONFIGS = {name: ROOT / "rebuttal/configs" / f"{name}.json" for name in EXPERIMENTS}


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_configuration(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError("configuration extends cycle")
    seen.add(path)
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a JSON object")
    if config.get("extends"):
        base = path.parent / config.pop("extends")
        config = merge(load_configuration(base, seen), config)
    return config


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    result = merge({}, config)
    for entry in overrides:
        if "=" not in entry:
            raise ValueError("--set expects dotted.key=value")
        name, value = entry.split("=", 1)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        target = result
        names = name.split(".")
        if any(not k or k.startswith("_") for k in names):
            raise ValueError("invalid or reserved configuration override")
        for key in names[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError("configuration override crosses a scalar value")
        target[names[-1]] = parsed
    return result


def dispatch(experiment: str, config: dict, stage: str, output_dir: Path) -> dict:
    if stage == "generate" and experiment in {"temporal_window", "motion"}:
        from rebuttal.adapters.generation import run
        return run(config, stage, output_dir)
    if stage == "quality" and experiment == "architecture":
        from rebuttal.experiments.temporal_window import run
        fitted = read_json(config.get("fit_report") or output_dir / "fit.json")
        if not config.get("reference_dir"):
            raise ValueError("architecture quality needs reference_dir or explicit video pairs")
        reports = {}
        for row in fitted["variants"]:
            name = row.get("variant", row.get("name"))
            axis = fitted.get("axis", config.get("axis", "temporal"))
            reports[name] = run({**config, "method": name,
                                 "candidate_dir": str(output_dir / "runs" / axis / name / "videos")},
                                "evaluate", output_dir / "quality" / name)
        result = {"status": "complete", "variants": reports}
        write_json(output_dir / "quality_summary.json", result)
        return result
    module = importlib.import_module(EXPERIMENTS[experiment])
    return module.run(config, stage, output_dir)


def execute(experiment: str, config: dict, stage: str, output_dir: Path, *, resume: bool = False) -> dict:
    config = {**config, "_resume": resume, "resume": resume}
    manifest = prepare_run(output_dir, config, experiment=experiment, stage=stage, resume=resume)
    try:
        result = dispatch(experiment, config, stage, output_dir)
    except Exception as error:
        finish_run(output_dir, manifest, stage, error=f"{type(error).__name__}: {error}")
        raise
    write_json(output_dir / f"result_{stage}.json", result)
    finish_run(output_dir, manifest, stage, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], required=True)
    parser.add_argument("--stage", choices=["plan", "fit", "replay", "capacity", "generate", "evaluate", "quality", "convert", "summarize"], default="plan")
    parser.add_argument("--config", type=Path, help="JSON config; relative input paths resolve against repository root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="materialize plan without fitting or generating")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.experiment == "all" and args.stage != "plan" and not args.dry_run:
        raise ValueError("all supports plan only; execute each experiment and stage explicitly")
    names = list(EXPERIMENTS)[:6] if args.experiment == "all" else [args.experiment]
    results = []
    for experiment in names:
        config_path = args.config or DEFAULT_CONFIGS[experiment]
        config = load_configuration(config_path)
        config = apply_overrides(config, args.set)
        if config.get("experiment") not in (None, experiment) and args.experiment != "all":
            raise ValueError("configuration experiment does not match --experiment")
        config["experiment"] = experiment
        studies = config.pop("studies", None) or [{}]
        for index, study in enumerate(studies):
            item = resolve_paths(merge(config, study))
            item.pop("resume", None)
            stage = "plan" if args.dry_run else args.stage
            if args.output_dir:
                output = args.output_dir.expanduser().resolve()
                if len(names)>1:
                    output = output / experiment
            elif item.get("output_dir"):
                output = Path(item["output_dir"])
            else:
                output = ROOT / "rebuttal/outputs" / experiment / digest(item)[:12]
            if len(studies)>1:
                output /= str(study.get("study_id", f"{item.get('model', 'study')}_{index}"))
            item["output_dir"] = str(output)
            result = execute(experiment, item, stage, output, resume=args.resume)
            results.append({"experiment": experiment, "stage": stage, "output_dir": str(output), "status": result.get("status")})
    print(json.dumps({"status": "complete", "runs": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, ImportError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
