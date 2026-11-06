"""Previous reconstructed-frame baselines and production predictor comparison."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time
from typing import Any

from rebuttal.common.manifest import write_json, read_json, file_sha256, check_prompt_split


def variant_configs(config: dict[str, Any]):
    from tempokv.config import load_config
    base = load_config(config["codec_config"], config.get("codec_overrides"))
    if base.mirrored:
        raise ValueError("previous-frame study requires k_to_v production direction")
    available = {
        "production": base,
        "k_identity": replace(base, k_predictor_mode="identity"),
        "kv_identity_cross": replace(base, k_predictor_mode="identity", v_predictor_mode="cross_kv_identity",
                                       v_history_length=max(base.v_history_length, 1)),
        "kv_identity_no_cross": replace(base, k_predictor_mode="identity", v_predictor_mode="identity",
                                          v_history_length=max(base.v_history_length, 1)),
        "identity": replace(base, k_predictor_mode="identity", v_predictor_mode="identity",
                              v_history_length=max(base.v_history_length, 1)),
        "direct": replace(base, k_predictor_mode="direct", k_history_length=0,
                            v_predictor_mode="direct", v_history_length=0),
    }
    names = config.get("variants", ["production", "k_identity", "kv_identity_cross"])
    unknown = set(names)-set(available)
    if unknown:
        raise ValueError(f"unknown previous-frame variants: {sorted(unknown)}")
    if base.k_history_length in (None, 0) and any(name not in {"direct", "production"} for name in names):
        raise ValueError("identity comparisons require a positive, explicit K history_length")
    return {name: available[name] for name in names}


def fit_variant(trace, codec, *, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    from tempokv.fitting import fit_predictors
    from tempokv.parameters import TemporalParameters
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = {"iterations": int(config.get("fit_iterations", 2)),
                "temporal_ridge": float(config.get("temporal_ridge", 1.0 if codec.fit_mode == "legacy" else 0.0)),
                "cross_ridge": float(config.get("cross_ridge", 0.1 if codec.fit_mode == "legacy" else 0.0)),
                "v_history_ridge": float(config.get("v_history_ridge", 0.0)),
                "v_history_mode": "diagonal" if codec.fit_mode == "legacy" else config.get("v_history_mode", "dense"),
                "fit_history_source": "reconstructed"}
    coordinate = trace.metadata.get("coordinate_system", config.get("coordinate_system", "pre_rope_normalized"))
    started = time.perf_counter()
    temporal, cross = fit_predictors(trace.keys, trace.values, config=codec,
                                    layer_ids=trace.layer_ids, coordinate_system=coordinate, **controls)
    seconds = time.perf_counter()-started
    k_path, v_path = output_dir / "k_temporal.npz", output_dir / ("v_temporal.npz" if isinstance(cross, TemporalParameters) else "v_cross_kv.npz")
    temporal.save(k_path)
    cross.save(v_path)
    codec_path = output_dir / "codec.json"
    write_json(codec_path, asdict(codec))
    payload = {"status": "complete", "calibration_trace": str(trace.path),
               "calibration_trace_sha256": file_sha256(trace.path), "trace_metadata": trace.metadata,
               "codec_config": str(codec_path), "k_params": str(k_path), "v_params": str(v_path),
               "parameter_set_id": temporal.parameter_set_id, "fit_seconds": seconds, "controls": controls,
               "k_sha256": file_sha256(k_path), "v_sha256": file_sha256(v_path),
               "production_family": codec.fit_mode}
    write_json(output_dir / "fit_report.json", payload)
    return payload


def _fit_record(root: Path, name: str) -> dict[str, Any]:
    path = root / "params" / name / "fit_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"fit the variant first: {path}")
    result = read_json(path)
    for field, hash_key in (("k_params", "k_sha256"), ("v_params", "v_sha256")):
        if file_sha256(result[field]) != result[hash_key]:
            raise ValueError("fitted parameters changed since their fit report")
    return result


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    configs = variant_configs(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = None
    if config.get("calibration_prompts") and config.get("evaluation_prompts"):
        split = check_prompt_split(config["calibration_prompts"], config["evaluation_prompts"], config.get("development_prompts"))
    if stage == "plan":
        plan = {"status": "planned", "experiment": "previous_frame", "model": config.get("model", "self_forcing"),
                "variants": {name: asdict(codec) for name, codec in configs.items()},
                "prompt_split": split, "stages": ["fit", "replay", "generate", "evaluate", "summarize"],
                "missing": [k for k in ("calibration_trace", "evaluation_trace", "checkpoint", "model_root") if not config.get(k)]}
        write_json(output_dir / "plan.json", plan)
        return plan
    if stage == "fit":
        from rebuttal.common.trace import load_trace, device_for
        trace = load_trace(config["calibration_trace"], device=device_for(config.get("device", "auto")),
                           identity_path=config.get("calibration_identity"), max_records=config.get("max_records"))
        reports = {}
        for name, codec in configs.items():
            marker = output_dir / "params" / name / "fit_report.json"
            if config.get("_resume") and marker.is_file():
                reports[name] = _fit_record(output_dir, name)
            else:
                reports[name] = fit_variant(trace, codec, output_dir=marker.parent, config=config)
        result = {"status": "complete", "variants": reports, "prompt_split": split}
    elif stage in {"replay", "cache"}:
        from rebuttal.common.trace import load_trace, device_for, check_trace_split
        from rebuttal.common.cache_metrics import load_pair, replay
        device = device_for(config.get("device", "auto"))
        trace = load_trace(config.get("evaluation_trace") or config["trace"], device=device,
                           identity_path=config.get("evaluation_identity") or config.get("trace_manifest"),
                           max_records=config.get("max_records"))
        if config.get("calibration_trace"):
            calibration = load_trace(config["calibration_trace"], max_records=1,
                                     identity_path=config.get("calibration_identity"))
            trace_split = check_trace_split(calibration, trace, allow_in_sample=config.get("allow_in_sample", False))
        else:
            trace_split = {"prompt_identity_verified": False}
        reports = {}
        for name, codec in configs.items():
            fitted = _fit_record(output_dir, name)
            k, v = load_pair(fitted["k_params"], fitted["v_params"], codec, device)
            folder = output_dir / "cache" / name
            report = replay(trace, codec, k, v, span_frames=config.get("span_frames"),
                            event_path=folder / "cache_events.jsonl", event_stride=int(config.get("event_stride", 1)),
                            model=config.get("model", "unknown"), variant=name)
            report["trace_split"] = trace_split
            write_json(folder / "summary.json", report)
            reports[name] = report
        result = {"status": "complete", "variants": reports, "prompt_split": split}
    elif stage == "generate":
        from rebuttal.adapters.generation import build_job, execute_jobs
        jobs = []
        for name in configs:
            fitted = _fit_record(output_dir, name)
            item = {**config, **{key: fitted[key] for key in ("codec_config", "k_params", "v_params")}}
            jobs.append(build_job(item, output_dir / "runs" / name, name=name))
        result = execute_jobs(jobs, resume=config.get("_resume", False))
    elif stage == "evaluate":
        from rebuttal.experiments.temporal_window import run as evaluate
        if not config.get("reference_dir"):
            raise ValueError("video evaluation requires reference_dir for matching BF16 videos")
        reports = {}
        for name in configs:
            candidate = output_dir / "runs" / name / "videos"
            candidate = Path(config.get("candidate_dirs", {}).get(name, candidate))
            item = {**config, "candidate_dir": str(candidate), "window_seconds": config.get("window_seconds", 15)}
            reports[name] = evaluate(item, "evaluate", output_dir / "quality" / name)
        result = {"status": "complete", "variants": reports}
    elif stage == "summarize":
        reports = {}
        for name in configs:
            reports[name] = {}
            for label, path in (("fit", output_dir / "params" / name / "fit_report.json"),
                                ("cache", output_dir / "cache" / name / "summary.json"),
                                ("quality", output_dir / "quality" / name / "summary.json")):
                reports[name][label] = read_json(path) if path.exists() else {"status": "missing", "reason": str(path)}
        result = {"status": "complete", "variants": reports}
    else:
        raise ValueError(f"unsupported previous-frame stage: {stage}")
    write_json(output_dir / f"{stage}_summary.json", result)
    return result
