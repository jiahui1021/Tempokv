"""Complete independent AquaKV experiment workflow; no TempoKV predictors/codecs."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

from rebuttal.common.manifest import ROOT, file_sha256, prompt_records, read_json, write_json

UPSTREAM_COMMIT = "e744183f2f3a4a90381f20a6975456871fd1ae29"
STAGES = ["collect", "fit", "replay", "generate", "evaluate", "summarize"]


def validate_independence(config):
    forbidden = {key for key in ("codec_config", "codec_overrides", "tempokv", "k_params", "v_params", "quantizer", "ridge")
                 if config.get(key) is not None}
    if forbidden:
        raise ValueError(f"independent AquaKV does not accept TempoKV/legacy settings: {sorted(forbidden)}; use higgs and reg_rate")
    if config.get("model", "self_forcing") != "self_forcing":
        raise ValueError("AquaKV video generation currently supports Self-Forcing")


def _source(config, output, split):
    return config.get(f"{split}_trace") or output / "collection" / split / "traces"


def _paths(source):
    if isinstance(source, (list, tuple)):
        return [p for item in source for p in _paths(item)]
    path = Path(source)
    paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError(f"AquaKV trace missing: {source}; run collect or supply an independent trace")
    return paths


def read_traces(source, *, device="cpu", require_identity=True):
    import torch
    from rebuttal.common.trace import load_trace
    traces = [load_trace(path, device=device) for path in _paths(source)]
    first = traces[0]
    if list(first.layer_ids) != list(range(first.shape[0])):
        raise ValueError("independent AquaKV requires every consecutive layer starting at layer 0")
    for trace in traces:
        if trace.layer_ids != first.layer_ids or trace.shape[2:] != first.shape[2:]:
            raise ValueError("AquaKV trace files must share layer/frame/head geometry")
        coordinate = trace.metadata.get("coordinate_system")
        if coordinate != "pre_rope_normalized":
            raise ValueError("AquaKV calibration requires normalized pre-RoPE Self-Forcing K")
        identity = trace.metadata["split_identity"]
        if require_identity and not identity["hashes_complete"]:
            raise ValueError("AquaKV traces require prompt hashes to verify calibration/heldout separation")
    sources = []
    for trace in traces:
        item = {"path": str(trace.path), "sha256": file_sha256(trace.path),
                "identity": trace.metadata["split_identity"]}
        sidecar = trace.path.with_suffix(".identity.json")
        if sidecar.is_file():
            item["identity_sha256"] = file_sha256(sidecar)
        sources.append(item)
    return torch.cat([t.keys for t in traces], dim=1), torch.cat([t.values for t in traces], dim=1), sources


def check_sources(calibration, evaluation):
    if {r["sha256"] for r in calibration} & {r["sha256"] for r in evaluation}:
        raise ValueError("AquaKV calibration and evaluation trace contents overlap")
    a = {h for r in calibration for h in r["identity"]["prompt_hashes"]}
    b = {h for r in evaluation for h in r["identity"]["prompt_hashes"]}
    if a & b:
        raise ValueError("AquaKV calibration and evaluation prompts overlap")
    return {"prompt_hashes_verified": bool(a and b), "calibration_prompts": len(a), "evaluation_prompts": len(b)}


def _predictor_path(config, output):
    return Path(config.get("predictor_path") or output / "aquakv_predictor.pt")


def load_artifact(config, output, device="cpu"):
    import torch
    from .predictors import load_predictor
    from .quantizer import HiggsConfig, Quantizer
    path = _predictor_path(config, output)
    report_path = output/"fit.json"
    if not config.get("predictor_path") and report_path.is_file():
        if read_json(report_path)["predictor_sha256"] != file_sha256(path):
            raise ValueError("AquaKV predictor changed after calibration")
    state = torch.load(path, map_location=device, weights_only=True)
    quantizer = Quantizer(state["heads"]*state["dim"], HiggsConfig(**config.get("higgs", {})))
    if state["quantizer"] != quantizer.settings():
        raise ValueError("AquaKV HIGGS configuration differs from fitted artifact")
    return load_predictor(state, device), quantizer, state


def build_job(config, output, *, mode, prompts, frames, predictor_path=None, trace_role=None):
    from rebuttal.adapters.generation import GenerationJob
    from .runtime import DEFAULT_ROPE_PRECISION
    if frames < 9 or (frames+3) % 12:
        raise ValueError("Self-Forcing frames must be >=9 and satisfy (frames+3)%12==0")
    missing = [key for key in ("checkpoint", "model_root") if not config.get(key)]
    if not prompts:
        missing.append("prompts/evaluation_prompts")
    if missing:
        raise ValueError("AquaKV generation requires " + ", ".join(missing))
    if int(config.get("num_gpus", 1)) < 1 or int(config.get("samples", 1)) < 1:
        raise ValueError("num_gpus and samples must be positive")
    rope_precision = config.get("rope_precision", DEFAULT_ROPE_PRECISION)
    worker = {**config, "output_dir": str(output), "mode": mode, "prompts": str(prompts),
              "frames": frames, "rope_precision": rope_precision}
    if predictor_path is not None:
        worker["predictor_path"] = str(predictor_path)
    if trace_role:
        worker["trace_role"] = trace_role
        worker["samples"] = int(config.get("calibration_samples", 1))
        worker["start_index"], worker["end_index"] = 0, -1
    path = write_json(output / "worker_config.json", worker)
    command = [str(config.get("python", sys.executable)), "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={int(config.get('num_gpus', 1))}", "--module",
               "rebuttal.experiments.aquakv.worker", "--config", str(path)]
    files = [str(config["checkpoint"]), str(prompts), str(path)]
    if config.get("model_config"):
        files.append(str(config["model_config"]))
    if mode in {"aquakv", "bf16_adapter"}:
        files.append(str(predictor_path))
    environment = {"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT/"src")]),
                   "SELF_FORCING_CACHE_ROPE_PRECISION": rope_precision}
    return GenerationJob(mode, command, str(output), environment, files, [str(config["model_root"])], frames)


def verify_videos(job, config, prompts, *, collection=False):
    from rebuttal.experiments.memory.identity import video_identities, read_prompts
    lines = read_prompts(Path(prompts))
    records, summary = video_identities(Path(job.output_dir)/"videos", prompts_path=Path(prompts), use_snapshot=False)
    start = 0 if collection else int(config.get("start_index", 0))
    stop = -1 if collection else int(config.get("end_index", -1))
    stop = len(lines) if stop < 0 else min(stop, len(lines))
    samples = int(config.get("calibration_samples", 1)) if collection else int(config.get("samples", 1))
    expected = {(str(p), int(config.get("seed", 0))+p*1_000_003, s) for p in range(start, stop) for s in range(samples)}
    actual = {(r["prompt_id"], r["noise_seed"], r["sample_id"]) for r in records}
    if actual != expected:
        raise ValueError("AquaKV generated video cohort does not match requested prompt/seed/sample identities")
    return {"summary": str(summary), "summary_sha256": file_sha256(summary), "videos": records}


def run(config, stage, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_independence(config)
    if stage == "plan":
        result = {"status": "planned", "experiment": "aquakv", "independent": True,
                  "upstream_commit": UPSTREAM_COMMIT, "quantizer": "official_HIGGS",
                  "stages": STAGES, "pipeline_stage": "pipeline",
                  "missing_for_collection": [k for k in ("checkpoint", "model_root", "calibration_prompts", "evaluation_prompts") if not config.get(k)],
                  "calibration_source": str(_source(config, output, "calibration")),
                  "evaluation_source": str(_source(config, output, "evaluation")),
                  "predictor_path": str(_predictor_path(config, output)),
                  "required_tempokv_artifacts": [], "result_values": []}
        write_json(output/"plan.json", result)
        return result
    if stage == "pipeline":
        results = {}
        for name in STAGES:
            if name == "collect" and config.get("calibration_trace") and config.get("evaluation_trace"):
                continue
            results[name] = run(config, name, output)
        return {"status": "complete", "stages": results}
    if stage == "collect":
        from rebuttal.adapters.generation import execute_jobs
        from rebuttal.common.manifest import check_prompt_split
        check_prompt_split(config["calibration_prompts"], config["evaluation_prompts"])
        jobs, cohorts = [], {}
        for split in ("calibration", "evaluation"):
            jobs.append(build_job(config, output/"collection"/split, mode="collect", prompts=config[f"{split}_prompts"],
                                  frames=int(config.get("calibration_frames", 81)), trace_role=split))
        result = execute_jobs(jobs, resume=config.get("_resume", False))
        for split, job in zip(("calibration", "evaluation"), jobs):
            cohort = verify_videos(job, config, config[f"{split}_prompts"], collection=True)
            paths = _paths(Path(job.output_dir)/"traces")
            if len(paths) != len({r["prompt_id"] for r in cohort["videos"]}):
                raise ValueError("incomplete AquaKV clean trace collection")
            cohorts[split] = {**cohort, "traces": [{"path": str(p), "sha256": file_sha256(p)} for p in paths]}
        result["cohorts"] = cohorts
        write_json(output/"collect.json", result)
        return result
    if stage == "fit":
        import torch
        from rebuttal.common.trace import device_for
        from .predictors import fit_predictors, predictor_state
        from .quantizer import Quantizer, HiggsConfig
        from .report import summarize
        if config.get("predictor_path"):
            raise ValueError("predictor_path selects an existing independent artifact; omit it when fitting into output_dir")
        path = _predictor_path(config, output)
        if path.exists():
            if config.get("_resume") and (output/"fit.json").is_file():
                report = read_json(output/"fit.json")
                if file_sha256(path) != report["predictor_sha256"]:
                    raise ValueError("AquaKV predictor changed after calibration")
                return report
            raise FileExistsError(f"AquaKV predictor already exists: {path}; use a new output directory")
        device = device_for(config.get("device", "auto"))
        k, v, sources = read_traces(_source(config, output, "calibration"), device=device,
                                    require_identity=config.get("require_prompt_identity", True))
        quantizer = Quantizer(k.shape[-2]*k.shape[-1], HiggsConfig(**config.get("higgs", {})))
        predictor, rows = fit_predictors(k, v, quantizer, parameter_dtype=config.get("parameter_dtype", "bfloat16"),
                reg_rate=float(config.get("reg_rate", 1e-3)), fit_token_stride=int(config.get("fit_token_stride", 1)),
                chunk_size=int(config.get("fit_chunk_size", 4096)), validation_records=int(config.get("validation_records", 0)),
                seed=int(config.get("seed", 0)))
        state = {**predictor_state(predictor), "layer_ids": list(range(k.shape[0])), "quantizer": quantizer.settings(),
                 "calibration_sources": sources, "upstream_commit": UPSTREAM_COMMIT}
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".aquakv-", dir=path.parent)
        os.close(fd)
        try:
            with open(temporary, "wb") as stream:
                torch.save(state, stream)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        result = {"status": "complete", "method": "aquakv", "independent": True,
                  "predictor_path": str(path), "predictor_sha256": file_sha256(path), "sources": sources,
                  "quantizer": quantizer.settings(), "parameter_dtype": config.get("parameter_dtype", "bfloat16"),
                  "upstream_commit": UPSTREAM_COMMIT, "calibration": summarize(rows, predictor.parameter_bytes())}
        write_json(output/"fit.json", result)
        return result
    if stage == "replay":
        from rebuttal.common.trace import device_for
        from .predictors import replay
        from .report import summarize
        device = device_for(config.get("device", "auto"))
        predictor, quantizer, state = load_artifact(config, output, device)
        k, v, sources = read_traces(_source(config, output, "evaluation"), device=device,
                                    require_identity=config.get("require_prompt_identity", True))
        split = check_sources(state["calibration_sources"], sources)
        result = {"status": "complete", "split": "heldout" if split["prompt_hashes_verified"] else "unverified_trace_split", "metric_scope": "trace_replay",
                  "split_check": split, "sources": sources,
                  "aquakv": summarize(replay(k, v, predictor, quantizer), predictor.parameter_bytes()),
                  "higgs_direct": summarize(replay(k, v, predictor, quantizer, direct=True))}
        write_json(output/"replay.json", result)
        return result
    if stage == "generate":
        from rebuttal.adapters.generation import execute_jobs
        predictor, quantizer, state = load_artifact(config, output)
        if str(predictor.dtype) != "torch.bfloat16":
            raise ValueError("Self-Forcing generation uses BF16; fit AquaKV with parameter_dtype=bfloat16")
        prompts = config.get("evaluation_prompts") or config.get("prompts")
        if not prompts:
            raise ValueError("AquaKV generate needs evaluation_prompts or prompts")
        hashes = {r["sha256"] for r in prompt_records(prompts)}
        calibration_hashes = {h for s in state["calibration_sources"] for h in s["identity"]["prompt_hashes"]}
        if hashes & calibration_hashes:
            raise ValueError("AquaKV generation prompts overlap predictor calibration prompts")
        methods = config.get("generation_methods", ["bf16", "aquakv"])
        if not methods or set(methods)-{"bf16", "bf16_adapter", "aquakv"} or len(set(methods)) != len(methods):
            raise ValueError("AquaKV generation_methods accepts bf16, bf16_adapter and aquakv once each")
        frames = int(config.get("frames", 1401))
        jobs = [build_job(config, output/"runs"/mode, mode=mode, prompts=prompts, frames=frames,
                          predictor_path=_predictor_path(config, output)) for mode in methods]
        result = execute_jobs(jobs, resume=config.get("_resume", False))
        result["cohorts"] = {job.name: verify_videos(job, config, prompts) for job in jobs}
        result["predictor_sha256"] = file_sha256(_predictor_path(config, output))
        write_json(output/"generate.json", result)
        return result
    if stage == "evaluate":
        from rebuttal.experiments.temporal_window import run as quality
        local_methods = {
            method for method, directory in (("bf16", "reference_dir"), ("aquakv", "candidate_dir"))
            if not config.get(directory)
        }
        if local_methods:
            generated = read_json(output/"generate.json")
            if generated.get("status") != "complete" or not local_methods.issubset(
                job["name"] for job in generated.get("jobs", [])
            ):
                raise ValueError("video evaluation requires completed generation for " + ", ".join(sorted(local_methods)))
            for job in generated["jobs"]:
                if job["name"] not in local_methods:
                    continue
                for artifact in job["artifacts"]:
                    if file_sha256(artifact["path"]) != artifact["sha256"]:
                        raise ValueError("generated video changed after completed generation")
        # The shared evaluator checks actual generation identities and video bytes.
        result = quality({**config, "model": "self_forcing", "method": "aquakv",
                          "reference_dir": str(config.get("reference_dir") or output/"runs/bf16/videos"),
                          "candidate_dir": str(config.get("candidate_dir") or output/"runs/aquakv/videos")},
                         "evaluate", output/"quality")
        write_json(output/"evaluate.json", result)
        return result
    if stage == "summarize":
        reports = {name: read_json(output/f"{name}.json") for name in ("fit", "replay", "generate", "evaluate")
                   if (output/f"{name}.json").is_file()}
        if not reports:
            raise ValueError("no completed AquaKV reports to summarize")
        result = {"status": "complete", "method": "aquakv", "independent": True,
                  "upstream_commit": UPSTREAM_COMMIT, "reports": reports}
        write_json(output/"summary.json", result)
        return result
    raise ValueError("AquaKV supports plan/pipeline/"+"/".join(STAGES))
