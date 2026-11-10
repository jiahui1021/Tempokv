"""Validated subprocess jobs for the bundled generators; no model downloads."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from rebuttal.common.manifest import ROOT, digest, file_sha256, read_json, write_json


@dataclass
class GenerationJob:
    name: str
    command: list[str]
    output_dir: str
    environment: dict[str, str] = field(default_factory=dict)
    required_files: list[str] = field(default_factory=list)
    required_directories: list[str] = field(default_factory=list)
    expected_frames: int | None = None

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _required(config: dict[str, Any], name: str) -> str:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"generation requires config.{name}")
    return str(value)


def build_job(config: dict[str, Any], output_dir: Path, *, mode: str = "tempokv",
              name: str | None = None) -> GenerationJob:
    model = config.get("model", "self_forcing")
    if model not in {"self_forcing", "causal_forcing", "longcat"}:
        raise ValueError(f"unsupported model: {model}")
    python = str(config.get("python", sys.executable))
    frames = int(config.get("frames", 1401))
    if frames < 1:
        raise ValueError("frames must be positive")
    seed = int(config.get("seed", 0))
    environment = {"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")]),
                   "TEMPOKV_BF16_LAG_FRAMES": str(config.get("bf16_lag_frames", 18)),
                   "TEMPOKV_FLUSH_TAIL": "1"}
    files: list[str] = []
    directories: list[str] = []
    custom = bool(config.get("research_params") or config.get("observer") or mode == "bf16_trace")
    if model in {"self_forcing", "causal_forcing"}:
        period = 12 if model == "self_forcing" else 4*int(config.get("num_frame_per_block", 3))
        if (frames+3) % period:
            raise ValueError(f"{model}: frames must satisfy (frames+3) % {period} == 0")
        checkpoint, prompts = _required(config, "checkpoint"), _required(config, "prompts")
        files.extend([checkpoint, prompts])
        if custom:
            if model != "self_forcing":
                raise ValueError("research codec/online observer generation currently uses Self-Forcing; use identified trace replay for other models")
            path = output_dir / "worker_config.json"
            worker_config = {**config, "mode": mode, "frames": frames, "output_dir": str(output_dir)}
            write_json(path, worker_config)
            command = [python, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={int(config.get('num_gpus', 1))}",
                       str(ROOT / "rebuttal/adapters/worker.py"), "--config", str(path)]
            if mode == "tempokv" and not config.get("research_params"):
                codec_path = _required(config, "codec_config")
                files.append(codec_path)
                from tempokv.config import load_config
                from tempokv.integration import is_parameter_free_identity
                fixed = Path(codec_path).is_file() and is_parameter_free_identity(load_config(codec_path))
                if not fixed:
                    files.extend([_required(config, k) for k in ("k_params", "v_params")])
            if config.get("research_params"):
                files.append(str(config["research_params"]))
        else:
            entry = ROOT / ("experiments/Self-Forcing/tempokv_adapter/generate.py" if model == "self_forcing"
                            else "experiments/Causal-Forcing/causal_forcing/tempokv_adapter/generate.py")
            command = [python, str(entry), "--mode", mode, "--checkpoint", checkpoint,
                       "--prompts", prompts, "--output-dir", str(output_dir), "--frames", str(frames),
                       "--seed", str(seed), "--samples", str(config.get("samples", 1)),
                       "--num-gpus", str(config.get("num_gpus", 1)),
                       "--start-index", str(config.get("start_index", 0)),
                       "--end-index", str(config.get("end_index", -1))]
            if model == "causal_forcing":
                model_root = _required(config, "model_root")
                directories.append(model_root)
                command += ["--model-root", model_root, "--num-frame-per-block", str(config.get("num_frame_per_block", 3))]
                if config.get("retain_final_cache", True):
                    command += ["--retain-final-cache"]
            if mode == "tempokv":
                for field, flag in (("codec_config", "--config"), ("k_params", "--k-params"), ("v_params", "--v-params")):
                    value = _required(config, field)
                    files.append(value)
                    command += [flag, value]
                if config.get("torch_decoder", False) and model == "self_forcing":
                    command += ["--torch-decoder"]
                if config.get("allow_triton_fallback", False):
                    command += ["--allow-triton-fallback"]
            for field, flag in (("local_attn_size", "--local-attn-size"), ("span_frames", "--span-frames"), ("model_config", "--model-config")):
                if config.get(field) is not None:
                    command += [flag, str(config[field])]
            if model == "self_forcing":
                command += ["--sink-size", str(config.get("sink_size", 0))]
        if model == "self_forcing":
            root = _required(config, "model_root")
            directories.append(root)
            environment["FORCING_WAN_MODEL_DIR"] = root
        if model == "causal_forcing" and config.get("local_attn_size", -1) not in (-1, None):
            if int(config["local_attn_size"]) < (frames+3)//4:
                if not config.get("sliding_window_ablation", False):
                    raise ValueError("short Causal-Forcing context requires explicit sliding_window_ablation=true")
                environment["CAUSAL_FORCING_BENCHMARK"] = "1"
    else:
        checkpoint = _required(config, "checkpoint_dir")
        directories.append(checkpoint)
        num_frames = int(config.get("num_frames", 93))
        cond = int(config.get("num_cond_frames", 53))
        segments = int(config.get("num_segments", 1))
        if num_frames < 1 or (num_frames-1)%4 or cond < 1 or cond > num_frames or (cond-1)%4 or segments < 1:
            raise ValueError("invalid LongCat frame/conditioning/segment configuration")
        actual_frames = num_frames + segments*(num_frames-cond)
        if "frames" in config and frames != actual_frames:
            raise ValueError(f"LongCat requested {frames} frames but segment configuration generates {actual_frames}")
        frames = actual_frames
        command = [python, str(ROOT / "experiments/LongCat/longcat/generate.py"), "--mode", mode,
                   "--checkpoint_dir", checkpoint, "--output_dir", str(output_dir),
                   "--num_frames", str(num_frames), "--num_cond_frames", str(cond),
                   "--num_segments", str(segments), "--prompt_source", "direct_prompt",
                   "--prompt", _required(config, "prompt"), "--prompt_idx", str(config.get("prompt_id", 0)),
                   "--seed", str(seed), "--context_parallel_size", str(config.get("context_parallel_size", 1)),
                   "--attn_sink_frames", str(config.get("sink_size", 0))]
        if mode == "tempokv":
            for field, flag in (("codec_config", "--config"), ("k_params", "--k-params"), ("v_params", "--v-params")):
                value = _required(config, field)
                files.append(value)
                command += [flag, value]
        if config.get("cumulative_cond", False):
            command.append("--cumulative_cond")
    return GenerationJob(name or f"{model}_{mode}", command, str(output_dir), environment,
                         files, directories, frames)


def execute_jobs(jobs: list[GenerationJob], *, resume: bool = False) -> dict[str, Any]:
    results = []
    for job in jobs:
        output = Path(job.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "rebuttal_generation.json"
        identity = digest(job.record())
        if marker.exists() and resume:
            old = read_json(marker)
            if old.get("job_sha256") != identity:
                raise ValueError("generation resume identity mismatch")
            if old.get("status") == "complete":
                for item in old.get("artifacts", []):
                    if not Path(item["path"]).is_file() or file_sha256(item["path"]) != item["sha256"]:
                        raise ValueError("generated video is missing or changed; choose a new run directory")
                if not old.get("artifacts"):
                    raise ValueError("completed generation marker contains no video artifacts")
                results.append(old)
                continue
        for path in job.required_files:
            if not Path(path).is_file():
                raise FileNotFoundError(f"generation input missing: {path}")
        for path in job.required_directories:
            if not Path(path).is_dir():
                raise FileNotFoundError(f"generation directory missing: {path}")
        if not resume and any(output.rglob("*.mp4")):
            raise FileExistsError(f"generation output already has videos: {output}")
        record = {"status": "running", "job_sha256": identity, **job.record()}
        write_json(marker, record)
        with (output / "generation.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(job.command, cwd=ROOT, env={**os.environ, **job.environment},
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        record.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
        videos = sorted(output.rglob("*.mp4"))
        if result.returncode == 0 and not videos:
            record.update(status="failed", error="generator exited without video artifacts")
        record["artifacts"] = [{"path": str(p), "sha256": file_sha256(p)} for p in videos]
        write_json(marker, record)
        if record["status"] != "complete":
            raise RuntimeError(f"generation failed: {job.name}; see {output / 'generation.log'}")
        results.append(record)
    return {"status": "complete", "jobs": results}


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    modes = config.get("generation_methods", ["bf16", "tempokv"])
    jobs = []
    missing = []
    for mode in modes:
        try:
            jobs.append(build_job(config, output_dir / "runs" / mode, mode=mode))
        except ValueError as error:
            if stage != "plan":
                raise
            missing.append(str(error))
    plan = {"status": "needs_inputs" if missing else "planned", "jobs": [x.record() for x in jobs], "missing": missing}
    write_json(output_dir / "generation_plan.json", plan)
    if stage == "plan":
        return plan
    if stage == "generate":
        return execute_jobs(jobs, resume=config.get("_resume", False))
    raise ValueError(f"unsupported generation stage: {stage}")
