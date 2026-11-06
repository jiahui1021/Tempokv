"""Paired fidelity and existing VBench-Long protocol for actual generated videos."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def generation_pairs(config: dict, output_dir: Path) -> list[dict]:
    reference_dir = Path(config.get("bf16_reference_dir") or config.get("reference_dir") or "")
    if not reference_dir.is_dir():
        raise ValueError("provide bf16_reference_dir for a complete-history BF16 reference")
    path = output_dir / "generation_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"generation plan is missing: {path}")
    plan = json.loads(path.read_text())
    pairs = []
    for job in plan["jobs"]:
        folder = Path(job["folder"]) / "videos"
        candidates = sorted(folder.glob("*.mp4"))
        if len(candidates) != job["expected_video_count"]:
            raise ValueError(f"incomplete generated video cohort: {folder}")
        for candidate in candidates:
            reference = reference_dir / candidate.name
            if not reference.is_file():
                raise FileNotFoundError(f"missing BF16 reference pair: {reference}")
            pairs.append({"model": "self_forcing", "prompt_id": candidate.stem.split("_")[0],
                          "seed": job["seed"], "sample_id": 0, "method": job["method"],
                          "comparison_id": job["name"], "reference": str(reference),
                          "candidate": str(candidate), "history_frames": job["history_frames"],
                          "budget_bytes": job["budget_bytes"]})
    return pairs


def _late_video(source: Path, output: Path, fraction: float):
    if not 0 < fraction < 1:
        raise ValueError("vbench_late_fraction must be in (0,1)")
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("late-video VBench requires ffmpeg for lossless FFV1 extraction")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("late-video extraction requires OpenCV metadata inspection") from error
    capture = cv2.VideoCapture(str(source))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count < 2:
        raise ValueError(f"cannot determine video frame count: {source}")
    start = int(count * (1 - fraction))
    subprocess.run([executable, "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
                    "-vf", rf"select=gte(n\,{start}),setpts=PTS-STARTPTS", "-an", "-c:v", "ffv1", str(output)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {"source": str(source), "first_frame": start, "source_frame_count": count,
            "output": str(output), "codec": "ffv1_lossless", "metric_scope": "late_video_consistency"}


def evaluate_vbench(config: dict, output_dir: Path) -> dict:
    plan = json.loads((output_dir / "generation_plan.json").read_text())
    prompts_path = config.get("prompts")
    if not prompts_path or not Path(prompts_path).is_file():
        raise ValueError("VBench evaluation requires the actual generation prompt file")
    prompts = [line.strip() for line in Path(prompts_path).read_text().splitlines() if line.strip()]
    settings = config.get("vbench", {})
    settings = settings if isinstance(settings, dict) else {}
    rows = []
    for job in plan["jobs"]:
        videos = Path(job["folder"]) / "videos"
        for scope in ("full", "late"):
            folder = output_dir / "vbench" / job["name"] / scope
            folder.mkdir(parents=True, exist_ok=True)
            input_dir = videos
            clips = []
            if scope == "late":
                input_dir = folder / "clips"
                input_dir.mkdir(exist_ok=True)
                for source in sorted(videos.glob("*.mp4")):
                    clips.append(_late_video(source, input_dir / (source.stem + ".avi"),
                                             float(settings.get("late_fraction", 0.5))))
            mapping = {}
            for path in sorted(input_dir.iterdir()):
                if path.suffix.lower() in (".mp4", ".avi", ".mov"):
                    index = int(path.stem.split("_")[0])
                    if index >= len(prompts):
                        raise ValueError(f"video index does not map to a generation prompt: {path}")
                    mapping[path.name] = prompts[index]
            prompt_file = folder / "prompts.json"
            prompt_file.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
            command = [str(settings.get("python", config.get("python", sys.executable))),
                       str(ROOT / "scripts/evaluation/evaluate_vbench.py"), str(input_dir),
                       "--output-dir", str(folder / "results"), "--prompt-file", str(prompt_file),
                       "--config-out", str(folder / "evaluation_config.json"),
                       "--device", str(settings.get("device", "auto"))]
            if settings.get("dimensions"):
                dimensions = settings["dimensions"]
                command += ["--dimensions", ",".join(dimensions) if isinstance(dimensions, list) else str(dimensions)]
            if settings.get("local_checkpoints", True):
                command.append("--local-checkpoints")
            if config.get("resume", False):
                command.append("--reuse-complete")
            with (folder / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            row = {"job": job["name"], "scope": scope, "command": command,
                   "status": "complete" if result.returncode == 0 else "failed",
                   "returncode": result.returncode, "results": str(folder / "results"), "clips": clips}
            rows.append(row)
            (output_dir / "vbench_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"VBench {scope} evaluation failed; see {folder / 'run.log'}")
    return {"status": "complete", "protocol": "existing_vbench_long", "rows": rows}


def evaluate(config: dict, output_dir: Path) -> dict:
    from rebuttal.experiments.temporal_window import run as windows
    quality_config = dict(config)
    if not (config.get("pairs") or config.get("pairs_manifest") or
            (config.get("reference_dir") and config.get("candidate_dir"))):
        quality_config["pairs"] = generation_pairs(config, output_dir)
    quality = windows(quality_config, "evaluate", output_dir / "quality")
    report = {"status": "evaluated", "fidelity": quality}
    if config.get("vbench"):
        report["vbench"] = evaluate_vbench(config, output_dir)
    (output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
