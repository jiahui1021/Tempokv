#!/usr/bin/env python3
"""Run a resumable VBench-Long matrix on independent GPU lanes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any


DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
    "clip_score",
)
PROTOCOL = "vbench_long_custom8_derived"
VIDEO_NAME = re.compile(r"^(?P<index>\d+)(?:_0)?\.mp4$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--devices", required=True, help="Comma-separated GPU UUIDs")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    parser.add_argument("--smoke-count", type=int, default=2)
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if not isinstance(payload.get("entries"), list) or not payload["entries"]:
        raise ValueError("manifest entries must be a non-empty list")
    names = [entry.get("name") for entry in payload["entries"]]
    if not all(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
        raise ValueError("entry names must be filesystem-safe strings")
    if len(names) != len(set(names)):
        raise ValueError("manifest entry names must be unique")
    for entry in payload["entries"]:
        if not isinstance(entry.get("lane"), int) or entry["lane"] < 0:
            raise ValueError(f"invalid lane for {entry['name']}")
        videos = Path(entry.get("videos", "")).expanduser()
        if not videos.is_absolute():
            raise ValueError(f"videos must be absolute for {entry['name']}")
    return payload


def _video_index(name: str) -> int:
    match = VIDEO_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"unsupported video filename: {name}")
    return int(match.group("index"))


def _inventory(videos_dir: Path, expected_count: int) -> list[Path]:
    videos = sorted(
        path for path in videos_dir.iterdir() if path.is_file() and path.suffix == ".mp4"
    )
    if len(videos) != expected_count:
        raise ValueError(
            f"{videos_dir} contains {len(videos)} MP4s; expected {expected_count}"
        )
    by_index = {_video_index(path.name): path for path in videos}
    if len(by_index) != expected_count or set(by_index) != set(range(expected_count)):
        raise ValueError(f"{videos_dir} does not cover indices 0..{expected_count - 1}")
    empty = [path.name for path in videos if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"empty videos in {videos_dir}: {', '.join(empty[:10])}")
    return [by_index[index] for index in range(expected_count)]


def _prompt_mapping(videos: list[Path], prompts: list[str]) -> dict[str, str]:
    return {video.name: prompts[_video_index(video.name)] for video in videos}


def _nvidia_memory() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory = {}
    for line in result.stdout.splitlines():
        uuid, used = (value.strip() for value in line.split(",", 1))
        memory[uuid] = int(used)
    return memory


def _wait_for_inputs(
    marker: Path | None,
    devices: list[str],
    *,
    poll_seconds: int,
    idle_memory_mib: int,
) -> None:
    last_message = 0.0
    while True:
        marker_ready = marker is None or marker.is_file()
        memory = _nvidia_memory()
        missing = [device for device in devices if device not in memory]
        if missing:
            raise RuntimeError(f"GPU UUIDs are not visible: {', '.join(missing)}")
        gpu_ready = all(memory[device] <= idle_memory_mib for device in devices)
        if marker_ready and gpu_ready:
            return
        now = time.time()
        if now - last_message >= 60:
            print(
                "waiting:",
                f"marker={marker_ready}",
                "gpu_memory_mib=" + ",".join(f"{device}:{memory[device]}" for device in devices),
                flush=True,
            )
            last_message = now
        time.sleep(poll_seconds)


def _validate_evaluation(output_dir: Path, name: str, expected_count: int) -> dict[str, Any]:
    report = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
    if report.get("status") not in {"complete", "reused"}:
        raise RuntimeError(f"{name} evaluation status is {report.get('status')}")
    if report.get("video_count") != expected_count:
        raise RuntimeError(f"{name} video_count is {report.get('video_count')}")
    if report.get("dimensions") != list(DIMENSIONS):
        raise RuntimeError(f"{name} dimensions do not match the frozen protocol")
    if report.get("completed_dimensions") != list(DIMENSIONS):
        raise RuntimeError(f"{name} completed_dimensions are incomplete")
    if report.get("protocol") != PROTOCOL:
        raise RuntimeError(f"{name} protocol is not {PROTOCOL}")
    result_files = []
    for dimension in DIMENSIONS:
        path = output_dir / f"{name}_{dimension}_eval_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = _numeric_values(payload.get(dimension))
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{name}/{dimension} is missing or non-finite")
        result_files.append(str(path))
    summary_path = output_dir / f"{name}_vbench_long_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != PROTOCOL:
        raise RuntimeError(f"{name} summary protocol is not {PROTOCOL}")
    display_scores = summary.get("display_scores")
    if not isinstance(display_scores, dict):
        raise RuntimeError(f"{name} summary has no display_scores")
    display_values = _numeric_values(display_scores)
    if len(display_values) != 8 or not all(
        math.isfinite(value) for value in display_values
    ):
        raise RuntimeError(f"{name} summary does not contain eight finite display scores")
    return {
        "name": name,
        "status": report["status"],
        "video_count": expected_count,
        "input_fingerprint": report["input_fingerprint"],
        "result_files": result_files,
        "summary_file": str(summary_path),
    }


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numeric_values(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numeric_values(item)]
    return []


def _run_one(
    *,
    evaluator: Path,
    python: str,
    vbench_root: Path,
    cache_dir: Path,
    device: str,
    name: str,
    videos: Path,
    prompt_file: Path,
    output_dir: Path,
    expected_count: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    command = [
        python,
        str(evaluator),
        str(videos),
        "--output-dir",
        str(output_dir),
        "--name",
        name,
        "--prompt-file",
        str(prompt_file),
        "--mode",
        "long_custom_input",
        "--device",
        "cuda",
        "--local-checkpoints",
        "--reuse-complete",
        "--config-out",
        str(output_dir / "evaluation.json"),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["VBENCH_CACHE_DIR"] = str(cache_dir)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(vbench_root), str(evaluator.parent), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} =====\n")
        log.write("command=" + json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    accepted = _validate_evaluation(output_dir, name, expected_count)
    accepted["elapsed_seconds"] = time.time() - started
    staging = output_dir / "_vbench_long_input"
    if staging.exists():
        shutil.rmtree(staging)
    return accepted


def _smoke_inputs(
    source: list[Path], destination: Path, count: int
) -> list[Path]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    selected = source[:count]
    for video in selected:
        (destination / video.name).symlink_to(video)
    return [destination / video.name for video in selected]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    vbench_root = args.vbench_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    max_lane = max(entry["lane"] for entry in manifest["entries"])
    if len(devices) <= max_lane:
        raise ValueError("--devices does not cover every manifest lane")
    if not vbench_root.is_dir() or not cache_dir.is_dir():
        raise FileNotFoundError("VBench root or cache directory does not exist")
    evaluator = Path(__file__).resolve().with_name("evaluate_vbench.py")
    expected_count = int(manifest.get("expected_count", 128))
    prompt_text = Path(manifest["prompt_text"]).expanduser().resolve()
    prompts = [line.strip() for line in prompt_text.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) != expected_count:
        raise ValueError(f"prompt file has {len(prompts)} non-empty lines; expected {expected_count}")

    output_root.mkdir(parents=True, exist_ok=True)
    provenance = output_root / "provenance"
    provenance.mkdir(exist_ok=True)
    manifest_copy = provenance / "manifest.json"
    if manifest_path != manifest_copy.resolve():
        shutil.copy2(manifest_path, manifest_copy)
    (provenance / "evaluator_sha256.txt").write_text(
        hashlib.sha256(evaluator.read_bytes()).hexdigest() + "  " + str(evaluator) + "\n",
        encoding="utf-8",
    )
    status_path = output_root / "status.json"
    _atomic_json(status_path, {"status": "waiting_for_inputs", "entries": len(manifest["entries"])})
    _wait_for_inputs(
        None,
        devices[: max_lane + 1],
        poll_seconds=args.poll_seconds,
        idle_memory_mib=args.idle_memory_mib,
    )

    def prepare_entry(entry: dict[str, Any]) -> tuple[list[Path], Path]:
        marker_text = entry.get("wait_for")
        marker = Path(marker_text).expanduser().resolve() if marker_text else None
        _wait_for_inputs(
            marker,
            [devices[entry["lane"]]],
            poll_seconds=args.poll_seconds,
            idle_memory_mib=args.idle_memory_mib,
        )
        videos_dir = Path(entry["videos"]).expanduser().resolve()
        videos = _inventory(videos_dir, expected_count)
        prompt_file = provenance / "prompts" / f"{entry['name']}.json"
        _atomic_json(prompt_file, _prompt_mapping(videos, prompts))
        return videos, prompt_file

    _atomic_json(status_path, {"status": "smoke_running", "entries": len(manifest["entries"])})
    smoke_entry = manifest["entries"][0]
    smoke_inventory, _ = prepare_entry(smoke_entry)
    smoke_name = "smoke_" + smoke_entry["name"]
    smoke_videos = output_root / "smoke" / "videos"
    selected = _smoke_inputs(smoke_inventory, smoke_videos, args.smoke_count)
    smoke_prompts = output_root / "smoke" / "prompts.json"
    _atomic_json(smoke_prompts, _prompt_mapping(selected, prompts))
    _run_one(
        evaluator=evaluator,
        python=args.python,
        vbench_root=vbench_root,
        cache_dir=cache_dir,
        device=devices[smoke_entry["lane"]],
        name=smoke_name,
        videos=smoke_videos,
        prompt_file=smoke_prompts,
        output_dir=output_root / "smoke" / "evaluation",
        expected_count=args.smoke_count,
    )
    (output_root / "smoke.accepted").touch()

    _atomic_json(status_path, {"status": "full_running", "entries": len(manifest["entries"])})

    def run_lane(lane: int) -> list[dict[str, Any]]:
        accepted = []
        for entry in (item for item in manifest["entries"] if item["lane"] == lane):
            name = entry["name"]
            _, prompt_file = prepare_entry(entry)
            accepted.append(
                _run_one(
                    evaluator=evaluator,
                    python=args.python,
                    vbench_root=vbench_root,
                    cache_dir=cache_dir,
                    device=devices[lane],
                    name=name,
                    videos=Path(entry["videos"]).expanduser().resolve(),
                    prompt_file=prompt_file,
                    output_dir=output_root / "full" / name,
                    expected_count=expected_count,
                )
            )
        return accepted

    lanes = sorted({entry["lane"] for entry in manifest["entries"]})
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [executor.submit(run_lane, lane) for lane in lanes]
        accepted = [item for future in futures for item in future.result()]
    accepted.sort(key=lambda item: item["name"])
    acceptance = {
        "status": "complete",
        "evaluator": "vbench_long",
        "mode": "long_custom_input",
        "protocol": PROTOCOL,
        "dimensions": list(DIMENSIONS),
        "entries": accepted,
    }
    _atomic_json(output_root / "acceptance.json", acceptance)
    _atomic_json(status_path, acceptance)
    (output_root / "evaluation.accepted").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
