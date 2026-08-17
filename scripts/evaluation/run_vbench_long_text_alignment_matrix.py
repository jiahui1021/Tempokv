#!/usr/bin/env python3
"""Wait for a visual VBench-Long matrix, then add text alignment and aggregate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import evaluate_vbench


TEXT_DIMENSIONS = ("overall_consistency", "clip_score")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--devices", required=True, help="Comma-separated GPU UUIDs")
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _wait(path: Path, poll_seconds: int) -> None:
    while not path.is_file():
        print(f"waiting for {path}", flush=True)
        time.sleep(poll_seconds)


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
    return {
        uuid.strip(): int(used.strip())
        for uuid, used in (line.split(",", 1) for line in result.stdout.splitlines())
    }


def _wait_for_idle(device: str, poll_seconds: int, idle_memory_mib: int) -> None:
    while True:
        memory = _nvidia_memory()
        if device not in memory:
            raise RuntimeError(f"GPU UUID is not visible: {device}")
        if memory[device] <= idle_memory_mib:
            return
        print(f"waiting for idle GPU {device}: {memory[device]} MiB", flush=True)
        time.sleep(poll_seconds)


def _scalar(path: Path, dimension: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        value = float(payload[dimension][0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(f"malformed {dimension} result: {path}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {dimension} result: {path}")
    return value


def _collect_scores(output_dir: Path, name: str) -> dict[str, float]:
    dimensions = evaluate_vbench.CUSTOM_INPUT_DIMENSIONS
    return {
        dimension: _scalar(
            output_dir / f"{name}_{dimension}_eval_results.json", dimension
        )
        for dimension in dimensions
    }


def _run_one(
    *,
    entry: dict[str, Any],
    output_root: Path,
    evaluator: Path,
    python: str,
    vbench_root: Path,
    cache_dir: Path,
    device: str,
    poll_seconds: int,
    idle_memory_mib: int,
) -> dict[str, Any]:
    name = entry["name"]
    output_dir = output_root / "full" / name
    visual_report = output_dir / "evaluation.json"
    if not visual_report.is_file():
        raise FileNotFoundError(f"missing completed visual report: {visual_report}")
    _wait_for_idle(device, poll_seconds, idle_memory_mib)
    prompt_file = output_root / "provenance" / "prompts" / f"{name}.json"
    command = [
        python,
        str(evaluator),
        entry["videos"],
        "--output-dir",
        str(output_dir),
        "--name",
        name,
        "--prompt-file",
        str(prompt_file),
        "--mode",
        "long_custom_input",
        "--dimensions",
        ",".join(TEXT_DIMENSIONS),
        "--device",
        "cuda",
        "--local-checkpoints",
        "--reuse-complete",
        "--config-out",
        str(output_dir / "text_alignment_evaluation.json"),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["VBENCH_CACHE_DIR"] = str(cache_dir)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(vbench_root), str(vbench_root / "competitions"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started = time.time()
    with (output_dir / "text_alignment.log").open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} =====\n")
        log.flush()
        subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    scores = _collect_scores(output_dir, name)
    summary = evaluate_vbench._write_summary(output_dir, name, scores)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    table_scores = dict(summary_payload["official_competition_aspects"])
    table_scores["visual_quality"] = summary_payload["display_scores"]["visual_quality"]
    staging = output_dir / evaluate_vbench.STAGING_DIR_NAME
    if staging.exists():
        shutil.rmtree(staging)
    return {
        "name": name,
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "summary": str(summary),
        "scores": table_scores,
    }


def _write_table(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# CF MovieGen128 VBench-Long aggregation",
        "",
        "| Method | Visual Quality | Temporal Quality | Frame-wise Quality | Text Alignment |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        scores = entry["scores"]
        lines.append(
            f"| {entry['name']} | {scores['visual_quality']:.6f} | "
            f"{scores['temporal_quality']:.6f} | "
            f"{scores['frame_wise_quality']:.6f} | {scores['text_alignment']:.6f} |"
        )
    lines.extend(
        [
            "",
            "> Visual Quality is the arithmetic mean of the six visual dimensions. "
            "It is not official VBench Total.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
    output_root = args.output_root.expanduser().resolve()
    vbench_root = args.vbench_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    lanes = sorted({entry["lane"] for entry in manifest["entries"]})
    if len(devices) <= max(lanes):
        raise ValueError("--devices does not cover every lane")
    _atomic_json(output_root / "competition_status.json", {"status": "waiting"})
    _wait(args.wait_for.expanduser().resolve(), args.poll_seconds)
    _atomic_json(output_root / "competition_status.json", {"status": "running"})
    evaluator = Path(__file__).resolve().with_name("evaluate_vbench.py")

    def run_lane(lane: int) -> list[dict[str, Any]]:
        return [
            _run_one(
                entry=entry,
                output_root=output_root,
                evaluator=evaluator,
                python=args.python,
                vbench_root=vbench_root,
                cache_dir=cache_dir,
                device=devices[lane],
                poll_seconds=args.poll_seconds,
                idle_memory_mib=args.idle_memory_mib,
            )
            for entry in manifest["entries"]
            if entry["lane"] == lane
        ]

    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [executor.submit(run_lane, lane) for lane in lanes]
        entries = [entry for future in futures for entry in future.result()]
    entries.sort(key=lambda entry: entry["name"])
    acceptance = {
        "status": "complete",
        "protocol": evaluate_vbench.PROTOCOL,
        "official_vbench_total": False,
        "entries": entries,
    }
    _atomic_json(output_root / "competition_acceptance.json", acceptance)
    _atomic_json(output_root / "competition_status.json", acceptance)
    _write_table(output_root / "visual_quality_text_alignment.md", entries)
    (output_root / "competition.accepted").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
