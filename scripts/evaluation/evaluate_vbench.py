#!/usr/bin/env python3
"""Validate a video directory and run the official VBench Python package."""

from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Any


# VBench states that customized videos and prompts are supported for exactly
# these dimensions.  `temporal_flickering` and `overall_consistency` are part
# of the standard suite only: the first additionally needs VBench's static
# filtering, and neither is defined for arbitrary input, so asking for them
# with `--mode custom_input` yields numbers that do not mean what the name
# suggests.  See "Evaluate Your Own Videos" in the VBench README.
CUSTOM_INPUT_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
STANDARD_ONLY_DIMENSIONS = (
    "temporal_flickering",
    "overall_consistency",
)
DEFAULT_DIMENSIONS = CUSTOM_INPUT_DIMENSIONS
DEFAULT_STANDARD_DIMENSIONS = CUSTOM_INPUT_DIMENSIONS + STANDARD_ONLY_DIMENSIONS
VIDEO_EXTENSIONS = (".mp4", ".gif")
DIMENSION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official VBench evaluation or validate and print its configuration."
    )
    parser.add_argument("videos", type=Path, help="Directory of MP4 or GIF videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", help="Result name (defaults to the video directory name)")
    parser.add_argument(
        "--dimensions",
        help=(
            "Comma-separated VBench dimensions; defaults to the six VBench "
            "supports for custom input, or all eight in standard mode"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("custom_input", "vbench_standard"),
        default="custom_input",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="JSON object mapping each video filename to its prompt (custom mode only)",
    )
    parser.add_argument(
        "--full-info",
        type=Path,
        help="Official VBench_full_info.json (required for standard mode)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--local-checkpoints", action="store_true")
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and emit configuration without importing VBench",
    )
    parser.add_argument("--config-out", type=Path, help="Write the validated JSON config")
    parser.add_argument(
        "--reuse-complete",
        action="store_true",
        help="Reuse --config-out when its completed input fingerprint still matches",
    )
    return parser


def _dimensions(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("--dimensions must contain at least one dimension")
    if len(values) != len(set(values)):
        raise ValueError("--dimensions contains duplicate values")
    invalid = [value for value in values if DIMENSION_NAME.fullmatch(value) is None]
    if invalid:
        raise ValueError(f"invalid VBench dimension name(s): {', '.join(invalid)}")
    return values


def _videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {directory}")
    videos = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise ValueError(f"no MP4 or GIF videos found in {directory}")
    empty = [path.name for path in videos if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"empty video file(s): {', '.join(empty[:10])}")
    return videos


def _prompts(path: Path | None, videos: list[Path], mode: str) -> dict[str, str]:
    if path is None:
        return {}
    if mode != "custom_input":
        raise ValueError("--prompt-file can only be used with --mode custom_input")
    if not path.is_file():
        raise FileNotFoundError(f"prompt file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"prompt file is not valid JSON: {path}: {error.msg}") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in payload.items()
    ):
        raise ValueError("prompt file must be a JSON object of filename-to-prompt strings")
    required = {video.name for video in videos}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"prompt file is missing video filename(s): {', '.join(missing[:10])}")
    return {name: payload[name] for name in sorted(required)}


def _safe_name(value: str) -> str:
    if not value or value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError("--name must contain only letters, digits, '.', '_' or '-'")
    return value


def _fingerprint(
    videos: list[Path],
    *,
    name: str,
    dimensions: tuple[str, ...],
    mode: str,
    prompts: dict[str, str],
    full_info: Path | None,
    device: str,
    local_checkpoints: bool,
    read_frame: bool,
) -> str:
    inventory = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in videos
    ]
    full_info_stat = None
    if full_info is not None:
        stat = full_info.stat()
        full_info_stat = {
            "path": str(full_info),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    payload = {
        "videos_path": str(videos[0].parent),
        "videos": inventory,
        "name": name,
        "dimensions": dimensions,
        "mode": mode,
        "prompts": prompts,
        "full_info": full_info_stat,
        "device": device,
        "local_checkpoints": local_checkpoints,
        "read_frame": read_frame,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    videos_dir = args.videos.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    video_paths = _videos(videos_dir)
    if args.dimensions:
        dimensions = _dimensions(args.dimensions)
    elif args.mode == "vbench_standard":
        dimensions = DEFAULT_STANDARD_DIMENSIONS
    else:
        dimensions = CUSTOM_INPUT_DIMENSIONS
    if args.mode == "custom_input":
        unsupported = [
            name for name in dimensions if name in STANDARD_ONLY_DIMENSIONS
        ]
        if unsupported:
            raise ValueError(
                "VBench does not support "
                + ", ".join(unsupported)
                + " for custom input; drop them or run --mode vbench_standard "
                "with --full-info"
            )
    name = _safe_name(args.name or videos_dir.name)
    prompt_path = args.prompt_file.expanduser().resolve() if args.prompt_file else None
    prompts = _prompts(prompt_path, video_paths, args.mode)
    full_info = args.full_info.expanduser().resolve() if args.full_info else None
    if args.mode == "vbench_standard":
        if full_info is None:
            raise ValueError("--full-info is required with --mode vbench_standard")
        if not full_info.is_file():
            raise FileNotFoundError(f"VBench full-info file does not exist: {full_info}")
    elif full_info is not None and not full_info.is_file():
        raise FileNotFoundError(f"VBench full-info file does not exist: {full_info}")
    if args.device != "auto" and re.fullmatch(r"cpu|cuda(?::\d+)?", args.device) is None:
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N")

    config = {
        "videos_path": str(videos_dir),
        "video_count": len(video_paths),
        "output_path": str(output_dir),
        "name": name,
        "dimensions": list(dimensions),
        "mode": args.mode,
        "prompt_file": str(prompt_path) if prompt_path else None,
        "full_info": str(full_info) if full_info else None,
        "device": args.device,
        "local_checkpoints": bool(args.local_checkpoints),
        "read_frame": bool(args.read_frame),
        "input_fingerprint": _fingerprint(
            video_paths,
            name=name,
            dimensions=dimensions,
            mode=args.mode,
            prompts=prompts,
            full_info=full_info,
            device=args.device,
            local_checkpoints=bool(args.local_checkpoints),
            read_frame=bool(args.read_frame),
        ),
    }
    return config, prompts


def _device(name: str):
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("VBench evaluation requires PyTorch to be installed") from error
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if selected == "auto":
        selected = "cpu"
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {selected!r} was requested but CUDA is unavailable")
    return torch.device(selected)


def run(config: dict[str, Any], prompts: dict[str, str]) -> Any:
    try:
        module = importlib.import_module("vbench")
        evaluator_type = getattr(module, "VBench")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "official VBench is not installed; install the VBench package or use --dry-run"
        ) from error

    output_dir = Path(config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    full_info = config["full_info"] or str(output_dir / "unused_full_info.json")
    evaluator = evaluator_type(_device(config["device"]), full_info, str(output_dir))
    return evaluator.evaluate(
        videos_path=config["videos_path"],
        name=config["name"],
        prompt_list=prompts,
        dimension_list=config["dimensions"],
        local=config["local_checkpoints"],
        read_frame=config["read_frame"],
        mode=config["mode"],
        imaging_quality_preprocessing_mode="longer",
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    config, prompts = prepare(args)
    destination = args.config_out.expanduser() if args.config_out else None
    if args.reuse_complete and destination is None:
        raise ValueError("--reuse-complete requires --config-out")
    if args.reuse_complete and destination is not None and destination.is_file():
        try:
            previous = json.loads(destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None
        if (
            isinstance(previous, dict)
            and previous.get("status") == "complete"
            and previous.get("input_fingerprint") == config["input_fingerprint"]
        ):
            previous["status"] = "reused"
            print(json.dumps(previous, indent=2, ensure_ascii=False))
            return previous
    if not args.dry_run:
        run(config, prompts)
        config["status"] = "complete"
    else:
        config["status"] = "dry_run"
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(config, indent=2, ensure_ascii=False))
    return config


if __name__ == "__main__":
    main()
