#!/usr/bin/env python3
"""Validate videos and run the official VBench-Long evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any


CUSTOM_INPUT_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
DEFAULT_STANDARD_DIMENSIONS = CUSTOM_INPUT_DIMENSIONS + (
    "temporal_flickering",
    "overall_consistency",
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
DIMENSION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
EVALUATOR = "vbench_long"
STAGING_DIR_NAME = "_vbench_long_input"
PREPROCESSING = {
    "sb_clip2clip_feat_extractor": "dinov2",
    "bg_clip2clip_feat_extractor": "dreamsim",
    "imaging_quality_preprocessing_mode": "longer",
    "clip_length_config": "clip_length_mix.yaml",
    "w_inclip": 1.0,
    "w_clip2clip": 0.0,
    "use_semantic_splitting": False,
    "slow_fast_eval_config": "slow_fast_params.yaml",
    "dev_flag": True,
    "sb_mapping_file": "subject_mapping_table.yaml",
    "bg_mapping_file": "background_mapping_table.yaml",
    "num_of_samples_per_prompt": 1,
    "split_rank": 0,
    "split_world_size": 1,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official VBench-Long or validate and print its configuration."
    )
    parser.add_argument("videos", type=Path, help="Directory of MP4, AVI, or MOV videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", help="Result name (defaults to the video directory name)")
    parser.add_argument(
        "--dimensions",
        help=(
            "Comma-separated VBench-Long dimensions; long_custom_input defaults "
            "to the six dimensions officially supported for custom videos"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("long_custom_input", "long_vbench_standard"),
        default="long_custom_input",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help=(
            "JSON object mapping every video filename to its prompt; required "
            "for long_custom_input"
        ),
    )
    parser.add_argument(
        "--full-info",
        type=Path,
        help="Official VBench_full_info.json (required for long_vbench_standard)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and emit configuration without importing VBench-Long",
    )
    parser.add_argument("--config-out", type=Path, help="Write the evaluation JSON")
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
        raise ValueError(f"invalid VBench-Long dimension name(s): {', '.join(invalid)}")
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
        raise ValueError(f"no MP4, AVI, or MOV videos found in {directory}")
    empty = [path.name for path in videos if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"empty video file(s): {', '.join(empty[:10])}")
    return videos


def _prompts(path: Path | None, videos: list[Path], mode: str) -> dict[str, str]:
    if mode == "long_custom_input" and path is None:
        raise ValueError("--prompt-file is required with --mode long_custom_input")
    if path is None:
        return {}
    if mode != "long_custom_input":
        raise ValueError("--prompt-file can only be used with --mode long_custom_input")
    if not path.is_file():
        raise FileNotFoundError(f"prompt file does not exist: {path}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"prompt file contains duplicate filename: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
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


def _staging_mapping(
    videos: list[Path], prompts: dict[str, str], mode: str
) -> list[dict[str, Any]]:
    mapping = []
    for index, video in enumerate(videos):
        staged_name = (
            f"{index}-0{video.suffix.lower()}"
            if mode == "long_custom_input"
            else video.name
        )
        mapping.append(
            {
                "source_name": video.name,
                "staged_name": staged_name,
                "prompt_index": index if mode == "long_custom_input" else None,
                "prompt": prompts.get(video.name),
            }
        )
    staged_names = [entry["staged_name"] for entry in mapping]
    if len(staged_names) != len(set(staged_names)):
        raise ValueError("staging would create duplicate video filenames")
    return mapping


def _fingerprint(
    videos: list[Path],
    *,
    name: str,
    dimensions: tuple[str, ...],
    mode: str,
    prompts: dict[str, str],
    staging_mapping: list[dict[str, Any]],
    full_info: Path | None,
    device: str,
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
        "evaluator": EVALUATOR,
        "videos_path": str(videos[0].parent),
        "videos": inventory,
        "name": name,
        "dimensions": dimensions,
        "mode": mode,
        "prompts": prompts,
        "staging_mapping": staging_mapping,
        "full_info": full_info_stat,
        "device": device,
        "read_frame": read_frame,
        "preprocessing": PREPROCESSING,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    videos_dir = args.videos.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    video_paths = _videos(videos_dir)
    if args.dimensions:
        dimensions = _dimensions(args.dimensions)
    elif args.mode == "long_vbench_standard":
        dimensions = DEFAULT_STANDARD_DIMENSIONS
    else:
        dimensions = CUSTOM_INPUT_DIMENSIONS
    if args.mode == "long_custom_input":
        unsupported = [name for name in dimensions if name not in CUSTOM_INPUT_DIMENSIONS]
        if unsupported:
            raise ValueError(
                "VBench-Long does not support "
                + ", ".join(unsupported)
                + " for custom input; use only the six official custom-input dimensions"
            )
    name = _safe_name(args.name or videos_dir.name)
    prompt_path = args.prompt_file.expanduser().resolve() if args.prompt_file else None
    prompts = _prompts(prompt_path, video_paths, args.mode)
    full_info = args.full_info.expanduser().resolve() if args.full_info else None
    if args.mode == "long_vbench_standard" and full_info is None:
        raise ValueError("--full-info is required with --mode long_vbench_standard")
    if full_info is not None and not full_info.is_file():
        raise FileNotFoundError(f"VBench full-info file does not exist: {full_info}")
    if args.device != "auto" and re.fullmatch(r"cpu|cuda(?::\d+)?", args.device) is None:
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N")

    mapping = _staging_mapping(video_paths, prompts, args.mode)
    config = {
        "evaluator": EVALUATOR,
        "videos_path": str(videos_dir),
        "video_count": len(video_paths),
        "output_path": str(output_dir),
        "staging_path": str(output_dir / STAGING_DIR_NAME),
        "staging_mapping": mapping,
        "name": name,
        "dimensions": list(dimensions),
        "mode": args.mode,
        "prompt_file": str(prompt_path) if prompt_path else None,
        "full_info": str(full_info) if full_info else None,
        "device": args.device,
        "read_frame": bool(args.read_frame),
        "preprocessing": dict(PREPROCESSING),
        "input_fingerprint": _fingerprint(
            video_paths,
            name=name,
            dimensions=dimensions,
            mode=args.mode,
            prompts=prompts,
            staging_mapping=mapping,
            full_info=full_info,
            device=args.device,
            read_frame=bool(args.read_frame),
        ),
    }
    return config, prompts


def _device(name: str):
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("VBench-Long evaluation requires PyTorch") from error
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if selected == "auto":
        selected = "cpu"
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {selected!r} was requested but CUDA is unavailable")
    return torch.device(selected)


def _stage_videos(config: dict[str, Any]) -> Path:
    staging_dir = Path(config["staging_path"])
    output_dir = Path(config["output_path"])
    if staging_dir.parent != output_dir:
        raise RuntimeError("VBench-Long staging directory escaped the output directory")
    if staging_dir.is_symlink() or staging_dir.is_file():
        staging_dir.unlink()
    elif staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    source_dir = Path(config["videos_path"])
    for entry in config["staging_mapping"]:
        source = source_dir / entry["source_name"]
        (staging_dir / entry["staged_name"]).symlink_to(source)
    return staging_dir


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numeric_values(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numeric_values(item)]
    return []


def _validate_result(path: Path, dimension: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"VBench-Long did not produce {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"VBench-Long result is invalid JSON: {path}") from error
    if not isinstance(payload, dict) or dimension not in payload:
        raise RuntimeError(f"VBench-Long result does not contain {dimension}: {path}")
    numbers = _numeric_values(payload[dimension])
    if not numbers or not all(math.isfinite(number) for number in numbers):
        raise RuntimeError(f"VBench-Long result for {dimension} is missing or non-finite")


def _runtime_full_info(module: Any, config: dict[str, Any]) -> Path:
    if config["full_info"]:
        return Path(config["full_info"])
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("cannot locate the installed vbench2_beta_long package")
    full_info = Path(module_file).resolve().parent / "VBench_full_info.json"
    if not full_info.is_file():
        raise RuntimeError(
            "VBench-Long requires VBench_full_info.json; pass --full-info or install "
            "the complete official VBench checkout"
        )
    return full_info


def run(config: dict[str, Any], prompts: dict[str, str]) -> list[str]:
    cache_dir_text = os.environ.get("VBENCH_CACHE_DIR")
    if not cache_dir_text:
        raise RuntimeError("VBENCH_CACHE_DIR must point to the VBench model cache")
    cache_dir = Path(cache_dir_text).expanduser().resolve()
    if not cache_dir.is_dir():
        raise RuntimeError(f"VBENCH_CACHE_DIR does not exist: {cache_dir}")
    try:
        module = importlib.import_module("vbench2_beta_long")
        evaluator_type = getattr(module, "VBenchLong")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "official VBench-Long is not installed; add the complete VBench checkout "
            "to PYTHONPATH or use --dry-run"
        ) from error

    output_dir = Path(config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    full_info = _runtime_full_info(module, config)
    preprocessing = config["preprocessing"]
    config_dir = Path(module.__file__).resolve().parent / "configs"
    required_configs = (
        preprocessing["clip_length_config"],
        preprocessing["slow_fast_eval_config"],
        preprocessing["sb_mapping_file"],
        preprocessing["bg_mapping_file"],
    )
    missing_configs = [
        name for name in required_configs if not (config_dir / name).is_file()
    ]
    if missing_configs:
        raise RuntimeError(
            "VBench-Long package is missing required config file(s): "
            + ", ".join(missing_configs)
        )
    staging_dir = _stage_videos(config)
    evaluator = evaluator_type(_device(config["device"]), str(full_info), str(output_dir))
    prompt_list = [
        prompts[entry["source_name"]]
        for entry in config["staging_mapping"]
        if entry["prompt_index"] is not None
    ]
    completed = []
    for dimension in config["dimensions"]:
        result_path = output_dir / f"{config['name']}_{dimension}_eval_results.json"
        result_path.unlink(missing_ok=True)
        evaluator.evaluate(
            videos_path=str(staging_dir),
            name=config["name"],
            prompt_list=prompt_list,
            dimension=dimension,
            read_frame=config["read_frame"],
            mode=config["mode"],
            sb_clip2clip_feat_extractor=preprocessing["sb_clip2clip_feat_extractor"],
            bg_clip2clip_feat_extractor=preprocessing["bg_clip2clip_feat_extractor"],
            imaging_quality_preprocessing_mode=preprocessing[
                "imaging_quality_preprocessing_mode"
            ],
            clip_length_config=preprocessing["clip_length_config"],
            w_inclip=preprocessing["w_inclip"],
            w_clip2clip=preprocessing["w_clip2clip"],
            use_semantic_splitting=preprocessing["use_semantic_splitting"],
            slow_fast_eval_config=str(
                config_dir / preprocessing["slow_fast_eval_config"]
            ),
            dev_flag=preprocessing["dev_flag"],
            sb_mapping_file_path=str(config_dir / preprocessing["sb_mapping_file"]),
            bg_mapping_file_path=str(config_dir / preprocessing["bg_mapping_file"]),
            num_of_samples_per_prompt=preprocessing["num_of_samples_per_prompt"],
            static_filter_flag=dimension == "temporal_flickering",
            split_rank=preprocessing["split_rank"],
            split_world_size=preprocessing["split_world_size"],
        )
        _validate_result(result_path, dimension)
        completed.append(dimension)
    return completed


def _write_config(destination: Path | None, config: dict[str, Any]) -> None:
    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
            and previous.get("evaluator") == EVALUATOR
            and previous.get("status") == "complete"
            and previous.get("input_fingerprint") == config["input_fingerprint"]
        ):
            previous["status"] = "reused"
            print(json.dumps(previous, indent=2, ensure_ascii=False))
            return previous
    if args.dry_run:
        config["status"] = "dry_run"
        _write_config(destination, config)
    else:
        config["status"] = "running"
        _write_config(destination, config)
        try:
            config["completed_dimensions"] = run(config, prompts)
        except Exception as error:
            config["status"] = "failed"
            config["error"] = {"type": type(error).__name__, "message": str(error)}
            _write_config(destination, config)
            raise
        config["status"] = "complete"
        _write_config(destination, config)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    return config


if __name__ == "__main__":
    main()
