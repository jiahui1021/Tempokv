from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _write_metrics(
    metrics_dir: Path,
    *,
    rank: int,
    wall_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    gpu_name: str,
    decode_routes: dict[str, int],
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"rank-{rank}.json"
    temporary = metrics_dir / f"rank-{rank}.json.tmp"
    payload = {
        "rank": rank,
        "wall_seconds": wall_seconds,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "gpu_name": gpu_name,
        "decode_routes": decode_routes,
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_profile(
    profile_dir: Path,
    *,
    rank: int,
    profile,
    instrumentation: dict[str, str],
    decode_routes: dict[str, int],
    wall_seconds: float,
    gpu_name: str,
) -> None:
    profile_dir = profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = profile.report()
    payload.update(
        {
            "rank": rank,
            "gpu_name": gpu_name,
            "wall_seconds": wall_seconds,
            "instrumentation": instrumentation,
            "decode_routes": decode_routes,
        }
    )
    path = profile_dir / f"rank-{rank}.json"
    temporary = profile_dir / f"rank-{rank}.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _install_profile(device_index: int | None):
    """Install CUDA-event profile sections for one worker."""
    import importlib

    import torch

    from tempokv.profiling import RuntimeProfile, install

    importlib.import_module("utils.wan_wrapper")
    importlib.import_module("pipeline.causal_inference")
    device = None if device_index is None else torch.device("cuda", device_index)
    profile = RuntimeProfile(device)
    return profile, install(profile)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one distributed Self-Forcing inference worker."
    )
    parser.add_argument(
        "--mode", choices=("tempokv", "bf16", "trace", "spectral", "spectral_kv"), default="tempokv"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="write a per-section CUDA-event runtime breakdown to this directory",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = list(args.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    runtime_root = PROJECT_ROOT / "experiments" / "Self-Forcing"
    inference = runtime_root / "inference.py"
    if not inference.is_file():
        parser.error(f"Self-Forcing inference script is missing: {inference}")
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    original_cwd = Path.cwd()
    os.chdir(runtime_root)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    metrics_value = os.environ.get("SELF_FORCING_METRICS_DIR")
    metrics_dir = (
        Path(metrics_value).expanduser().resolve() if metrics_value else None
    )

    import torch

    cuda_available = torch.cuda.is_available()
    cuda_device = None
    gpu_name = "CUDA unavailable"
    if cuda_available:
        torch.cuda.set_device(local_rank)
        cuda_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        torch.cuda.synchronize(cuda_device)

    started = time.perf_counter()
    generation_wall_seconds: float | None = None
    profile = None
    instrumentation: dict[str, str] = {}
    try:
        if args.mode == "tempokv":
            cache_root = os.environ.get("TEMPOKV_TRITON_CACHE")
            if cache_root and not os.environ.get("TRITON_CACHE_DIR"):
                cache_dir = (
                    Path(cache_root).expanduser().resolve() / f"rank-{local_rank}"
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
            from bridge import install

            install(runtime_root)
        elif args.mode == "spectral_kv":
            from tempokv.spectral_kv.bridge import install
            install(runtime_root, runtime="self_forcing")
        elif args.mode == "trace":
            from trace_bridge import install

            install(runtime_root)
            arguments.append("--skip_video_save")
        elif args.mode == "spectral":
            from spectral_bridge import install

            install(runtime_root)
        if os.environ.get("RESEARCH_ENABLE") == "1":
            from research_bridge import install as install_research
            install_research(runtime_root)
        if args.profile is not None:
            profile, instrumentation = _install_profile(
                cuda_device if cuda_available else None
            )
        sys.argv = [str(inference), *arguments]
        runtime_namespace = runpy.run_path(str(inference), run_name="__main__")
        measured = (
            runtime_namespace.get("generation_wall_seconds")
            if isinstance(runtime_namespace, dict)
            else None
        )
        if isinstance(measured, (int, float)) and measured >= 0:
            generation_wall_seconds = float(measured)
        return 0
    finally:
        os.chdir(original_cwd)
        if cuda_available:
            try:
                torch.cuda.synchronize(cuda_device)
            except Exception:
                pass
        wall_seconds = (
            generation_wall_seconds
            if generation_wall_seconds is not None
            else time.perf_counter() - started
        )
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
        if cuda_available:
            try:
                peak_allocated_bytes = torch.cuda.max_memory_allocated(cuda_device)
                peak_reserved_bytes = torch.cuda.max_memory_reserved(cuda_device)
            except Exception:
                pass
        decode_routes: dict[str, int] = {}
        if args.mode == "tempokv":
            try:
                from tempokv.triton_ops import route_report

                decode_routes = route_report()
            except Exception:
                pass
        if profile is not None:
            # A profile that fails to serialize must not lose the generation it
            # measured, so the metrics write below still runs.
            try:
                _write_profile(
                    args.profile,
                    rank=rank,
                    profile=profile,
                    instrumentation=instrumentation,
                    decode_routes=decode_routes,
                    wall_seconds=wall_seconds,
                    gpu_name=gpu_name,
                )
            except OSError as error:
                print(f"[worker] could not write profile: {error}", file=sys.stderr)
        if metrics_dir is not None:
            try:
                _write_metrics(
                    metrics_dir,
                    rank=rank,
                    wall_seconds=wall_seconds,
                    peak_allocated_bytes=peak_allocated_bytes,
                    peak_reserved_bytes=peak_reserved_bytes,
                    gpu_name=gpu_name,
                    decode_routes=decode_routes,
                )
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
