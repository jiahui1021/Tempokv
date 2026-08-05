from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Sequence


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one distributed Self-Forcing inference worker."
    )
    parser.add_argument(
        "--mode", choices=("tempokv", "bf16"), default="tempokv"
    )
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = list(args.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    model_repo = args.model_repo.resolve()
    experiment_root = model_repo / "experiments" / "Self-Forcing"
    inference = (
        model_repo
        / "experiments"
        / "Self-Forcing"
        / "inference.py"
    )
    if not inference.is_file():
        parser.error(f"Self-Forcing inference script is missing: {inference}")
    for directory in (model_repo, experiment_root):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

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

            install(model_repo)
        sys.argv = [str(inference), *arguments]
        runpy.run_path(str(inference), run_name="__main__")
        return 0
    finally:
        if cuda_available:
            try:
                torch.cuda.synchronize(cuda_device)
            except Exception:
                pass
        wall_seconds = time.perf_counter() - started
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
