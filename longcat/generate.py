#!/usr/bin/env python3
"""Launch the external LongCat runner with the TempoKV bridge installed."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from longcat.bridge import (  # noqa: E402
    LongCatRuntimeSpec,
    install_longcat_bridge,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the external LongCat pipeline with TempoKV or collect clean traces."
    )
    parser.add_argument("--mode", choices=("tempokv", "trace"), default="tempokv")
    parser.add_argument(
        "--longcat-repo",
        type=Path,
        required=True,
        help="Quant-VideoGen checkout containing experiments/LongCat",
    )
    parser.add_argument("--config", type=Path, help="TempoKV codec JSON")
    parser.add_argument("--k-params", type=Path, help="fitted temporal K parameters")
    parser.add_argument("--v-params", type=Path, help="fitted CrossKV V parameters")
    parser.add_argument(
        "--triton-cache", type=Path, help="optional Triton compilation cache directory"
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="directory for clean LongCat layer traces (trace mode or optional with tempokv)",
    )
    parser.add_argument("--max-trace-spatial-positions", type=int, default=256)
    parser.add_argument(
        "--stride",
        type=int,
        help=(
            "expected local LongCat cache stride; omit to use the default "
            "1560 for CP=1 or runtime validation for CP>1"
        ),
    )
    return parser


def _option_value(arguments: list[str], name: str) -> str | None:
    for index in range(len(arguments) - 1, -1, -1):
        argument = arguments[index]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def _validate_stage_options(arguments: list[str]) -> None:
    cp_size = _option_value(arguments, "--context_parallel_size")
    if cp_size is not None and int(cp_size) <= 0:
        raise ValueError("context_parallel_size must be positive")
    sink_frames = _option_value(arguments, "--attn_sink_frames")
    if sink_frames is not None and int(sink_frames) < 0:
        raise ValueError("attn_sink_frames must be non-negative")


def _external_arguments(
    arguments: list[str], *, trace: bool = False
) -> list[str]:
    _validate_stage_options(arguments)
    result = list(arguments)
    if (
        "--no_offload_kv_cache" not in result
        and "--offload_kv_cache" not in result
    ):
        result.append("--no_offload_kv_cache")
    if _option_value(result, "--context_parallel_size") is None:
        result.extend(("--context_parallel_size", "1"))
    if _option_value(result, "--attn_sink_frames") is None:
        result.extend(("--attn_sink_frames", "0"))
    if trace:
        # Parameter fitting must see the clean LongCat cache.  Put this last
        # so an accidentally forwarded qvg quantizer option cannot alter the
        # trace run.
        result.extend(("--quant_type", "none"))
    return result


def run(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    known, external = parser.parse_known_args(argv)
    if known.stride is not None and known.stride <= 0:
        parser.error("--stride must be positive")
    external = _external_arguments(external, trace=known.mode == "trace")
    context_parallel_size = int(_option_value(external, "--context_parallel_size") or "1")
    attention_sink_frames = int(_option_value(external, "--attn_sink_frames") or "0")
    rank = int(os.environ.get("RANK", "0"))
    trace_dir = known.trace_dir
    if trace_dir is not None:
        # Every CP rank sees a different spatial shard.  Keep the shards in
        # separate directories so event creation remains race-free; the trace
        # preparation script merges them recursively.
        trace_dir = trace_dir / f"rank_{rank:05d}"

    if known.mode == "tempokv":
        if known.config is None or known.k_params is None or known.v_params is None:
            parser.error("tempokv mode requires --config, --k-params, and --v-params")
        bridge = install_longcat_bridge(
            config_path=known.config,
            k_parameters_path=known.k_params,
            v_parameters_path=known.v_params,
            triton_cache=known.triton_cache,
            mode="tempokv",
            trace_dir=trace_dir,
            max_trace_spatial_positions=known.max_trace_spatial_positions,
            spec=(
                LongCatRuntimeSpec(stride=known.stride)
                if known.stride is not None
                else None
            ),
            context_parallel_size=context_parallel_size,
            attention_sink_frames=attention_sink_frames,
            model_repo=known.longcat_repo,
        )
    else:
        bridge = install_longcat_bridge(
            mode="trace",
            trace_dir=trace_dir
            or PROJECT_ROOT / "parameters" / "longcat_raw_trace" / f"rank_{rank:05d}",
            max_trace_spatial_positions=known.max_trace_spatial_positions,
            spec=(
                LongCatRuntimeSpec(stride=known.stride)
                if known.stride is not None
                else None
            ),
            context_parallel_size=context_parallel_size,
            attention_sink_frames=attention_sink_frames,
            model_repo=known.longcat_repo,
        )

    repo = known.longcat_repo.expanduser().resolve()
    project_root = repo
    if repo.name == "LongCat" and repo.parent.name == "experiments":
        project_root = repo.parent.parent
    runner_root = project_root / "experiments" / "LongCat"
    runner_path = runner_root / "run_long_t2v.py"
    if not runner_path.is_file():
        parser.error(f"LongCat runner not found: {runner_path}")
    if str(runner_root) not in sys.path:
        sys.path.insert(0, str(runner_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    runner = importlib.import_module("run_long_t2v")
    original_get_model_and_pipe = runner.get_model_and_pipe

    def get_model_and_pipe(*args, **kwargs):
        dit, pipeline = original_get_model_and_pipe(*args, **kwargs)
        bridge.attach(dit, pipeline)
        return dit, pipeline

    runner.get_model_and_pipe = get_model_and_pipe
    os.environ.setdefault("TEMPOKV_LONGCAT_MODE", known.mode)
    sys.argv = [str(runner_path), *external]
    external_args = runner._parse_args()
    Path(external_args.output_dir).expanduser().resolve().mkdir(
        parents=True, exist_ok=True
    )
    runner.generate(external_args)
    report_path = (
        Path(external_args.output_dir).expanduser().resolve()
        / f"tempokv_bridge_report_rank_{rank:05d}.json"
    )
    report_path.write_text(
        json.dumps(bridge.report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except ImportError as error:
        print(
            "LongCat dependencies are unavailable; install the LongCat extras "
            f"for Quant-VideoGen first: {error}",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"LongCat TempoKV error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
