#!/usr/bin/env python3
"""Launch a pinned Rolling Forcing checkout with the TempoKV bridge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run external Rolling Forcing with TempoKV or native BF16 KV."
    )
    parser.add_argument(
        "--mode", choices=("tempokv", "bf16", "trace"), default=os.environ.get("MODE", "tempokv")
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Pinned Rolling Forcing checkout containing inference.py",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="TempoKV runtime report path (defaults to TEMPOKV_ROLLING_REPORT)",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="clean_trace.npz path (required for --mode trace)",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime_root = args.runtime_root.expanduser().resolve()
    inference = runtime_root / "inference.py"
    if not inference.is_file():
        raise FileNotFoundError(f"Rolling Forcing inference.py is missing: {inference}")
    if args.report is not None:
        os.environ["TEMPOKV_ROLLING_REPORT"] = str(args.report.expanduser().resolve())
    if args.trace_output is not None:
        os.environ["TEMPOKV_TRACE_OUTPUT"] = str(args.trace_output.expanduser().resolve())
    if args.mode in {"tempokv", "trace"}:
        from .bridge import install

        install(runtime_root, mode=args.mode)
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    original_cwd = Path.cwd()
    os.chdir(runtime_root)
    try:
        forwarded = list(args.arguments)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        sys.argv = [str(inference), *forwarded]
        runpy.run_path(str(inference), run_name="__main__")
    finally:
        os.chdir(original_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
