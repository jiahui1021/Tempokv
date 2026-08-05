from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
from typing import Sequence

from bridge import install


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = list(args.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    inference = (
        args.model_repo.resolve()
        / "experiments"
        / "Self-Forcing"
        / "inference.py"
    )
    if not inference.is_file():
        parser.error(f"Self-Forcing inference script is missing: {inference}")
    cache_root = os.environ.get("TEMPOKV_TRITON_CACHE")
    if cache_root and not os.environ.get("TRITON_CACHE_DIR"):
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        cache_dir = Path(cache_root).expanduser().resolve() / f"rank-{rank}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    install(args.model_repo)
    sys.argv = [str(inference), *arguments]
    runpy.run_path(str(inference), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
