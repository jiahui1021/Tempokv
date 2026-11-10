"""Run one Python script while holding the experiment's verified CUDA limit."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: strict_python.py SCRIPT [ARG ...]")
        return 0 if len(sys.argv) >= 2 else 2
    target = str(Path(sys.argv[1]).resolve())
    arguments = sys.argv[2:]
    from rebuttal.experiments.memory.hard_memory_limit import (
        acquire_verified_limit, apply_torch_limit, verify_current_client,
    )
    with acquire_verified_limit():
        apply_torch_limit()
        verify_current_client()
        sys.argv = [target, *arguments]
        # Match direct Python script imports, while retaining repository imports.
        sys.path.insert(0, str(Path(target).parent))
        runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
