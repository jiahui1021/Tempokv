"""Run the original memory worker with incremental accounting only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rebuttal.experiments.memory import worker
from rebuttal.experiments.memory.fast_accounting import (
    FastBudgetManager, TrackedReleasableKVCache,
)

worker.BudgetManager = FastBudgetManager
worker.ReleasableKVCache = TrackedReleasableKVCache

if __name__ == "__main__":
    worker.main()
