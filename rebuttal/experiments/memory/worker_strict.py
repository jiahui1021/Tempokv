"""Original memory worker, admitted only under a verified CUDA hard limit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main():
    from rebuttal.experiments.memory.hard_memory_limit import acquire_verified_limit, apply_torch_limit, verify_current_client
    with acquire_verified_limit():
        apply_torch_limit()
        verify_current_client()
        if len(sys.argv) == 3 and sys.argv[1] == "--strict-memory-smoke":
            import json
            import torch
            from rebuttal.experiments.memory.hard_memory_limit import MIB, gpu_rows
            allocation = torch.empty(64 * MIB, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            audit = verify_current_client()
            result = {"verified": True, "allocation_bytes": allocation.numel(),
                      "torch_fraction": torch.cuda.get_per_process_memory_fraction(0),
                      "audit": audit, "all_devices": gpu_rows()}
            Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
            del allocation
            torch.cuda.synchronize()
            return 0
        from rebuttal.experiments.memory import worker
        from rebuttal.experiments.memory.fast_accounting import FastBudgetManager, TrackedReleasableKVCache
        worker.BudgetManager = FastBudgetManager
        worker.ReleasableKVCache = TrackedReleasableKVCache
        return worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
