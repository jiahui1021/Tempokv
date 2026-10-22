#!/usr/bin/env python3
"""Recompute E3 independently from an opt-in, single-head E2 snapshot."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tempokv.spectral.compression import evaluate_budgets, load_snapshot
from tempokv.spectral.config import SpectralConfig
from tempokv.spectral.sensitivity import reference_attention


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-attention", action="store_true", help="use the Self-Forcing attention backend (CUDA environment)")
    args = parser.parse_args(argv)
    q, k, v, blocks, scores, metadata = load_snapshot(args.input, device=args.device)
    config = SpectralConfig.from_dict(metadata["config"])
    attention_fn = reference_attention
    if args.runtime_attention:
        sys.path.insert(0, str(ROOT / "experiments/Self-Forcing"))
        from wan.modules.attention import attention
        attention_fn = attention
    # LOO scores remain the recorded diagnostic scores; joint-deletion errors
    # and the reference are recomputed with the selected replay backend.
    rows = evaluate_budgets(q, k, v, blocks, scores, config.compression,
                            protected_indices=metadata["protected_indices"], attention_fn=attention_fn,
                            seed=metadata.get("selection_seed", config.seed), eps=config.eps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        columns = [*rows[0], "replay_backend", "replay_device"] if rows else ["status"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({**row, "replay_backend": attention_fn.__module__, "replay_device": args.device} for row in rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
