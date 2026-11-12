#!/usr/bin/env python3
"""Launch a paired native Self-Forcing cache-recovery experiment on one GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tempokv.research.cache_recovery import BRANCHES, SCHEMA, RecoverySettings
from tempokv.research.io import atomic_json, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output-dir", type=Path, help="new output directory")
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bits", type=int, choices=(2, 4, 16), default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--fork-chunk", type=int, default=6)
    parser.add_argument("--future-chunks", type=int, default=8)
    parser.add_argument("--protect-chunks", type=int, default=2)
    parser.add_argument("--warm-start-chunk", type=int,
                        help="first low-bit read in the warm prefix, or the fork for a zero-exposure control")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = RecoverySettings(
            args.bits, args.group_size, args.fork_chunk, args.future_chunks, args.protect_chunks,
            args.warm_start_chunk,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.prompt_start < 0 or args.prompt_count < 1 or args.seed < 0:
        parser.error("prompt start and seed must be nonnegative; prompt count must be positive")
    runtime = ROOT / "experiments" / "Self-Forcing"
    output = args.output_dir.expanduser().resolve() if args.output_dir else None
    paths = {
        name: str(path.expanduser().resolve()) if path else None
        for name, path in (("checkpoint", args.checkpoint), ("model_root", args.model_root), ("prompts", args.prompts))
    }
    latent_frames = 3 * (settings.fork_chunk + settings.future_chunks)
    plan = {
        "schema_version": SCHEMA, "settings": asdict(settings), "branches": BRANCHES,
        "environment": environment(), "output_dir": str(output) if output else None,
        "prompt_start": args.prompt_start, "prompt_count": args.prompt_count, "base_seed": args.seed,
        "latent_frames": latent_frames, "runtime": paths,
        "coordinate": "native BF16 post-RoPE K / raw V",
        "quantization": "symmetric groupwise rounding and BF16 reconstruction; no packed storage",
        "attention": "full history; native kernel; no sink; recent and current chunks protected during rounding",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if not args.checkpoint or not Path(paths["checkpoint"]).is_file():
        parser.error("--checkpoint must identify an existing local file")
    if not args.model_root or not Path(paths["model_root"]).is_dir():
        parser.error("--model-root must identify an existing local directory")
    if not args.prompts or not Path(paths["prompts"]).is_file():
        parser.error("--prompts must identify an existing local file")
    if output is None:
        parser.error("--output-dir is required")
    prompt_data = Path(paths["prompts"]).read_bytes()
    if args.prompt_start + args.prompt_count > len(prompt_data.splitlines()):
        parser.error("selected prompt indices exceed the file")
    plan["runtime"]["prompt_file_sha256"] = hashlib.sha256(prompt_data).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    launch = output / "launch.json"
    atomic_json(launch, plan)
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("TEMPOKV_") and key not in ("RESEARCH_ENABLE", "SELF_FORCING_HORIZON_PROBE")
    }
    env.update(RESEARCH_CACHE_RECOVERY_LAUNCH=str(launch), SELF_FORCING_CACHE_ROPE_PRECISION="complex64",
               FORCING_WAN_MODEL_DIR=paths["model_root"], WAN_MODEL_DIR=paths["model_root"])
    command = [
        sys.executable, str(runtime / "tempokv_adapter" / "worker.py"), "--mode", "cache_recovery", "--",
        "--config_path", str(runtime / "configs" / "self_forcing_dmd.yaml"),
        "--checkpoint_path", paths["checkpoint"], "--data_path", paths["prompts"],
        "--output_folder", str(output / "inference"),
        "--num_output_frames", str(latent_frames), "--num_samples", "1",
        "--seed", str(args.seed), "--start_index", str(args.prompt_start),
        "--end_index", str(args.prompt_start + args.prompt_count),
        "--save_with_index", "--use_ema", "--local_attn_size", "-1", "--sink_size", "0",
    ]
    result = subprocess.run(command, cwd=runtime, env=env, check=False)
    if result.returncode:
        return result.returncode
    for index in range(args.prompt_start, args.prompt_start + args.prompt_count):
        path = output / f"prompt-{index}-seed-{args.seed + index * 1_000_003}.json"
        if not path.is_file() or json.loads(path.read_text())["status"] != "completed":
            raise RuntimeError(f"missing completed result for prompt {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
