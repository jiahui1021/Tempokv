#!/usr/bin/env python3
"""Separate suite B launcher; no suite A config or output is reused."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from tempokv.spectral_kv.config import load_config
from tempokv.spectral_kv.method_spec import parse_method


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime", choices=("self_forcing", "causal_forcing"), default="self_forcing")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--frames", type=int, default=189)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--sink", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if set(config.experiments) == {"B_E5_RESIDUAL"}:
        parser.error("B_E5_RESIDUAL uses residual_spectrum.py with an explicit clean-grid NPZ input")
    runtime = "Self-Forcing" if args.runtime == "self_forcing" else "Causal-Forcing/causal_forcing"
    script = ROOT / "experiments" / runtime / "tempokv_adapter/generate.py"
    output = args.output.expanduser().resolve()
    window_label = "full_history" if args.window == -1 else f"window-{args.window}"
    video = (args.video_output or output / "videos" / args.runtime / window_label / config.profile / ("run-" + uuid.uuid4().hex[:12])).expanduser().resolve()
    if not args.dry_run and video.exists() and any(video.iterdir()):
        parser.error("video-output already contains files; choose a new directory to preserve experiment separation")
    command = [sys.executable, str(script), "--mode", "spectral_kv", "--checkpoint", str(args.checkpoint.resolve()),
               "--prompts", str(args.prompts.resolve()), "--output-dir", str(video), "--frames", str(args.frames),
               "--local-attn-size", str(args.window), "--seed", str(args.seed),
               "--start-index", str(args.start_index), "--end-index", str(args.end_index), "--num-gpus", str(args.gpus),
               "--samples", str(args.samples)]
    if args.runtime == "causal_forcing":
        if args.sink != 0:
            parser.error("the Causal-Forcing launcher currently supports --sink 0 only")
        command += ["--model-root", str(args.model_root.resolve())]
    else:
        command += ["--sink-size", str(args.sink)]
    environment = {**os.environ, "SPECTRAL_KV_CONFIG": str(args.config.resolve()),
                   "SPECTRAL_KV_OUTPUT_DIR": str(output)}
    if args.runtime == "self_forcing":
        environment["WAN_MODEL_DIR"] = str(args.model_root.resolve())
    elif args.window != -1:
        # This explicitly selected suite-B window is the repository's
        # sliding-window ablation, which its existing launcher requires labeled.
        environment["CAUSAL_FORCING_BENCHMARK"] = "1"
    if args.dry_run:
        command.append("--dry-run")
    print(json.dumps({"suite_id": config.suite_id, "experiments": config.experiments, "profile": config.profile,
                      "trajectory_changed": config.online and parse_method(config.method).family != "full", "command": command}, ensure_ascii=False), flush=True)
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
