#!/usr/bin/env python3
"""Run the remaining 717-frame steps of a span manifest, skipping completed ones.

Mirrors run_study.py's _execute: skips steps whose expected artifact exists,
writes per-step logs under <output>/logs/, and injects FORCING_WAN_MODEL_DIR.
Also sets a dedicated TRITON_CACHE_DIR (via TEMPOKV_TRITON_CACHE) so parallel
experiments on other GPUs never share a Triton cache with this run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triton-cache", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.output.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    if "WAN_MODEL_DIR" not in env and "FORCING_WAN_MODEL_DIR" not in env:
        raise ValueError("set WAN_MODEL_DIR or FORCING_WAN_MODEL_DIR")
    if "WAN_MODEL_DIR" in env:
        env["FORCING_WAN_MODEL_DIR"] = env["WAN_MODEL_DIR"]
    if args.triton_cache is not None:
        cache = args.triton_cache.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        env["TEMPOKV_TRITON_CACHE"] = str(cache)
        env["TRITON_CACHE_DIR"] = str(cache / "span")

    skipped = 0
    failed: list[str] = []
    for step in manifest["steps"]:
        expected = Path(step["expected"])
        if expected.is_file():
            skipped += 1
            continue
        log = args.output.resolve() / "logs" / f"{step['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[run] {step['id']} -> {log.name}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                step["command"],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            print(f"[FAIL] {step['id']} exit {result.returncode} (see {log.name})", flush=True)
            failed.append(step["id"])
            continue
        if not expected.is_file():
            print(f"[FAIL] {step['id']} did not create {expected}", flush=True)
            failed.append(step["id"])
    print(f"\nDONE: {len(manifest['steps'])} steps, {skipped} skipped, "
          f"{len(manifest['steps']) - skipped - len(failed)} ran, {len(failed)} failed",
          flush=True)
    if failed:
        print("FAILED:", *failed, sep="\n  ", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
