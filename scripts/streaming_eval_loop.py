#!/usr/bin/env python3
"""Streaming evaluation loop: evaluate each generated video against the BF16
reference as soon as it appears in the target directory.

Monitors TGT_DIR for `N_0.mp4` files, evaluates the new ones against the
matching reference in REF_DIR, and appends one CSV row per video so the
metrics stream out while generation is still running.

Usage:
  python streaming_eval_loop.py \
      --ref-dir /path/to/bf16 \
      --target-dir /path/to/videos \
      --output /path/to/streaming_psnr_ssim.csv \
      --expected 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = Path("/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/evaluation/scripts")
sys.path.insert(0, str(EVAL_DIR))

from eval_quality import LpipsMetric, eval_video  # noqa: E402


def _video_index(name: str) -> int | None:
    stem = name[:-4] if name.endswith(".mp4") else name
    if "_" not in stem:
        return None
    prefix, suffix = stem.split("_", 1)
    if not prefix.isdigit():
        return None
    return int(prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-interval", type=float, default=20.0)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    tgt_dir = Path(args.target_dir)
    out_path = Path(args.output)
    max_frames = args.max_frames or None

    lp = LpipsMetric(net=args.lpips_net, device=args.device)
    done: set[int] = set()

    fieldnames = [
        "video", "psnr", "ssim", "lpips", "frames", "mse", "identical_frames",
    ]

    # Legacy CSVs may lack the lpips column; appending 7-field rows to a
    # 6-field header shifts every value.  Migrate such files in place by
    # re-writing them under the current schema (missing fields become "").
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open() as f:
            header = f.readline().strip().split(",")
        if header != fieldnames:
            with out_path.open() as f:
                legacy_rows = list(csv.DictReader(f))
            tmp = out_path.with_suffix(out_path.suffix + ".migrating")
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in legacy_rows:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})
            tmp.replace(out_path)
            print(f"migrated {out_path} from header {header} to {fieldnames}", flush=True)

    # Load previously evaluated indices (resume across restarts).
    if out_path.exists():
        with out_path.open() as f:
            for row in csv.DictReader(f):
                idx = _video_index(row["video"])
                if idx is not None:
                    done.add(idx)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    print(
        f"[streaming_eval] ref={ref_dir} tgt={tgt_dir} out={out_path} "
        f"already_done={len(done)}",
        flush=True,
    )

    last_scan: set[int] = set()
    idle_polls = 0
    # Targets whose reference video has not appeared yet. Unlike `done`, these
    # are retried on later polls so a dynamically-growing reference directory
    # (e.g. the BF16 baseline is still generating) can be caught up.
    waiting: set[int] = set()
    while True:
        present: set[int] = set()
        for p in tgt_dir.glob("*_0.mp4"):
            idx = _video_index(p.name)
            if idx is not None:
                present.add(idx)

        new_videos = sorted(present - done - waiting)
        for idx in new_videos:
            name = f"{idx}_0.mp4"
            ref_path = ref_dir / name
            if not ref_path.exists():
                print(f"  [wait] {name}: reference not ready yet", flush=True)
                waiting.add(idx)
                continue
            print(f"  [eval] {name}", flush=True)
            try:
                mm = eval_video(
                    str(ref_path),
                    str(tgt_dir / name),
                    lpips_metric=lp,
                    frame_stride=args.frame_stride,
                    max_frames=max_frames,
                    lpips_batch_size=8,
                    strict=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {name}: {exc}", flush=True)
                continue
            if mm is None:
                print(f"  [warn] {name}: eval_video returned None", flush=True)
                continue
            row = {
                "video": name,
                "psnr": f"{mm.psnr:.6f}",
                "ssim": f"{mm.ssim:.6f}",
                "lpips": f"{mm.lpips:.6f}" if mm.lpips is not None else "",
                "frames": str(mm.frames),
                "mse": f"{mm.mse:.4f}",
                "identical_frames": str(mm.identical_frames),
            }
            with out_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow(row)
            done.add(idx)
            waiting.discard(idx)
            print(
                f"    -> {name}: psnr={row['psnr']} ssim={row['ssim']} "
                f"lpips={row['lpips'] or 'n/a'} "
                f"(done {len(done)}/{args.expected or '?'})",
                flush=True,
            )

        if args.expected and len(done) >= args.expected:
            print(
                f"[streaming_eval] reached expected {args.expected} videos; "
                "continuing to watch for stragglers",
                flush=True,
            )

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[streaming_eval] interrupted", file=sys.stderr)
        raise SystemExit(0)
