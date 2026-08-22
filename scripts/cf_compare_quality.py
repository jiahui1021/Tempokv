#!/usr/bin/env python3
"""Compare a target video set against a reference set for Causal-Forcing.

Computes per-video psnr/ssim/lpips/mse plus global metrics following the
project convention: global PSNR from frame-weighted mean MSE, frame-weighted
SSIM/LPIPS. Supports two naming conventions:

  --mode index   videos named {idx}_0.mp4 in both dirs (TempoKV)
  --mode prompt  videos named {prompt[:100]}.mp4 in both dirs (QVG)

Usage:
  python scripts/cf_compare_quality.py \
      --ref-dir results/causal_forcing/chunkwise/bf16-bench5/videos \
      --tgt-dir results/causal_forcing/chunkwise/kh2-vh1-int2-bench5/videos \
      --out-csv results/causal_forcing/chunkwise/kh2-vh1-int2-bench5/quality_per_video.csv \
      --name TempoKV_kh2vh1_int2 --mode index
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/evaluation")
from eval_quality import eval_video, LpipsMetric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--tgt-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--name", required=True, help="method label written into the csv")
    p.add_argument("--mode", choices=["index", "prompt"], required=True)
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p.parse_args()


def match_pairs(ref_dir: Path, tgt_dir: Path, mode: str) -> list[tuple[str, str, str]]:
    """Return [(video_name, ref_path, tgt_path)] sorted deterministically."""
    if mode == "index":
        refs = {p.stem.split("_")[0]: p for p in ref_dir.glob("*_0.mp4")}
        tgts = {p.stem.split("_")[0]: p for p in tgt_dir.glob("*_0.mp4")}
        keys = sorted(refs.keys() & tgts.keys(), key=int)
        return [(k, str(refs[k]), str(tgts[k])) for k in keys]
    refs = {p.name: p for p in ref_dir.glob("*.mp4")}
    tgts = {p.name: p for p in tgt_dir.glob("*.mp4")}
    keys = sorted(refs.keys() & tgts.keys())
    return [(k, str(refs[k]), str(tgts[k])) for k in keys]


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dir)
    tgt_dir = Path(args.tgt_dir)
    pairs = match_pairs(ref_dir, tgt_dir, args.mode)
    if not pairs:
        print(f"no matching video pairs between {ref_dir} and {tgt_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"computing {len(pairs)} videos...", flush=True)

    lp = LpipsMetric(net="alex", device=args.device)
    rows = []
    for i, (name, ref_p, tgt_p) in enumerate(pairs):
        try:
            mm = eval_video(
                ref_p, tgt_p, lpips_metric=lp, frame_stride=args.frame_stride,
                max_frames=args.max_frames, lpips_batch_size=8, strict=True,
            )
        except Exception as exc:
            print(f"  [warn] {name}: {exc}", flush=True)
            continue
        rows.append({
            "target_name": args.name,
            "video": name,
            "psnr": f"{mm.psnr:.6f}",
            "ssim": f"{mm.ssim:.6f}",
            "lpips": f"{mm.lpips:.6f}" if mm.lpips is not None else "",
            "mse": f"{mm.mse:.4f}",
            "frames": str(mm.frames),
            "identical_frames": str(mm.identical_frames),
        })
        if (i + 1) % 5 == 0:
            print(f"  ... {len(rows)}", flush=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target_name", "video", "psnr", "ssim",
                                          "lpips", "mse", "frames", "identical_frames"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out} ({len(rows)} rows)", flush=True)

    if not rows:
        sys.exit(1)
    frames = sum(float(r["frames"]) for r in rows)
    mse_sum = sum(float(r["mse"]) * float(r["frames"]) for r in rows)
    ssim_sum = sum(float(r["ssim"]) * float(r["frames"]) for r in rows)
    global_psnr = 10 * np.log10(255.0**2 / (mse_sum / frames))
    global_ssim = ssim_sum / frames
    lp_rows = [r for r in rows if r["lpips"]]
    lp_sum = sum(float(r["lpips"]) * float(r["frames"]) for r in lp_rows)
    global_lpips = lp_sum / frames if lp_rows else None
    arith_psnr = sum(float(r["psnr"]) for r in rows) / len(rows)

    print("\n=== GLOBAL (MSE-averaged) ===")
    print(f"global_psnr = {global_psnr:.4f} dB  (arith_mean_psnr was {arith_psnr:.4f})")
    print(f"global_ssim = {global_ssim:.6f}  (frame-weighted)")
    if global_lpips is not None:
        print(f"global_lpips = {global_lpips:.4f}  (frame-weighted)")

    result = {
        "method": args.name,
        "n_videos": len(rows),
        "total_frames": int(frames),
        "global_psnr_mse_averaged": round(float(global_psnr), 4),
        "global_ssim_frame_weighted": round(float(global_ssim), 6),
        "global_lpips_frame_weighted": (
            round(float(global_lpips), 4) if global_lpips is not None else None
        ),
    }
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"summary -> {json_out}", flush=True)


if __name__ == "__main__":
    main()
