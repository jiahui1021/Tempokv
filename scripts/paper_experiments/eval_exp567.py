#!/usr/bin/env python3
"""Compute per-video quality (vs Self-Forcing bf16 reference) for paper exp5/6/7.

Writes, per method:
  results/<exp>/<method>/quality_per_video.csv   (video,psnr,ssim,lpips,mse,frames)
and prints global MSE-averaged PSNR / frame-weighted SSIM & LPIPS.

Reference is the first N videos of ~/kv-quant-result/Self-Forcing/bf16,
matched by name (<idx>_0.mp4).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/evaluation")
from eval_quality import eval_video, LpipsMetric  # noqa: E402

import numpy as np  # noqa: E402

REF = Path("/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/Self-Forcing/bf16")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ref", type=Path, default=REF)
    args = parser.parse_args()

    ref_dir = args.ref.expanduser().resolve()
    ref_names = sorted(
        (p.name for p in ref_dir.glob("*_0.mp4")),
        key=lambda n: int(n.split("_")[0]),
    )
    lp = LpipsMetric(net="alex", device=args.device)

    for method in args.methods:
        tgt_dir = args.result_root / method / "videos"
        out_csv = args.result_root / method / "quality_per_video.csv"
        if not tgt_dir.is_dir():
            print(f"[skip] no videos dir: {tgt_dir}")
            continue
        # Match reference videos by numeric index.
        tgt_by_idx = {}
        for p in tgt_dir.glob("*_0.mp4"):
            try:
                idx = int(p.name.split("_")[0])
            except ValueError:
                continue
            tgt_by_idx[idx] = p
        names = [n for n in ref_names if int(n.split("_")[0]) in tgt_by_idx]
        print(f"[{method}] computing {len(names)} videos...", flush=True)
        rows = []
        for i, n in enumerate(names):
            tgt = tgt_by_idx[int(n.split("_")[0])]
            try:
                mm = eval_video(
                    str(ref_dir / n), str(tgt), lpips_metric=lp, frame_stride=4,
                    max_frames=None, lpips_batch_size=8, strict=True,
                )
            except Exception as exc:
                print(f"  [warn] {n}: {exc}", flush=True)
                continue
            rows.append({
                "video": n,
                "psnr": f"{mm.psnr:.6f}",
                "ssim": f"{mm.ssim:.6f}",
                "lpips": f"{mm.lpips:.6f}" if mm.lpips is not None else "",
                "mse": f"{mm.mse:.4f}",
                "frames": str(mm.frames),
            })
            if (i + 1) % 10 == 0:
                print(f"  ... {len(rows)}", flush=True)
        rows.sort(key=lambda r: int(r["video"].split("_")[0]))
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video", "psnr", "ssim", "lpips", "mse", "frames"])
            w.writeheader()
            w.writerows(rows)
        print(f"saved -> {out_csv} ({len(rows)} rows)", flush=True)

        valid = [r for r in rows if r["lpips"]]
        frames = sum(float(r["frames"]) for r in valid)
        global_mse = sum(float(r["mse"]) * float(r["frames"]) for r in valid) / frames
        global_psnr = 10 * np.log10(255.0 ** 2 / global_mse)
        global_ssim = sum(float(r["ssim"]) * float(r["frames"]) for r in valid) / frames
        global_lpips = sum(float(r["lpips"]) * float(r["frames"]) for r in valid) / frames
        print(f"[{method}] GLOBAL psnr={global_psnr:.4f} ssim={global_ssim:.6f} "
              f"lpips={global_lpips:.4f} (n={len(valid)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
