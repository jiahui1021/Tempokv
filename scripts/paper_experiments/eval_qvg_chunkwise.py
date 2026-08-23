#!/usr/bin/env python3
"""Evaluate the first video of each QVG dir under causal_forcing/chunkwise
against the chunkwise/bf16 reference."""
import sys, glob, json
from pathlib import Path
sys.path.insert(0, "/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/evaluation")
from eval_quality import eval_video, LpipsMetric

REF = "/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/causal_forcing/chunkwise/bf16/videos/0_0.mp4"
ROOT = Path("/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/causal_forcing/chunkwise")

def main():
    lp = LpipsMetric(net="alex", device="cuda")
    dirs = ["QVG_BF16-bench5","QVG_INT2-bench5","QVG_INT4-bench5",
            "QVG_INT2-w180","QVG_INT4-w180","QVG_INT4"]
    rows = []
    for d in dirs:
        mp4s = sorted(glob.glob(str(ROOT / d / "*.mp4")))
        if not mp4s:
            print(f"{d}: 无视频", flush=True)
            continue
        tgt = mp4s[0]
        mm = eval_video(REF, tgt, lpips_metric=lp, frame_stride=4,
                        max_frames=None, lpips_batch_size=8, strict=True)
        name = Path(tgt).name[:30]
        rows.append({"dir": d, "video": name, "psnr": mm.psnr,
                     "ssim": mm.ssim, "lpips": mm.lpips})
        print(f"{d:18} [{name}] psnr={mm.psnr:.4f} ssim={mm.ssim:.4f} "
              f"lpips={mm.lpips:.4f}", flush=True)
    out = ROOT / "qvg_vs_chunkwise_bf16.csv"
    with out.open("w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["dir", "video", "psnr", "ssim", "lpips"])
        w.writeheader(); w.writerows(rows)
    print(f"saved -> {out}", flush=True)

if __name__ == "__main__":
    main()
