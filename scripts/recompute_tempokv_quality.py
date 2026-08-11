#!/usr/bin/env python3
"""Recompute per-video quality (psnr/ssim/lpips/mse) for TempoKV videos vs bf16 ref.
Saves full per-video csv and prints global (MSE-averaged) PSNR + weighted SSIM.
"""
import sys, csv
from pathlib import Path
sys.path.insert(0, "/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/evaluation")
from eval_quality import eval_video, LpipsMetric

REF = Path("/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/Self-Forcing/bf16")
TGT = Path("/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/self_forcing/kh3-vh1-int2/videos")
OUT = Path("/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/self_forcing/kh3-vh1-int2/quality_per_video.csv")

def main():
    lp = LpipsMetric(net="alex", device="cuda")
    names = sorted((p.name for p in REF.glob("*_0.mp4") if (TGT/p.name).exists()),
                   key=lambda n: int(n.split("_")[0]))
    print(f"computing {len(names)} videos...", flush=True)
    rows = []
    for i, n in enumerate(names):
        try:
            mm = eval_video(str(REF/n), str(TGT/n), lpips_metric=lp, frame_stride=4,
                            max_frames=None, lpips_batch_size=8, strict=True)
        except Exception as exc:
            print(f"  [warn] {n}: {exc}", flush=True)
            continue
        rows.append({"target_name":"int2-videos","target_dir":str(TGT),"video":n,
                     "psnr":f"{mm.psnr:.6f}","ssim":f"{mm.ssim:.6f}",
                     "lpips":f"{mm.lpips:.6f}" if mm.lpips is not None else "",
                     "mse":f"{mm.mse:.4f}","frames":str(mm.frames),
                     "identical_frames":str(mm.identical_frames)})
        if (i+1) % 20 == 0:
            print(f"  ... {len(rows)}", flush=True)
    rows.sort(key=lambda r: int(r["video"].split("_")[0]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target_name","target_dir","video","psnr","ssim","lpips","mse","frames","identical_frames"])
        w.writeheader(); w.writerows(rows)
    print(f"saved -> {OUT} ({len(rows)} rows)", flush=True)

    # Global metrics: MSE-averaged PSNR, frame-weighted SSIM
    mse_sum = sum(float(r["mse"])*float(r["frames"]) for r in rows)
    frames = sum(float(r["frames"]) for r in rows)
    ssim_sum = sum(float(r["ssim"])*float(r["frames"]) for r in rows)
    global_mse = mse_sum / frames
    import numpy as np
    global_psnr = 10*np.log10(255.0**2/global_mse)
    global_ssim = ssim_sum/frames
    arith_psnr = sum(float(r["psnr"]) for r in rows)/len(rows)
    print(f"\n=== GLOBAL (MSE-averaged) ===")
    print(f"global_psnr = {global_psnr:.4f} dB  (arith_mean_psnr was {arith_psnr:.4f})")
    print(f"global_ssim = {global_ssim:.6f}  (frame-weighted)")
    lp_sum = sum(float(r["lpips"])*float(r["frames"]) for r in rows if r["lpips"])
    print(f"global_lpips = {lp_sum/frames:.4f}  (frame-weighted)")

if __name__ == "__main__":
    main()
