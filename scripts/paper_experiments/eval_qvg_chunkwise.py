#!/usr/bin/env python3
"""Evaluate every index-paired QVG video against the chunkwise BF16 reference."""
import csv
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/cpfs/epic-user/hujiahui-20260612/kv-quant-result/evaluation")
from eval_quality import eval_video, LpipsMetric

REF_DIR = Path("/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/causal_forcing/chunkwise/bf16/videos")
ROOT = Path("/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/causal_forcing/chunkwise")


def _indexed_videos(directory: Path) -> dict[int, Path]:
    videos: dict[int, Path] = {}
    for path in directory.glob("*_0.mp4"):
        parts = path.stem.split("_")
        if len(parts) != 2 or parts[1] != "0":
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        videos[index] = path
    return videos


def _match_pairs(reference_dir: Path, target_dir: Path) -> list[tuple[int, Path, Path]]:
    references = _indexed_videos(reference_dir)
    targets = _indexed_videos(target_dir)
    return [
        (index, references[index], targets[index])
        for index in sorted(references.keys() & targets.keys())
    ]

def main():
    lp = LpipsMetric(net="alex", device="cuda")
    dirs = ["QVG_BF16-bench5","QVG_INT2-bench5","QVG_INT4-bench5",
            "QVG_INT2-w180","QVG_INT4-w180","QVG_INT4"]
    rows = []
    for d in dirs:
        pairs = _match_pairs(REF_DIR, ROOT / d)
        if not pairs:
            print(f"{d}: 无匹配 index 视频", flush=True)
            continue
        for index, reference, target in pairs:
            mm = eval_video(str(reference), str(target), lpips_metric=lp,
                            frame_stride=4, max_frames=None,
                            lpips_batch_size=8, strict=True)
            name = target.name
            rows.append({"dir": d, "index": index, "video": name,
                         "psnr": mm.psnr, "ssim": mm.ssim, "lpips": mm.lpips})
            print(f"{d:18} [{name}] psnr={mm.psnr:.4f} ssim={mm.ssim:.4f} "
                  f"lpips={mm.lpips:.4f}", flush=True)
    out = ROOT / "qvg_vs_chunkwise_bf16.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dir", "index", "video", "psnr", "ssim", "lpips"])
        w.writeheader(); w.writerows(rows)
    print(f"saved -> {out}", flush=True)

if __name__ == "__main__":
    main()
