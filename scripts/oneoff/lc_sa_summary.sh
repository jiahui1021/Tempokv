#!/bin/bash
# 汇总 single_anchor 的压缩比 + paired 指标 → single_anchor/SUMMARY.md
set -u
W=${WORKSPACE:?set WORKSPACE to the experiment root}
$W/miniconda3/envs/qvg/bin/python - "$W/kv-quant-result/LongCat/single_anchor" "$W/kv-quant-result/LongCat" <<'PY'
import json, sys, time
from pathlib import Path

out, r = Path(sys.argv[1]), Path(sys.argv[2])
TAGS = ("k2v2", "k4v2", "k5v3", "k6v4")
BASE3 = {"k2v2": "TEMPOKV_KH2VH1_INT2/gen/0", "k4v2": "hybrid-bits/k4v2/gen/0",
         "k5v3": "hybrid-bits/k5v3/gen/0", "k6v4": "hybrid-bits/k6v4/gen/0"}


def load(p):
    return json.loads(p.read_text()) if p.is_file() else {}


def rep(d):
    f = sorted(d.rglob("tempokv_bridge_report*.json"))
    return load(f[0]) if f else {}


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


head = ("| 组 | K/V/anchor | 未压缩缓存 GB | 压缩比 | 峰值显存 GB | 帧数 | "
        "vs BF16 PSNR | SSIM | LPIPS | vs 3-anchor PSNR | SSIM | LPIPS | 3-anchor 压缩比 |")
sep = "|" + "---|" * (len(head.split("|")) - 2)
lines = [
    "# LongCat single_anchor(单 anchor)汇总",
    "",
    "口径:idx0、480p、NUM_SEGMENTS=1、kh2-vh1 + layer_head 协议(配置 `results/paper/span-int2/configs/sa_*.json`);"
    "SA 出片 493 帧,参考(BF16/0 与 3-anchor)273 帧,paired 指标取前 273 帧前缀、MSE-pooled、frame-stride 8(35 帧)。"
    "n=1 条,FID/VBench 不适用。",
    "",
    head, sep,
]
for tag in TAGS:
    sa = rep(out / tag)
    bf = load(out / "eval" / f"{tag}_vs_bf16.json")
    th = load(out / "eval" / f"{tag}_vs_3a.json")
    b3 = rep(r / BASE3[tag])
    fr = sum(v.get("frames", 0) for v in [bf, th] if v) or None
    cells = [
        tag,
        "{}/{}/{}".format(sa.get("k_bits", "?"), sa.get("v_bits", "?"), sa.get("anchor_bits", "?")),
        f((sa.get("uncompressed_cache_bytes") or None) and sa["uncompressed_cache_bytes"] / 2**30, 1),
        f(sa.get("compression_ratio")),
        f((sa.get("peak_gpu_allocated_bytes") or None) and sa["peak_gpu_allocated_bytes"] / 2**30, 1),
        "493" if sa else "—",
        f(bf.get("psnr_db")), f(bf.get("ssim"), 4), f(bf.get("lpips"), 4),
        f(th.get("psnr_db")), f(th.get("ssim"), 4), f(th.get("lpips"), 4),
        f(b3.get("compression_ratio")),
    ]
    lines.append("| " + " | ".join(str(c) for c in cells) + " |")
lines += ["", "更新:" + time.strftime("%Y-%m-%d %H:%M") + "(lc_sa_summary.sh)"]
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
