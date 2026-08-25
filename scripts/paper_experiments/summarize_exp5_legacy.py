#!/usr/bin/env python3
"""Aggregate exp5 legacy streaming results into a summary CSV + markdown."""
import csv
import json
from pathlib import Path

ROOT = Path("/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv/results/paper_exp5_legacy")
METHODS = ["k4v4", "k2v4", "k4v2"]
# 压缩比来自各配置 pipeline_report.json 的 kv_cache_memory
COMPRESSION = {
    "k4v4": None,
    "k2v4": None,
    "k4v2": None,
}


def load_compression(label: str):
    report = ROOT / label / "pipeline_report.json"
    if not report.is_file():
        return None
    d = json.loads(report.read_text())
    batches = d.get("prompt_batches") or []
    mem = (batches[0].get("kv_cache_memory") or {}) if batches else {}
    comp = mem.get("total_kv_cache_bytes")
    uncomp = mem.get("total_bf16_kv_bytes")
    if isinstance(comp, int) and isinstance(uncomp, int) and comp > 0:
        return uncomp / comp
    rate = mem.get("compression_rate")
    return float(rate) if rate else None


def main():
    summary_rows = []
    for m in METHODS:
        csv_path = ROOT / m / "streaming_psnr_ssim.csv"
        if not csv_path.is_file():
            print(f"[skip] {m}: no streaming csv")
            continue
        rows = list(csv.DictReader(open(csv_path)))
        psnrs = [float(r["psnr"]) for r in rows]
        ssims = [float(r["ssim"]) for r in rows]
        n = len(rows)
        ratio = load_compression(m)
        summary_rows.append({
            "config": m,
            "k_bits": int(m[1]),
            "v_bits": int(m[3]),
            "n_videos": n,
            "avg_psnr": sum(psnrs) / n,
            "avg_ssim": sum(ssims) / n,
            "min_psnr": min(psnrs),
            "max_psnr": max(psnrs),
            "compression_ratio": round(ratio, 3) if ratio else "",
        })

    # CSV
    out_csv = ROOT / "exp5_legacy_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"saved -> {out_csv}")

    # Markdown
    out_md = ROOT / "exp5_legacy_summary.md"
    lines = [
        "# exp5 Legacy 混合位宽结果",
        "",
        "配置：legacy + shared, kh2, vh1, anchor4。30 条 moviegen 前 30, 717 帧。",
        "参考：kv-quant-result/Self-Forcing/bf16。",
        "",
        "| 配置 | K bits | V bits | 平均 PSNR | 平均 SSIM | 压缩比 |",
        "|------|--------|--------|-----------|-----------|--------|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['config']} | {r['k_bits']} | {r['v_bits']} | "
            f"{r['avg_psnr']:.2f} | {r['avg_ssim']:.4f} | {r['compression_ratio']} |"
        )
    lines += [
        "",
        "## 逐视频 PSNR",
        "",
    ]
    for m in METHODS:
        csv_path = ROOT / m / "streaming_psnr_ssim.csv"
        if not csv_path.is_file():
            continue
        rows = list(csv.DictReader(open(csv_path)))
        lines.append(f"### {m}")
        lines.append("")
        lines.append("| video | psnr | ssim |")
        lines.append("|-------|------|------|")
        for r in rows:
            lines.append(f"| {r['video']} | {float(r['psnr']):.3f} | {float(r['ssim']):.4f} |")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {out_md}")


if __name__ == "__main__":
    main()
