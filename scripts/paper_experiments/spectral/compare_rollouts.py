#!/usr/bin/env python3
"""Compare paired rollout NPZs with values=[T,...], prompt_id, and seed.

The first values axis is always time.  Batched [B,T,...] archives must be split
into one paired file per sample before use.  PSNR is measured only with
--space pixel and an explicitly positive --data-range; latent PSNR is omitted.
This tool does not measure LPIPS/FID/VBench or execute a video model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from numbers import Integral
from pathlib import Path

import numpy as np


def _read(path):
    with np.load(path, allow_pickle=False) as archive:
        if not {"values", "prompt_id", "seed"} <= set(archive.files):
            raise ValueError("rollout NPZ requires values=[T,...], scalar prompt_id, and scalar seed")
        values = np.asarray(archive["values"])
        if values.ndim < 2 or min(values.shape) < 1 or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("values must be a finite real array with shape [T,...]")
        metadata = {}
        for name in ("prompt_id", "seed", "space", "layout"):
            if name in archive:
                value = np.asarray(archive[name])
                if value.ndim != 0:
                    raise ValueError(f"{name} must be scalar")
                metadata[name] = value.item()
        seed = metadata["seed"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(metadata["prompt_id"], (str, int)) or isinstance(metadata["prompt_id"], bool):
            raise ValueError("prompt_id must be a scalar string or integer")
        if metadata.get("layout", "T...") not in ("T...", "T,*", "time_first"):
            raise ValueError("rollout layout must be time-first [T,...]; split batched archives")
    return values.astype(np.float64, copy=True), metadata


def compare_rollouts(reference, candidate, output_dir, *, space="latent", data_range=None, experiment="B_E3_ON_ROLLOUT"):
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if experiment not in ("B_E3_ON_ROLLOUT", "B_E2B_REFRESH"):
        raise ValueError("experiment must be B_E3_ON_ROLLOUT or B_E2B_REFRESH")
    if space not in ("latent", "pixel"):
        raise ValueError("space must be latent or pixel")
    if space == "pixel":
        if data_range is None or isinstance(data_range, bool) or not np.isfinite(data_range) or data_range <= 0:
            raise ValueError("pixel PSNR requires an explicit finite positive data_range")
    elif data_range is not None:
        raise ValueError("data_range/PSNR is accepted only for explicit pixel-space comparisons")
    ref, ref_meta = _read(reference)
    cand, cand_meta = _read(candidate)
    if ref.shape != cand.shape:
        raise ValueError("paired rollouts must have the same shape")
    for field in ("prompt_id", "seed"):
        if type(ref_meta[field]) is not type(cand_meta[field]) or ref_meta[field] != cand_meta[field]:
            raise ValueError(f"paired rollouts must have matching {field}")
    if any(meta.get("space", space) != space for meta in (ref_meta, cand_meta)):
        raise ValueError("archive space metadata disagrees with the requested comparison space")
    rows = []
    for frame in range(ref.shape[0]):
        error = cand[frame] - ref[frame]
        error_energy, reference_energy = float(np.square(error).sum()), float(np.square(ref[frame]).sum())
        mse = float(np.square(error).mean())
        nmse = error_energy / reference_energy if reference_energy > 0 else 0.0 if error_energy == 0 else None
        row = {"suite_id": "B_SPECTRAL_KV", "experiment_id": experiment, "frame_idx": frame,
            "relative_l2": math.sqrt(nmse) if nmse is not None else None,
            "rmse": math.sqrt(mse), "nmse": nmse,
            "reference_energy": reference_energy,
            "relative_status": "zero_reference_nonzero_error" if nmse is None else "ok"}
        if space == "pixel":
            row["psnr_db"] = 10 * math.log10(data_range ** 2 / mse) if mse > 0 else float("inf")
        rows.append(row)
    # Import the plotting dependency before creating the exclusive directory.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tempokv-spectral-kv-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = {"suite_id": "B_SPECTRAL_KV", "experiment_id": experiment,
        "reference": str(reference), "candidate": str(candidate), "prompt_id": ref_meta["prompt_id"],
        "seed": ref_meta["seed"], "shape": list(ref.shape), "layout": "[T,...]", "space": space,
        "data_range": data_range, "frames": len(rows),
        "metrics": {"relative_l2": "measured", "rmse": "measured", "nmse": "measured",
            "psnr": "measured_pixel_space" if space == "pixel" else "not_measured_latent_space",
            "lpips": "not_measured_use_existing_evaluation_framework", "fid": "not_measured_use_existing_evaluation_framework",
            "vbench": "not_measured_use_existing_evaluation_framework"},
        "zero_reference_rule": "relative_l2/nmse are 0 for two zero frames; missing for nonzero error against a zero reference",
        "perfect_pixel_match_psnr": "positive infinity, written as inf in CSV",
        "model_or_video_generation_executed": False}
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "drift.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "metadata.json").open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    indices = [row["frame_idx"] for row in rows]
    axes[0].plot(indices, [row["relative_l2"] for row in rows], label="Relative L2")
    axes[1].plot(indices, [row["rmse"] for row in rows], label="RMSE")
    for axis in axes:
        axis.set_xlabel("Frame index")
        axis.set_ylabel(axis.get_legend_handles_labels()[1][0])
        axis.grid(alpha=0.25)
    fig.suptitle(f"{experiment}: paired {space} rollout drift")
    fig.savefig(destination / "drift.png", dpi=150)
    plt.close(fig)
    return metadata, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--space", choices=("latent", "pixel"), default="latent")
    parser.add_argument("--data-range", type=float)
    parser.add_argument("--experiment", choices=("B_E3_ON_ROLLOUT", "B_E2B_REFRESH"), default="B_E3_ON_ROLLOUT")
    args = parser.parse_args(argv)
    metadata, _ = compare_rollouts(args.reference, args.candidate, args.output,
        space=args.space, data_range=args.data_range, experiment=args.experiment)
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
