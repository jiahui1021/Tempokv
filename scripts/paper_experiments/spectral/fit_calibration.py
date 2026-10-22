#!/usr/bin/env python3
"""Merge suite B calibration_state NPZ files and save a covered affine map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tempokv.spectral_kv.calibration import LinearCalibration


def fit_calibration(input_path: str | Path, output_path: str | Path) -> dict:
    """Merge matching archives recursively; unrelated NPZ files are reported."""
    source, destination = Path(input_path), Path(output_path)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if not source.exists():
        raise FileNotFoundError(source)
    paths = sorted(source.rglob("*.npz")) if source.is_dir() else [source]
    calibration = None
    accepted, ignored = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            kind = str(np.asarray(archive["format"]).item()) if "format" in archive else None
            if kind != "tempokv.spectral_kv.linear_calibration.v1":
                ignored.append(str(path))
                continue
            if calibration is None:
                calibration = LinearCalibration(float(np.asarray(archive["ridge"]).item()))
            calibration.merge(archive)
        accepted.append(str(path))
    if calibration is None:
        raise ValueError("no matching LinearCalibration state archives were found")
    fitted = calibration.fit()
    fitted.save(destination)
    return {"suite_id": "B_SPECTRAL_KV", "experiment_id": "B_E2B_REFRESH",
            "output": str(destination), "input_states": accepted, "ignored_npz": ignored,
            "calibration": fitted.metadata,
            "coverage_status": "complete_for_observed_keys" if not fitted.metadata["skipped"] else "incomplete",
            "runtime_coverage": "must be checked against every runtime layer/role/head before KVSKIP_lin"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="state NPZ or directory scanned recursively")
    parser.add_argument("--output", required=True, help="new CalibrationMap NPZ; existing files are rejected")
    args = parser.parse_args(argv)
    print(json.dumps(fit_calibration(args.input, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
