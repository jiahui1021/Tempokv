from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.paper_experiments.direction import (
    collect_longcat_four_way_cosine as four_way,
    collect_longcat_k_identity_mean_cosine as experiment,
)


ROOT = Path(__file__).resolve().parents[3]


def _write_inputs(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(23)
    layers, events, frames, positions, heads, dim = 2, 2, 6, 3, 4, 128
    raw = root / "raw"
    merged_layers: list[list[np.ndarray]] = [[] for _ in range(layers)]
    for event in range(events):
        event_dir = raw / f"event_{event:05d}"
        event_dir.mkdir(parents=True)
        for layer in range(layers):
            key = rng.normal(
                size=(1, frames, positions, heads, dim)
            ).astype(np.float32)
            # Make the sequence temporally correlated so both metrics are meaningful.
            for frame in range(1, frames):
                key[:, frame] = 0.85 * key[:, frame - 1] + 0.15 * key[:, frame]
            merged_layers[layer].append(
                key.transpose(0, 2, 1, 3, 4).reshape(
                    positions, frames, heads, dim
                )
            )
            np.savez_compressed(
                event_dir / f"layer_{layer:02d}.npz",
                k_pre_rope=key,
                v=np.zeros_like(key),
                layer_index=np.asarray(layer, dtype=np.int32),
                temporal_units=np.asarray(frames, dtype=np.int32),
                trace_stride=np.asarray(16, dtype=np.int32),
                coordinate_system=np.asarray("pre_rope_normalized"),
            )
    merged = root / "clean_trace.npz"
    keys = np.stack(
        [np.concatenate(records, axis=0) for records in merged_layers]
    )
    np.savez_compressed(
        merged,
        keys=keys,
        values=np.zeros_like(keys),
        layer_ids=np.arange(layers, dtype=np.int32),
        trace_stride=np.asarray(16, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
    )
    return raw, merged


def _args(trace: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        trace=trace,
        output=output,
        config=ROOT / "configs" / "native.json",
        frames=6,
        holdout_step=3,
        layers=None,
        allow_dynamic_shape=True,
        device="cpu",
        threads=1,
    )


def _four_args(trace: Path, output: Path) -> argparse.Namespace:
    args = _args(trace, output)
    args.iterations = 1
    return args


def test_raw_and_merged_longcat_traces_match(tmp_path: Path) -> None:
    raw, merged = _write_inputs(tmp_path)
    raw_report = experiment.run(_args(raw, tmp_path / "raw_result.json"))
    merged_report = experiment.run(
        _args(merged, tmp_path / "merged_result.json")
    )

    assert raw_report["trace"]["evaluated_shape"] == [2, 2, 6, 4, 128]
    assert raw_report["trace"]["source_format"] == "raw_event_directory"
    assert merged_report["trace"]["source_format"] == "merged_clean_trace"
    for method in (experiment.IDENTITY, experiment.TIME_MEAN):
        assert raw_report["methods"][method]["mean_cosine"] == pytest.approx(
            merged_report["methods"][method]["mean_cosine"], abs=1e-12
        )
        assert len(raw_report["methods"][method]["per_layer_cosine"]) == 2
        assert len(raw_report["methods"][method]["per_frame"]) == 5

    for stem in ("raw_result", "merged_result"):
        assert (tmp_path / f"{stem}_summary.csv").is_file()
        assert (tmp_path / f"{stem}_per_layer.csv").is_file()
        assert (tmp_path / f"{stem}_per_frame.csv").is_file()


def test_four_way_raw_and_merged_longcat_traces_match(tmp_path: Path) -> None:
    raw, merged = _write_inputs(tmp_path)
    raw_report = four_way.run(_four_args(raw, tmp_path / "four_raw.json"))
    merged_report = four_way.run(
        _four_args(merged, tmp_path / "four_merged.json")
    )

    assert set(raw_report["cosine_comparison"]["mean_cosine"]) == set(
        four_way.METHODS
    )
    assert len(raw_report["cosine_comparison"]["per_layer"]) == 2
    assert len(raw_report["cosine_comparison"]["per_frame"]) == 5
    assert raw_report["matched_k_fit"]["max_abs_weight_difference"] == 0.0
    assert raw_report["matched_k_fit"]["max_abs_bias_difference"] == 0.0
    for method in four_way.METHODS:
        assert raw_report["cosine_comparison"]["mean_cosine"][method] == pytest.approx(
            merged_report["cosine_comparison"]["mean_cosine"][method], abs=1e-12
        )

    for stem in ("four_raw", "four_merged"):
        assert (tmp_path / f"{stem}_summary.csv").is_file()
        assert (tmp_path / f"{stem}_per_layer.csv").is_file()
        assert (tmp_path / f"{stem}_per_frame.csv").is_file()
