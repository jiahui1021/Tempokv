from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path: Path, *, fit_mode: str = "native") -> SimpleNamespace:
    trace = tmp_path / "trace.npz"
    checkpoint = tmp_path / "model.pt"
    prompts = tmp_path / "prompts.txt"
    for path in (trace, checkpoint):
        path.write_bytes(b"fixture")
    prompts.write_text("a prompt\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "k_bits": 2,
                "v_bits": 2,
                "anchor_bits": 4,
                "group_size": 64,
                "stride": 1560,
                "k_history_length": 2,
                "v_history_length": 1,
                "fit_mode": fit_mode,
                "k_predictor_mode": "shared" if fit_mode == "legacy" else "layer_head",
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        trace=trace,
        base_config=config,
        anchor_control_config=None,
        checkpoint=checkpoint,
        prompts=prompts,
        output=tmp_path / "study",
        spans=(2, 4, 8, 12, 18),
        bits=(2, 4),
        seed=0,
        prompt_index=0,
        fit_device="cpu",
        nmse_device="cpu",
        attention_steps=None,
    )


def test_manifest_has_independent_fit_before_every_nmse_and_generation(tmp_path):
    run_study = _module(
        "span_run_study", "scripts/paper_experiments/span/run_study.py"
    )
    manifest = run_study.build_manifest(_inputs(tmp_path))
    steps = manifest["steps"]
    assert len(steps) == 62
    positions = {step["id"]: index for index, step in enumerate(steps)}
    for bits, anchor in ((2, 4), (4, 4), (2, 2)):
        for span in (2, 4, 8, 12, 18):
            label = f"int{bits}_anchor{anchor}_span{span}"
            assert positions[f"fit_{label}"] < positions[f"nmse_{label}"]
            assert positions[f"fit_{label}"] < positions[f"generate_{label}_pixel717"]
    assert manifest["long_requested_pixel_frames"] == 1400
    assert manifest["long_computed_pixel_frames"] == 1401
    assert manifest["long_steps_deferred_until_fixed_scan"] is True


def test_long_selection_uses_measured_best_and_worst_not_fixed_endpoints(tmp_path):
    run_study = _module(
        "span_run_study_selection", "scripts/paper_experiments/span/run_study.py"
    )
    scores = {
        2: (30.0, 0.30),
        4: (33.0, 0.20),
        8: (35.0, 0.10),
        12: (34.0, 0.15),
        18: (31.0, 0.25),
    }
    for bits in (2, 4):
        for span, (psnr, lpips) in scores.items():
            path = (
                tmp_path
                / "runs"
                / f"int{bits}_anchor4_span{span}"
                / "pixel_717"
                / "quality.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"psnr_db": psnr, "lpips": lpips}),
                encoding="utf-8",
            )
    selection = run_study.select_long_spans(
        tmp_path, [2, 4], [2, 4, 8, 12, 18]
    )
    selected = {
        (row["bits"], row["role"]): row["span_frames"]
        for row in selection["selections"]
    }
    assert selected == {
        (2, "best"): 8,
        (2, "worst"): 2,
        (4, "best"): 8,
        (4, "worst"): 2,
    }


def test_legacy_config_requires_native_anchor_control(tmp_path):
    run_study = _module(
        "span_run_study_legacy", "scripts/paper_experiments/span/run_study.py"
    )
    with pytest.raises(ValueError, match="anchor INT2"):
        run_study.build_manifest(_inputs(tmp_path, fit_mode="legacy"))


def test_synthetic_nmse_has_every_position_without_forcing_monotonicity():
    analyze = _module(
        "span_analyze", "scripts/paper_experiments/span/analyze.py"
    )
    quality, long_rows, nmse = analyze._synthetic_rows(0)
    assert len(quality) == 15
    assert len(long_rows) == 4
    rows = [
        row
        for row in nmse
        if row["bits"] == 2
        and row["anchor_bits"] == 4
        and row["span_frames"] == 18
        and row["tensor"] == "K"
    ]
    assert [row["position"] for row in rows] == list(range(18))
    assert all(row["reference_energy"] > 0 for row in rows)
