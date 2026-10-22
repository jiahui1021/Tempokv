from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tempokv.parameters import TemporalParameters
from tempokv.spectral_kv.calibration import CalibrationMap, LinearCalibration

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location("suite_b_" + name, ROOT / "scripts/paper_experiments/spectral" / (name + ".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def grid_file(path, predictions=True):
    rng = np.random.default_rng(5)
    k = rng.normal(size=(4, 3, 5, 2, 8)).astype(np.float32)
    values = {"k_pre": k, "v": k * 2, "grid": [4, 3, 5], "stride": 15, "layer": 0,
              "coordinate_system": "pre_rope_normalized", "prompt_id": "synthetic", "seed": 5}
    if predictions: values.update(predicted_k=k, predicted_v=k * 2)
    np.savez(path, **values)


def test_e5_uses_provided_predictions_and_preserves_separation(tmp_path):
    source = tmp_path / "input.npz"; grid_file(source)
    out = tmp_path / "e5"
    meta = script("residual_spectrum").analyze_residual(source, out, bins=3)
    assert meta["suite_id"] == "B_SPECTRAL_KV" and meta["observed_frame_role_pairs"] == 8
    rows = [json.loads(line) for line in (out / "B_E5_RESIDUAL/records.jsonl").read_text().splitlines()]
    assert {r["source_kind"] for r in rows} == {"original", "residual"}
    assert all(r["power_incl_dc"] == 0 for r in rows if r["source_kind"] == "residual")
    with pytest.raises(FileExistsError): script("residual_spectrum").analyze_residual(source, out)
    analyzed = script("analyze").analyze(out, tmp_path / "summary", "B_E5_RESIDUAL")
    assert (analyzed / "B_E5_RESIDUAL/results.png").exists()


def test_e5_reuses_existing_predictor_and_marks_warmup(tmp_path):
    source = tmp_path / "input.npz"; grid_file(source, predictions=False)
    params = TemporalParameters(torch.ones(1, 1, 2, 8), torch.zeros(1, 2, 8),
                                stride=15, coordinate_system="pre_rope_normalized")
    path = tmp_path / "params.npz"; params.save(path)
    meta = script("residual_spectrum").analyze_residual(source, tmp_path / "e5", k_params=path, bins=3)
    assert any(r["reason"] == "full_order_history_warmup" and r["frames"] == 1 for r in meta["skipped"])
    assert any(r["role"] == "v" and r["reason"] == "no_prediction_or_parameters" for r in meta["skipped"])
    assert meta["codec_or_quantizer_executed"] is False


def test_calibration_cli_merges_states_and_reports_missing_maps(tmp_path):
    rng = torch.Generator().manual_seed(8)
    for part in range(2):
        state = LinearCalibration(1e-8)
        x = torch.randn(32, 2, 8, generator=rng)
        for role in ("k", "v"): state.update(0, role, x, x * 2 + .25)
        with (tmp_path / f"calibration_{part}.npz").open("xb") as handle: np.savez_compressed(handle, **state.state_dict())
    target = tmp_path / "map.npz"
    report = script("fit_calibration").fit_calibration(tmp_path, target)
    assert report["coverage_status"] == "complete_for_observed_keys"
    mapping = CalibrationMap.load(target)
    query = torch.randn(1, 3, 2, 8, generator=rng)
    torch.testing.assert_close(mapping.apply(0, "k", query), query * 2 + .25)
    with pytest.raises(FileExistsError): script("fit_calibration").fit_calibration(tmp_path, target)


def test_paired_rollout_metrics_and_rejected_wrong_pair(tmp_path):
    reference, candidate = tmp_path / "ref.npz", tmp_path / "cand.npz"
    np.savez(reference, values=np.ones((4, 3)), prompt_id=7, seed=10)
    np.savez(candidate, values=np.ones((4, 3)) * 1.5, prompt_id=7, seed=10)
    tool = script("compare_rollouts")
    meta, rows = tool.compare_rollouts(reference, candidate, tmp_path / "drift")
    assert all(r["relative_l2"] == .5 and r["rmse"] == .5 for r in rows)
    assert "not_measured" in meta["metrics"]["psnr"]
    assert (tmp_path / "drift/drift.png").exists()
    _, pixel_rows = tool.compare_rollouts(reference, candidate, tmp_path / "pixel", space="pixel", data_range=2)
    assert pixel_rows[0]["psnr_db"] == pytest.approx(12.041199826559248)
    np.savez(candidate, values=np.ones((4, 3)), prompt_id=8, seed=10)
    with pytest.raises(ValueError, match="prompt_id"): tool.compare_rollouts(reference, candidate, tmp_path / "invalid")
    with pytest.raises(ValueError, match="data_range"): tool.compare_rollouts(reference, candidate, tmp_path / "invalid", space="pixel")


def test_a_only_metadata_never_enters_b_reports(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"suite_id": "A_FREQUENCY_ORACLE", "config": {}, "coordinates": {}}))
    with pytest.raises(ValueError, match="suite A"): script("analyze").analyze(tmp_path, tmp_path / "out")
