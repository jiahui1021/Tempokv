"""Calibration tests use synthetic tensors only to check the implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


_path = Path(__file__).resolve().parents[2] / "src/tempokv/spectral_kv/calibration.py"
_spec = importlib.util.spec_from_file_location("spectral_kv_calibration_under_test", _path)
assert _spec is not None and _spec.loader is not None
_calibration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_calibration)
LinearCalibration = _calibration.LinearCalibration
CalibrationMap = _calibration.CalibrationMap


def _affine_data(n=80, heads=2, d=4):
    generator = torch.Generator().manual_seed(82)
    x = torch.randn(n, heads, d, generator=generator, dtype=torch.float64)
    weight = torch.randn(heads, d, d, generator=generator, dtype=torch.float64)
    bias = torch.randn(heads, d, generator=generator, dtype=torch.float64)
    y = torch.einsum("nhd,hde->nhe", x, weight) + bias
    return x, y, weight, bias


def test_streamed_known_affine_recovery_and_head_ids():
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration(ridge=1e-8)
    for start in range(0, 80, 7):
        calibration.update(5, "k", x[start:start + 7], y[start:start + 7], head_ids=[3, 9])
    fitted = calibration.fit()
    assert fitted.keys == ((5, "k", 3), (5, "k", 9))
    assert fitted.metadata["skipped"] == []
    assert fitted.metadata["coverage"][0]["samples"] == 80
    torch.testing.assert_close(fitted.apply(5, "k", x, [3, 9]), y, atol=2e-9, rtol=2e-9)
    # The requested tensor order, rather than sorted key order, determines heads.
    torch.testing.assert_close(fitted.apply(5, "k", x[:, [1, 0]], [9, 3]), y[:, [1, 0]], atol=2e-9, rtol=2e-9)


def test_ridge_stability_and_unpenalized_bias():
    x, _, _, _ = _affine_data(n=40, heads=1, d=3)
    y = torch.full_like(x, 12.5)
    calibration = LinearCalibration(ridge=1e6)
    calibration.update(0, "v", x, y)
    fitted = calibration.fit()
    torch.testing.assert_close(fitted.apply(0, "v", x), y, atol=2e-13, rtol=0)


def test_insufficient_and_rank_deficient_entries_are_explicitly_skipped():
    calibration = LinearCalibration()
    calibration.update(0, "k", torch.ones(3, 1, 4), torch.ones(3, 1, 4))
    calibration.update(1, "v", torch.ones(20, 1, 4), torch.ones(20, 1, 4))
    fitted = calibration.fit()
    assert fitted.keys == ()
    assert [r["reason"] for r in fitted.metadata["skipped"]] == ["insufficient_samples", "rank_deficient"]
    assert fitted.metadata["skipped"][0]["required_samples"] == 5
    assert fitted.metadata["skipped"][1]["rank"] == 1
    with pytest.raises(ValueError, match="no fitted entries"):
        fitted.apply(0, "k", torch.ones(3, 1, 4))


def test_minimum_identifiable_sample_count_is_accepted():
    # Augmented [x,1] is square and full rank, exactly at D+1 samples.
    x = torch.tensor([[0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)[:, None]
    y = 2 * x + 3
    calibration = LinearCalibration(ridge=1e-10)
    calibration.update(0, "k", x, y)
    fitted = calibration.fit()
    assert fitted.keys == ((0, "k", 0),)
    torch.testing.assert_close(fitted.apply(0, "k", x), y, atol=1e-9, rtol=0)


def test_merge_matches_single_accumulation_and_is_npz_friendly(tmp_path):
    x, y, _, _ = _affine_data()
    whole = LinearCalibration()
    whole.update(2, "v", x, y)
    first, second = LinearCalibration(), LinearCalibration()
    first.update(2, "v", x[:31], y[:31])
    second.update(2, "v", x[31:], y[31:])
    state_path = tmp_path / "moments.npz"
    np.savez(state_path, **second.state_dict())
    with np.load(state_path, allow_pickle=False) as state:
        assert all(state[name].dtype.kind != "O" for name in state.files)
        assert first.merge(state) is first
    torch.testing.assert_close(first.fit().apply(2, "v", x), whole.fit().apply(2, "v", x), atol=1e-14, rtol=1e-13)
    for name in ("xtx", "xty"):
        np.testing.assert_allclose(first.state_dict()[name], whole.state_dict()[name], atol=5e-13, rtol=1e-13)


def test_merge_object_disjoint_keys_and_empty_states():
    x, y, _, _ = _affine_data()
    first, second = LinearCalibration(), LinearCalibration()
    first.merge(second)
    first.update(1, "k", x, y)
    second.update(1, "v", x, y)
    first.merge(second)
    assert first.fit().keys == ((1, "k", 0), (1, "k", 1), (1, "v", 0), (1, "v", 1))


def test_state_has_bounded_size_and_does_not_alias():
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration()
    calibration.update(0, "k", x[:10], y[:10])
    before = sum(value.nbytes for value in calibration.state_dict().values())
    calibration.update(0, "k", x, y)
    after = sum(value.nbytes for value in calibration.state_dict().values())
    assert before == after
    copied = calibration.state_dict()
    copied["xtx"].fill(0)
    assert calibration.state_dict()["xtx"].sum() != 0
    assert set(copied) == {"format", "ridge", "feature_dim", "layers", "roles", "heads", "counts", "xtx", "xty"}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_inference_dtype_shape_no_mutation_and_autocast(dtype):
    x, y, weight, bias = _affine_data()
    calibration = LinearCalibration(ridge=1e-10)
    calibration.update(3, "k", x, y)
    fitted = calibration.fit()
    samples = x[:12].reshape(3, 4, 2, 4).to(dtype)
    original = samples.clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = fitted.apply(3, "k", samples)
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    expected = (torch.einsum("...hd,hde->...he", samples.to(compute), weight.to(compute)) + bias.to(compute)).to(dtype)
    assert result.shape == samples.shape and result.dtype == dtype and result.device == samples.device
    assert torch.equal(samples, original)
    torch.testing.assert_close(result, expected, atol=2e-10 if dtype == torch.float64 else 2e-6, rtol=2e-9 if dtype == torch.float64 else 2e-6)


def test_inference_keeps_gradients():
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration()
    calibration.update(0, "k", x, y)
    query = x.float().requires_grad_()
    calibration.fit().apply(0, "k", query).square().sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_npz_roundtrip_exact_inference_and_no_overwrite(tmp_path):
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration()
    calibration.update(2, "k", x, y)
    calibration.update(4, "v", x[:2], y[:2])
    fitted = calibration.fit()
    path = fitted.save(tmp_path / "nested" / "map.npz")
    original_bytes = path.read_bytes()
    loaded = CalibrationMap.load(path)
    assert loaded.metadata == fitted.metadata and loaded.keys == fitted.keys
    assert torch.equal(loaded.apply(2, "k", x), fitted.apply(2, "k", x))
    with np.load(path, allow_pickle=False) as state:
        assert all(state[name].dtype.kind != "O" for name in state.files)
    with pytest.raises(FileExistsError):
        fitted.save(path)
    assert path.read_bytes() == original_bytes


def test_empty_map_saved_and_unfitted_failure(tmp_path):
    fitted = LinearCalibration().fit()
    loaded = CalibrationMap.load(fitted.save(tmp_path / "empty.npz"))
    assert loaded.metadata == fitted.metadata
    with pytest.raises(ValueError, match="no fitted entries"):
        loaded.apply(0, "k", torch.zeros(1, 1, 3))


def test_missing_role_layer_head_and_shape_are_errors():
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration()
    calibration.update(3, "k", x, y, head_ids=[2, 5])
    fitted = calibration.fit()
    for layer, role, ids in [(4, "k", [2, 5]), (3, "v", [2, 5]), (3, "k", [2, 6])]:
        with pytest.raises(ValueError, match="missing fitted calibration"):
            fitted.apply(layer, role, x, ids)
    with pytest.raises(ValueError, match="feature dimension"):
        fitted.apply(3, "k", x[..., :3], [2, 5])
    with pytest.raises(ValueError, match="head_ids"):
        fitted.apply(3, "k", x, [2, 2])
    with pytest.raises(ValueError, match="shape"):
        fitted.apply(3, "k", x[0])


@pytest.mark.parametrize("ridge", [0, -1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_ridge(ridge):
    with pytest.raises(ValueError, match="ridge"):
        LinearCalibration(ridge)


@pytest.mark.parametrize("mutation", ["nan", "inf", "shape", "integer", "heads", "role", "layer", "dimension"])
def test_update_validation_is_atomic(mutation):
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration()
    calibration.update(1, "k", x[:10], y[:10])
    original = calibration.state_dict()
    bad_x, bad_y, layer, role, ids = x.clone(), y.clone(), 1, "k", None
    if mutation == "nan":
        bad_x[0, 0, 0] = float("nan")
    elif mutation == "inf":
        bad_y[0, 0, 0] = float("inf")
    elif mutation == "shape":
        bad_y = y[:1]
    elif mutation == "integer":
        bad_x = x.long()
    elif mutation == "heads":
        ids = [0]
    elif mutation == "role":
        role = "q"
    elif mutation == "layer":
        layer = -1
    elif mutation == "dimension":
        bad_x, bad_y = x[..., :2], y[..., :2]
    with pytest.raises(ValueError):
        calibration.update(layer, role, bad_x, bad_y, head_ids=ids)
    for name, value in original.items():
        np.testing.assert_array_equal(calibration.state_dict()[name], value)


@pytest.mark.parametrize("mutation", ["ridge", "dimension", "count", "nan", "duplicate", "format", "asymmetric"])
def test_merge_validation_is_atomic(mutation):
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration()
    calibration.update(1, "k", x, y)
    original = calibration.state_dict()
    state = calibration.state_dict()
    if mutation == "ridge":
        state["ridge"] = np.asarray(0.2)
    elif mutation == "dimension":
        state["feature_dim"] = np.asarray(3)
    elif mutation == "count":
        state["counts"][0] = 0
    elif mutation == "nan":
        state["xty"][0, 0, 0] = np.nan
    elif mutation == "duplicate":
        state["heads"][1] = state["heads"][0]
    elif mutation == "format":
        state["format"] = np.asarray("wrong")
    elif mutation == "asymmetric":
        state["xtx"][0, 0, 1] += 1
    with pytest.raises(ValueError):
        calibration.merge(state)
    for name, value in original.items():
        np.testing.assert_array_equal(calibration.state_dict()[name], value)


def test_load_rejects_pickled_and_malformed_parameters(tmp_path):
    object_path = tmp_path / "object.npz"
    np.savez(object_path, format=np.asarray({"arbitrary": "object"}, dtype=object))
    with pytest.raises(ValueError):
        CalibrationMap.load(object_path)
    x, y, _, _ = _affine_data()
    calibration = LinearCalibration()
    calibration.update(0, "k", x, y)
    state = calibration.fit().state_dict()
    state["weights"][0, 0, 0] = np.nan
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, **state)
    with pytest.raises(ValueError, match="finite"):
        CalibrationMap.load(bad_path)
