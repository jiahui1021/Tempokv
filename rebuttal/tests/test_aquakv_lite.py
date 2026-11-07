"""Synthetic smoke/unit tests for the AQUA-KV video baseline. No real model/video data."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rebuttal.experiments.aquakv_lite.model import Quantizer, encode, direct, ridge, apply, AquaKVPredictor
from rebuttal.experiments.aquakv_lite.report import storage, summarize

ROOT = Path(__file__).resolve().parents[2]


def synth(seed=0, L=3, R=2, T=5, H=2, D=64):
    g = torch.Generator().manual_seed(seed)
    k = torch.empty(L, R, T, H, D)
    k[0] = torch.randn(R, T, H, D, generator=g)
    for l in range(1, L):
        k[l] = 0.9 * k[l - 1] + 0.2 * torch.randn(R, T, H, D, generator=g)
    v = 0.5 * k + 0.1 * torch.randn(k.shape, generator=g)
    return k, v


Q = Quantizer(bits=2, group_size=64, mode="asymmetric", scale_dtype="bfloat16", reconstruction_dtype="float32")


def test_ridge_exact_linear_and_bias():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 3.]])
    w = torch.tensor([[2., 0.], [0., -1.]])
    y = x @ w + torch.tensor([0.5, 1.0])
    sol = ridge(x, y, 0.0)
    assert sol.shape == (3, 2)
    assert torch.allclose(sol[:2], w.double(), atol=1e-9) and torch.allclose(sol[2], torch.tensor([.5, 1.], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(apply(sol, x.double()), y.double(), atol=1e-9)


def test_shapes_and_param_bytes():
    k, v = synth()
    p, rows, kh, vh = encode(k, v, Q, fit=True)
    F = 2 * 64
    assert kh.shape == k.shape and vh.shape == v.shape and len(rows) == 3
    assert [w.shape for w in p.key_weights] == [(F + 1, F)] * 2
    assert [w.shape for w in p.value_weights] == [(2 * F + 1, F)] * 2
    n = 2 * ((F + 1) * F + (2 * F + 1) * F)
    assert p.parameter_count() == n and p.parameter_bytes("bfloat16") == 2 * n
    s = storage(k.numel(), Q, p.parameter_bytes())
    # asymmetric 2-bit, group 64, bf16 scale + 2-bit zero: (2 + (16+2)/64) bits/elt
    assert s["payload_bits_per_element"] == pytest.approx(2 + 18 / 64)
    assert s["total_bytes"] == pytest.approx(s["payload_bytes"] + 2 * n)


def test_layer0_is_plain_quantization():
    k, v = synth()
    _, rows, kh, vh = encode(k, v, Q, fit=True)
    assert torch.equal(kh[0], Q(k[0])) and torch.equal(vh[0], Q(v[0]))
    assert rows[0]["k"]["prediction_sse"] == pytest.approx(rows[0]["k"]["signal_sse"])


def test_recursion_uses_reconstructed_previous_layer_and_current_k():
    k, v = synth(1)
    p, _, kh, vh = encode(k, v, Q, fit=True)
    flat = lambda x: x.reshape(-1, 128).double()
    pk = apply(p.key_weights[0], flat(kh[0])).float().reshape(k[1].shape)
    assert torch.allclose(kh[1], pk + Q(k[1] - pk), atol=1e-5)
    pv = apply(p.value_weights[0], torch.cat([flat(vh[0]), flat(kh[1])], 1)).float().reshape(v[1].shape)
    assert torch.allclose(vh[1], pv + Q(v[1] - pv), atol=1e-5)
    # Fitted on reconstructed (not clean) inputs: refit on K̂_0 reproduces the weight.
    assert torch.allclose(p.key_weights[0], ridge(flat(kh[0]), flat(k[1]), p.ridge), atol=1e-8)
    assert not torch.allclose(p.key_weights[0], ridge(flat(k[0]), flat(k[1]), p.ridge), atol=1e-4)
    # Teacher forcing produces a different reconstruction (diagnostic only).
    _, _, kt, _ = encode(k, v, Q, predictor=p, teacher_forcing=True)
    assert not torch.allclose(kt[2], kh[2])


def test_heldout_does_not_refit_and_beats_direct_on_correlated_layers():
    k, v = synth(0, R=4, T=100)
    p, _, _, _ = encode(k, v, Q, fit=True)
    before = [w.clone() for w in p.key_weights + p.value_weights]
    k2, v2 = synth(7, R=2, T=50)
    _, rows, _, _ = encode(k2, v2, Q, predictor=p)
    assert all(torch.equal(a, b) for a, b in zip(before, p.key_weights + p.value_weights))
    a = summarize(rows)["aggregate_predicted_layers_only"]["k"]["actual_reconstruction_nmse"]
    d = summarize(direct(k2, v2, Q)[1:])["aggregate"]["k"]["actual_reconstruction_nmse"]
    assert a < d


@pytest.mark.parametrize("bad", ["ndim", "mismatch", "one_layer", "group"])
def test_invalid_inputs(bad):
    k, v = synth()
    with pytest.raises(ValueError):
        if bad == "ndim":
            encode(k[0], v[0], Q, fit=True)
        elif bad == "mismatch":
            encode(k, v[:, :1], Q, fit=True)
        elif bad == "one_layer":
            encode(k[:1], v[:1], Q, fit=True)
        else:
            encode(k[..., :48], v[..., :48], Q, fit=True)
    p, _, _, _ = encode(k, v, Q, fit=True)
    with pytest.raises(ValueError):
        encode(k[:, :, :, :1], v[:, :, :, :1], Q, predictor=p)
    with pytest.raises(ValueError):
        encode(k, v, Q, predictor=p, fit=True)


def test_cli_smoke(tmp_path):
    for name, seed in (("cal", 0), ("ev", 3)):
        k, v = synth(seed)
        np.savez(tmp_path / f"{name}.npz", keys=k.numpy(), values=v.numpy())
    cfg = {"experiment": "aquakv_lite", "device": "cpu", "quantizer": {"bits": 2, "group_size": 64},
           "calibration_trace": str(tmp_path / "cal.npz"), "evaluation_trace": str(tmp_path / "ev.npz"),
           "teacher_forcing_diagnostic": True}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    out = tmp_path / "out"
    for stage in ("fit", "evaluate"):
        r = subprocess.run([sys.executable, "rebuttal/run.py", "--experiment", "aquakv_lite", "--stage", stage,
                            "--config", str(tmp_path / "c.json"), "--output-dir", str(out)],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    rep = json.loads((out / "evaluate.json").read_text())
    assert rep["split"] == "heldout" and len(rep["aquakv"]["per_layer"]) == 3
    assert rep["aquakv"]["storage"]["predictor_parameter_bytes"] > 0
    assert rep["tempokv"]["status"] == "not_run"
