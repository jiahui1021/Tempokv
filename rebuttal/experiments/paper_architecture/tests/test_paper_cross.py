import torch

from rebuttal.adapters.experimental_codec import ResearchCross
from rebuttal.experiments.paper_architecture.cross import _fit, _make, build_cross, load_cross
from rebuttal.experiments.paper_architecture.storage import storage_bytes


def fixture(d=6):
    torch.manual_seed(8)
    w = torch.randn(2, 2, d, d) / d
    b = torch.randn(2, 2, d) / 10
    h = torch.randn(2, 1, 2, d) / 10
    baseline = ResearchCross('dense', 'layer_head', b.reshape(4, d), h, 2, 2, w.reshape(4, d, d))
    return baseline, w, b


def test_paper_projection_and_state_round_trip():
    baseline, w, b = fixture()
    keys = torch.randn(2, 3, 4, 2, 6)
    model, report = build_cross(keys, keys, baseline, None, {'kind': 'paper'})
    assert report['rounds'] == 0
    for layer in range(2):
        source = keys[layer].permute(0, 2, 1, 3)
        torch.testing.assert_close(model.project(source, layer=layer), baseline.project(source, layer=layer))
        torch.testing.assert_close(load_cross(model.state()).project(source, layer=layer), model.project(source, layer=layer))
    assert model.history_length == baseline.history_length


def test_svd_low_rank_and_tucker_execute_factors():
    baseline, w, b = fixture()
    x = torch.randn(3, 2, 4, 6)
    for kind in ('svd', 'low_rank_refit', 'shared_tucker'):
        model = _make(kind, w, b, baseline, 'fp32', rank=3)
        assert model.weight is None
        assert len(model.factors) == (3 if kind == 'shared_tucker' else 2)
        got = model.project(x, layer=1)
        assert got.shape == x.shape and torch.isfinite(got).all()
        if kind == 'shared_tucker':
            assert storage_bytes(model.storage_tensors()) < storage_bytes([w, b, baseline.history_weights])


def test_rank_d_ridge_matches_dense_target():
    baseline, _, _ = fixture(4)
    keys = torch.randn(2, 5, 5, 2, 4)
    values = torch.randn_like(keys)
    dense, dense_bias = _fit(keys, values, values, baseline, 'dense_refit', .1, 4, None)
    low, low_bias = _fit(keys, values, values, baseline, 'low_rank_refit', .1, 4, None)
    torch.testing.assert_close(low, dense, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(low_bias, dense_bias, atol=1e-5, rtol=1e-5)


def test_packed_quant_and_no_dense_mirror():
    baseline, w, b = fixture(8)
    model = _make('dense_quant', w, b, baseline, 'int4')
    payload = model.weights[0]
    assert payload.codes.dtype == torch.uint8
    assert payload.codes.numel() == w.numel() // 2
    assert model.weight is None
    assert storage_bytes(model.storage_tensors()) == sum(t.untyped_storage().nbytes() for t in model.storage_tensors())
    assert torch.isfinite(model.project(torch.randn(3, 2, 4, 8), layer=0)).all()
