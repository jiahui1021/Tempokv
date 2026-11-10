import torch

from tempokv.config import CodecConfig
from rebuttal.adapters.experimental_codec import ResearchCross
from rebuttal.experiments.paper_architecture.cross import build_cross
from rebuttal.experiments.paper_architecture.history import build_history, load_history_cross
from rebuttal.experiments.paper_architecture.storage import storage_bytes


def fixture(d=8):
    torch.manual_seed(4)
    keys = torch.randn(1, 3, 5, 2, d)
    values = .7 * keys + torch.randn_like(keys) * .05
    w = torch.randn(2, d, d) / d
    h = torch.randn(1, 1, 2, d) / 10
    baseline = ResearchCross('dense', 'layer_head', torch.randn(2, d) / 10, h,
                             1, 2, w)
    paper, _ = build_cross(keys, values, baseline, None, {'kind': 'paper'})
    codec = CodecConfig(k_bits=4, v_bits=4, group_size=d, stride=1, k_history_length=2,
                        v_history_length=1, fit_mode='native', use_triton=False)
    return keys, values, paper, codec


def test_paper_history_preserves_cross_projection_and_vh1():
    keys, values, paper, codec = fixture()
    model, report = build_history(keys, values, paper, codec, {'kind': 'paper'})
    source = keys[0].permute(0, 2, 1, 3)
    torch.testing.assert_close(model.project(source, layer=0), paper.project(source, layer=0))
    projected = model.project(source, layer=0)
    previous = values[0, :, 0:1].permute(0, 2, 1, 3)
    torch.testing.assert_close(model.add_history(projected[:, :, 1:2], [previous], layer=0),
                               paper.add_history(projected[:, :, 1:2], [previous], layer=0))
    assert report['rounds'] == 0 and model.history_length == 1
    assert model.history_weights.numel() == 0


def test_refits_are_finite_causal_and_keep_paper_weight():
    keys, values, paper, codec = fixture()
    source = keys[0].permute(0, 2, 1, 3)
    for kind, precision in [('scalar', 'fp32'), ('dense', 'fp32'),
                            ('block', 'fp16'), ('low_rank', 'fp16')]:
        for rounds in (1, 2):
            model, report = build_history(keys, values, paper, codec,
                                          {'kind': kind, 'precision': precision,
                                           'block_size': 2, 'rank': 1}, rounds=rounds)
            torch.testing.assert_close(model.weights[0].materialize(), paper.weights[0].materialize())
            assert torch.isfinite(model.project(source, layer=0)).all()
            previous = torch.randn(3, 2, 1, 8)
            projection = model.project(source, layer=0)[:, :, 1:2]
            result = model.add_history(projection, [previous], layer=0)
            assert torch.isfinite(result).all() and result.shape == projection.shape
            assert report['rounds'] == rounds
            restored = load_history_cross(model.state()).to('cpu')
            torch.testing.assert_close(restored.add_history(projection, [previous], layer=0), result)
            assert storage_bytes(restored.storage_tensors()) == sum(
                t.untyped_storage().nbytes() for t in restored.storage_tensors())
            assert restored.history_weights.numel() == 0


def test_block_history_has_no_cross_block_effect():
    keys, values, paper, codec = fixture()
    model, _ = build_history(keys, values, paper, codec,
                             {'kind': 'block', 'block_size': 2, 'precision': 'fp16'}, rounds=1)
    zero = torch.zeros(1, 2, 1, 8)
    impulse = zero.clone(); impulse[..., 0] = 1
    response = model.add_history(zero, [impulse], layer=0)
    assert torch.count_nonzero(response[..., 2:]) == 0
