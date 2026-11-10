import torch

from rebuttal.adapters.experimental_codec import ResearchTemporal
from rebuttal.experiments.paper_architecture.storage import storage_bytes
from rebuttal.experiments.paper_architecture.temporal import build_temporal, load_temporal, _replay
from tempokv.config import CodecConfig


def fixtures(d=4, p=2, t=7):
    torch.manual_seed(3)
    keys = torch.randn(2, 3, t, 2, d)
    weights = torch.randn(4, p, p, d) * .03
    weights[:, :, 0] += .8
    bias = torch.randn(4, p, d) * .01
    baseline = ResearchTemporal('channel_linear', 'layer_head', weights, bias, 2, 2)
    codec = CodecConfig(k_bits=4, v_bits=4, anchor_bits=4, group_size=d,
                        reconstruction_dtype='bfloat16', scale_dtype='bfloat16', fit_mode='native')
    return keys, baseline, codec


def test_paper_exact_and_independent_heads():
    keys, baseline, codec = fixtures()
    model, report = build_temporal(keys, baseline, codec, {'kind': 'paper', 'precision': 'fp32'})
    history = [torch.randn(3, 2, 1, 4), torch.randn(3, 2, 1, 4)]
    for layer in range(2):
        torch.testing.assert_close(model.predict(history, layer=layer), baseline.predict(history, layer=layer))
    altered = [h.clone() for h in history]
    altered[0][:, 1] += 100
    torch.testing.assert_close(model.predict(altered, layer=0)[:, 0], model.predict(history, layer=0)[:, 0])
    torch.testing.assert_close(load_temporal(model.state()).predict(history, layer=1), model.predict(history, layer=1))
    assert report['rounds'] == 0


def test_identity_has_no_stored_parameters():
    keys, baseline, codec = fixtures()
    model, _ = build_temporal(keys, baseline, codec, {'kind': 'identity'})
    history = [torch.randn(3, 2, 1, 4)]
    assert model.storage_tensors() == []
    torch.testing.assert_close(model.predict(history, layer=1), history[0])


def test_block_one_matches_channel_connectivity():
    keys, baseline, codec = fixtures()
    block, _ = build_temporal(keys, baseline, codec, {'kind': 'block', 'block_size': 1}, rounds=1)
    channel, _ = build_temporal(keys, baseline, codec, {'kind': 'channel_refit'}, rounds=1)
    history = [torch.randn(3, 2, 1, 4), torch.randn(3, 2, 1, 4)]
    torch.testing.assert_close(block.predict(history, layer=0), channel.predict(history, layer=0), atol=1e-5, rtol=1e-5)


def test_precision_payload_and_closed_loop():
    keys, baseline, codec = fixtures(t=8)
    full, _ = build_temporal(keys, baseline, codec, {'kind': 'dense', 'precision': 'fp32'}, rounds=1)
    half, _ = build_temporal(keys, baseline, codec, {'kind': 'dense', 'precision': 'fp16'}, rounds=1)
    assert storage_bytes(half.storage_tensors()) < storage_bytes(full.storage_tensors())
    rank, _ = build_temporal(keys, baseline, codec, {'kind': 'low_rank', 'rank': 1, 'precision': 'fp16'}, rounds=1)
    assert rank.weight is None
    assert storage_bytes(rank.storage_tensors()) < storage_bytes(half.storage_tensors())
    rebuilt = _replay(keys, rank, codec)
    assert rebuilt.shape == keys.shape and torch.isfinite(rebuilt).all()
    torch.testing.assert_close(load_temporal(rank.state()).predict([rebuilt[0, :, 1].unsqueeze(2)], layer=0),
                               rank.predict([rebuilt[0, :, 1].unsqueeze(2)], layer=0))


def test_int4_storage_and_causal_replay():
    keys, baseline, codec = fixtures(t=8)
    model, _ = build_temporal(keys, baseline, codec, {'kind': 'dense', 'precision': 'int4'}, rounds=1)
    assert model.weight.codes.dtype == torch.uint8
    assert model.weight.scales.dtype == torch.float16
    early = _replay(keys, model, codec)
    changed = keys.clone()
    changed[:, :, 6:] += 20
    later = _replay(changed, model, codec)
    torch.testing.assert_close(early[:, :, :6], later[:, :, :6])
