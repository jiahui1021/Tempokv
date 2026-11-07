"""Independent HIGGS/calibration/cache/CLI tests; no model weights or real results."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from rebuttal.experiments.aquakv.predictors import (
    AquaKVPredictor, fit_predictors, load_predictor, predictor_state, reconstruct_layer, replay,
)
from rebuttal.experiments.aquakv.quantizer import HiggsConfig, Quantizer, pack_indices, unpack_indices
from rebuttal.experiments.aquakv.runtime import AquaKVCache, rotate_pre_rope
from rebuttal.experiments.aquakv.upstream.hadamard import reference_hadamard
from rebuttal.experiments.aquakv.upstream.linear_utils import fit_linear_regression
from rebuttal.experiments.aquakv.workflow import run, build_job, verify_videos

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT/'experiments/Self-Forcing'
HIGGS = dict(hadamard_groupsize=8, edenn_d=2, edenn_n=16, chunk_size=8)


def synth(seed=0, records=3, frames=5):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(3, records, frames, 2, 4, generator=g)
    k[1] = .8*k[0]+.2*k[1]
    k[2] = .8*k[1]+.2*k[2]
    v = .5*k+.1*torch.randn(k.shape, generator=g)
    return k, v


def make_predictor(dtype='float32'):
    k, v = synth(frames=20)
    q = Quantizer(8, HiggsConfig(**HIGGS))
    p, _ = fit_predictors(k, v, q, parameter_dtype=dtype)
    return p, q


@pytest.mark.parametrize('bits', [4, 6, 8])
def test_centroid_indices_are_physically_bitpacked(bits):
    x = torch.arange(39).reshape(3, 13).remainder(1 << bits).to(torch.uint8)
    packed = pack_indices(x, bits)
    assert packed.numel() == 3*((13*bits+7)//8)
    assert torch.equal(unpack_indices(packed, bits, 13), x)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('layer', [0, 1])
def test_higgs_decode_matches_upstream_and_counts_real_padding(dtype, layer):
    q = Quantizer(12, HiggsConfig(**HIGGS))
    x = torch.linspace(-3, 3, 60).reshape(1, 5, 2, 6).to(dtype)
    native = q._backend(layer).quantize(x.flatten(0, -3).reshape(-1, 12))
    expected = q._backend(layer).dequantize(native).reshape_as(x).to(dtype)
    state = q.encode(x, layer)
    assert torch.equal(q.decode(state, layer), expected)
    bits = 8 if layer == 0 else 4
    # 12 channels padded to 16, two groups of four 2D vectors.
    assert state.storage_bytes() == 5*2*(4*bits//8+4)
    assert all(torch.isfinite(q.roundtrip(torch.zeros_like(x), layer)).flatten())


def test_upstream_calibration_uses_reconstructed_inputs_and_original_v_order():
    k, v = synth()
    q = Quantizer(8, HiggsConfig(**HIGGS))
    p, rows = fit_predictors(k, v, q, parameter_dtype='float32', validation_records=1, seed=4)
    order = torch.randperm(3, generator=torch.Generator().manual_seed(4))[1:]
    k0 = q.roundtrip(k[0], 0)
    x = k0[order].reshape(-1, 8)
    y = k[1, order].reshape(-1, 8)
    w, b = fit_linear_regression(x, y, reg_rate=1e-3, fit_intercept=True,
                                compute_dtype=torch.float32, compute_device=k.device, chunk_size=4096)
    torch.testing.assert_close(p.keys['1'].weight, w, atol=0, rtol=0)
    torch.testing.assert_close(p.keys['1'].bias, b, atol=0, rtol=0)
    pk = p.predict_key(k0, 1)
    k1 = pk+q.roundtrip(k[1]-pk, 1)
    v0 = q.roundtrip(v[0], 0)
    expected_v = p.values['1'](torch.cat([k1.flatten(-2), v0.flatten(-2)], -1)).reshape_as(v0)
    torch.testing.assert_close(p.predict_value(k1, v0, 1), expected_v, atol=0, rtol=0)
    assert rows[1]['calibration']['v']['validation_prediction_nmse'] is not None
    assert p.parameter_bytes() == sum(t.numel()*t.element_size() for t in p.parameters())


def test_predictor_loading_preserves_dtype_and_rng():
    p, _ = make_predictor('bfloat16')
    rng = torch.random.get_rng_state().clone()
    restored = load_predictor(predictor_state(p))
    assert restored.dtype == torch.bfloat16
    assert torch.equal(rng, torch.random.get_rng_state())
    with pytest.raises(ValueError, match='independent'):
        load_predictor({'key_weights': []})


def run_pass(cache, keys, values, start, captured=None, rotate=None):
    def attention(q, k, v):
        if captured is not None:
            captured.append((k.clone(), v.clone()))
        return q
    for layer in range(cache.predictor.layers):
        cache.attend(layer, keys[layer], keys[layer], values[layer], current_start=start,
                     grid_sizes=torch.tensor([[keys.shape[2]//cache.frame_seqlen, 1, cache.frame_seqlen]]),
                     freqs=None, rotate=rotate or (lambda x, *a, **kw: x.clone()), attention=attention)


def test_runtime_replaces_noisy_block_commits_clean_and_decodes_cross_layer():
    p, q = make_predictor()
    cache = AquaKVCache(p, q, frame_seqlen=1, prefix_frames=0, recent_buffer_frames=0)
    k, v = synth(frames=3)
    k, v = k[:, :1], v[:, :1]
    run_pass(cache, k+9, v+9, 0)
    run_pass(cache, k+5, v+5, 0)
    run_pass(cache, k, v, 0)
    assert cache.counts['replacements'] == 2
    cache.commit_clean(0, 3)
    assert all(c.packed for c in cache.chunks)
    captured = []
    run_pass(cache, k+1, v+1, 3, captured)
    previous = None
    for layer in range(3):
        previous, _, _ = reconstruct_layer(k[layer], v[layer], layer, p, q, previous)
        assert torch.equal(captured[layer][0][:, :3], previous[0])
        assert torch.equal(captured[layer][1][:, :3], previous[1])
    assert cache.previous == []
    assert cache.report()['persistent_reconstructed_history'] is False
    cache.commit_clean(3, 6)
    cache.reset()
    assert cache.chunks == [] and cache.active is None and cache.committed_end == 0


def test_prefix_buffer_storage_and_partial_layer_rejection():
    p, q = make_predictor()
    cache = AquaKVCache(p, q, frame_seqlen=1, prefix_frames=1, recent_buffer_frames=6)
    k, v = synth(frames=3)
    run_pass(cache, k[:, :1], v[:, :1], 0)
    cache.commit_clean(0, 3)
    assert [(c.prefix, c.packed) for c in cache.chunks] == [(True, False), (False, False)]
    assert cache.chunks[0].layers[0][0].untyped_storage().nbytes() == 8*4
    assert cache.chunks[1].layers[0][0].untyped_storage().nbytes() == 2*8*4
    cache.flush()
    assert cache.chunks[0].prefix and not cache.chunks[0].packed and cache.chunks[1].packed
    with pytest.raises(ValueError, match='current uncommitted'):
        run_pass(cache, k[:, :1], v[:, :1], 0)


def test_sliding_window_keeps_absolute_rotary_positions_and_sink():
    p, q = make_predictor()
    cache = AquaKVCache(p, q, frame_seqlen=1, prefix_frames=1, recent_buffer_frames=100,
                        local_attn_size=3, sink_size=1)
    k = torch.zeros(3, 1, 3, 2, 4)
    run_pass(cache, k, k, 0)
    cache.commit_clean(0, 3)
    captured = []
    def rotate(x, sizes, freqs, *, start_frame, **kw):
        markers = torch.arange(start_frame, start_frame+x.shape[1]).reshape(1, -1, 1, 1)
        return x+markers
    run_pass(cache, k, k, 3, captured, rotate=rotate)
    assert captured[0][0][0, :, 0, 0].tolist() == [0, 4, 5]
    cache.commit_clean(3, 6)
    assert all(c.prefix or c.end > 4 for c in cache.chunks)


def test_rotary_matches_direct_complex_multiplication():
    x = torch.arange(48).reshape(1, 3, 2, 8).float()
    rates = torch.tensor([1., .1, .01, .001], dtype=torch.float64)
    freqs = torch.polar(torch.ones(12, 4, dtype=torch.float64), torch.arange(12)[:, None]*rates)
    y = rotate_pre_rope(x, torch.tensor([[3, 1, 1]]), freqs, start_frame=4)
    phases = torch.cat([freqs[4:7, :2], freqs[0, 2:].expand(3, 2)], -1)
    expected = torch.view_as_real(torch.view_as_complex(x.double().reshape(1, 3, 2, 4, 2))*phases[None, :, None]).flatten(-2).float()
    assert torch.equal(y, expected)
    # The same positions work beyond the initial rotary lookup table.
    extended = rotate_pre_rope(x, torch.tensor([[3, 1, 1]]), freqs[:5], start_frame=4)
    torch.testing.assert_close(y, extended, rtol=1e-6, atol=1e-6)


def save_trace(path, seed, prompt):
    k, v = synth(seed)
    np.savez(path, keys=k.numpy(), values=v.numpy(), layer_ids=np.arange(3),
             prompt_id=np.asarray(prompt), prompt_sha256=np.asarray(hashlib.sha256(prompt.encode()).hexdigest()),
             coordinate_system=np.asarray('pre_rope_normalized'))


def test_cli_independent_collect_plan_fit_replay_and_summary(tmp_path):
    for name, seed in [('cal', 0), ('ev', 3)]:
        save_trace(tmp_path/f'{name}.npz', seed, name)
    cfg = {'experiment': 'aquakv', 'device': 'cpu', 'parameter_dtype': 'float32', 'higgs': HIGGS,
           'calibration_trace': str(tmp_path/'cal.npz'), 'evaluation_trace': str(tmp_path/'ev.npz')}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    output = tmp_path/'out'
    for stage in ['plan', 'fit', 'replay', 'summarize']:
        result = subprocess.run([sys.executable, str(ROOT/'rebuttal/run.py'), '--experiment', 'aquakv',
                                 '--config', str(path), '--stage', stage, '--output-dir', str(output)],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    report = json.loads((output/'replay.json').read_text())
    assert report['split_check']['prompt_hashes_verified']
    assert report['aquakv']['storage']['predictor_parameter_bytes'] > 0
    assert set(report) == {'status', 'split', 'metric_scope', 'split_check', 'sources', 'aquakv', 'higgs_direct'}
    # Imports and fitting work with no installed TempoKV package.
    code = 'from rebuttal.experiments.aquakv.workflow import run; import sys; run('+repr(cfg)+', "replay", '+repr(str(output))+'); assert not any(k == "tempokv" or k.startswith("tempokv.") for k in sys.modules)'
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_default_plan_is_independent_and_legacy_settings_are_rejected(tmp_path):
    config = json.loads((ROOT/'rebuttal/configs/aquakv.json').read_text())
    result = run(config, 'plan', tmp_path)
    assert result['required_tempokv_artifacts'] == [] and result['independent']
    assert result['missing_for_collection'] == ['checkpoint', 'model_root']
    with pytest.raises(ValueError, match='legacy'):
        run({**config, 'codec_config': 'configs/legacy.json'}, 'fit', tmp_path)


def test_replay_rejects_calibration_prompts_without_needing_calibration_path(tmp_path):
    save_trace(tmp_path/'cal.npz', 0, 'same')
    save_trace(tmp_path/'ev.npz', 5, 'same')
    config = {'calibration_trace': str(tmp_path/'cal.npz'), 'evaluation_trace': str(tmp_path/'ev.npz'),
              'device': 'cpu', 'parameter_dtype': 'float32', 'higgs': HIGGS}
    run(config, 'fit', tmp_path/'out')
    config.pop('calibration_trace')
    with pytest.raises(ValueError, match='prompts overlap'):
        run(config, 'replay', tmp_path/'out')


def test_job_isolated_under_rebuttal_and_uses_no_tempokv_arguments(tmp_path):
    config = {'checkpoint': '/models/sf.pt', 'model_root': '/models/wan', 'higgs': HIGGS}
    job = build_job(config, tmp_path/'run', mode='aquakv', prompts='/data/eval.txt', frames=81, predictor_path=tmp_path/'aquakv.pt')
    assert 'rebuttal.experiments.aquakv.worker' in job.command
    worker = json.loads((tmp_path/'run/worker_config.json').read_text())
    assert not {'codec_config', 'k_params', 'v_params'} & worker.keys()
    assert not any('tempokv_adapter' in arg for arg in job.command)


def test_multi_sample_video_identity_uses_actual_batch_seed(tmp_path):
    folder = tmp_path/'run/videos'
    folder.mkdir(parents=True)
    prompts = tmp_path/'prompts.txt'
    prompts.write_text('A scene\n')
    records = []
    for sample in range(2):
        path = folder/f'0_{sample}.mp4'
        path.write_bytes(b'video')
        records.append({'path': str(path), 'prompt_index': 0, 'sample_index': sample, 'noise_seed': 7})
    (folder/'summary.json').write_text(json.dumps({'videos': records}))
    job = SimpleNamespace(output_dir=str(tmp_path/'run'))
    assert len(verify_videos(job, {'seed': 7, 'samples': 2}, prompts)['videos']) == 2


def test_native_attention_calls_independent_cache_handler():
    tree = ast.parse((RUNTIME/'wan/modules/causal_model.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CausalWanSelfAttention')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    namespace = {'torch': torch, 'attention': lambda *args: None}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'native_self_attention', 'exec'), namespace)
    layer = SimpleNamespace(num_heads=2, head_dim=4, norm_q=torch.nn.Identity(), norm_k=torch.nn.Identity(),
                            q=torch.nn.Identity(), k=torch.nn.Identity(), v=torch.nn.Identity(), o=torch.nn.Identity())
    seen = []
    def handler(q, k, v, **kwargs):
        seen.append(kwargs['current_start'])
        return v*3
    x = torch.ones(1, 2, 8)
    out = namespace['forward'](layer, x, None, None, None, None, kv_cache={'aquakv_handler': handler}, current_start=4)
    assert seen == [4] and torch.equal(out, 3*x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/FHT numerical parity requires GPU')
def test_cuda_hadamard_matches_cpu_reference():
    from fast_hadamard_transform import hadamard_transform
    x = torch.linspace(-1, 1, 1024, device='cuda').to(torch.float16)
    torch.testing.assert_close(hadamard_transform(x), reference_hadamard(x), atol=.05, rtol=.005)


def native_pipeline_base():
    """Execute the repository's actual inference loop with tiny CPU model objects."""
    tree = ast.parse((RUNTIME/'pipeline/causal_inference.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CausalInferencePipeline')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'inference')
    class Progress:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args): pass
    namespace = {'torch': torch, 'List': list, 'Optional': __import__('typing').Optional,
                 'os': os, 'tqdm': Progress}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'native_pipeline_loop', 'exec'), namespace)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads, self.head_dim = 2, 4
            self.norm_k, self.v = torch.nn.Identity(), torch.nn.Identity()
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
            for block in self.blocks:
                block.self_attn = Attention()
            self.model = SimpleNamespace(blocks=self.blocks, local_attn_size=-1)
        def forward(self, noisy_image_or_video, kv_cache, current_start, **kwargs):
            x = noisy_image_or_video.reshape(1, -1, 8)
            grid = torch.tensor([[x.shape[1]//2, 1, 2]])
            for layer, block in enumerate(self.blocks):
                k = block.self_attn.norm_k(x).reshape(1, -1, 2, 4)
                v = block.self_attn.v(x).reshape_as(k)
                cache = kv_cache[layer]
                if 'aquakv_handler' in cache:
                    cache['aquakv_handler'](k, k, v, current_start=current_start, grid_sizes=grid,
                                            freqs=torch.ones(64, 2, dtype=torch.complex128),
                                            attention=lambda q, k, v: q)
                cache['local_end_index'] = current_start+x.shape[1]
            return None, noisy_image_or_video
    class Base(torch.nn.Module):
        inference = namespace['inference']
        def __init__(self):
            super().__init__()
            self.generator = Generator()
            self.num_transformer_blocks, self.num_heads, self.head_dim = 3, 2, 4
            self.frame_seq_length, self.local_attn_size = 2, -1
            self.num_frame_per_block, self.independent_first_frame = 3, False
            self.denoising_step_list = torch.tensor([100, 0])
            self.args = SimpleNamespace(context_noise=0)
            self.text_encoder = lambda **kwargs: {}
            self.scheduler = SimpleNamespace(add_noise=lambda prediction, *args: prediction)
            self.vae = SimpleNamespace(decode_to_pixel=lambda output, **kwargs: output)
            self.kv_cache1, self._uses_pre_rope_cache = None, False
            self.summary = {}
        def _initialize_kv_cache(self, **kwargs):
            from rebuttal.experiments.aquakv.worker import CacheSlot
            self.kv_cache1 = [{'k': CacheSlot(), 'v': CacheSlot(), 'global_end_index': 0, 'local_end_index': 0} for _ in range(3)]
        def _initialize_crossattn_cache(self, **kwargs):
            self.crossattn_cache = [{'is_init': False} for _ in range(3)]
        def _sample_resident_kv(self): pass
        def _resident_kv_bytes(self): return 0, 0
    return Base


@pytest.mark.parametrize('mode', ['aquakv', 'collect'])
def test_worker_runs_native_clean_refresh_loop_and_resets_per_prompt(tmp_path, monkeypatch, mode):
    from rebuttal.experiments.aquakv.worker import install
    p, q = make_predictor('bfloat16')
    artifact = {**predictor_state(p), 'quantizer': q.settings(), 'layer_ids': [0, 1, 2]}
    torch.save(artifact, tmp_path/'predictor.pt')
    prompts = tmp_path/'prompts.txt'
    prompts.write_text('An independent scene\nAnother independent scene\n')
    package, module = ModuleType('pipeline'), ModuleType('pipeline.causal_inference')
    module.CausalInferencePipeline = native_pipeline_base()
    monkeypatch.setitem(sys.modules, 'pipeline', package)
    monkeypatch.setitem(sys.modules, 'pipeline.causal_inference', module)
    config = {'mode': mode, 'predictor_path': str(tmp_path/'predictor.pt'), 'higgs': HIGGS,
              'prompts': str(prompts), 'output_dir': str(tmp_path),
              'cache': {'prefix_frames': 0, 'recent_buffer_frames': 0}, 'trace_tokens_per_frame': 2}
    pipeline = install(config, RUNTIME)()
    for prompt in range(2):
        pipeline.set_research_context(prompt_index=prompt, seed=prompt+10)
        pipeline.inference(noise=torch.ones(1, 6, 2, 1, 8, dtype=torch.bfloat16),
                           text_prompts=['scene'], decode_output=False)
        if mode == 'aquakv':
            assert pipeline.aquakv_cache.counts['clean_commits'] == 2
            assert pipeline.aquakv_cache.counts['replacements'] == 4
            assert pipeline.aquakv_cache.counts['decode_k'] > 0
            assert pipeline.aquakv_cache.committed_end == 12
        else:
            path = next((tmp_path/'traces').glob(f'prompt_{prompt:06d}*.npz'))
            with np.load(path) as data:
                assert data['keys'].shape == (3, 2, 6, 2, 4)
                assert data['coordinate_system'].item() == 'pre_rope_normalized'
                assert data['seed'].item() == prompt+10


def test_internal_predictor_mutation_is_rejected(tmp_path):
    save_trace(tmp_path/'cal.npz', 0, 'cal')
    save_trace(tmp_path/'ev.npz', 3, 'heldout')
    config = {'calibration_trace': str(tmp_path/'cal.npz'), 'evaluation_trace': str(tmp_path/'ev.npz'),
              'device': 'cpu', 'parameter_dtype': 'float32', 'higgs': HIGGS}
    output = tmp_path/'out'
    run(config, 'fit', output)
    path = output/'aquakv_predictor.pt'
    state = torch.load(path, weights_only=True)
    state['weights']['keys.1.weight'].zero_()
    torch.save(state, path)
    with pytest.raises(ValueError, match='changed after calibration'):
        run(config, 'replay', output)


def test_evaluate_does_not_accept_changed_generated_videos(tmp_path):
    candidate = tmp_path/'candidate.mp4'
    candidate.write_bytes(b'changed')
    report = {'status': 'complete', 'jobs': [
        {'name': 'bf16', 'artifacts': []},
        {'name': 'aquakv', 'artifacts': [{'path': str(candidate), 'sha256': 'old'}]}]}
    (tmp_path/'generate.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='video changed'):
        run({}, 'evaluate', tmp_path)
