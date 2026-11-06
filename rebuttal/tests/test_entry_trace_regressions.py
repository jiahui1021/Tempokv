"""Regression coverage for executable plans, resume, and full-cohort identity."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rebuttal.common.manifest import input_fingerprints, read_json
from rebuttal.common.trace import load_trace, load_trace_identity, check_trace_split
from rebuttal.run import DEFAULT_CONFIGS, execute, load_configuration, main


def save_trace(path, prompts, *, sidecar=True):
    x = np.random.default_rng(7).normal(size=(1, len(prompts), 9, 1, 64)).astype(np.float32)
    data = {} if sidecar else {"prompt_ids": np.asarray(prompts)}
    np.savez(path, keys=x, values=x * .8, **data)
    if sidecar:
        path.with_suffix('.identity.json').write_text(json.dumps({"records": [
            {"record_index": i, "prompt_id": prompt, "prompt_sha256": prompt, "seed": 0}
            for i, prompt in enumerate(prompts)]}))


def test_all_default_configs_resolve_and_six_plans_execute(tmp_path):
    for experiment, path in DEFAULT_CONFIGS.items():
        assert load_configuration(path)['experiment'] == experiment
    assert main(['--experiment', 'all', '--stage', 'plan', '--output-dir', str(tmp_path)]) == 0
    assert len(list(tmp_path.glob('*/manifest.json'))) == 6


def test_resume_is_execution_control_and_changed_input_still_rejected(tmp_path):
    source = tmp_path / 'input.txt'
    source.write_text('original')
    config = {'calibration_prompts': str(source)}
    output = tmp_path / 'run'
    execute('quant_aware_fit', config, 'plan', output)
    first = read_json(output / 'manifest.json')['config_sha256']
    execute('quant_aware_fit', config, 'plan', output, resume=True)
    assert read_json(output / 'manifest.json')['config_sha256'] == first
    source.write_text('changed')
    with pytest.raises(ValueError, match='input contents changed'):
        execute('quant_aware_fit', config, 'plan', output, resume=True)


def test_sampling_cannot_hide_overlapping_calibration_or_evaluation_records(tmp_path):
    cal, test = tmp_path/'cal.npz', tmp_path/'test.npz'
    save_trace(cal, ['cal', 'overlap'])
    save_trace(test, ['test', 'overlap'])
    for calibration in (load_trace(cal, max_records=1), load_trace_identity(cal)):
        with pytest.raises(ValueError, match='overlap'):
            check_trace_split(calibration, load_trace(test, max_records=1))


def test_identity_only_reader_never_loads_kv_arrays(tmp_path, monkeypatch):
    cal = tmp_path/'cal.npz'
    save_trace(cal, ['A', 'B'])
    original = np.lib.npyio.NpzFile.__getitem__
    def metadata_only(self, key):
        assert key not in {'keys', 'values'}
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', metadata_only)
    assert load_trace_identity(cal).metadata['split_identity']['prompt_ids'] == ['A', 'B']


def test_incomplete_sidecar_cannot_claim_verified_split(tmp_path):
    cal, test = tmp_path/'cal.npz', tmp_path/'test.npz'
    save_trace(cal, ['cal-a', 'cal-b'])
    save_trace(test, ['test-a', 'test-b'])
    cal.with_suffix('.identity.json').write_text(json.dumps({'records': [
        {'record_index': 0, 'prompt_id': 'cal-a', 'seed': 0}]}))
    result = check_trace_split(load_trace_identity(cal), load_trace(test, max_records=1))
    assert result['prompt_identity_verified'] is False


def test_plural_embedded_identity_and_sidecar_fingerprints(tmp_path):
    cal, test = tmp_path/'cal.npz', tmp_path/'test.npz'
    save_trace(cal, ['A', 'B'], sidecar=False)
    save_trace(test, ['C', 'D'])
    assert check_trace_split(load_trace(cal), load_trace(test))['prompt_identity_verified']
    config = {'evaluation_trace': str(test)}
    first = input_fingerprints(config)
    assert len(first) == 2
    test.with_suffix('.identity.json').write_text('{}')
    assert input_fingerprints(config) != first


def test_previous_frame_real_fit_replay_includes_learned_one_frame(tmp_path):
    from dataclasses import asdict
    from tempokv.config import CodecConfig
    from rebuttal.experiments.previous_frame import run, variant_configs
    cal, test = tmp_path/'cal.npz', tmp_path/'test.npz'
    save_trace(cal, ['A', 'B'])
    save_trace(test, ['C', 'D'])
    cp = tmp_path/'codec.json'
    cp.write_text(json.dumps(asdict(CodecConfig(k_history_length=2, v_history_length=1,
        group_size=64, stride=1, fit_mode='legacy', use_triton=False))))
    config = {'codec_config': str(cp), 'calibration_trace': str(cal),
        'evaluation_trace': str(test), 'variants': ['production', 'k_one_frame', 'k_identity'],
        'require_prompt_identity': True, 'max_records': 1, 'span_frames': 6, 'device': 'cpu'}
    assert variant_configs(config)['k_one_frame'].k_history_length == 1
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run(config, 'fit', tmp_path/'output')
        result = run(config, 'replay', tmp_path/'output')
    finally:
        torch.set_num_threads(prior_threads)
    assert set(result['variants']) == set(config['variants'])
    for report in result['variants'].values():
        assert report['trace_split']['prompt_identity_verified']
    cal.with_suffix('.identity.json').write_text('{}')
    with pytest.raises(ValueError, match='identity|identities'):
        run(config, 'replay', tmp_path/'output')


def test_known_hash_overlap_cannot_hide_behind_partial_hashes(tmp_path):
    cal, test = tmp_path/'cal.npz', tmp_path/'test.npz'
    save_trace(cal, ['a', 'b'])
    save_trace(test, ['c', 'd'])
    cal.with_suffix('.identity.json').write_text(json.dumps({'records': [
        {'record_index': 0, 'prompt_id': 'a', 'prompt_sha256': 'same'},
        {'record_index': 1, 'prompt_id': 'b'}]}))
    test.with_suffix('.identity.json').write_text(json.dumps({'records': [
        {'record_index': 0, 'prompt_id': 'c', 'prompt_sha256': 'same'},
        {'record_index': 1, 'prompt_id': 'd', 'prompt_sha256': 'different'}]}))
    with pytest.raises(ValueError, match='hashes overlap'):
        check_trace_split(load_trace_identity(cal), load_trace_identity(test))


def test_sidecar_cannot_relabel_conflicting_embedded_prompt_ids(tmp_path):
    trace = tmp_path/'trace.npz'
    save_trace(trace, ['a', 'b'], sidecar=False)
    trace.with_suffix('.identity.json').write_text(json.dumps({'records': [
        {'record_index': 0, 'prompt_id': 'different'}]}))
    for reader in (load_trace, load_trace_identity):
        with pytest.raises(ValueError, match='conflicts with embedded'):
            reader(trace)


def test_new_stage_must_keep_inputs_and_code_snapshot(tmp_path, monkeypatch):
    from rebuttal.common import manifest
    source = tmp_path/'trace.txt'
    source.write_text('original')
    config = {'evaluation_trace': str(source)}
    output = tmp_path/'run'
    monkeypatch.setattr(manifest, 'code_fingerprint', lambda: 'first')
    record = manifest.prepare_run(output, config, experiment='test', stage='fit', resume=False)
    manifest.finish_run(output, record, 'fit', {'status': 'complete'})
    source.write_text('changed')
    with pytest.raises(ValueError, match='input contents changed'):
        manifest.prepare_run(output, config, experiment='test', stage='replay', resume=False)
    source.write_text('original')
    monkeypatch.setattr(manifest, 'code_fingerprint', lambda: 'second')
    with pytest.raises(ValueError, match='experiment code changed'):
        manifest.prepare_run(output, config, experiment='test', stage='replay', resume=False)


def test_plan_promotion_freezes_newly_available_inputs(tmp_path):
    from rebuttal.common import manifest
    source = tmp_path/'later.npz'
    config = {'evaluation_trace': str(source)}
    output = tmp_path/'run'
    record = manifest.prepare_run(output, config, experiment='test', stage='plan', resume=False)
    manifest.finish_run(output, record, 'plan', {'status': 'planned'})
    assert record['inputs'] == []
    source.write_bytes(b'now present')
    promoted = manifest.prepare_run(output, config, experiment='test', stage='fit', resume=False)
    assert promoted['inputs'] == manifest.input_fingerprints(config)
    assert promoted['inputs']


def test_quant_aware_export_checks_validation_against_final_test(tmp_path):
    from rebuttal.experiments.quant_calibration import selection_split
    from rebuttal.common.manifest import file_sha256, write_json
    paths = {}
    for name, prompts in [('cal', ['A', 'B']), ('val', ['C', 'D']), ('test', ['E', 'F'])]:
        paths[name] = tmp_path/f'{name}.npz'
        save_trace(paths[name], prompts)
    k, v = tmp_path/'k.npz', tmp_path/'v.npz'
    k.write_bytes(b'k parameters'); v.write_bytes(b'v parameters')
    write_json(tmp_path/'fit_report.json', {
        'method': 'bounded_local_random_search_real_quantizer',
        'k_sha256': file_sha256(k), 'v_sha256': file_sha256(v),
        'selection_sources': [
            {'role': role, 'path': str(paths[name]), 'sha256': file_sha256(paths[name]),
             'split_identity': load_trace_identity(paths[name]).metadata['split_identity']}
            for role, name in [('calibration', 'cal'), ('validation', 'val')]]})
    config = {'k_params': str(k), 'v_params': str(v), 'require_prompt_identity': True}
    # Data bytes coincide in this fixture; vary final test values for a genuinely separate file.
    with np.load(paths['test']) as src:
        keys, values = src['keys'].copy(), src['values'].copy()
    np.savez(paths['test'], keys=keys+1, values=values+1)
    assert selection_split(config, load_trace(paths['test']))['status'] == 'recorded_selection_sources_checked'
    with pytest.raises(ValueError, match='overlap'):
        selection_split(config, load_trace(paths['val']))


def test_quant_aware_fit_exports_selection_provenance_for_real_parameters(tmp_path):
    from dataclasses import asdict
    from tempokv.config import CodecConfig
    from rebuttal.experiments.previous_frame import run as predictors
    from rebuttal.experiments.quant_aware_fit import run as quant_fit
    from rebuttal.experiments.quant_calibration import selection_split
    cp = tmp_path/'codec.json'
    cp.write_text(json.dumps(asdict(CodecConfig(k_bits=2, v_bits=2, k_history_length=2,
        v_history_length=1, group_size=64, stride=1, fit_mode='legacy', use_triton=False))))
    cal, val, test = (tmp_path/f'{name}.npz' for name in ('cal', 'val', 'test'))
    for path, prompts in ((cal, ['A', 'B']), (val, ['C', 'D']), (test, ['E', 'F'])):
        save_trace(path, prompts)
    # Keep independent synthetic tensors as well as prompt identities.
    for index, path in enumerate((val, test), start=1):
        x = np.random.default_rng(40+index).normal(size=(1, 2, 9, 1, 64)).astype(np.float32)
        np.savez(path, keys=x, values=x*.8)
    settings = {'codec_config': str(cp), 'calibration_trace': str(cal), 'validation_trace': str(val),
                'variants': ['production'], 'require_prompt_identity': True,
                'optimization_steps': 1, 'device': 'cpu', 'span_frames': 6}
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        fitted = predictors(settings, 'fit', tmp_path/'initial')['variants']['production']
        tuned = quant_fit({**settings, 'k_params': fitted['k_params'], 'v_params': fitted['v_params']},
                         'fit', tmp_path/'tuned')
    finally:
        torch.set_num_threads(prior_threads)
    assert {row['role'] for row in tuned['selection_sources']} == {'calibration', 'validation'}
    config = {**settings, 'k_params': tuned['k_params'], 'v_params': tuned['v_params']}
    assert selection_split(config, load_trace(test))['status'] == 'recorded_selection_sources_checked'
    with pytest.raises(ValueError, match='overlap'):
        selection_split(config, load_trace(val))
