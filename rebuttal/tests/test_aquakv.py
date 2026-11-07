"""CPU checks for the faithful AQUA-KV port. No GPU and no quality numbers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from rebuttal.experiments.aquakv.cache import AquaRuntime
from rebuttal.experiments.aquakv.fit import chunks, fit_official_chunked, fit_streaming
from rebuttal.experiments.aquakv.hadamard import hadamard_transform
from rebuttal.experiments.aquakv.method import (
    AquaSpec,
    apply_linear,
    calibrate,
    flatten_tokens,
    make_quantizers,
    project_layer,
    unflatten_tokens,
)
from rebuttal.experiments.aquakv.official import HiggsHandle, official_modules, pack_indices, unpack_indices
from rebuttal.experiments.aquakv.store import (
    ListSampleSource,
    MemmapSampleSource,
    calibrate_streaming,
    check_collection_against_prompts,
    write_sample,
)
from rebuttal.run import main

ROOT = Path(__file__).resolve().parents[2]


def _spec(**overrides) -> AquaSpec:
    payload = dict(
        num_layers=2, num_heads=2, head_dim=4, hadamard_groupsize=8,
        edenn_d=2, edenn_n=16, first_layer_edenn_d=2, first_layer_edenn_n=16,
        prefix_tokens=2, recent_buffer_tokens=2, percdamp=1e-3, valid_nsamples=1,
        seed=0, activation_dtype="float32", index_packing="packed", chunk_size=8,
        higgs_chunk_size=4,
    )
    payload.update(overrides)
    return AquaSpec(**payload)


def _videos(n=4, tokens=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    layer0_k = torch.randn(n, tokens, 2, 4, generator=g)
    layer0_v = torch.randn(n, tokens, 2, 4, generator=g)
    # A recoverable linear map plus noise, so train corruption moves the fit.
    weight = torch.randn(8, 8, generator=g)
    bias = torch.randn(8, generator=g)
    flat0 = layer0_k.reshape(n, tokens, 8)
    layer1_k = (flat0 @ weight.T + bias).reshape(n, tokens, 2, 4)
    layer1_k = layer1_k + 0.05 * torch.randn(n, tokens, 2, 4, generator=g)
    layer1_v = 0.3 * layer1_k + 0.2 * layer0_v
    keys = torch.stack([layer0_k, layer1_k], dim=1)
    values = torch.stack([layer0_v, layer1_v], dim=1)
    return [(keys[i], values[i]) for i in range(n)]


def _sylvester(n: int) -> torch.Tensor:
    block = torch.ones(1, 1)
    while block.shape[0] < n:
        block = torch.cat([
            torch.cat([block, block], dim=1),
            torch.cat([block, -block], dim=1),
        ], dim=0)
    return block


def test_hadamard_matches_sylvester_and_scales():
    x = torch.randn(3, 5, 8)
    matrix = _sylvester(8)
    assert torch.allclose(hadamard_transform(x), x @ matrix, atol=1e-5)
    assert torch.allclose(hadamard_transform(x, scale=0.25), (x @ matrix) * 0.25, atol=1e-5)
    restored = hadamard_transform(hadamard_transform(x), scale=1 / 8)
    assert torch.allclose(restored, x, atol=1e-5)


def test_higgs_matches_official_quantize_dequantize():
    _, _, quantizers = official_modules()
    x = torch.randn(6, 8)
    for packing in ("packed", "uint8"):
        handle = HiggsHandle(channel=8, hadamard_groupsize=8, edenn_d=2, edenn_n=16, packing=packing)
        recon, stored, scales = handle.quantize(x)
        official = quantizers.HiggsQuantizer(8, 2, 16, channel_size=8, chunk_size=4)
        reference = official.quantize_dequantize(x)
        assert torch.allclose(recon, reference, atol=0, rtol=0)
        assert torch.allclose(handle.dequantize(stored, scales, x.dtype), reference, atol=0, rtol=0)
        raw = official.quantize(x).idx
        flat_idx = raw.reshape(x.shape[0], -1)
        flat_stored = stored.reshape(x.shape[0], -1)
        if packing == "uint8":
            assert torch.equal(flat_stored, flat_idx)
        else:
            unpacked = unpack_indices(flat_stored, handle.codes_per_token, handle.index_bits)
            assert torch.equal(unpacked, flat_idx)
            assert torch.equal(pack_indices(raw, handle.index_bits).reshape(x.shape[0], -1), flat_stored)


def test_streaming_fit_matches_official_chunked_regression():
    inputs = torch.randn(80, 5)
    targets = inputs @ torch.randn(5, 3) + torch.tensor([0.2, -0.4, 0.7])
    weight, bias = fit_streaming(
        chunks(inputs, targets, 7), in_features=5, out_features=3,
        reg_rate=1e-3, out_dtype=torch.float32,
    )
    official_w, official_b = fit_official_chunked(inputs, targets, reg_rate=1e-3, chunk_size=7)
    assert torch.allclose(weight, official_w, atol=1e-5)
    assert torch.allclose(bias, official_b, atol=1e-5)


def test_calibrate_matches_official_fit_on_reconstructed_train_rows_only():
    spec = _spec()
    samples = _videos()
    bank, info = calibrate(samples, spec)
    streamed, streamed_info = calibrate_streaming(ListSampleSource(samples), spec)
    assert info["train_indices"] == streamed_info["train_indices"]
    assert info["valid_indices"] == streamed_info["valid_indices"]
    for layer in bank.key_weight:
        assert torch.allclose(bank.key_weight[layer], streamed.key_weight[layer], atol=1e-5)
        assert torch.allclose(bank.value_weight[layer], streamed.value_weight[layer], atol=1e-5)
    residual, first = make_quantizers(spec)
    hats = []
    for keys, values in samples:
        hat_k, _, _ = project_layer(keys[0], None, first)
        hats.append(hat_k)
    rows_x = torch.cat([flatten_tokens(hats[i]) for i in info["train_indices"]], dim=0)
    rows_y = torch.cat([flatten_tokens(samples[i][0][1]) for i in info["train_indices"]], dim=0)
    weight, bias = fit_official_chunked(rows_x, rows_y, reg_rate=spec.percdamp, chunk_size=spec.chunk_size,
                                        out_dtype=spec.dtype)
    assert torch.allclose(bank.key_weight[1], weight, atol=1e-5)
    assert torch.allclose(bank.key_bias[1], bias, atol=1e-5)
    # The same fit on clean layer-0 keys is a different predictor.
    clean = torch.cat([flatten_tokens(samples[i][0][0]) for i in info["train_indices"]], dim=0)
    other, _ = fit_official_chunked(clean, rows_y, reg_rate=spec.percdamp, chunk_size=spec.chunk_size)
    assert not torch.allclose(bank.key_weight[1], other, atol=1e-3)
    del residual


def test_validation_videos_do_not_enter_the_normal_equations():
    spec = _spec()
    samples = _videos()
    bank, info = calibrate(samples, spec)
    valid = info["valid_indices"][0]
    train = info["train_indices"][0]
    corrupted_valid = [(k.clone(), v.clone()) for k, v in samples]
    corrupted_valid[valid] = (corrupted_valid[valid][0] + 50, corrupted_valid[valid][1] - 50)
    bank_valid, _ = calibrate(corrupted_valid, spec)
    assert torch.allclose(bank.key_weight[1], bank_valid.key_weight[1], atol=0, rtol=0)
    assert torch.allclose(bank.value_weight[1], bank_valid.value_weight[1], atol=0, rtol=0)
    corrupted_train = [(k.clone(), v.clone()) for k, v in samples]
    corrupted_train[train] = (corrupted_train[train][0] + 3, corrupted_train[train][1] + 3)
    bank_train, _ = calibrate(corrupted_train, spec)
    assert not torch.allclose(bank.key_weight[1], bank_train.key_weight[1], atol=1e-3)


def test_cache_roundtrip_matches_project_layer_and_does_not_alias():
    spec = _spec()
    samples = _videos()
    bank, _ = calibrate(samples, spec)
    keys, values = samples[0]
    runtime = AquaRuntime(
        spec, bank, batch_size=1, frame_seq_length=4, max_frames=3,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    for frame in range(3):
        start, end = frame * 4, (frame + 1) * 4
        for layer in range(2):
            runtime.write(layer, "k", start, end, keys[layer, start:end].unsqueeze(0))
            runtime.write(layer, "v", start, end, values[layer, start:end].unsqueeze(0))
    stats = runtime.compress_finalized(12)
    assert stats["quant_end"] == 10
    assert 1 not in runtime.exact[0]["k"]
    assert 0 in runtime.exact[0]["k"] and 2 in runtime.exact[0]["k"]

    residual, first = make_quantizers(spec)
    quant_k = keys[:, 2:10]
    quant_v = values[:, 2:10]
    hat_k0, _, _ = project_layer(quant_k[0], None, first)
    pred_k = unflatten_tokens(apply_linear(bank.key_weight[1], bank.key_bias[1], flatten_tokens(hat_k0)), 2, 4)
    hat_k1, _, _ = project_layer(quant_k[1], pred_k, residual)
    hat_v0, _, _ = project_layer(quant_v[0], None, first)
    pred_v = unflatten_tokens(apply_linear(
        bank.value_weight[1], bank.value_bias[1],
        torch.cat([flatten_tokens(hat_k1), flatten_tokens(hat_v0)], dim=-1),
    ), 2, 4)
    hat_v1, _, _ = project_layer(quant_v[1], pred_v, residual)
    expected_k = torch.cat([keys[1, :2], hat_k1, keys[1, 10:]], dim=0)
    expected_v = torch.cat([values[1, :2], hat_v1, values[1, 10:]], dim=0)
    got_k = runtime.read(1, "k", 0, 12)
    got_v = runtime.read(1, "v", 0, 12, source_tensor=runtime.read(1, "k", 0, 12))
    assert torch.allclose(got_k[0], expected_k, atol=1e-5)
    assert torch.allclose(got_v[0], expected_v, atol=1e-5)
    assert torch.equal(runtime.read(0, "k", 0, 2)[0], keys[0, :2])

    prefix = runtime.read(0, "k", 0, 2)
    assert prefix.data_ptr() != runtime.exact[0]["k"][0].data_ptr()
    prefix.add_(1)
    assert torch.equal(runtime.read(0, "k", 0, 2)[0], keys[0, :2])


def test_exact_first_layer_is_not_stored_as_codes():
    spec = _spec(not_quantize_first_layer=True)
    samples = _videos()
    bank, _ = calibrate(samples, spec)
    keys, values = samples[0]
    runtime = AquaRuntime(
        spec, bank, batch_size=1, frame_seq_length=4, max_frames=3,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    for frame in range(3):
        start, end = frame * 4, (frame + 1) * 4
        runtime.write(0, "k", start, end, keys[0, start:end].unsqueeze(0))
        runtime.write(0, "v", start, end, values[0, start:end].unsqueeze(0))
        runtime.write(1, "k", start, end, keys[1, start:end].unsqueeze(0))
        runtime.write(1, "v", start, end, values[1, start:end].unsqueeze(0))
    runtime.compress_finalized(12)
    assert runtime.codes[0]["k"] is None
    assert torch.equal(runtime.read(0, "k", 0, 12)[0], keys[0])


def _shift_memmap(path: Path) -> None:
    raw = np.memmap(path, dtype=np.int16, mode="r+")
    values = torch.from_numpy(np.array(raw, copy=True)).view(torch.bfloat16)
    values.add_(4)
    raw[:] = values.view(torch.int16).numpy()
    raw.flush()
    del raw


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_collection_rejects_evaluation_prompts_and_roundtrips_bf16(tmp_path):
    spec = _spec(activation_dtype="bfloat16")
    samples = [(k.to(torch.bfloat16), v.to(torch.bfloat16)) for k, v in _videos()]
    cal_lines = [f"calibration prompt {i}" for i in range(4)]
    ev_lines = ["heldout prompt 0", "heldout prompt 1"]
    cal = tmp_path / "cal.txt"
    ev = tmp_path / "ev.txt"
    cal.write_text("\n".join(cal_lines) + "\n", encoding="utf-8")
    ev.write_text("\n".join(ev_lines) + "\n", encoding="utf-8")
    root = tmp_path / "collection"
    for index, (keys, values) in enumerate(samples):
        write_sample(root / "rank-0" / f"prompt-{index:04d}", keys, values, {
            "prompt_index": index, "prompt_sha256": _sha(cal_lines[index]), "seed": 0,
        }, frame_seq_length=4)
    source = MemmapSampleSource(root)
    assert check_collection_against_prompts(source, cal, ev)["status"] == "disjoint"
    back_k, back_v = source.layer_kv(0, 0)
    assert torch.equal(back_k, samples[0][0][0])
    assert torch.equal(back_v, samples[0][1][0])
    leaked = tmp_path / "leaked.txt"
    leaked.write_text(cal_lines[0] + "\nheldout prompt 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leak"):
        check_collection_against_prompts(source, cal, leaked)


def test_plan_prints_commands_and_does_not_invent_quality(tmp_path):
    completed = subprocess.run(
        [sys.executable, "rebuttal/run.py", "--experiment", "aquakv", "--stage", "plan",
         "--output-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["quality"] is None
    assert plan["official_commit"] == "e744183f2f3a4a90381f20a6975456871fd1ae29"
    assert "--aquakv-collect" in plan["commands"]["collect_calibration"]
    assert "aquakv" in plan["commands"]["generate"]
    rows = {row["method"]: row for row in plan["comparison"]["rows"]}
    assert rows["bf16"]["average_bits_per_element"] == 16
    assert rows["aquakv"]["predictor_bytes"] > 0
    assert rows["aquakv"]["quality"] is None and rows["tempokv"]["quality"] is None
    assert "psnr_db" not in json.dumps(plan["comparison"]["rows"])
    gap = plan["comparison"]["bit_gap_aquakv_minus_tempokv"]
    assert isinstance(gap, float)


def test_evaluate_without_videos_stays_not_run(tmp_path):
    assert main(["--experiment", "aquakv", "--stage", "evaluate", "--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "evaluate.json").read_text(encoding="utf-8"))
    assert report["status"] == "not_run" and report["quality"] is None


def test_disk_calibrate_excludes_validation_and_cli_generate_is_commands_only(tmp_path):
    spec = _spec(activation_dtype="bfloat16", valid_nsamples=1)
    samples = [(k.to(torch.bfloat16), v.to(torch.bfloat16)) for k, v in _videos()]
    lines = [f"calibration prompt {i}" for i in range(4)]
    cal = tmp_path / "cal.txt"
    ev = tmp_path / "ev.txt"
    cal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ev.write_text("heldout a\nheldout b\n", encoding="utf-8")
    root = tmp_path / "collection"
    folders = []
    for index, (keys, values) in enumerate(samples):
        folders.append(write_sample(root / "rank-0" / f"prompt-{index:04d}", keys, values, {
            "prompt_index": index, "prompt_sha256": _sha(lines[index]), "seed": 0,
        }, frame_seq_length=4))
    config = {
        "experiment": "aquakv",
        "num_layers": 2, "num_heads": 2, "head_dim": 4,
        "frame_seq_length": 4, "latent_frames": 3, "frames": 9,
        "hadamard_groupsize": 8, "edenn_d": 2, "edenn_n": 16,
        "first_layer_edenn_d": 2, "first_layer_edenn_n": 16,
        "prefix_tokens": 2, "recent_buffer_tokens": 2, "percdamp": 0.001,
        "valid_nsamples": 1, "calibration_seed": 0, "activation_dtype": "bfloat16",
        "index_packing": "packed", "chunk_size": 8, "higgs_chunk_size": 4,
        "codec_config": "configs/legacy.json", "span_frames": 24,
        "calibration_prompts": str(cal), "evaluation_prompts": str(ev),
        "collection_dir": str(root), "expected_prompt_count": 4,
        "allow_short_collection": True,
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "fit"
    completed = subprocess.run(
        [sys.executable, "rebuttal/run.py", "--experiment", "aquakv", "--stage", "calibrate",
         "--config", str(path), "--output-dir", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    first = torch.load(out / "aquakv_predictors.pt", map_location="cpu")
    valid = first["valid_indices"][0]
    train = first["train_indices"][0]
    _shift_memmap(folders[valid] / "k.i16")
    out_valid = tmp_path / "fit_valid"
    again = subprocess.run(
        [sys.executable, "rebuttal/run.py", "--experiment", "aquakv", "--stage", "calibrate",
         "--config", str(path), "--output-dir", str(out_valid)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert again.returncode == 0, again.stderr
    second = torch.load(out_valid / "aquakv_predictors.pt", map_location="cpu")
    layer = next(iter(first["key_predictors"]))
    assert torch.equal(first["key_predictors"][layer]["weight"], second["key_predictors"][layer]["weight"])
    _shift_memmap(folders[train] / "k.i16")
    out_train = tmp_path / "fit_train"
    moved = subprocess.run(
        [sys.executable, "rebuttal/run.py", "--experiment", "aquakv", "--stage", "calibrate",
         "--config", str(path), "--output-dir", str(out_train)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert moved.returncode == 0, moved.stderr
    third = torch.load(out_train / "aquakv_predictors.pt", map_location="cpu")
    assert not torch.equal(first["key_predictors"][layer]["weight"], third["key_predictors"][layer]["weight"])
    generated = subprocess.run(
        [sys.executable, "rebuttal/run.py", "--experiment", "aquakv", "--stage", "generate",
         "--config", str(path), "--output-dir", str(tmp_path / "gen")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert generated.returncode == 0, generated.stderr
    report = json.loads((tmp_path / "gen" / "generate.json").read_text(encoding="utf-8"))
    assert report["status"] == "commands_only" and report["quality"] is None
