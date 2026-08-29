from __future__ import annotations

import importlib.util
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tempokv import (
    CodecConfig,
    CrossKVParameters,
    PackedKV,
    TempoKVCodec,
    TemporalParameters,
)
from tempokv.fitting import fit_predictors
from tempokv.quantization import roundtrip


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = _load("history_ablation_analyze", ROOT / "scripts/tools/analyze_predictor.py")
SWEEP = _load(
    "history_ablation_sweep",
    ROOT / "scripts/tools/sweep_history_length.py",
)
RUNNER = _load(
    "history_ablation_runner",
    ROOT / "scripts/paper_experiments/history_ablation/run_final.py",
)
CACHE = _load(
    "history_ablation_cache",
    ROOT / "scripts/paper_experiments/history_ablation/evaluate_cache.py",
)


def _direct_config(**updates: object) -> CodecConfig:
    values: dict[str, object] = {
        "k_bits": 2,
        "v_bits": 2,
        "anchor_bits": 4,
        "group_size": 4,
        "stride": 2,
        "k_history_length": 0,
        "k_predictor_mode": "direct",
        "v_history_length": 0,
        "v_predictor_mode": "cross_kv",
        "scale_dtype": "float32",
        "reconstruction_dtype": "float32",
        "use_triton": False,
        "strict_triton": False,
    }
    values.update(updates)
    return CodecConfig(**values)


def test_direct_k_codec_quantizes_each_non_anchor_unit_without_prediction() -> None:
    config = _direct_config()
    temporal = TemporalParameters(
        torch.empty(0, 0, 4),
        torch.empty(0, 4),
        stride=2,
        codec_signature=config.format_signature(),
    )
    cross = CrossKVParameters(
        torch.eye(4).reshape(1, 4, 4),
        torch.zeros(1, 4),
        stride=2,
        codec_signature=config.format_signature(),
    )
    codec = TempoKVCodec(temporal, cross, config)
    key = torch.randn(1, 1, 8, 4, generator=torch.Generator().manual_seed(3))

    packed, rebuilt = codec.encode_key(key)
    expected = torch.empty_like(key)
    expected[:, :, :2] = roundtrip(
        key[:, :, :2], bits=4, group_size=4, scale_dtype=torch.float32, symmetric=True
    )
    for start in (2, 4, 6):
        expected[:, :, start : start + 2] = roundtrip(
            key[:, :, start : start + 2],
            bits=2,
            group_size=4,
            scale_dtype=torch.float32,
            symmetric=False,
        )

    torch.testing.assert_close(rebuilt, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        codec.decode(PackedKV(packed, codec.encode_value(torch.zeros_like(key), rebuilt)))[0],
        expected,
        rtol=0,
        atol=0,
    )
    assert packed.temporal is not None
    assert packed.temporal.history_length == 0


def test_direct_k_fit_produces_empty_k_parameters_and_fits_cross_v_from_rebuilt_k() -> None:
    generator = torch.Generator().manual_seed(4)
    keys = torch.randn(2, 3, 6, 1, 4, generator=generator)
    values = torch.randn_like(keys)
    config = _direct_config()

    temporal, cross = fit_predictors(
        keys,
        values,
        config=config,
        iterations=1,
        temporal_ridge=0.1,
        cross_ridge=0.1,
        v_history_mode="diagonal",
    )

    assert temporal.history_length == 0
    assert tuple(temporal.history_weights.shape) == (0, 0, 4)
    assert tuple(temporal.history_bias.shape) == (0, 4)
    assert tuple(cross.weight.shape) == (2, 1, 4, 4)
    assert temporal.parameter_set_id == cross.parameter_set_id
    _, _, rebuilt = ANALYZE.reconstruct_k(keys, temporal, config)
    assert torch.isfinite(rebuilt).all()


def test_cache_evaluator_reports_direct_k_and_cross_v_nmse(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(6)
    keys = torch.randn(2, 4, 5, 1, 4, generator=generator)
    values = torch.randn_like(keys)
    config = _direct_config()
    temporal, cross = fit_predictors(
        keys,
        values,
        config=config,
        iterations=1,
        temporal_ridge=0.1,
        cross_ridge=0.1,
        v_history_mode="diagonal",
    )
    trace = tmp_path / "trace.npz"
    np.savez(trace, keys=keys.numpy(), values=values.numpy())
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"codec": {key: value for key, value in asdict(config).items()}}),
        encoding="utf-8",
    )
    k_path = tmp_path / "k_temporal.npz"
    v_path = tmp_path / "v_cross_kv.npz"
    temporal.save(k_path)
    cross.save(v_path)

    report = CACHE.evaluate(
        trace,
        config_path,
        k_path,
        v_path,
        device_name="cpu",
    )
    assert report["k"]["prediction_nmse"] == pytest.approx(1.0)
    assert np.isfinite(report["v"]["reconstruction_nmse"])


def test_history_sweep_accepts_direct_k_and_temporal_v() -> None:
    generator = torch.Generator().manual_seed(5)
    keys = torch.randn(2, 8, 5, 1, 4, generator=generator)
    values = torch.randn_like(keys)
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=4,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        scale_dtype="float32",
        reconstruction_dtype="float32",
        use_triton=False,
    )

    result = SWEEP.measure(
        keys,
        values,
        config,
        iterations=1,
        temporal_ridge=0.1,
        cross_ridge=0.1,
        max_gain=1.0,
    )
    assert result["v_predictor_mode"] == "temporal"
    assert result["v"]["cross_term_energy_ratio"] is None


def test_final_runner_dry_run_materializes_all_four_gpu_jobs(tmp_path: Path) -> None:
    trace = tmp_path / "trace.npz"
    np.savez(trace, keys=np.zeros((1, 1, 3, 1, 4)), values=np.zeros((1, 1, 3, 1, 4)))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    wan_model_dir = tmp_path / "wan"
    wan_model_dir.mkdir()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(50)) + "\n")
    output = tmp_path / "study"

    args = RUNNER._parser().parse_args(
        [
            "--trace",
            str(trace),
            "--base-config",
            str(ROOT / "configs/legacy.json"),
            "--checkpoint",
            str(checkpoint),
            "--wan-model-dir",
            str(wan_model_dir),
            "--prompts",
            str(prompts),
            "--output",
            str(output),
            "--gpus",
            "0,1,2,3",
            "--dry-run",
        ]
    )
    assert RUNNER.run(args) is None

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"]["videos_per_variant"] == 50
    assert manifest["protocol"]["gpus"] == ["0", "1", "2", "3"]
    assert len(manifest["variants"]) == 14
    assert len(manifest["commands"]["fit"]) == 14
    assert len(manifest["commands"]["cache"]) == 14
    assert len(manifest["commands"]["generation"]) == 15
    assert len(manifest["commands"]["quality"]) == 14
    direct = next(item for item in manifest["variants"] if item["k_history_length"] == 0)
    assert direct["k_predictor_mode"] == "direct"
    no_cross = next(item for item in manifest["variants"] if item["axis"] == "without_cross")
    assert no_cross["v_predictor_mode"] == "temporal"
    assert no_cross["v_history_length"] == 1
    v_no_cross = [
        item for item in manifest["variants"] if item["axis"] == "v_history_no_cross"
    ]
    assert sorted(item["v_history_length"] for item in v_no_cross) == [2, 4, 5]
    assert all(item["v_predictor_mode"] == "temporal" for item in v_no_cross)
