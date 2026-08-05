from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch

from tempokv import CodecConfig, CrossKVParameters, TempoKVCodec, TemporalParameters


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "self-forcing"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load("tempokv_self_forcing_bridge", DIRECTORY / "bridge.py")
generate = _load("tempokv_self_forcing_generate", DIRECTORY / "generate.py")


def test_bridge_decodes_k_before_cross_kv_v() -> None:
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=4,
        group_size=4,
        stride=2,
        residual_mode="asymmetric",
        scale_dtype="float32",
        reconstruction_dtype="float32",
        use_triton=False,
    )
    temporal = TemporalParameters(
        torch.full((1, 1, 1, 4), 0.8), torch.zeros(1, 1, 4), stride=2
    )
    cross = CrossKVParameters(
        torch.eye(4).reshape(1, 4, 4), torch.zeros(1, 4), stride=2
    )
    codec = TempoKVCodec(temporal, cross, config)
    keys = torch.randn(1, 1, 6, 4)
    values = torch.randn_like(keys)
    packed = codec.encode(keys, values, layer=0)
    expected_keys, expected_values = codec.decode(packed)

    rebuilt_keys = bridge.decode_cache_state(packed.key, codec=codec)
    rebuilt_values = bridge.decode_cache_state(
        packed.value, source_tensor=rebuilt_keys, codec=codec
    )
    torch.testing.assert_close(rebuilt_keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(rebuilt_values, expected_values, rtol=0, atol=0)


def _self_forcing_inputs(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "model"
    required = (
        "experiments/Self-Forcing/inference.py",
        "experiments/Self-Forcing/pipeline/__init__.py",
        "experiments/Self-Forcing/pipeline/causal_inference.py",
        "experiments/Self-Forcing/utils/wan_wrapper.py",
        "experiments/Self-Forcing/wan/modules/causal_model.py",
        "experiments/Self-Forcing/configs/self_forcing_dmd.yaml",
    )
    for relative in required:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    config = tmp_path / "codec.json"
    config.write_text(
        json.dumps(
            {
                "k_bits": 2,
                "v_bits": 2,
                "anchor_bits": 4,
                "group_size": 64,
                "stride": 1560,
                "residual_mode": "asymmetric",
                "scale_dtype": "bfloat16",
                "reconstruction_dtype": "bfloat16",
                "use_triton": True,
                "strict_triton": True,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a test video\n", encoding="utf-8")
    k_params = tmp_path / "k_temporal.npz"
    v_params = tmp_path / "v_cross_kv.npz"
    k_params.write_bytes(b"fixture")
    v_params.write_bytes(b"fixture")
    return {
        "repo": repo,
        "config": config,
        "checkpoint": checkpoint,
        "prompts": prompts,
        "k_params": k_params,
        "v_params": v_params,
    }


def _generation_args(paths: dict[str, Path], output: Path, frames: int = 27):
    return generate.build_parser().parse_args(
        [
            "--config",
            str(paths["config"]),
            "--k-params",
            str(paths["k_params"]),
            "--v-params",
            str(paths["v_params"]),
            "--model-repo",
            str(paths["repo"]),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--prompts",
            str(paths["prompts"]),
            "--output-dir",
            str(output),
            "--frames",
            str(frames),
            "--dry-run",
        ]
    )


def test_self_forcing_generation_dry_run_never_fits(
    tmp_path, monkeypatch
) -> None:
    paths = _self_forcing_inputs(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(
        generate,
        "_validate_parameters",
        lambda *args: {"status": "validated"},
    )
    report = generate.run(_generation_args(paths, output))
    assert report["status"] == "dry_run"
    assert "worker.py" in " ".join(report["generation_command"])
    assert "fit_params.py" not in " ".join(report["generation_command"])
    assert not output.exists()


def test_self_forcing_rejects_too_short_generation(tmp_path) -> None:
    paths = _self_forcing_inputs(tmp_path)
    try:
        generate.run(_generation_args(paths, tmp_path / "run", frames=21))
    except ValueError as error:
        assert "at least 27 latent frames" in str(error)
    else:
        raise AssertionError("short Self-Forcing generation was accepted")
