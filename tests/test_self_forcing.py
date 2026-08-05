from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
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


bridge = _load("bridge", DIRECTORY / "bridge.py")
generate = _load("tempokv_self_forcing_generate", DIRECTORY / "generate.py")
worker = _load("tempokv_self_forcing_worker", DIRECTORY / "worker.py")


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


def _generation_args(
    paths: dict[str, Path],
    output: Path,
    *,
    mode: str = "tempokv",
    frames: int = 27,
    dry_run: bool = True,
):
    arguments = [
        "--mode",
        mode,
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
    ]
    if dry_run:
        arguments.append("--dry-run")
    if mode == "tempokv":
        arguments[2:2] = [
            "--config",
            str(paths["config"]),
            "--k-params",
            str(paths["k_params"]),
            "--v-params",
            str(paths["v_params"]),
        ]
    return generate.build_parser().parse_args(arguments)


def test_tempokv_generation_dry_run_validates_parameters_and_never_fits(
    tmp_path, monkeypatch
) -> None:
    paths = _self_forcing_inputs(tmp_path)
    output = tmp_path / "run"
    calls = []

    def validate(*args):
        calls.append(args)
        return {"status": "validated"}

    monkeypatch.setattr(generate, "_validate_parameters", validate)
    report = generate.run(_generation_args(paths, output))
    assert report["status"] == "dry_run"
    assert report["mode"] == "tempokv"
    assert len(calls) == 1
    command = " ".join(report["generation_command"])
    assert "worker.py" in command
    assert "--mode tempokv" in command
    assert "fit_params.py" not in command
    assert not output.exists()


def test_bf16_generation_needs_no_codec_inputs_or_parameter_validation(
    tmp_path, monkeypatch
) -> None:
    paths = _self_forcing_inputs(tmp_path)
    output = tmp_path / "bf16-run"

    def unexpected_validation(*args):
        raise AssertionError("bf16 must not validate TempoKV parameters")

    monkeypatch.setattr(generate, "_validate_parameters", unexpected_validation)
    report = generate.run(_generation_args(paths, output, mode="bf16"))

    assert report["status"] == "dry_run"
    assert report["mode"] == "bf16"
    assert report.get("parameters") is None
    command = " ".join(report["generation_command"])
    assert "worker.py" in command
    assert "--mode bf16" in command
    assert "fit_params.py" not in command
    assert "k_temporal.npz" not in command
    assert "v_cross_kv.npz" not in command
    assert not output.exists()


@pytest.mark.parametrize("mode", ["tempokv", "bf16"])
def test_generation_only_exports_codec_environment_for_tempokv(
    tmp_path, monkeypatch, mode: str
) -> None:
    paths = _self_forcing_inputs(tmp_path)
    output = tmp_path / f"{mode}-run"
    launches = []
    for name in tuple(generate.os.environ):
        if name.startswith("TEMPOKV_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(
        generate,
        "_validate_parameters",
        lambda *args: {"status": "validated"},
    )

    def launch(command, *, cwd, env, check):
        launches.append((command, cwd, env, check))
        videos = output / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        (videos / "sample.mp4").write_bytes(b"video")
        metrics = Path(env["SELF_FORCING_METRICS_DIR"])
        metrics.mkdir(parents=True, exist_ok=True)
        (metrics / "rank-0.json").write_text(
            json.dumps(
                {
                    "rank": 0,
                    "wall_seconds": 1.0,
                    "peak_allocated_bytes": 10,
                    "peak_reserved_bytes": 20,
                    "gpu_name": "GPU 0",
                }
            ),
            encoding="utf-8",
        )
        return generate.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(generate.subprocess, "run", launch)
    report = generate.run(
        _generation_args(paths, output, mode=mode, dry_run=False)
    )

    assert report["status"] == "complete"
    assert len(launches) == 1
    environment = launches[0][2]
    codec_environment = {
        key: value for key, value in environment.items() if key.startswith("TEMPOKV_")
    }
    if mode == "tempokv":
        assert codec_environment["TEMPOKV_CONFIG_PATH"].endswith(
            "runtime_config.json"
        )
        assert codec_environment["TEMPOKV_K_PARAMS_PATH"] == str(
            paths["k_params"].resolve()
        )
        assert codec_environment["TEMPOKV_V_PARAMS_PATH"] == str(
            paths["v_params"].resolve()
        )
    else:
        assert codec_environment == {}


def test_self_forcing_rejects_too_short_generation(tmp_path) -> None:
    paths = _self_forcing_inputs(tmp_path)
    try:
        generate.run(_generation_args(paths, tmp_path / "run", frames=21))
    except ValueError as error:
        assert "at least 27 latent frames" in str(error)
    else:
        raise AssertionError("short Self-Forcing generation was accepted")


@pytest.mark.parametrize(
    ("mode", "expected_installs"), (("tempokv", 1), ("bf16", 0))
)
def test_worker_only_installs_bridge_for_tempokv(
    tmp_path, monkeypatch, mode: str, expected_installs: int
) -> None:
    paths = _self_forcing_inputs(tmp_path)
    installs = []
    launches = []
    monkeypatch.delenv("SELF_FORCING_METRICS_DIR", raising=False)
    monkeypatch.setattr(bridge, "install", lambda repo: installs.append(repo))
    monkeypatch.setattr(
        worker.runpy,
        "run_path",
        lambda path, run_name: launches.append((path, run_name, list(sys.argv))),
    )

    assert (
        worker.main(
            [
                "--mode",
                mode,
                "--model-repo",
                str(paths["repo"]),
                "--",
                "--seed",
                "7",
            ]
        )
        == 0
    )

    assert len(installs) == expected_installs
    assert len(launches) == 1
    assert launches[0][1] == "__main__"
    assert launches[0][2][1:] == ["--seed", "7"]


def test_bf16_worker_writes_metrics_when_inference_exits(
    tmp_path, monkeypatch
) -> None:
    paths = _self_forcing_inputs(tmp_path)
    metrics_dir = tmp_path / "worker-metrics"
    monkeypatch.setenv("SELF_FORCING_METRICS_DIR", str(metrics_dir))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    def stop(*args, **kwargs):
        raise SystemExit(3)

    monkeypatch.setattr(worker.runpy, "run_path", stop)
    with pytest.raises(SystemExit) as error:
        worker.main(
            ["--mode", "bf16", "--model-repo", str(paths["repo"])]
        )
    assert error.value.code == 3
    payload = json.loads(
        (metrics_dir / "rank-0.json").read_text(encoding="utf-8")
    )
    assert payload["rank"] == 0
    assert payload["wall_seconds"] >= 0
    assert payload["peak_allocated_bytes"] >= 0
    assert payload["peak_reserved_bytes"] >= 0


def test_worker_metrics_take_maximum_across_all_ranks(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    payloads = [
        {
            "rank": 0,
            "wall_seconds": 8.5,
            "peak_allocated_bytes": 120,
            "peak_reserved_bytes": 300,
            "gpu_name": "GPU 0",
        },
        {
            "rank": 1,
            "wall_seconds": 9.25,
            "peak_allocated_bytes": 240,
            "peak_reserved_bytes": 280,
            "gpu_name": "GPU 1",
        },
    ]
    for payload in payloads:
        (metrics_dir / f"rank-{payload['rank']}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    aggregated = generate._load_worker_metrics(metrics_dir, expected_ranks=2)

    assert aggregated["max_worker_seconds"] == 9.25
    assert aggregated["peak_allocated_bytes"] == 240
    assert aggregated["peak_reserved_bytes"] == 300
    assert aggregated["peak_allocated_gib"] == pytest.approx(240 / 1024**3)
    assert aggregated["peak_reserved_gib"] == pytest.approx(300 / 1024**3)
    assert aggregated["ranks"] == payloads


def test_worker_metrics_fail_when_an_expected_rank_is_missing(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "rank-0.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "wall_seconds": 1.0,
                "peak_allocated_bytes": 10,
                "peak_reserved_bytes": 20,
                "gpu_name": "GPU 0",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"rank-1\.json"):
        generate._load_worker_metrics(metrics_dir, expected_ranks=2)
