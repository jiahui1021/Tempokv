from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import longcat.preflight as longcat_preflight
from longcat.bridge import LongCatTempoKVBridge
from longcat.preflight import REQUIRED_CHECKPOINT_SUBDIRECTORIES, run as run_preflight
from tempokv.codec import TempoKVCodec
from tempokv.config import CodecConfig
from tempokv.integration import load_codec
from tempokv.parameters import CrossKVParameters, TemporalParameters


ROOT = Path(__file__).resolve().parents[2]
FIT_SCRIPT = ROOT / "scripts" / "common" / "fit_params.py"
CONFIG = ROOT / "configs" / "legacy.json"


def _load_module(name: str, path: Path):
    previous_protocol = sys.modules.get("protocol")
    if path.parent.name == "direction":
        protocol_spec = importlib.util.spec_from_file_location(
            "protocol", path.parent / "protocol.py"
        )
        assert protocol_spec is not None and protocol_spec.loader is not None
        protocol = importlib.util.module_from_spec(protocol_spec)
        sys.modules["protocol"] = protocol
        protocol_spec.loader.exec_module(protocol)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if path.parent.name == "direction":
            if previous_protocol is None:
                sys.modules.pop("protocol", None)
            else:
                sys.modules["protocol"] = previous_protocol
    return module


def _trace(path: Path) -> Path:
    generator = np.random.default_rng(20260903)
    shape = (1, 2, 5, 1, 64)
    np.savez(
        path,
        keys=generator.standard_normal(shape, dtype=np.float32),
        values=generator.standard_normal(shape, dtype=np.float32),
        layer_ids=np.asarray([0], dtype=np.int32),
        trace_stride=np.asarray(1560, dtype=np.int32),
    )
    return path


@pytest.mark.parametrize(("k_bits", "v_bits"), [(1, 7), (3, 5)])
def test_mixed_odd_bits_fit_save_load_and_codec_roundtrip(
    tmp_path: Path, k_bits: int, v_bits: int
) -> None:
    output = tmp_path / f"k{k_bits}-v{v_bits}"
    completed = subprocess.run(
        [
            sys.executable,
            str(FIT_SCRIPT),
            str(_trace(tmp_path / "trace.npz")),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output),
            "--k-bits",
            str(k_bits),
            "--v-bits",
            str(v_bits),
            "--iterations",
            "2",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    report = json.loads((output / "fit_report.json").read_text(encoding="utf-8"))
    assert report["codec_config"]["k_bits"] == k_bits
    assert report["codec_config"]["v_bits"] == v_bits
    temporal = TemporalParameters.load(output / "k_temporal.npz")
    cross = CrossKVParameters.load(output / "v_cross_kv.npz")
    assert temporal.codec_signature == cross.codec_signature
    assert temporal.parameter_set_id == cross.parameter_set_id

    codec = load_codec(
        CONFIG,
        output / "k_temporal.npz",
        output / "v_cross_kv.npz",
        overrides={
            "k_bits": k_bits,
            "v_bits": v_bits,
            "use_triton": False,
            "strict_triton": False,
        },
    )
    generator = torch.Generator().manual_seed(100 + k_bits * 10 + v_bits)
    key = torch.randn(1, 1, 3121, 64, generator=generator)
    value = torch.randn(1, 1, 3121, 64, generator=generator)
    packed = codec.encode(key, value)
    rebuilt_key, rebuilt_value = codec.decode(packed, output_dtype=torch.bfloat16)

    assert packed.key.residual_quant is not None
    assert packed.value.residual_quant is not None
    assert packed.key.residual_quant.shape[-1] == (64 * k_bits + 7) // 8
    assert packed.value.residual_quant.shape[-1] == (64 * v_bits + 7) // 8
    assert rebuilt_key.shape == key.shape and torch.isfinite(rebuilt_key).all()
    assert rebuilt_value.shape == value.shape and torch.isfinite(rebuilt_value).all()


@pytest.mark.parametrize("model", ["Self-Forcing", "Causal-Forcing", "LongCat"])
@pytest.mark.parametrize(("k_bits", "v_bits"), [(1, 7), (3, 5)])
def test_model_parameter_slug_distinguishes_mixed_odd_bits(
    model: str, k_bits: int, v_bits: int
) -> None:
    common = ROOT / "scripts" / model / "_common.sh"
    command = (
        f"source {common!s}; "
        f"codec_slug {CONFIG!s} '' '' {k_bits} {v_bits} '' '' reconstructed"
    )
    completed = subprocess.run(
        ["bash", "-c", command], cwd=ROOT, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert f"-k{k_bits}-v{v_bits}-" in completed.stdout.strip()


def test_mixed_parameter_signature_rejects_a_different_bit_pair(tmp_path: Path) -> None:
    output = tmp_path / "parameters"
    subprocess.run(
        [
            sys.executable,
            str(FIT_SCRIPT),
            str(_trace(tmp_path / "trace.npz")),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output),
            "--k-bits",
            "1",
            "--v-bits",
            "7",
            "--iterations",
            "2",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(ValueError, match="fitted for codec"):
        load_codec(
            CONFIG,
            output / "k_temporal.npz",
            output / "v_cross_kv.npz",
            overrides={"k_bits": 3, "v_bits": 5, "use_triton": False},
        )


def _longcat_parameter_pair(
    directory: Path, *, k_bits: int = 3, v_bits: int = 5
) -> tuple[Path, Path, CodecConfig]:
    config = CodecConfig(
        k_bits=k_bits,
        v_bits=v_bits,
        k_history_length=2,
        v_history_length=1,
        use_triton=False,
        strict_triton=False,
    )
    signature = config.format_signature()
    coordinate_system = "pre_rope_normalized"
    parameter_set_id = f"mixed-k{k_bits}-v{v_bits}"
    temporal = TemporalParameters(
        torch.zeros(2, 2, 128),
        torch.zeros(2, 128),
        stride=1560,
        codec_signature=signature,
        parameter_set_id=parameter_set_id,
        coordinate_system=coordinate_system,
    )
    cross = CrossKVParameters(
        torch.eye(128).unsqueeze(0),
        torch.zeros(1, 128),
        stride=1560,
        codec_signature=signature,
        parameter_set_id=parameter_set_id,
        coordinate_system=coordinate_system,
        history_weights=torch.zeros(1, 1, 128),
    )
    directory.mkdir(parents=True)
    k_path = directory / "k_temporal.npz"
    v_path = directory / "v_cross_kv.npz"
    temporal.save(k_path)
    cross.save(v_path)
    return k_path, v_path, config


def _longcat_checkpoint(directory: Path) -> Path:
    for name in REQUIRED_CHECKPOINT_SUBDIRECTORIES:
        (directory / name).mkdir(parents=True, exist_ok=True)
    (directory / "dit" / "config.json").write_text(
        json.dumps({"depth": 48, "num_heads": 32, "hidden_size": 4096}),
        encoding="utf-8",
    )
    return directory


def _preflight_args(
    tmp_path: Path, k_path: Path, v_path: Path, *, k_bits: int, v_bits: int
) -> argparse.Namespace:
    return argparse.Namespace(
        mode="tempokv",
        checkpoint_dir=_longcat_checkpoint(tmp_path / "checkpoint"),
        config=CONFIG,
        k_parameters=k_path,
        v_parameters=v_path,
        context_parallel_size=1,
        stride=None,
        k_bits=k_bits,
        v_bits=v_bits,
        v_predictor_mode=None,
    )


def test_longcat_preflight_checks_the_resolved_mixed_bit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    k_path, v_path, config = _longcat_parameter_pair(tmp_path / "parameters")
    monkeypatch.setattr(
        longcat_preflight,
        "_parameter_file_check",
        lambda path, **kwargs: {"ok": True, "path": str(path)},
    )
    report = run_preflight(
        _preflight_args(tmp_path, k_path, v_path, k_bits=3, v_bits=5)
    )
    identity = report["checks"]["parameter_identity"]
    assert identity["ok"] is True
    assert (identity["k_bits"], identity["v_bits"]) == (3, 5)

    codec = TempoKVCodec(
        TemporalParameters.load(k_path), CrossKVParameters.load(v_path), config
    )
    bridge_report = LongCatTempoKVBridge(codec=codec).report()
    assert (bridge_report["k_bits"], bridge_report["v_bits"]) == (3, 5)
    assert bridge_report["codec_config"]["k_bits"] == 3
    assert bridge_report["codec_config"]["v_bits"] == 5


def test_longcat_preflight_rejects_parameters_from_another_bit_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    k_path, v_path, _ = _longcat_parameter_pair(tmp_path / "parameters")
    monkeypatch.setattr(
        longcat_preflight,
        "_parameter_file_check",
        lambda path, **kwargs: {"ok": True, "path": str(path)},
    )
    report = run_preflight(
        _preflight_args(tmp_path, k_path, v_path, k_bits=1, v_bits=7)
    )
    assert report["checks"]["parameter_identity"]["ok"] is False
    assert "parameter_identity" in report["failed_checks"]


def test_mixed_efficiency_records_keep_both_widths_and_render_them() -> None:
    record_module = _load_module(
        "mixed_efficiency_record",
        ROOT
        / "experiments"
        / "Self-Forcing"
        / "tempokv_adapter"
        / "efficiency_record.py",
    )
    collect_module = _load_module(
        "mixed_collect_efficiency",
        ROOT / "scripts" / "paper_experiments" / "efficiency" / "collect_efficiency.py",
    )
    assert record_module._resolved_bit_widths(None, 3, 5) == (None, 3, 5)
    assert record_module._resolved_bit_widths(3, None, None) == (3, 3, 3)
    row = {"method": "TempoKV", "bits": None, "k_bits": 3, "v_bits": 5}
    assert collect_module._method_label(row) == "TempoKV K3/V5"


def _experiment_inputs(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "checkpoint"
    wan_model = tmp_path / "wan-model"
    checkpoint.mkdir()
    wan_model.mkdir()
    trace = tmp_path / "trace.npz"
    trace.write_bytes(b"trace")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"prompt {index}" for index in range(60)) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "wan_model": wan_model,
        "trace": trace,
        "prompts": prompts,
    }


@pytest.mark.parametrize(
    ("relative_path", "parser_name", "first_bits", "second_bits"),
    [
        ("predictor_ablation/run.py", "build_parser", (3, 5), (1, 7)),
        ("innovation_ablation/run.py", "_parser", (3, 5), (1, 7)),
        ("direction/run_quality.py", "build_parser", (3, 3), (1, 1)),
    ],
)
def test_resume_rejects_results_from_another_bit_pair(
    tmp_path: Path,
    relative_path: str,
    parser_name: str,
    first_bits: tuple[int, int],
    second_bits: tuple[int, int],
) -> None:
    module = _load_module(
        "resume_" + relative_path.replace("/", "_").replace(".", "_"),
        ROOT / "scripts" / "paper_experiments" / relative_path,
    )
    inputs = _experiment_inputs(tmp_path)
    output = tmp_path / "output"
    common = [
        "--checkpoint",
        str(inputs["checkpoint"]),
        "--wan-model-dir",
        str(inputs["wan_model"]),
        "--trace",
        str(inputs["trace"]),
        "--prompts",
        str(inputs["prompts"]),
        "--output-root",
        str(output),
        "--dry-run",
        "--no-quality",
    ]
    if relative_path != "predictor_ablation/run.py":
        common.append("--no-include-bf16")
    parser = getattr(module, parser_name)()
    first = parser.parse_args(
        [*common, "--k-bits", str(first_bits[0]), "--v-bits", str(first_bits[1])]
    )
    module.run(first)

    second = parser.parse_args(
        [
            *common,
            "--resume",
            "--k-bits",
            str(second_bits[0]),
            "--v-bits",
            str(second_bits[1]),
        ]
    )
    with pytest.raises(ValueError, match="different K/V bit widths"):
        module.run(second)


@pytest.mark.parametrize("model", ["Self-Forcing", "Causal-Forcing"])
def test_mixed_cross_kv_attention_refinement_defaults_to_the_supported_key_phase(
    tmp_path: Path, model: str
) -> None:
    trace = tmp_path / "trace.npz"
    trace.write_bytes(b"trace")
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / model / "fit_params.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "TRACE_FILE": str(trace),
            "OUTPUT_DIR": str(tmp_path / "parameters"),
            "CONFIG": str(CONFIG),
            "K_BITS": "3",
            "V_BITS": "5",
            "ATTENTION_STEPS": "1",
            "DRY_RUN": "1",
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--k-bits 3" in completed.stdout
    assert "--v-bits 5" in completed.stdout
    assert "--attention-refine-tensors key" in completed.stdout


def test_longcat_default_results_are_isolated_by_mixed_bit_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "LongCat" / "generate.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "CHECKPOINT_DIR": str(checkpoint),
            "K_BITS": "3",
            "V_BITS": "5",
            "DRY_RUN": "1",
            "PREFLIGHT": "0",
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    identity = "legacy-kh2-vh1-k3-v5-layershared"
    assert f"results/longcat/tempokv/{identity}" in completed.stdout
    assert f"data/longcat/params/{identity}" in completed.stdout


def test_history_sweep_uses_a_mixed_bit_specific_default_report(tmp_path: Path) -> None:
    trace = tmp_path / "trace.npz"
    trace.write_bytes(b"trace")
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "common" / "sweep_history.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL": "self-forcing",
            "TRACE_FILE": str(trace),
            "K_BITS": "3",
            "V_BITS": "5",
            "DRY_RUN": "1",
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        "history_sweep-legacy-kh2-vh1-k3-v5-layershared.json"
        in completed.stdout
    )
