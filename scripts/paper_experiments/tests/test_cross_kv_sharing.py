from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tempokv.config import config_from_mapping
from tempokv.parameters import CrossKVParameters, TemporalParameters


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> SimpleNamespace:
    base_config = tmp_path / "legacy.json"
    base_config.write_text(
        json.dumps(
            {
                "k_bits": 4,
                "v_bits": 4,
                "k_history_length": 2,
                "v_history_length": 1,
                "k_predictor_mode": "shared",
                "cross_matmul_precision": "ieee",
                "fit_mode": "legacy",
            }
        ),
        encoding="utf-8",
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("one\ntwo\n", encoding="utf-8")
    return SimpleNamespace(
        base_config=base_config,
        calibration_trace=tmp_path / "calibration.npz",
        held_out_trace=tmp_path / "held_out.npz",
        checkpoint=tmp_path / "model.pt",
        prompts=prompts,
        model_config=None,
        output=tmp_path / "study",
        frames=717,
        samples=1,
        seed=17,
        start_index=0,
        end_index=1,
        num_gpus=1,
        local_attn_size=12,
        span_frames=24,
        fit_device="cpu",
        reconstruction_device="cpu",
        quality_device="cpu",
        fid_batch_size=8,
        max_quality_frames=16,
        attention_steps=None,
        vbench_long=False,
        vbench_root=None,
        vbench_cache_dir=None,
        vbench_devices=None,
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_variants_only_change_cross_kv_weight_mode(tmp_path: Path) -> None:
    study = _module(
        "cross_kv_sharing_study_configs",
        "scripts/paper_experiments/cross_kv_sharing/run_study.py",
    )
    variants = study.make_variant_configs(_args(tmp_path).base_config, tmp_path / "out")
    configs = {
        mode: json.loads(path.read_text(encoding="utf-8"))["codec"]
        for mode, path in variants.items()
    }
    assert set(configs) == {
        "layer_head",
        "layer_shared",
        "head_shared",
        "shared",
    }
    baseline = {
        key: value
        for key, value in configs["layer_head"].items()
        if key != "cross_kv_weight_mode"
    }
    for mode, config in configs.items():
        assert config["cross_kv_weight_mode"] == mode
        treatment_free = {
            key: value
            for key, value in config.items()
            if key != "cross_kv_weight_mode"
        }
        assert treatment_free == baseline
    assert baseline["k_predictor_mode"] == "shared"
    assert baseline["fit_mode"] == "legacy"


def test_manifest_freezes_fair_generation_and_contains_full_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _module(
        "cross_kv_sharing_study_manifest",
        "scripts/paper_experiments/cross_kv_sharing/run_study.py",
    )
    wan = tmp_path / "wan-model"
    monkeypatch.setenv("WAN_MODEL_DIR", str(wan))
    manifest = study.build_manifest(_args(tmp_path))
    kinds = [step["kind"] for step in manifest["steps"]]
    assert kinds.count("fit") == 4
    assert kinds.count("reconstruction") == 4
    assert kinds.count("generation") == 5
    assert kinds.count("quality") == 4

    commands = [" ".join(step["command"]) for step in manifest["steps"]]
    assert sum("scripts/common/fit_params.py" in command for command in commands) == 4
    assert sum("evaluate_reconstruction.py" in command for command in commands) == 4
    assert sum("tempokv_adapter/generate.py" in command for command in commands) == 5
    assert sum("evaluate_quality.py" in command for command in commands) == 4
    assert all("--no-fid" not in command for command in commands)

    generation = [step for step in manifest["steps"] if step["kind"] == "generation"]
    assert all(step["fairness"] == manifest["fair_generation_protocol"] for step in generation)
    assert manifest["fair_generation_protocol"]["wan_model_dir"] == str(wan.resolve())
    fair_options = (
        "--checkpoint",
        "--prompts",
        "--frames",
        "--samples",
        "--seed",
        "--start-index",
        "--end-index",
        "--num-gpus",
        "--local-attn-size",
        "--span-frames",
    )
    for option in fair_options:
        assert len({_option(step["command"], option) for step in generation}) == 1
    quality_steps = [step for step in manifest["steps"] if step["kind"] == "quality"]
    assert {step["reference"] for step in quality_steps} == {"bf16"}
    assert manifest["reconstruction_trace_role"] == "held_out"
    assert manifest["fair_generation_protocol"]["samples"] == 1
    assert manifest["fair_generation_protocol"]["start_index"] == 0
    assert manifest["fair_generation_protocol"]["end_index"] == 1


def test_dry_run_writes_manifest_and_planned_comparison_without_model_files(
    tmp_path: Path,
) -> None:
    study = _module(
        "cross_kv_sharing_study_dry_run",
        "scripts/paper_experiments/cross_kv_sharing/run_study.py",
    )
    args = _args(tmp_path)
    result = study.main(
        [
            "--base-config",
            str(args.base_config),
            "--calibration-trace",
            str(args.calibration_trace),
            "--held-out-trace",
            str(args.held_out_trace),
            "--checkpoint",
            str(args.checkpoint),
            "--prompts",
            str(args.prompts),
            "--output",
            str(args.output),
            "--end-index",
            "1",
        ]
    )
    assert result == 0
    assert (args.output / "manifest.json").is_file()
    comparison = json.loads((args.output / "comparison.json").read_text(encoding="utf-8"))
    assert [row["method"] for row in comparison["rows"]] == [
        "bf16",
        "layer_head",
        "layer_shared",
        "head_shared",
        "shared",
    ]
    assert (args.output / "comparison.csv").is_file()


def test_synthetic_trace_reconstruction_has_pooled_layer_and_head_nmse(tmp_path: Path) -> None:
    evaluator = _module(
        "cross_kv_sharing_reconstruction",
        "scripts/paper_experiments/cross_kv_sharing/evaluate_reconstruction.py",
    )
    codec_payload = {
        "k_bits": 4,
        "v_bits": 4,
        "k_history_length": 2,
        "v_history_length": 1,
        "k_predictor_mode": "shared",
        "cross_matmul_precision": "ieee",
        "fit_mode": "legacy",
        "cross_kv_weight_mode": "shared",
    }
    config = config_from_mapping(codec_payload)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(codec_payload), encoding="utf-8")

    generator = torch.Generator().manual_seed(5)
    keys = torch.randn(2, 5, 2, 64, generator=generator)
    values = 0.7 * keys + 0.2 * torch.randn(2, 5, 2, 64, generator=generator)
    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        keys=keys.numpy(),
        values=values.numpy(),
        layer_ids=np.asarray([7, 11], dtype=np.int32),
    )

    signature = config.format_signature()
    k_parameters = TemporalParameters(
        torch.zeros(2, 2, 64),
        torch.zeros(2, 64),
        stride=config.stride,
        codec_signature=signature,
        parameter_set_id="synthetic-pair",
    )
    weight = torch.eye(64).view(1, 1, 64, 64).expand(2, 2, 64, 64).clone() * 0.7
    v_parameters = CrossKVParameters(
        weight,
        torch.zeros(2, 2, 64),
        stride=config.stride,
        codec_signature=signature,
        parameter_set_id="synthetic-pair",
        history_weights=torch.zeros(2, 1, 2, 64),
    )
    k_path = tmp_path / "k_temporal.npz"
    v_path = tmp_path / "v_cross_kv.npz"
    k_parameters.save(k_path)
    v_parameters.save(v_path)

    result = evaluator.evaluate(trace, config_path, k_path, v_path)
    for tensor_name in ("k", "v"):
        tensor = result[tensor_name]
        assert tensor["pooled"]["nmse"] is not None
        assert tensor["pooled"]["nmse"] >= 0.0
        assert len(tensor["per_layer"]) == 2
        assert len(tensor["per_head"]) == 2
        assert len(tensor["per_layer_head"]) == 4
        assert {row["layer_id"] for row in tensor["per_layer"]} == {7, 11}
        assert sum(row["error_energy"] for row in tensor["per_layer"]) == pytest.approx(
            tensor["pooled"]["error_energy"]
        )
        assert sum(row["reference_energy"] for row in tensor["per_head"]) == pytest.approx(
            tensor["pooled"]["reference_energy"]
        )

    mismatched_v = CrossKVParameters(
        weight,
        torch.zeros(2, 2, 64),
        stride=config.stride,
        codec_signature=signature,
        parameter_set_id="different-fit",
        history_weights=torch.zeros(2, 1, 2, 64),
    )
    mismatched_v.save(v_path)
    with pytest.raises(ValueError, match="same parameter_set_id"):
        evaluator.evaluate(trace, config_path, k_path, v_path)


def test_rejects_refinement_that_would_unshare_value_weights(tmp_path: Path) -> None:
    study = _module(
        "cross_kv_sharing_study_refinement",
        "scripts/paper_experiments/cross_kv_sharing/run_study.py",
    )
    args = _args(tmp_path)
    args.attention_steps = 10
    with pytest.raises(ValueError, match="value refinement"):
        study.build_manifest(args)


def test_vbench_manifest_uses_evaluator_prompt_text_field(tmp_path: Path) -> None:
    study = _module(
        "cross_kv_sharing_study_vbench",
        "scripts/paper_experiments/cross_kv_sharing/run_study.py",
    )
    args = _args(tmp_path)
    args.vbench_long = True
    args.vbench_root = tmp_path / "vbench"
    args.vbench_cache_dir = tmp_path / "cache"
    args.vbench_devices = "GPU-0"
    manifest = study.build_manifest(args)
    vbench = json.loads(
        Path(manifest["vbench_long_manifest"]).read_text(encoding="utf-8")
    )
    assert vbench["prompt_text"].endswith("prompts.txt")
    assert "prompt_file" not in vbench
    vbench_step = next(
        step for step in manifest["steps"] if step["kind"] == "vbench_long"
    )
    assert _option(vbench_step["command"], "--smoke-count") == "1"


def test_comparison_collects_materialized_and_logical_parameter_sizes(tmp_path: Path) -> None:
    study = _module(
        "cross_kv_sharing_study_comparison",
        "scripts/paper_experiments/cross_kv_sharing/run_study.py",
    )
    sizes = (
        ("layer_head", 8.0, 8.0),
        ("layer_shared", 8.0, 0.5),
        ("head_shared", 8.0, 0.25),
        ("shared", 8.0, 0.01),
    )
    for mode, materialized, logical in sizes:
        fit = tmp_path / "params" / mode / "fit_report.json"
        fit.parent.mkdir(parents=True, exist_ok=True)
        fit.write_text(
            json.dumps({"parameters": {"mib": materialized, "logical_mib": logical}}),
            encoding="utf-8",
        )
    study.write_comparison(tmp_path)
    rows = {
        row["method"]: row
        for row in json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))["rows"]
    }
    for mode, materialized, logical in sizes:
        assert rows[mode]["materialized_parameter_mib"] == materialized
        assert rows[mode]["logical_parameter_mib"] == logical


def test_formal_protocol_rejects_more_than_one_video(tmp_path: Path) -> None:
    study = _module(
        "cross_kv_sharing_study_one_video",
        "scripts/paper_experiments/cross_kv_sharing/run_study.py",
    )
    args = _args(tmp_path)
    args.samples = 2
    with pytest.raises(ValueError, match="samples=1"):
        study._validate_args(args)
    args.samples = 1
    args.end_index = 2
    with pytest.raises(ValueError, match="exactly one prompt"):
        study._validate_args(args)


def test_run_one_video_entrypoint_is_valid_and_pins_the_protocol(tmp_path: Path) -> None:
    script = (
        ROOT
        / "scripts"
        / "paper_experiments"
        / "cross_kv_sharing"
        / "run_one_video.sh"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    capture = tmp_path / "arguments.txt"
    stub = tmp_path / "python-stub.sh"
    stub.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": str(stub),
            "CAPTURE": str(capture),
            "CALIBRATION_TRACE": str(tmp_path / "calibration.npz"),
            "HELD_OUT_TRACE": str(tmp_path / "held-out.npz"),
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "WAN_MODEL_DIR": str(tmp_path / "wan"),
            "PROMPTS": str(tmp_path / "prompts.txt"),
            "OUTPUT_DIR": str(tmp_path / "results"),
        }
    )
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[0].endswith("cross_kv_sharing/run_study.py")
    assert _option(arguments, "--samples") == "1"
    assert _option(arguments, "--start-index") == "0"
    assert _option(arguments, "--end-index") == "1"
    assert "--execute" in arguments
