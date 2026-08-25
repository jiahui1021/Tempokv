from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIT_SCRIPT = PROJECT_ROOT / "scripts" / "Causal-Forcing" / "fit_params.sh"
TRACE_SCRIPT = PROJECT_ROOT / "scripts" / "Causal-Forcing" / "collect_trace.sh"
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "Causal-Forcing" / "run_tempokv.sh"
COMMON_FITTER = PROJECT_ROOT / "scripts" / "common" / "fit_params.py"


def _run(script: Path, **values: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "DRY_RUN": "1",
        "PYTHON_BIN": os.environ.get("PYTHON", "python3"),
        **values,
    }
    return subprocess.run(
        ["bash", str(script)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_legacy_fit_uses_layer_shared_cross_kv_end_to_end() -> None:
    completed = _run(FIT_SCRIPT, CONFIG="configs/legacy.json")

    assert completed.returncode == 0, completed.stderr
    assert "trace/chunkwise_legacy/clean_trace.npz" in completed.stdout
    assert "legacy-kh2-vh1-int4-layershared" in completed.stdout
    assert "--v-predictor-mode cross_kv" in completed.stdout


def test_native_fit_keeps_cross_kv_identity() -> None:
    completed = _run(FIT_SCRIPT, CONFIG="configs/native.json")

    assert completed.returncode == 0, completed.stderr
    assert "trace/chunkwise/clean_trace.npz" in completed.stdout
    assert "params/chunkwise/kh2-vh1-int4" in completed.stdout
    assert "--v-predictor-mode cross_kv" in completed.stdout


def test_clean_fitting_state_has_an_isolated_parameter_directory() -> None:
    completed = _run(
        FIT_SCRIPT,
        CONFIG="configs/native.json",
        FIT_HISTORY_SOURCE="clean",
    )

    assert completed.returncode == 0, completed.stderr
    assert "params/chunkwise/kh2-vh1-int4-fitclean" in completed.stdout
    assert "--fit-history-source clean" in completed.stdout


def test_generate_selects_clean_fitted_parameters_without_runtime_override() -> None:
    completed = _run(
        GENERATE_SCRIPT,
        CHECKPOINT="checkpoint.pt",
        MODEL_ROOT="model-root",
        PROMPTS="prompts.txt",
        CONFIG="configs/native.json",
        FIT_HISTORY_SOURCE="clean",
    )

    assert "params/chunkwise/kh2-vh1-int4-fitclean/k_temporal.npz" in completed.stdout
    assert "--fit-history-source" not in completed.stdout


def test_invalid_fitting_state_is_rejected_by_the_shell_contract() -> None:
    completed = _run(
        FIT_SCRIPT,
        CONFIG="configs/native.json",
        FIT_HISTORY_SOURCE="runtime",
    )

    assert completed.returncode != 0
    assert "FIT_HISTORY_SOURCE must be reconstructed or clean" in completed.stderr


@pytest.mark.parametrize(
    ("config", "parameter_name", "v_artifact", "v_predictor"),
    [
        (
            "configs/legacy.json",
            "legacy-kh2-vh1-int4-layershared",
            "v_cross_kv.npz",
            "cross_kv",
        ),
        (
            "configs/native.json",
            "kh2-vh1-int4",
            "v_cross_kv.npz",
            "cross_kv",
        ),
    ],
)
def test_generate_uses_the_same_v_predictor_identity_as_fitting(
    config: str, parameter_name: str, v_artifact: str, v_predictor: str
) -> None:
    completed = _run(
        GENERATE_SCRIPT,
        CHECKPOINT="checkpoint.pt",
        MODEL_ROOT="model-root",
        PROMPTS="prompts.txt",
        CONFIG=config,
    )

    # The Python dry-run still validates placeholder paths and may exit 2;
    # the shell contract is the fully resolved command printed first.
    assert f"params/chunkwise/{parameter_name}/k_temporal.npz" in completed.stdout
    assert f"params/chunkwise/{parameter_name}/{v_artifact}" in completed.stdout
    assert f"--v-predictor-mode {v_predictor}" in completed.stdout


def test_attention_controls_are_forwarded_for_legacy_default() -> None:
    completed = _run(
        FIT_SCRIPT,
        CONFIG="configs/legacy.json",
        ATTENTION_STEPS="4",
        ATTENTION_LR="0.002",
        ATTENTION_PATIENCE="3",
        ATTENTION_MAX_TOKENS="16",
        ATTENTION_LAYER_CHUNK="2",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--attention-steps 4" in completed.stdout
    assert "--attention-lr 0.002" in completed.stdout
    assert "--attention-patience 3" in completed.stdout
    assert "--attention-max-tokens 16" in completed.stdout
    assert "--attention-layer-chunk 2" in completed.stdout


def test_attention_controls_are_forwarded_for_legacy_cross_kv() -> None:
    completed = _run(
        FIT_SCRIPT,
        CONFIG="configs/legacy.json",
        V_PREDICTOR="cross_kv",
        ATTENTION_STEPS="2",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--v-predictor-mode cross_kv" in completed.stdout
    assert "--attention-steps 2" in completed.stdout


def test_attention_refinement_controls_are_forwarded_for_native() -> None:
    completed = _run(
        FIT_SCRIPT,
        CONFIG="configs/native.json",
        V_PREDICTOR="temporal",
        ATTENTION_STEPS="2",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--v-predictor-mode temporal" in completed.stdout
    assert "--attention-steps 2" in completed.stdout
    assert "--attention-optimizer adam" in completed.stdout


def test_attention_ablation_controls_are_forwarded() -> None:
    completed = _run(
        FIT_SCRIPT,
        CONFIG="configs/native.json",
        ATTENTION_STEPS="1",
        ATTENTION_OPTIMIZER="sign_gradient",
        ATTENTION_RECONSTRUCTION_WEIGHT="1.0",
        ATTENTION_OUTPUT_WEIGHT="0.0",
        ATTENTION_REFINE_TENSORS="value",
        ATTENTION_SIGN_STEP_SIZES="0.001 0.01",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--attention-optimizer sign_gradient" in completed.stdout
    assert "--attention-reconstruction-weight 1.0" in completed.stdout
    assert "--attention-output-weight 0.0" in completed.stdout
    assert "--attention-refine-tensors value" in completed.stdout
    assert "--attention-sign-step-sizes 0.001 0.01" in completed.stdout


@pytest.mark.parametrize(
    ("config", "expected_protocol", "expected_trace"),
    [
        ("configs/legacy.json", "--legacy-protocol", "trace/chunkwise_legacy"),
        ("configs/native.json", None, "trace/chunkwise/raw"),
    ],
)
def test_trace_protocol_follows_config(
    config: str, expected_protocol: str | None, expected_trace: str
) -> None:
    completed = _run(
        TRACE_SCRIPT,
        CHECKPOINT="checkpoint.pt",
        MODEL_ROOT="model-root",
        PROMPTS="prompts.txt",
        CONFIG=config,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_trace in completed.stdout
    assert ("--legacy-protocol" in completed.stdout) == bool(expected_protocol)


def test_trace_protocol_override_cannot_conflict_with_config() -> None:
    completed = _run(
        TRACE_SCRIPT,
        CHECKPOINT="checkpoint.pt",
        MODEL_ROOT="model-root",
        PROMPTS="prompts.txt",
        CONFIG="configs/native.json",
        LEGACY_PROTOCOL="1",
    )

    assert completed.returncode == 2
    assert "与 CONFIG 的 fit_mode=native 冲突" in completed.stderr


def test_common_fitter_accepts_layered_attention_refinement_controls(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("tempokv_fit_params", COMMON_FITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            str(tmp_path / "missing-trace.npz"),
            "--config",
            str(PROJECT_ROOT / "configs" / "native.json"),
            "--output-dir",
            str(tmp_path / "params"),
            "--v-predictor-mode",
            "temporal",
            "--attention-steps",
            "2",
        ]
    )

    assert args.attention_steps == 2
    assert args.attention_optimizer == "adam"
    assert args.attention_refine_tensors == "key_value"


def test_attention_token_limit_is_deterministic() -> None:
    spec = importlib.util.spec_from_file_location("tempokv_fit_params", COMMON_FITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensor = module.torch.arange(2 * 3 * 4 * 5 * 1 * 2).reshape(2, 3, 4, 5, 1, 2)

    limited = module._limit_attention_tracks(tensor, 5)

    assert limited.shape == (2, 5, 5, 1, 2)
    expected = tensor.reshape(2, 12, 5, 1, 2)[:, :5]
    assert module.torch.equal(limited, expected)
