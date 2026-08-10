import importlib.util
import importlib
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).with_name("generate.py")
SPEC = importlib.util.spec_from_file_location("causal_tempokv_generate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_generation_command = MODULE._generation_command
build_parser = MODULE.build_parser


def test_bundled_runtime_is_importable_without_external_pythonpath():
    assert str(MODULE.RUNTIME_ROOT) in sys.path
    bridge = importlib.import_module("causal_forcing.tempokv_adapter.bridge")

    assert bridge.DEFAULT_SPAN_FRAMES > 0


def _parse(*extra: str):
    return build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--model-root",
            "model-root",
            "--prompts",
            "prompts.txt",
            "--output-dir",
            "output",
            *extra,
        ]
    )


def _paths():
    return {
        "model_config": Path("config.yaml"),
        "checkpoint": Path("checkpoint.pt"),
        "model_root": Path("model-root"),
        "prompts": Path("prompts.txt"),
        "videos": Path("output/videos"),
    }


def test_official_checkpoint_uses_generator_by_default():
    command = _generation_command(_parse(), _paths(), latent_frames=21)

    assert "--use_ema" not in command


def test_ema_checkpoint_can_be_selected_explicitly():
    command = _generation_command(_parse("--use-ema"), _paths(), latent_frames=21)

    assert "--use_ema" in command
