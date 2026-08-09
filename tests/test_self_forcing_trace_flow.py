from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "Self-Forcing"
RUNTIME_ROOT = PROJECT_ROOT / "experiments" / "Self-Forcing"


def _load_script(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dry_run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "WAN_MODEL_DIR": str(tmp_path / "wan"),
            "PROMPTS": "assets/fit_prompts.txt",
            "DRY_RUN": "1",
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT_ROOT / "collect_trace.sh")],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_collect_trace_dry_run_uses_bundled_runtime_and_defaults(tmp_path: Path):
    completed = _dry_run(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "--mode trace" in completed.stdout
    assert "experiments/Self-Forcing" in completed.stdout
    assert "data/self-forcing/trace/raw" in completed.stdout
    assert "data/self-forcing/trace/clean_trace.npz" in completed.stdout
    assert "prepare_trace.py" in completed.stdout
    assert "TRACE_RUNNER" not in completed.stdout
    assert "MODEL_REPO" not in completed.stdout
    assert "--start_index" not in completed.stdout
    assert "--local_attn_size" not in completed.stdout
    assert "--quant_type" not in completed.stdout


def test_collect_trace_keeps_explicit_path_overrides(tmp_path: Path):
    completed = _dry_run(
        tmp_path,
        TRACE_DIR=str(tmp_path / "custom-trace-dir"),
        TRACE_FILE=str(tmp_path / "custom-trace.npz"),
    )

    assert completed.returncode == 0, completed.stderr
    assert str(tmp_path / "custom-trace-dir" / "raw") in completed.stdout
    assert str(tmp_path / "custom-trace.npz") in completed.stdout


def test_bf16_entry_uses_only_bundled_runtime(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    prompts = tmp_path / "prompts.txt"
    wan = tmp_path / "wan"
    checkpoint.write_bytes(b"checkpoint")
    prompts.write_text("a test video\n", encoding="utf-8")
    wan.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "CHECKPOINT": str(checkpoint),
            "WAN_MODEL_DIR": str(wan),
            "PROMPTS": str(prompts),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "FRAMES": "81",
            "DRY_RUN": "1",
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT_ROOT / "run_bf16.sh")],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--mode bf16" in completed.stdout
    assert "experiments/Self-Forcing" not in completed.stderr
    assert "MODEL_REPO" not in completed.stdout
    assert "model-repo" not in completed.stdout


def test_model_script_directories_share_the_same_public_layout():
    expected = {
        "README.md",
        "_common.sh",
        "collect_trace.sh",
        "fit_params.sh",
        "generate.sh",
        "run_bf16.sh",
        "run_tempokv.sh",
    }
    directories = [
        PROJECT_ROOT / "scripts" / "Self-Forcing",
        PROJECT_ROOT / "scripts" / "LongCat",
    ]
    for directory in directories:
        assert {path.name for path in directory.iterdir() if path.is_file()} == expected
        completed = subprocess.run(
            [
                "bash",
                "-n",
                *(str(directory / name) for name in sorted(expected) if name.endswith(".sh")),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


class _FakeSelfAttention:
    def __init__(self) -> None:
        self.norm_k = torch.nn.Identity()
        self.v = torch.nn.Identity()


class _FakeBlock:
    def __init__(self) -> None:
        self.self_attn = _FakeSelfAttention()


class _FakeGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(blocks=[_FakeBlock() for _ in range(30)])

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
    ) -> torch.Tensor:
        del conditional_dict, kv_cache, crossattn_cache, current_start, timestep
        output = noisy_image_or_video
        for block in self.model.blocks:
            block.self_attn.norm_k(output)
            block.self_attn.v(output)
        return output


def test_bundled_trace_bridge_captures_clean_pre_rope_kv(monkeypatch, tmp_path: Path):
    pipeline_package = ModuleType("pipeline")
    pipeline_module = ModuleType("pipeline.causal_inference")

    class FakePipeline:
        def __init__(self) -> None:
            self.frame_seq_length = 1560
            self.args = SimpleNamespace(context_noise=0)
            self.generator = _FakeGenerator()

        def inference(self, projection: torch.Tensor) -> str:
            self.generator(
                projection,
                {},
                torch.tensor([[750]]),
                kv_cache=[{}],
                current_start=0,
            )
            self.generator(
                projection,
                {},
                torch.tensor([[0]]),
                kv_cache=[{}],
                current_start=0,
            )
            self.generator(
                projection + 1,
                {},
                torch.tensor([[0]]),
                kv_cache=[{}],
                current_start=1560,
            )
            return "base-result"

    pipeline_package.CausalInferencePipeline = FakePipeline
    pipeline_module.CausalInferencePipeline = FakePipeline
    pipeline_module.WanVAEWrapper = object
    monkeypatch.setitem(sys.modules, "pipeline", pipeline_package)
    monkeypatch.setitem(sys.modules, "pipeline.causal_inference", pipeline_module)

    trace_root = tmp_path / "raw_trace"
    monkeypatch.setenv("TEMPOKV_TRACE_DIR", str(trace_root))
    monkeypatch.setenv("TEMPOKV_TRACE_TOKENS_PER_FRAME", "2")
    monkeypatch.setenv("RANK", "0")

    trace_bridge = _load_script(
        "tempokv_test_trace_bridge",
        "experiments/Self-Forcing/tempokv_adapter/trace_bridge.py",
    )
    traced_pipeline = trace_bridge.install(tmp_path)()
    projection = torch.arange(1560, dtype=torch.float32).reshape(1, 1560, 1)
    projection = projection.expand(1, 1560, 12 * 128)

    assert traced_pipeline.inference(projection) == "base-result"
    assert traced_pipeline.inference(projection + 2) == "base-result"

    layer_files = sorted(trace_root.rglob("layer_*.npz"))
    assert len(layer_files) == 60
    with np.load(layer_files[0], allow_pickle=False) as layer:
        assert layer["k_pre_rope"].shape == (1, 2, 2, 12, 128)
        assert layer["v"].shape == (1, 2, 2, 12, 128)
        assert layer["token_indices"].tolist() == [0, 780]
        assert int(layer["tokens_start"]) == 0
        assert int(layer["tokens_end"]) == 3120
        assert int(layer["layer_index"]) == 0

    prepare_trace = _load_script(
        "tempokv_test_prepare_trace",
        "experiments/Self-Forcing/tempokv_adapter/prepare_trace.py",
    )
    merged = tmp_path / "self_forcing_trace.npz"
    report = prepare_trace.convert_trace_root(trace_root, merged)

    assert report["status"] == "complete"
    assert report["events"] == 2
    assert report["shape"] == [30, 4, 2, 12, 128]
    with np.load(merged, allow_pickle=False) as trace:
        assert trace["keys"].shape == (30, 4, 2, 12, 128)
        assert trace["values"].shape == (30, 4, 2, 12, 128)
        assert trace["layer_ids"].tolist() == list(range(30))
        assert int(trace["stride"]) == 1560
        assert str(trace["coordinate_system"]) == "pre_rope_normalized"


def test_bundled_runtime_is_pinned_and_complete():
    runtime = RUNTIME_ROOT
    assert (runtime / "inference.py").is_file()
    assert (runtime / "pipeline" / "causal_inference.py").is_file()
    assert (runtime / "wan" / "modules" / "causal_model.py").is_file()
    assert "33593df3e81fa3ec10239271dd2c100facac6de1" in (
        runtime / "UPSTREAM.md"
    ).read_text(encoding="utf-8")


def test_bundled_cache_supports_bf16_and_tempokv_spans(monkeypatch):
    cache_module = _load_script(
        "tempokv_test_self_forcing_cache",
        "experiments/Self-Forcing/kv_cache.py",
    )
    cache = cache_module.ChunkedKVCache(
        batch_size=1,
        frame_seq_length=2,
        num_heads=1,
        head_dim=2,
        max_num_chunks=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layout="BSHD",
        cache_role="K",
    )
    first = torch.arange(8, dtype=torch.float32).reshape(1, 4, 1, 2)
    second = torch.arange(8, 16, dtype=torch.float32).reshape(1, 4, 1, 2)
    cache.write(0, 4, first)
    cache.write(4, 8, second)
    torch.testing.assert_close(cache.read(0, 8), torch.cat((first, second), dim=1))

    packed = {"rebuilt": first.permute(0, 2, 1, 3).contiguous()}

    def decode(state, *, source_tensor=None, output_tensor=None):
        assert source_tensor is None
        assert output_tensor is None
        return state["rebuilt"]

    cache_module.set_quantized_decoder(decode)
    cache.store_quantized(0, 4, packed)
    assert cache.chunks[0] is None and cache.chunks[1] is None
    torch.testing.assert_close(cache.read(0, 8), torch.cat((first, second), dim=1))
