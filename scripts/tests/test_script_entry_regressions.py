from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from tempokv import CodecConfig, TemporalParameters


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_self_forcing_generator_accepts_direct_v_predictor(tmp_path: Path) -> None:
    module = _load(
        "self_forcing_direct_parser",
        ROOT / "experiments/Self-Forcing/tempokv_adapter/generate.py",
    )

    args = module.build_parser().parse_args(
        [
            "--v-predictor-mode",
            "direct",
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--prompts",
            str(tmp_path / "prompts.txt"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.v_predictor_mode == "direct"


def test_cache_reconstruction_tool_loads_temporal_v_parameters(tmp_path: Path) -> None:
    module = _load(
        "tempokv_cache_generate_temporal_v",
        ROOT / "scripts/tools/generate.py",
    )
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        fit_mode="legacy",
        k_predictor_mode="shared",
        cross_kv_weight_mode="layer_shared",
        use_triton=False,
    )
    identity = {
        "stride": 2,
        "codec_signature": config.format_signature(),
        "parameter_set_id": "temporal-v-pair",
    }
    k_parameters = TemporalParameters(
        torch.zeros(1, 1, 64), torch.zeros(1, 64), **identity
    )
    v_parameters = TemporalParameters(
        torch.zeros(1, 1, 64), torch.zeros(1, 64), **identity
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    k_path = tmp_path / "k_temporal.npz"
    v_path = tmp_path / "v_temporal.npz"
    k_parameters.save(k_path)
    v_parameters.save(v_path)
    input_path = tmp_path / "cache.npz"
    generator = np.random.default_rng(4)
    np.savez(
        input_path,
        keys=generator.standard_normal((1, 1, 4, 64), dtype=np.float32),
        values=generator.standard_normal((1, 1, 4, 64), dtype=np.float32),
    )

    args = module._parser().parse_args(
        [
            str(input_path),
            str(tmp_path / "output"),
            "--config",
            str(config_path),
            "--k-params",
            str(k_path),
            "--v-params",
            str(v_path),
            "--device",
            "cpu",
        ]
    )
    report = module.run(args)

    assert report["status"] == "complete"
    assert (tmp_path / "output/reconstructed.npz").is_file()


def test_longcat_trace_refuses_existing_events_without_overwrite(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    trace_root = tmp_path / "trace"
    (trace_root / "raw/rank_00000/event_00000").mkdir(parents=True)
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/LongCat/collect_trace.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "CHECKPOINT_DIR": str(checkpoint),
            "TRACE_DIR": str(trace_root),
            "PREFLIGHT": "0",
            "TORCHRUN_BIN": "/bin/true",
            "PYTHON_BIN": "/bin/true",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "OVERWRITE=1" in completed.stderr


def test_longcat_default_output_contains_only_the_final_video() -> None:
    source = (ROOT / "experiments/LongCat/inference.py").read_text(encoding="utf-8")

    assert "and args.save_segment_snapshots" in source
    assert 'snapshot_dir / "segment_000.mp4"' in source
    assert 'output_dir / "segment_000.mp4"' not in source
    assert 'output_dir / "longcat.mp4"' in source

    shell = (ROOT / "scripts/LongCat/generate.sh").read_text(encoding="utf-8")
    assert 'SAVE_SEGMENT_SNAPSHOTS="${SAVE_SEGMENT_SNAPSHOTS:-0}"' in shell
    assert "args+=(--save_segment_snapshots)" in shell
