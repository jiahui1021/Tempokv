#!/usr/bin/env python3
"""Minimal complete run of the K->V direction study.

Runs the reconstruction half for all three arms and renders the figure, on CPU,
in seconds, against a small synthetic trace pair.  It also drives the mirrored
codec through the real ``ChunkedKVCache`` so the runtime half -- the part that
needs a GPU and a checkpoint to exercise for real -- is at least proven to
encode and decode in both directions here.

The generation-quality half cannot run without a model, so it is exercised in
``--dry-run`` form: the full command matrix is recorded and checked, which is
what catches a wiring mistake before it costs GPU hours.

    python scripts/paper_experiments/direction/run_minimal.py \\
        --output-root /tmp/direction-minimal

The numbers describe the synthetic generator and say nothing about TempoKV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(HERE))

from protocol import ARMS, arm as get_arm  # noqa: E402
from tempokv.codec import TempoKVCodec, decode_tensor  # noqa: E402
from tempokv.fitting import fit_predictors  # noqa: E402
from tempokv.runtime import cache as cache_module  # noqa: E402


LAYERS, RECORDS, FRAMES, HEADS, DIM = 4, 12, 6, 2, 128
STRIDE = 4
_STATE_KEY = "tempokv_state"


def synthesize(path: Path, *, seed: int) -> Path:
    """Write a trace whose V genuinely depends on K, so direction can matter."""
    rng = np.random.default_rng(seed)
    shape = (LAYERS, RECORDS, HEADS, DIM)
    keys = np.zeros((LAYERS, RECORDS, FRAMES, HEADS, DIM), dtype=np.float32)
    values = np.zeros_like(keys)
    keys[:, :, 0] = rng.normal(size=shape)
    values[:, :, 0] = rng.normal(size=shape)
    for frame in range(1, FRAMES):
        keys[:, :, frame] = (
            0.9 * keys[:, :, frame - 1] + 0.1 * rng.normal(size=shape)
        )
        values[:, :, frame] = (
            0.6 * keys[:, :, frame]
            + 0.3 * values[:, :, frame - 1]
            + 0.1 * rng.normal(size=shape)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keys=keys,
        values=values,
        layer_ids=np.arange(LAYERS, dtype=np.int32),
        trace_stride=np.asarray(STRIDE, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
        trace_role=np.asarray("calibration"),
    )
    return path


def _wrap(packed):
    return {
        _STATE_KEY: packed,
        "predictor_kind": packed.predictor_kind,
        "method": "tempokv",
    }


def check_runtime_directions(calibration: Path) -> dict[str, dict[str, float]]:
    """Encode and decode through the real cache in both directions.

    This is the runtime contract the model bridges rely on: the primary slot is
    written first and the secondary slot decodes from the primary's
    reconstruction.  Exercising it here means a direction bug does not have to
    wait for a GPU run to show up.
    """
    with np.load(calibration, allow_pickle=False) as archive:
        keys = torch.from_numpy(np.array(archive["keys"], copy=True))
        values = torch.from_numpy(np.array(archive["values"], copy=True))

    results: dict[str, dict[str, float]] = {}
    for name in ("k_to_v", "v_to_k"):
        arm_spec = get_arm(name)
        config = arm_spec.codec(STRIDE)
        temporal, secondary = fit_predictors(
            keys,
            values,
            config=config,
            iterations=2,
            temporal_ridge=1.0,
            cross_ridge=0.1,
            v_history_mode="diagonal",
            coordinate_system="pre_rope_normalized",
        )
        codec = TempoKVCodec(temporal, secondary, config)

        def decoder(state, source_tensor=None, output_tensor=None,
                    output_layout="BSHD", _codec=codec):
            packed = state[_STATE_KEY] if isinstance(state, dict) else state
            kwargs = {
                "source": source_tensor,
                "output_tensor": output_tensor,
                "use_triton": False,
                "strict_triton": False,
            }
            if output_layout != "BSHD":
                kwargs["output_layout"] = output_layout
            return decode_tensor(packed, **kwargs)

        cache_module.set_quantized_decoder(decoder)
        caches = {
            role: cache_module.ChunkedKVCache(
                batch_size=1,
                frame_seq_length=STRIDE,
                num_heads=HEADS,
                head_dim=DIM,
                max_num_chunks=FRAMES,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                layout="BSHD",
                cache_role=role,
            )
            for role in ("k", "v")
        }
        tokens = FRAMES * STRIDE
        written = {}
        for role, tensor in (("k", keys), ("v", values)):
            frames = tensor[0, 0]
            expanded = (
                frames.unsqueeze(1)
                .expand(FRAMES, STRIDE, HEADS, DIM)
                .reshape(1, tokens, HEADS, DIM)
                .to(torch.bfloat16)
                .contiguous()
            )
            written[role] = expanded
            caches[role].write(0, tokens, expanded)

        primary_role, secondary_role = codec.primary_role, codec.secondary_role
        primary = caches[primary_role].read_bhsd(0, tokens)
        packed_primary, rebuilt = codec.encode_primary(
            primary, layer=0, reuse_input=True
        )
        caches[primary_role].store_quantized(0, tokens, _wrap(packed_primary))
        dependent = caches[secondary_role].read_bhsd(0, tokens)
        packed_secondary = codec.encode_secondary(
            dependent, rebuilt, layer=0, reuse_input=True
        )
        caches[secondary_role].store_quantized(0, tokens, _wrap(packed_secondary))

        primary_all = caches[primary_role].read(0, tokens)
        secondary_all = caches[secondary_role].read(
            0, tokens, source_tensor=primary_all
        )
        got = {primary_role: primary_all, secondary_role: secondary_all}

        def nmse(rebuilt_tensor, reference):
            reference = reference.double()
            error = rebuilt_tensor.double() - reference
            return float(error.square().sum() / reference.square().sum())

        results[name] = {
            "primary": primary_role,
            "k_nmse": nmse(got["k"], written["k"]),
            "v_nmse": nmse(got["v"], written["v"]),
        }
    return results


def _run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    if subprocess.run(command, text=True).returncode != 0:
        raise SystemExit(f"step failed: {' '.join(command)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.output_root.expanduser().resolve()
    calibration = synthesize(root / "traces" / "calibration.npz", seed=args.seed)
    evaluation = synthesize(root / "traces" / "evaluation.npz", seed=args.seed + 1)

    print("== runtime: both directions through the real cache ==")
    runtime = check_runtime_directions(calibration)
    for name, measured in runtime.items():
        print(
            f"  {name:<8} primary={measured['primary']}  "
            f"K nmse={measured['k_nmse']:.6f}  V nmse={measured['v_nmse']:.6f}"
        )
    (root / "runtime_directions.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n== reconstruction: all three arms ==")
    reconstruction_dir = root / "reconstruction"
    _run(
        [args.python_bin, str(HERE / "run_reconstruction.py"),
         "--calibration-trace", str(calibration),
         "--evaluation-trace", str(evaluation),
         "--output-dir", str(reconstruction_dir)]
    )

    print("\n== quality: command matrix only (needs a model to run for real) ==")
    fake = root / "fake"
    (fake / "wan").mkdir(parents=True, exist_ok=True)
    checkpoint = fake / "checkpoint.pt"
    checkpoint.touch()
    quality_dir = root / "quality"
    _run(
        [args.python_bin, str(HERE / "run_quality.py"),
         "--checkpoint", str(checkpoint),
         "--wan-model-dir", str(fake / "wan"),
         "--prompts", str(PROJECT_ROOT / "assets" / "moviegen128_first30.txt"),
         "--trace", str(calibration),
         "--output-root", str(quality_dir),
         "--dry-run"]
    )

    _run(
        [args.python_bin, str(HERE / "plot_figure.py"),
         "--reconstruction-report",
         str(reconstruction_dir / "direction_reconstruction.json"),
         "--quality-report", str(quality_dir / "summary.json"),
         "--output-dir", str(root / "figure")]
    )

    expected = [
        root / "runtime_directions.json",
        reconstruction_dir / "direction_reconstruction.json",
        quality_dir / "manifest.json",
        quality_dir / "summary.json",
        root / "figure" / "panel_a_reconstruction.csv",
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise SystemExit(
            "the run finished but did not produce: "
            + ", ".join(str(path) for path in missing)
        )
    # A dry run must still describe every arm, or the matrix is incomplete.
    matrix = json.loads((quality_dir / "manifest.json").read_text(encoding="utf-8"))
    covered = {item["name"] for item in matrix["arms"]}
    if covered != {item.name for item in ARMS}:
        raise SystemExit(f"quality matrix is missing arms: {covered}")

    print("\nminimal direction study complete")
    for path in expected:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
