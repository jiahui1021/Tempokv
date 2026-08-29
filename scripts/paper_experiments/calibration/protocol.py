#!/usr/bin/env python3
"""Shared machinery for the TempoKV calibration study.

Every module in this study runs one protocol.  Collect a single calibration
*pool* trace per model, slice that pool by prompt, refit the predictor on the
slice, and score the refit predictor on a fixed held-out slice of the same
pool.  Because both the calibration and the evaluation records come from one
collection run, the quantizer, the anchor schedule, the sampling protocol and
the model checkpoint are identical across every point; the prompt sets are the
only free variable.

The three trace bridges (Self-Forcing, Causal-Forcing, LongCat) all merge their
per-event records along axis 1, so a pool trace is ``[L, S, T, H, D]`` with the
records of prompt ``p`` occupying one contiguous block of rows.  That makes the
slicing model-agnostic: the block size is derived from the archive, never
hard-coded per model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIT_SCRIPT = PROJECT_ROOT / "scripts" / "common" / "fit_params.py"
EVALUATE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "paper_experiments"
    / "cross_kv_sharing"
    / "evaluate_reconstruction.py"
)

# The fixed INT2 codec every fit and every evaluation in the study uses.  It is
# written to one config file per run and passed to both the fitter and the
# evaluator, so a codec difference can never be mistaken for a calibration
# effect.  ``stride`` is the one model-dependent field and is read from the
# pool trace instead of being pinned here.
CODEC: dict[str, Any] = {
    "k_bits": 2,
    "v_bits": 2,
    "anchor_bits": 4,
    "group_size": 64,
    "residual_mode": "asymmetric",
    "scale_dtype": "bfloat16",
    "reconstruction_dtype": "bfloat16",
    "k_history_length": 2,
    "v_history_length": 1,
    "k_predictor_mode": "shared",
    "v_predictor_mode": "cross_kv",
    "cross_kv_weight_mode": "layer_shared",
    "cross_matmul_precision": "ieee",
    "fit_mode": "legacy",
    "use_triton": True,
    "strict_triton": False,
}
# Legacy Cross(K) fitting accepts only the diagonal V-history map, so this is
# not a free choice; it is stated once here rather than at each call site.
V_HISTORY_MODE = "diagonal"

SCHEMA_VERSION = 1


def load_module(path: Path, name: str) -> ModuleType:
    """Import a repository script that is not importable as a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def codec_config(stride: int) -> dict[str, Any]:
    """Return the study codec bound to one model's trace stride."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    return {**CODEC, "stride": int(stride)}


def write_codec_config(path: Path, stride: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(codec_config(stride), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# Prompt groups
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptGroups:
    """Named prompt sets, addressed by their position in the pool."""

    path: Path
    prompts_file: str
    prompt_count: int
    prompt_indices: tuple[int, ...]
    groups: dict[str, tuple[int, ...]]

    @classmethod
    def load(cls, path: Path) -> "PromptGroups":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"prompt groups must be a JSON object: {path}")
        try:
            prompt_count = int(payload["prompt_count"])
            raw_groups = payload["groups"]
        except KeyError as error:
            raise ValueError(f"{path} is missing {error.args[0]!r}") from error
        if prompt_count <= 0:
            raise ValueError(f"{path} prompt_count must be positive")
        if not isinstance(raw_groups, dict) or not raw_groups:
            raise ValueError(f"{path} must define at least one prompt group")
        groups: dict[str, tuple[int, ...]] = {}
        for name, members in raw_groups.items():
            if not isinstance(members, list) or not members:
                raise ValueError(f"{path} group {name!r} must be a non-empty list")
            positions = tuple(int(item) for item in members)
            if len(set(positions)) != len(positions):
                raise ValueError(f"{path} group {name!r} repeats a prompt")
            out_of_range = [p for p in positions if not 0 <= p < prompt_count]
            if out_of_range:
                raise ValueError(
                    f"{path} group {name!r} refers to prompts outside the pool: "
                    f"{sorted(out_of_range)}"
                )
            groups[name] = positions
        prompt_indices = tuple(
            int(item) for item in payload.get("prompt_indices", range(prompt_count))
        )
        if len(prompt_indices) != prompt_count:
            raise ValueError(f"{path} prompt_indices must have prompt_count entries")
        return cls(
            path=path,
            prompts_file=str(payload.get("prompts_file", "")),
            prompt_count=prompt_count,
            prompt_indices=prompt_indices,
            groups=groups,
        )

    def group(self, name: str) -> tuple[int, ...]:
        try:
            return self.groups[name]
        except KeyError:
            available = ", ".join(sorted(self.groups))
            raise ValueError(
                f"{self.path} has no prompt group {name!r}; defines {available}"
            ) from None

    def require_disjoint(self, first: str, second: str) -> None:
        """Refuse a calibration/evaluation pair that shares prompts."""
        overlap = sorted(set(self.group(first)) & set(self.group(second)))
        if overlap:
            raise ValueError(
                f"prompt groups {first!r} and {second!r} overlap at {overlap}; "
                "calibration and evaluation prompts must be disjoint"
            )


# --------------------------------------------------------------------------
# Pool traces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolTrace:
    """A calibration pool trace, sliceable by prompt."""

    path: Path
    keys: np.ndarray
    values: np.ndarray
    layer_ids: np.ndarray
    stride: int
    prompt_count: int
    rows_per_prompt: int
    coordinate_system: str
    source_layout: str

    @classmethod
    def load(cls, path: Path, prompt_count: int) -> "PoolTrace":
        path = path.expanduser().resolve()
        if prompt_count <= 0:
            raise ValueError("prompt_count must be positive")
        with np.load(path, allow_pickle=False) as archive:
            missing = [n for n in ("keys", "values") if n not in archive]
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
            keys = np.array(archive["keys"], copy=True)
            values = np.array(archive["values"], copy=True)
            layer_ids = (
                np.array(archive["layer_ids"], copy=True)
                if "layer_ids" in archive
                else None
            )
            stride = None
            for name in ("trace_stride", "stride"):
                if name in archive:
                    stride = int(np.asarray(archive[name]).item())
                    break
            coordinate_system = (
                str(np.asarray(archive["coordinate_system"]).item())
                if "coordinate_system" in archive
                else "unspecified"
            )
            layout = (
                str(np.asarray(archive["trace_layout"]).item())
                if "trace_layout" in archive
                else "standard_lsthd"
            )
        if keys.shape != values.shape:
            raise ValueError(f"{path} K and V must share a shape")
        if stride is None:
            raise ValueError(f"{path} does not record a trace stride")
        # The two Forcing bridges can emit the seven-dimensional legacy event
        # layout [L,E,B,P,T,H,D].  Its records already run event-major, so
        # collapsing E/B/P yields the same row order the flat bridges write.
        if keys.ndim == 7:
            layers, events, batch, tokens = keys.shape[:4]
            keys = keys.reshape(layers, events * batch * tokens, *keys.shape[4:])
            values = values.reshape(layers, events * batch * tokens, *values.shape[4:])
        if keys.ndim != 5:
            raise ValueError(
                f"{path} must use [L,S,T,H,D] or the legacy [L,E,B,P,T,H,D] "
                f"layout, got {tuple(keys.shape)}"
            )
        rows = int(keys.shape[1])
        if rows % prompt_count:
            raise ValueError(
                f"{path} holds {rows} records, which {prompt_count} prompts do "
                "not divide evenly; the pool trace and the prompt groups "
                "describe different collection runs"
            )
        if layer_ids is None:
            layer_ids = np.arange(int(keys.shape[0]), dtype=np.int32)
        return cls(
            path=path,
            keys=np.ascontiguousarray(keys, dtype=np.float32),
            values=np.ascontiguousarray(values, dtype=np.float32),
            layer_ids=np.asarray(layer_ids, dtype=np.int32),
            stride=stride,
            prompt_count=prompt_count,
            rows_per_prompt=rows // prompt_count,
            coordinate_system=coordinate_system,
            source_layout=layout,
        )

    def rows_for(self, prompts: Sequence[int]) -> np.ndarray:
        """Return the record indices belonging to the given pool prompts."""
        if not len(prompts):
            raise ValueError("a prompt slice must contain at least one prompt")
        span = self.rows_per_prompt
        blocks = []
        for prompt in prompts:
            if not 0 <= int(prompt) < self.prompt_count:
                raise ValueError(f"prompt {prompt} is outside the pool")
            start = int(prompt) * span
            blocks.append(np.arange(start, start + span))
        return np.concatenate(blocks)

    def write_slice(self, path: Path, prompts: Sequence[int]) -> Path:
        """Write one prompt slice as a standalone fitting trace.

        The slice is written in the plain ``[L,S,T,H,D]`` layout with no
        ``trace_layout`` key.  That is the layout the legacy fitter accepts for
        an arbitrary record count, which is what a variable-size calibration
        set needs; the pinned 12-prompt event protocol cannot express it.
        """
        rows = self.rows_for(prompts)
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            keys=self.keys[:, rows],
            values=self.values[:, rows],
            layer_ids=self.layer_ids,
            trace_stride=np.asarray(self.stride, dtype=np.int32),
            coordinate_system=np.asarray(self.coordinate_system),
            cross_source_kind=np.asarray("normalized"),
            trace_role=np.asarray("calibration"),
            # Provenance, ignored by the fitter but enough to rebuild the slice.
            pool_trace=np.asarray(str(self.path)),
            pool_prompts=np.asarray(list(int(p) for p in prompts), dtype=np.int32),
        )
        return path


# --------------------------------------------------------------------------
# Fit and evaluate
# --------------------------------------------------------------------------


def fit_predictor(
    trace: Path,
    output_dir: Path,
    config: Path,
    *,
    device: str = "auto",
    iterations: int = 2,
    python_bin: str = sys.executable,
) -> dict[str, Any]:
    """Refit K/V predictors on one calibration slice via the shared fitter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        str(FIT_SCRIPT),
        str(trace),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
        "--v-history-mode",
        V_HISTORY_MODE,
        "--iterations",
        str(int(iterations)),
        "--device",
        device,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "fitting failed\n"
            + " ".join(command)
            + "\n"
            + (completed.stderr or completed.stdout)
        )
    report = output_dir / "fit_report.json"
    if not report.is_file():
        raise RuntimeError(f"fitter did not write {report}")
    return json.loads(report.read_text(encoding="utf-8"))


_EVALUATOR: ModuleType | None = None


def _evaluator() -> ModuleType:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = load_module(EVALUATE_SCRIPT, "calibration_evaluate_reconstruction")
    return _EVALUATOR


def evaluate_predictor(
    trace: Path,
    config: Path,
    parameters: Path,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Return pooled K and V normalized reconstruction MSE on a held-out slice.

    This replays the production closed-loop reconstruction, so the number is
    the error the deployed codec would make, not a teacher-forced regression
    residual.
    """
    result = _evaluator().evaluate(
        trace,
        config,
        parameters / "k_temporal.npz",
        parameters / "v_cross_kv.npz",
        device=device,
    )
    measured = {
        "k_nmse": result["k"]["pooled"]["nmse"],
        "v_nmse": result["v"]["pooled"]["nmse"],
    }
    for name, value in measured.items():
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"{name} is not finite on {trace}")
    return {name: float(value) for name, value in measured.items()}


@dataclass(frozen=True)
class Measurement:
    """One calibration set scored against one held-out set."""

    calibration_prompts: tuple[int, ...]
    k_nmse: float
    v_nmse: float
    fit_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_prompts": list(self.calibration_prompts),
            "calibration_size": len(self.calibration_prompts),
            "k_nmse": self.k_nmse,
            "v_nmse": self.v_nmse,
            "fit_seconds": self.fit_seconds,
        }


def measure(
    pool: PoolTrace,
    calibration_prompts: Sequence[int],
    heldout_trace: Path,
    config: Path,
    work_dir: Path,
    *,
    fit_device: str = "auto",
    eval_device: str = "cpu",
    iterations: int = 2,
    python_bin: str = sys.executable,
    keep_artifacts: bool = False,
) -> Measurement:
    """Fit on ``calibration_prompts`` and score on a prebuilt held-out trace."""
    work_dir.mkdir(parents=True, exist_ok=True)
    slice_trace = work_dir / "calibration_trace.npz"
    parameters = work_dir / "parameters"
    pool.write_slice(slice_trace, calibration_prompts)
    try:
        report = fit_predictor(
            slice_trace,
            parameters,
            config,
            device=fit_device,
            iterations=iterations,
            python_bin=python_bin,
        )
        scores = evaluate_predictor(
            heldout_trace, config, parameters, device=eval_device
        )
    finally:
        if not keep_artifacts:
            # A pool slice and its Cross(K) weights are gigabyte-scale; the
            # study keeps the metrics, not the intermediates.
            slice_trace.unlink(missing_ok=True)
            shutil.rmtree(parameters, ignore_errors=True)
    return Measurement(
        calibration_prompts=tuple(int(p) for p in calibration_prompts),
        k_nmse=scores["k_nmse"],
        v_nmse=scores["v_nmse"],
        fit_seconds=float(report.get("fit", {}).get("seconds", 0.0)),
    )


# --------------------------------------------------------------------------
# Draws and statistics
# --------------------------------------------------------------------------


def draw_subsets(
    population: Sequence[int], size: int, repeats: int, *, seed: int
) -> list[tuple[int, ...]]:
    """Return up to ``repeats`` distinct calibration sets of the given size.

    Which prompts land in a small calibration set matters, so every size is
    measured over several draws.  Drawing the whole population admits exactly
    one set, and the duplicates are dropped rather than reported as
    independent repeats.
    """
    if size <= 0:
        raise ValueError("calibration size must be positive")
    if size > len(population):
        raise ValueError(
            f"cannot draw {size} prompts from a pool of {len(population)}"
        )
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    ordered = sorted(int(item) for item in population)
    seen: set[tuple[int, ...]] = set()
    draws: list[tuple[int, ...]] = []
    for repeat in range(repeats):
        rng = random.Random(seed * 1_000_003 + size * 1_009 + repeat)
        candidate = tuple(sorted(rng.sample(ordered, size)))
        if candidate in seen:
            continue
        seen.add(candidate)
        draws.append(candidate)
    return draws


def summarize(values: Iterable[float]) -> dict[str, float]:
    collected = [float(item) for item in values]
    if not collected:
        raise ValueError("cannot summarize an empty sample")
    mean = sum(collected) / len(collected)
    if len(collected) == 1:
        deviation = 0.0
    else:
        variance = sum((item - mean) ** 2 for item in collected) / (
            len(collected) - 1
        )
        deviation = math.sqrt(variance)
    return {
        "mean": mean,
        "std": deviation,
        "min": min(collected),
        "max": max(collected),
        "samples": len(collected),
    }


def sweep_sizes(
    pool: PoolTrace,
    groups: PromptGroups,
    sizes: Sequence[int],
    *,
    calibration_group: str,
    heldout_group: str,
    output_dir: Path,
    repeats: int = 3,
    seed: int = 0,
    fit_device: str = "auto",
    eval_device: str = "cpu",
    iterations: int = 2,
    python_bin: str = sys.executable,
    keep_artifacts: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Measure held-out K/V NMSE as a function of calibration set size."""
    groups.require_disjoint(calibration_group, heldout_group)
    population = groups.group(calibration_group)
    heldout = groups.group(heldout_group)
    requested = sorted({int(size) for size in sizes})
    too_large = [size for size in requested if size > len(population)]
    if too_large:
        raise ValueError(
            f"calibration sizes {too_large} exceed the {len(population)} prompts "
            f"in group {calibration_group!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = write_codec_config(output_dir / "codec.json", pool.stride)
    heldout_trace = output_dir / "heldout_trace.npz"
    pool.write_slice(heldout_trace, heldout)
    points: list[dict[str, Any]] = []
    try:
        for size in requested:
            draws = draw_subsets(population, size, repeats, seed=seed)
            measurements = []
            for index, calibration in enumerate(draws):
                if progress:
                    print(
                        f"[calibration] N={size} draw {index + 1}/{len(draws)}",
                        file=sys.stderr,
                        flush=True,
                    )
                measurements.append(
                    measure(
                        pool,
                        calibration,
                        heldout_trace,
                        config,
                        output_dir / "work" / f"n{size}_r{index}",
                        fit_device=fit_device,
                        eval_device=eval_device,
                        iterations=iterations,
                        python_bin=python_bin,
                        keep_artifacts=keep_artifacts,
                    )
                )
            points.append(
                {
                    "calibration_size": size,
                    "draws": [item.as_dict() for item in measurements],
                    "k_nmse": summarize(item.k_nmse for item in measurements),
                    "v_nmse": summarize(item.v_nmse for item in measurements),
                }
            )
    finally:
        if not keep_artifacts:
            heldout_trace.unlink(missing_ok=True)
            shutil.rmtree(output_dir / "work", ignore_errors=True)
    return {
        "calibration_group": calibration_group,
        "calibration_pool_size": len(population),
        "heldout_group": heldout_group,
        "heldout_prompts": list(heldout),
        "repeats": repeats,
        "seed": seed,
        "points": points,
    }


def provenance(
    pool: PoolTrace, groups: PromptGroups, *, model: str
) -> dict[str, Any]:
    """Return the run identity every module records alongside its numbers."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "codec": codec_config(pool.stride),
        "v_history_mode": V_HISTORY_MODE,
        "pool_trace": str(pool.path),
        "pool_trace_layout": pool.source_layout,
        "pool_shape": list(pool.keys.shape),
        "prompt_count": pool.prompt_count,
        "records_per_prompt": pool.rows_per_prompt,
        "prompt_groups": str(groups.path),
        "prompts_file": groups.prompts_file,
    }


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pool-trace", type=Path, required=True)
    parser.add_argument("--prompt-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-artifacts", action="store_true")
