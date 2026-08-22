"""Shared contracts for paper experiment harnesses.

The benchmark scripts deliberately exchange small JSON reports with the model
runtimes.  Keeping that boundary strict prevents an incomplete profile, a
different prompt, or an asynchronously timed section from silently entering a
paper table.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHODS = (
    "BF16",
    "RTN_INT2",
    "RTN_INT4",
    "QVG_INT2",
    "QVG_INT4",
    "TempoKV_INT2",
    "TempoKV_INT4",
)
HORIZONS = (180, 360, 540, 717, 1050, 1400)
FIXED_FRAMES = 717
REPEATS = 5

MANIFEST_SCHEMA = "tempokv.latency-manifest.v1"
RUN_SCHEMA = "tempokv.latency-run.v1"
COMPILATION_SCHEMA = "tempokv.compilation-run.v1"

TIMING_KEYS = (
    "e2e",
    "cache_encode",
    "cache_decode",
    "vae_decode",
    "denoising_other",
)
FROZEN_KEYS = (
    "model",
    "width",
    "height",
    "prompt_sha256",
    "seed",
    "gpu_name",
    "gpu_uuid",
    "cuda_version",
    "attention_backend",
    "attention_lifecycle",
)


class ContractError(ValueError):
    """A report cannot be admitted to the experiment."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile (NumPy's default definition)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    location = fraction * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: Sequence[float]) -> dict[str, float]:
    if len(values) != REPEATS:
        raise ContractError(f"expected exactly {REPEATS} repeats, got {len(values)}")
    return {
        "median": percentile(values, 0.50),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
    }


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    missing = expected_set - actual
    extra = actual - expected_set
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if extra:
            details.append("extra=" + ",".join(sorted(extra)))
        raise ContractError(f"{label} has the wrong fields ({'; '.join(details)})")


def validate_frozen(frozen: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(frozen, dict):
        raise ContractError("frozen must be an object")
    _exact_keys(frozen, FROZEN_KEYS, "frozen")
    _exact_keys(expected, FROZEN_KEYS, "manifest frozen")
    mismatches = {
        key: (expected[key], frozen[key])
        for key in FROZEN_KEYS
        if frozen[key] != expected[key]
    }
    if mismatches:
        rendered = ", ".join(
            f"{key}: expected {before!r}, got {after!r}"
            for key, (before, after) in mismatches.items()
        )
        raise ContractError(f"frozen-field mismatch: {rendered}")
    if frozen["gpu_name"] != "NVIDIA A100-SXM4-80GB":
        raise ContractError(
            "formal protocol requires gpu_name='NVIDIA A100-SXM4-80GB'"
        )
    return dict(frozen)


RUN_FIELDS = (
    "schema_version",
    "method",
    "phase",
    "repeat",
    "requested_frames",
    "effective_frames",
    "session_id",
    "session_scope",
    "runtime_instance_id",
    "frozen",
    "timings_s",
    "synchronization",
    "exclusions",
)


def validate_run_report(
    report: Any,
    *,
    expected_method: str,
    expected_phase: str,
    expected_repeat: int,
    expected_frames: int,
    expected_session_id: str,
    expected_session_scope: str,
    expected_frozen: Mapping[str, Any],
    expected_runtime_instance_id: str | None = None,
    formal: bool = True,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ContractError("run report must be an object")
    _exact_keys(report, RUN_FIELDS, "run report")
    expectations = {
        "schema_version": RUN_SCHEMA,
        "method": expected_method,
        "phase": expected_phase,
        "repeat": expected_repeat,
        "requested_frames": expected_frames,
        "effective_frames": expected_frames,
        "session_id": expected_session_id,
        "session_scope": expected_session_scope,
    }
    for key, expected in expectations.items():
        if report[key] != expected:
            raise ContractError(
                f"{key} mismatch: expected {expected!r}, got {report[key]!r}"
            )
    runtime_instance_id = report["runtime_instance_id"]
    if not isinstance(runtime_instance_id, str) or not runtime_instance_id:
        raise ContractError("runtime_instance_id must be a non-empty string")
    if (
        expected_runtime_instance_id is not None
        and runtime_instance_id != expected_runtime_instance_id
    ):
        raise ContractError(
            "warmup and measurement did not use the same loaded runtime instance: "
            f"expected {expected_runtime_instance_id!r}, got {runtime_instance_id!r}"
        )
    validate_frozen(report["frozen"], expected_frozen)

    timings = report["timings_s"]
    if not isinstance(timings, dict):
        raise ContractError("timings_s must be an object")
    _exact_keys(timings, TIMING_KEYS, "timings_s")
    numeric = {}
    for key in TIMING_KEYS:
        value = timings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"timings_s.{key} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ContractError(f"timings_s.{key} must be finite and non-negative")
        numeric[key] = value
    parts = sum(numeric[key] for key in TIMING_KEYS if key != "e2e")
    tolerance = max(0.005, numeric["e2e"] * 0.01)
    if abs(parts - numeric["e2e"]) > tolerance:
        raise ContractError(
            "timing decomposition must sum to e2e within 1% (or 5 ms): "
            f"parts={parts:.6f}, e2e={numeric['e2e']:.6f}"
        )

    sync = report["synchronization"]
    if not isinstance(sync, dict):
        raise ContractError("synchronization must be an object")
    _exact_keys(sync, ("api", "before_each_segment", "after_each_segment"), "synchronization")
    if formal and sync != {
        "api": "torch.cuda.synchronize",
        "before_each_segment": True,
        "after_each_segment": True,
    }:
        raise ContractError(
            "formal reports require torch.cuda.synchronize before and after every timing segment"
        )

    exclusions = report["exclusions"]
    if not isinstance(exclusions, dict):
        raise ContractError("exclusions must be an object")
    _exact_keys(exclusions, ("model_loading", "compilation"), "exclusions")
    if formal and exclusions != {"model_loading": True, "compilation": True}:
        raise ContractError("e2e must exclude model loading and compilation")
    normalized = dict(report)
    normalized["timings_s"] = numeric
    return normalized


COMPILATION_FIELDS = (
    "schema_version",
    "method",
    "one_time_compilation_s",
    "session_id",
    "session_scope",
    "frozen",
)


def validate_compilation_report(
    report: Any,
    *,
    expected_method: str,
    expected_session_id: str,
    expected_session_scope: str,
    expected_frozen: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ContractError("compilation report must be an object")
    _exact_keys(report, COMPILATION_FIELDS, "compilation report")
    expectations = {
        "schema_version": COMPILATION_SCHEMA,
        "method": expected_method,
        "session_id": expected_session_id,
        "session_scope": expected_session_scope,
    }
    for key, expected in expectations.items():
        if report[key] != expected:
            raise ContractError(
                f"{key} mismatch: expected {expected!r}, got {report[key]!r}"
            )
    validate_frozen(report["frozen"], expected_frozen)
    seconds = report["one_time_compilation_s"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ContractError("one_time_compilation_s must be numeric")
    if not math.isfinite(float(seconds)) or float(seconds) < 0:
        raise ContractError("one_time_compilation_s must be finite and non-negative")
    normalized = dict(report)
    normalized["one_time_compilation_s"] = float(seconds)
    return normalized
