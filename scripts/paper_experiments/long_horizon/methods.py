#!/usr/bin/env python3
"""Comparison methods and per-frame reconstruction error.

A pooled reconstruction number hides the thing a long horizon is asked about:
whether the error is flat across the sequence or grows as the predictor rolls
further from its anchor.  This module measures both -- the error at each frame
index, and the pooled figure over the whole trace -- with every method on one
reconstruction path so a difference between methods is a difference between
predictors and not between two pieces of measurement code.

Only methods this repository actually implements are listed.  Externally
published baselines that are not implemented here are deliberately absent
rather than approximated.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_temporal_trace,
    _reconstructed_key_innovation_trace,
    fit_predictors,
)


SCHEMA_VERSION = 1

# Shared by every method; only the predictor fields below are allowed to vary,
# so a difference between methods cannot come from the quantizer.
BASE_CODEC: dict[str, Any] = {
    "anchor_bits": 4,
    "group_size": 64,
    "residual_mode": "asymmetric",
    "scale_dtype": "bfloat16",
    "reconstruction_dtype": "bfloat16",
    "cross_kv_weight_mode": "layer_shared",
    "cross_matmul_precision": "ieee",
    "fit_mode": "legacy",
    "use_triton": True,
    "strict_triton": False,
}
V_HISTORY_MODE = "diagonal"


@dataclass(frozen=True)
class Method:
    """One reconstruction method under comparison."""

    name: str
    label: str
    k_predictor_mode: str
    v_predictor_mode: str
    k_history_length: int
    v_history_length: int
    predictor_direction: str
    description: str

    def codec_mapping(self, stride: int, bits: int) -> dict[str, Any]:
        return {
            **BASE_CODEC,
            "stride": int(stride),
            "k_bits": int(bits),
            "v_bits": int(bits),
            "k_predictor_mode": self.k_predictor_mode,
            "v_predictor_mode": self.v_predictor_mode,
            "k_history_length": self.k_history_length,
            "v_history_length": self.v_history_length,
            "predictor_direction": self.predictor_direction,
        }

    def codec(self, stride: int, bits: int) -> CodecConfig:
        return config_from_mapping(self.codec_mapping(stride, bits))

    @property
    def has_cross_term(self) -> bool:
        return self.v_predictor_mode in {"cross_kv", "innovation"}


METHODS: tuple[Method, ...] = (
    Method(
        name="rtn",
        label="RTN",
        k_predictor_mode="direct",
        v_predictor_mode="direct",
        k_history_length=0,
        v_history_length=0,
        predictor_direction="k_to_v",
        description="round-to-nearest per group; no predictor on either tensor",
    ),
    Method(
        name="temporal",
        label="temporal only",
        k_predictor_mode="shared",
        v_predictor_mode="temporal",
        k_history_length=2,
        v_history_length=1,
        predictor_direction="k_to_v",
        description="each tensor predicted from its own reconstructed history",
    ),
    Method(
        name="tempokv",
        label="TempoKV",
        k_predictor_mode="shared",
        v_predictor_mode="cross_kv",
        k_history_length=2,
        v_history_length=1,
        predictor_direction="k_to_v",
        description="production: V projected from the reconstructed K",
    ),
    Method(
        name="innovation",
        label="innovation",
        k_predictor_mode="shared",
        v_predictor_mode="innovation",
        k_history_length=2,
        v_history_length=1,
        predictor_direction="k_to_v",
        description="V projected from the decoder-visible K temporal innovation",
    ),
    Method(
        name="v_to_k",
        label="V->K mirror",
        k_predictor_mode="shared",
        v_predictor_mode="cross_kv",
        k_history_length=2,
        v_history_length=1,
        predictor_direction="v_to_k",
        description="mirrored: K projected from the reconstructed V",
    ),
)

METHODS_BY_NAME = {item.name: item for item in METHODS}
DEFAULT_METHODS = ("rtn", "temporal", "tempokv")


def method(name: str) -> Method:
    try:
        return METHODS_BY_NAME[name]
    except KeyError:
        available = ", ".join(METHODS_BY_NAME)
        raise ValueError(
            f"unknown method {name!r}; this repository implements {available}"
        ) from None


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def reconstruct(
    spec: Method,
    keys: torch.Tensor,
    values: torch.Tensor,
    temporal,
    secondary,
    config: CodecConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the closed loop for one method and return rebuilt (K, V).

    RTN and the temporal method share this path because a zero-length temporal
    map *is* direct quantization; the codec already routes it that way, so the
    baseline is the shipped code rather than a second implementation of it.
    """
    if not spec.has_cross_term:
        rebuilt_keys = _reconstruct_temporal_trace(keys, temporal, config)
        rebuilt_values = _reconstruct_temporal_trace(
            values, secondary, config, bits=config.v_bits
        )
        return rebuilt_keys, rebuilt_values

    primary, dependent = (
        (values, keys) if config.mirrored else (keys, values)
    )
    rebuilt_primary = _reconstruct_temporal_trace(primary, temporal, config)
    source = rebuilt_primary
    if spec.v_predictor_mode == "innovation":
        # Match the fitter: the projection reads the innovation the decoder can
        # reproduce, not the reconstruction itself.
        source = _reconstructed_key_innovation_trace(rebuilt_primary, temporal, config)
    rebuilt_dependent = _reconstruct_cross_kv_trace(
        source, dependent, secondary, config
    )
    if config.mirrored:
        return rebuilt_dependent, rebuilt_primary
    return rebuilt_primary, rebuilt_dependent


def fit(
    spec: Method,
    keys: torch.Tensor,
    values: torch.Tensor,
    config: CodecConfig,
    *,
    iterations: int = 2,
    temporal_ridge: float = 1.0,
    cross_ridge: float = 0.1,
    coordinate_system: str = "pre_rope_normalized",
):
    return fit_predictors(
        keys,
        values,
        config=config,
        iterations=iterations,
        temporal_ridge=temporal_ridge,
        cross_ridge=cross_ridge,
        v_history_mode=V_HISTORY_MODE,
        coordinate_system=coordinate_system,
    )


# --------------------------------------------------------------------------
# Per-frame error
# --------------------------------------------------------------------------


def _frame_energies(
    clean: torch.Tensor, rebuilt: torch.Tensor
) -> tuple[list[float], list[float]]:
    """Return per-frame (error energy, reference energy) along the T axis."""
    frames = int(clean.shape[-3])
    errors: list[float] = []
    references: list[float] = []
    for index in range(frames):
        reference = clean[..., index, :, :].double()
        error = rebuilt[..., index, :, :].double() - reference
        errors.append(float(error.square().sum()))
        references.append(float(reference.square().sum()))
    return errors, references


def frame_curve(clean: torch.Tensor, rebuilt: torch.Tensor) -> dict[str, Any]:
    """Per-frame, cumulative, and pooled NMSE for one tensor.

    ``cumulative[t]`` is the NMSE over frames 0..t, so the last entry is the
    pooled figure.  Reporting both separates a method that degrades late from
    one that is uniformly worse -- a distinction the pooled number erases.
    """
    errors, references = _frame_energies(clean, rebuilt)
    per_frame: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    running_error = 0.0
    running_reference = 0.0
    for index, (error, reference) in enumerate(zip(errors, references)):
        running_error += error
        running_reference += reference
        per_frame.append(
            {
                "frame": index,
                "nmse": error / reference if reference > 0.0 else None,
                "error_energy": error,
                "reference_energy": reference,
            }
        )
        cumulative.append(
            {
                "frame": index,
                "nmse": (
                    running_error / running_reference
                    if running_reference > 0.0
                    else None
                ),
            }
        )
    return {
        "frames": len(per_frame),
        "per_frame": per_frame,
        "cumulative": cumulative,
        "pooled_nmse": (
            running_error / running_reference if running_reference > 0.0 else None
        ),
    }


def summarize_curve(curve: dict[str, Any], *, tail_fraction: float = 0.25) -> dict[str, Any]:
    """Condense a curve into the few numbers a table needs.

    The tail figure answers the long-horizon question directly: what the error
    looks like once the predictor has rolled far past its anchor.
    """
    values = [row["nmse"] for row in curve["per_frame"] if row["nmse"] is not None]
    if not values:
        raise ValueError("curve contains no finite frames")
    frames = len(values)
    tail_length = min(frames, max(1, int(round(frames * tail_fraction))))
    tail_start = frames - tail_length
    tail = values[tail_start:]
    # Everything before the tail window.  When the tail covers the whole
    # sequence there is no head to compare against, so the tail stands in for
    # it and the ratio is 1 by construction rather than a slice of the tail.
    head = values[:tail_start] if tail_start > 0 else tail
    return {
        "pooled_nmse": curve["pooled_nmse"],
        "first_frame_nmse": values[0],
        "last_frame_nmse": values[-1],
        "max_frame_nmse": max(values),
        "tail_start_frame": tail_start,
        "tail_mean_nmse": sum(tail) / len(tail),
        "head_mean_nmse": sum(head) / len(head),
        "tail_over_head": (
            (sum(tail) / len(tail)) / (sum(head) / len(head))
            if sum(head) > 0.0
            else None
        ),
    }


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
