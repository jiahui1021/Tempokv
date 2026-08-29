#!/usr/bin/env python3
"""Shared definitions for the K->V direction study.

TempoKV predicts V from the reconstructed K of the same span.  That is an
asymmetric design, and asymmetry needs a reason.  This study supplies one by
measuring the two directions and the no-dependency baseline under an otherwise
identical codec:

    temporal_only   K and V are each predicted from their own temporal history.
                    No cross dependency in either direction.
    k_to_v          V is projected from the reconstructed K.  Production.
    v_to_k          K is projected from the reconstructed V.  The mirror.

The codec equations never mention K or V, so ``v_to_k`` is not a second codec:
it is the same one with the tensor roles exchanged, selected by
``CodecConfig.predictor_direction``.  That matters for the argument -- if the
mirror needed different math, a difference between the arms could be a
difference between two implementations rather than between two directions.

Every arm is measured on one code path here for the same reason.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_temporal_trace,
    fit_predictors,
)


SCHEMA_VERSION = 1

# The codec every arm shares.  Only ``v_predictor_mode`` and
# ``predictor_direction`` are allowed to vary, and both are set per arm below.
# Equal K/V bit widths are required by the mirror and are also what makes the
# comparison about direction rather than about bit allocation.
BASE_CODEC: dict[str, Any] = {
    "k_bits": 2,
    "v_bits": 2,
    "anchor_bits": 4,
    "group_size": 64,
    "residual_mode": "asymmetric",
    "scale_dtype": "bfloat16",
    "reconstruction_dtype": "bfloat16",
    # Role-scoped, not tensor-scoped: the primary (temporally predicted) tensor
    # gets ``k_history_length`` lags and the secondary (cross-projected) tensor
    # gets ``v_history_length`` lags of its own history, whichever way round the
    # two tensors are.  That keeps the mirror a structural reflection.
    "k_history_length": 2,
    "v_history_length": 1,
    "k_predictor_mode": "shared",
    "cross_kv_weight_mode": "layer_shared",
    "cross_matmul_precision": "ieee",
    "fit_mode": "legacy",
    "use_triton": True,
    "strict_triton": False,
}
V_HISTORY_MODE = "diagonal"


@dataclass(frozen=True)
class Arm:
    """One point of the direction comparison."""

    name: str
    v_predictor_mode: str
    predictor_direction: str
    label: str
    interpretation: str

    @property
    def has_cross_term(self) -> bool:
        return self.v_predictor_mode == "cross_kv"

    @property
    def primary_tensor(self) -> str:
        return "v" if self.predictor_direction == "v_to_k" else "k"

    def codec(self, stride: int) -> CodecConfig:
        return config_from_mapping(self.codec_mapping(stride))

    def codec_mapping(self, stride: int) -> dict[str, Any]:
        return {
            **BASE_CODEC,
            "stride": int(stride),
            "v_predictor_mode": self.v_predictor_mode,
            "predictor_direction": self.predictor_direction,
        }


ARMS: tuple[Arm, ...] = (
    Arm(
        name="temporal_only",
        v_predictor_mode="temporal",
        predictor_direction="k_to_v",
        label="temporal only",
        interpretation="no cross dependency in either direction",
    ),
    Arm(
        name="k_to_v",
        v_predictor_mode="cross_kv",
        predictor_direction="k_to_v",
        label=r"$K_t \rightarrow V_t$",
        interpretation="V projected from reconstructed K (production)",
    ),
    Arm(
        name="v_to_k",
        v_predictor_mode="cross_kv",
        predictor_direction="v_to_k",
        label=r"$V_t \rightarrow K_t$",
        interpretation="K projected from reconstructed V (mirror)",
    ),
)

ARMS_BY_NAME = {arm.name: arm for arm in ARMS}


def arm(name: str) -> Arm:
    try:
        return ARMS_BY_NAME[name]
    except KeyError:
        available = ", ".join(ARMS_BY_NAME)
        raise ValueError(f"unknown arm {name!r}; known arms are {available}") from None


def write_codec_config(path: Path, arm_spec: Arm, stride: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(arm_spec.codec_mapping(stride), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def _nmse(clean: torch.Tensor, rebuilt: torch.Tensor) -> float:
    reference = clean.double()
    error = rebuilt.double() - reference
    energy = float(reference.square().sum().item())
    if energy <= 0.0:
        raise ValueError("cannot normalize against a zero-energy reference")
    return float(error.square().sum().item()) / energy


def reconstruct(
    arm_spec: Arm,
    keys: torch.Tensor,
    values: torch.Tensor,
    temporal,
    secondary,
    config: CodecConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the closed loop for one arm and return rebuilt (K, V).

    All three arms go through this one function.  A comparison whose arms were
    measured by different code would not separate the direction from the
    measurement.
    """
    if arm_spec.v_predictor_mode == "temporal":
        # No cross term: each tensor closes its own loop, V with its own bits.
        rebuilt_keys = _reconstruct_temporal_trace(keys, temporal, config)
        rebuilt_values = _reconstruct_temporal_trace(
            values, secondary, config, bits=config.v_bits
        )
        return rebuilt_keys, rebuilt_values

    # A cross term exists; ``primary`` leads and ``secondary`` is projected
    # from the primary's reconstruction.  Swapping the pair is exactly what
    # ``predictor_direction`` means, so the mirror reuses the same two calls.
    primary, dependent = (
        (values, keys) if config.mirrored else (keys, values)
    )
    rebuilt_primary = _reconstruct_temporal_trace(primary, temporal, config)
    rebuilt_dependent = _reconstruct_cross_kv_trace(
        rebuilt_primary, dependent, secondary, config
    )
    if config.mirrored:
        return rebuilt_dependent, rebuilt_primary
    return rebuilt_primary, rebuilt_dependent


def measure_arm(
    arm_spec: Arm,
    calibration: tuple[torch.Tensor, torch.Tensor],
    evaluation: tuple[torch.Tensor, torch.Tensor],
    stride: int,
    *,
    iterations: int = 2,
    temporal_ridge: float = 1.0,
    cross_ridge: float = 0.1,
    coordinate_system: str = "pre_rope_normalized",
) -> dict[str, Any]:
    """Fit one arm on the calibration trace and score it on the held-out one."""
    config = arm_spec.codec(stride)
    calibration_keys, calibration_values = calibration
    temporal, secondary = fit_predictors(
        calibration_keys,
        calibration_values,
        config=config,
        iterations=iterations,
        temporal_ridge=temporal_ridge,
        cross_ridge=cross_ridge,
        v_history_mode=V_HISTORY_MODE,
        coordinate_system=coordinate_system,
    )
    evaluation_keys, evaluation_values = evaluation
    rebuilt_keys, rebuilt_values = reconstruct(
        arm_spec, evaluation_keys, evaluation_values, temporal, secondary, config
    )
    return {
        "arm": arm_spec.name,
        "v_predictor_mode": arm_spec.v_predictor_mode,
        "predictor_direction": arm_spec.predictor_direction,
        "primary_tensor": arm_spec.primary_tensor,
        "interpretation": arm_spec.interpretation,
        "codec_signature": config.format_signature(),
        "k_nmse": _nmse(evaluation_keys, rebuilt_keys),
        "v_nmse": _nmse(evaluation_values, rebuilt_values),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Report each arm against the no-dependency baseline.

    The cross term is only worth its parameters if it beats ``temporal_only``,
    and the direction is only a design decision if the two directions differ.
    Both comparisons are made explicit here rather than left to the reader.
    """
    by_name = {item["arm"]: item for item in results}
    baseline = by_name.get("temporal_only")
    rows = []
    for item in results:
        row = {"arm": item["arm"]}
        for tensor in ("k_nmse", "v_nmse"):
            row[tensor] = item[tensor]
            if baseline is not None and baseline[tensor] > 0.0:
                row[f"{tensor}_vs_temporal_only"] = (
                    item[tensor] / baseline[tensor] - 1.0
                )
        rows.append(row)
    comparison = None
    if "k_to_v" in by_name and "v_to_k" in by_name:
        forward, mirror = by_name["k_to_v"], by_name["v_to_k"]
        # Total closed-loop error is what the attention layer actually sees, so
        # the directions are ranked on the sum rather than on either tensor.
        forward_total = forward["k_nmse"] + forward["v_nmse"]
        mirror_total = mirror["k_nmse"] + mirror["v_nmse"]
        comparison = {
            "k_to_v_total_nmse": forward_total,
            "v_to_k_total_nmse": mirror_total,
            "mirror_penalty": (
                mirror_total / forward_total - 1.0 if forward_total > 0.0 else None
            ),
            "preferred_direction": (
                "k_to_v" if forward_total <= mirror_total else "v_to_k"
            ),
        }
    return {"per_arm": rows, "direction_comparison": comparison}
