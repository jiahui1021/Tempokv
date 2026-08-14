from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


# Settings that existed in an earlier format and no longer do.  A hand-written
# config, or a ``runtime_config.json`` stored alongside an older run, would
# otherwise fail with an unexpected-keyword ``TypeError`` that does not say what
# to remove.
WITHDRAWN_SETTINGS = {
    "k_max_history_length": "shared-history K parameters were removed",
    "v_max_history_length": "shared-history V parameters were removed",
}


def config_from_mapping(payload: Mapping[str, Any]) -> "CodecConfig":
    """Build a codec config from parsed JSON, naming withdrawn settings."""
    withdrawn = sorted(set(payload) & set(WITHDRAWN_SETTINGS))
    if withdrawn:
        raise ValueError(
            "this codec config uses settings that no longer exist: "
            + "; ".join(
                f"{name} ({WITHDRAWN_SETTINGS[name]})" for name in withdrawn
            )
            + " — remove them and refit if the parameters were shared"
        )
    return CodecConfig(**payload)


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """Runtime quantization settings shared by the K and V codecs."""

    k_bits: int = 2
    v_bits: int = 2
    anchor_bits: int = 4
    group_size: int = 64
    stride: int = 1560
    residual_mode: str = "asymmetric"
    scale_dtype: str = "bfloat16"
    reconstruction_dtype: str = "bfloat16"
    use_triton: bool = True
    strict_triton: bool = False
    # Appended to preserve the positional argument order of older callers.
    v_history_length: int = 0
    # None keeps legacy parameter-driven K history selection.
    k_history_length: int | None = None
    # The legacy predictor uses IEEE FP32 CrossKV projections. ``tf32``
    # retains the native TempoKV execution policy for other configurations.
    cross_matmul_precision: str = "tf32"
    # ``shared`` pools all Self-Forcing layers and heads into [P,P,D].
    k_predictor_mode: str = "layer_head"
    # ``temporal`` predicts V from its own reconstructed history with the same
    # predictor K uses, so V needs no reconstructed K and no CrossKV matrix.
    # ``cross_kv`` keeps the Cross(K) projection.
    v_predictor_mode: str = "cross_kv"
    # Select the legacy fixed-point/shared-K and direct temporal-V fitting math.
    fit_mode: str = "native"

    def __post_init__(self) -> None:
        for name in ("k_bits", "v_bits", "anchor_bits"):
            value = int(getattr(self, name))
            if value not in (2, 4, 8):
                raise ValueError(f"{name} must be 2, 4, or 8, got {value}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if (
            isinstance(self.v_history_length, bool)
            or int(self.v_history_length) != self.v_history_length
            or not 0 <= int(self.v_history_length) <= 5
        ):
            raise ValueError("v_history_length must be an integer from 0 to 5")
        if self.k_history_length is not None and (
            isinstance(self.k_history_length, bool)
            or int(self.k_history_length) != self.k_history_length
            or int(self.k_history_length) <= 0
            or int(self.k_history_length) > 8
        ):
            raise ValueError("k_history_length must be an integer from 1 to 8 or null")
        if self.residual_mode not in {"symmetric", "asymmetric"}:
            raise ValueError("residual_mode must be symmetric or asymmetric")
        supported_dtypes = {"float32", "float16", "bfloat16"}
        if self.scale_dtype not in supported_dtypes:
            raise ValueError("scale_dtype must be float32, float16, or bfloat16")
        if self.reconstruction_dtype not in supported_dtypes:
            raise ValueError(
                "reconstruction_dtype must be float32, float16, or bfloat16"
            )
        if self.cross_matmul_precision not in {"ieee", "tf32"}:
            raise ValueError("cross_matmul_precision must be ieee or tf32")
        if self.k_predictor_mode not in {"layer_head", "shared"}:
            raise ValueError(
                "k_predictor_mode must be layer_head or shared"
            )
        if self.v_predictor_mode not in {"cross_kv", "temporal"}:
            raise ValueError("v_predictor_mode must be cross_kv or temporal")
        # The temporal V predictor reads only its own reconstructed history, so
        # with no history it has nothing to predict from.  Say so here; both
        # fitters would otherwise fail on a zero-length parameter axis.
        if self.v_predictor_mode == "temporal" and self.v_history_length < 1:
            raise ValueError(
                "v_predictor_mode=temporal needs v_history_length >= 1"
            )
        if self.fit_mode not in {"native", "legacy"}:
            raise ValueError("fit_mode must be native or legacy")
        if (
            self.k_predictor_mode == "shared"
            and self.fit_mode != "legacy"
        ):
            raise ValueError(
                "k_predictor_mode=shared requires fit_mode=legacy"
            )
        if self.fit_mode == "legacy":
            if self.k_predictor_mode != "shared":
                raise ValueError(
                    "legacy fitting requires k_predictor_mode=shared"
                )
            if self.k_history_length is None:
                raise ValueError(
                    "legacy fitting requires an explicit k_history_length"
                )
            if not 0 <= self.v_history_length <= 5:
                raise ValueError(
                    "legacy fitting supports V history length 0 through 5"
                )
            # ieee reproduces the reference arithmetic and stays the default;
            # tf32 trades the last mantissa bits of the Cross projection for
            # tensor cores, which is worth measuring end to end.
            if self.k_bits != self.v_bits:
                raise ValueError("legacy fitting requires matching K/V bits")
            # anchor_bits and group_size fix the quantizer's bit layout, so the
            # reference implementation is only reproduced at these values.
            # stride is not one of them: it only decides where the encoder cuts
            # units, and every legacy fitter merely records it, so pinning it
            # here stated a parity requirement that does not exist.  The real
            # limit sits in scripts/common/fit_params.py, which accepts the
            # fixed legacy event protocol produced by the Self-Forcing and
            # Causal-Forcing trace bridges; that protocol pins stride to 1560.
            if self.anchor_bits != 4 or self.group_size != 64:
                raise ValueError(
                    "legacy fitting requires anchor_bits=4 and group_size=64"
                )
            if self.residual_mode != "asymmetric":
                raise ValueError("legacy fitting requires asymmetric residuals")
            if (
                self.scale_dtype != "bfloat16"
                or self.reconstruction_dtype != "bfloat16"
            ):
                raise ValueError(
                    "legacy fitting requires bfloat16 scales and reconstruction"
                )

    def format_signature(self) -> str:
        """Return a stable identity for settings used during fitting."""
        payload = {
            "anchor_bits": self.anchor_bits,
            "cross_matmul_precision": self.cross_matmul_precision,
            "fit_mode": self.fit_mode,
            "group_size": self.group_size,
            "k_bits": self.k_bits,
            "k_predictor_mode": self.k_predictor_mode,
            "quantizer_version": "scale_floor_v1",
            "reconstruction_dtype": self.reconstruction_dtype,
            "residual_mode": self.residual_mode,
            "scale_dtype": self.scale_dtype,
            "stride": self.stride,
            "v_bits": self.v_bits,
            "v_predictor_mode": self.v_predictor_mode,
        }
        # History lengths join the identity only when they are set, so a
        # signature stays byte-identical to the one fitted parameters carry.
        if self.k_history_length is not None:
            payload["k_history_length"] = int(self.k_history_length)
        if self.v_history_length:
            payload["v_history_length"] = int(self.v_history_length)
        return "tempokv-codec-v5:" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
