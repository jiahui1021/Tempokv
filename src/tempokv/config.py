from __future__ import annotations

from dataclasses import dataclass
import json


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
    # Shared parameter files can serve any active length up to these maxima.
    k_max_history_length: int | None = None
    v_max_history_length: int | None = None
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
        if (
            self.k_max_history_length is not None
            or self.v_max_history_length is not None
        ):
            object.__setattr__(
                self,
                "k_max_history_length",
                self.k_max_history_length
                if self.k_max_history_length is not None
                else self.k_history_length or 3,
            )
            object.__setattr__(
                self,
                "v_max_history_length",
                self.v_max_history_length
                if self.v_max_history_length is not None
                else self.v_history_length,
            )
        if self.k_max_history_length is not None and (
            isinstance(self.k_max_history_length, bool)
            or int(self.k_max_history_length) != self.k_max_history_length
            or not 1 <= int(self.k_max_history_length) <= 8
        ):
            raise ValueError(
                "k_max_history_length must be an integer from 1 to 8 or null"
            )
        if self.v_max_history_length is not None and (
            isinstance(self.v_max_history_length, bool)
            or int(self.v_max_history_length) != self.v_max_history_length
            or not 0 <= int(self.v_max_history_length) <= 5
        ):
            raise ValueError("v_max_history_length must be an integer from 0 to 5, or null")
        if (
            self.k_history_length is not None
            and self.k_max_history_length is not None
            and self.k_history_length > self.k_max_history_length
        ):
            raise ValueError("k_history_length must not exceed k_max_history_length")
        if (
            self.v_max_history_length is not None
            and self.v_history_length > self.v_max_history_length
        ):
            raise ValueError("v_history_length must not exceed v_max_history_length")
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
            if (
                self.k_max_history_length is not None
                or self.v_max_history_length is not None
            ):
                raise ValueError(
                    "legacy fitting does not use shared maximum-history parameters"
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
            if self.anchor_bits != 4 or self.group_size != 64 or self.stride != 1560:
                raise ValueError(
                    "legacy fitting requires anchor_bits=4, group_size=64, "
                    "and stride=1560"
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
            if self.v_history_length and self.v_predictor_mode == "cross_kv" and self.strict_triton:
                raise ValueError(
                    "legacy Cross+previous-V decodes in Torch and has no "
                    "Triton path; use v_predictor_mode=temporal for one"
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
        if self.k_history_length is not None:
            payload["k_history_length"] = int(self.k_history_length)
            if self.v_history_length:
                payload["v_history_length"] = int(self.v_history_length)
            version = "v5"
        elif self.v_history_length:
            payload["v_history_length"] = int(self.v_history_length)
            version = "v5"
        else:
            version = "v5"
        return f"tempokv-codec-{version}:" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )

    def shared_format_signature(self) -> str:
        """Identity for parameters reusable across active history lengths."""
        payload = {
            "anchor_bits": self.anchor_bits,
            "cross_matmul_precision": self.cross_matmul_precision,
            "fit_mode": self.fit_mode,
            "group_size": self.group_size,
            "k_bits": self.k_bits,
            "k_max_history_length": self.k_max_history_length,
            "k_predictor_mode": self.k_predictor_mode,
            "quantizer_version": "scale_floor_v1",
            "reconstruction_dtype": self.reconstruction_dtype,
            "residual_mode": self.residual_mode,
            "scale_dtype": self.scale_dtype,
            "stride": self.stride,
            "v_bits": self.v_bits,
            "v_max_history_length": self.v_max_history_length,
            "v_predictor_mode": self.v_predictor_mode,
        }
        return "tempokv-codec-v6:" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
