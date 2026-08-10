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

    def format_signature(self) -> str:
        """Return a stable identity for settings used during fitting."""
        payload = {
            "anchor_bits": self.anchor_bits,
            "group_size": self.group_size,
            "k_bits": self.k_bits,
            "reconstruction_dtype": self.reconstruction_dtype,
            "residual_mode": self.residual_mode,
            "scale_dtype": self.scale_dtype,
            "stride": self.stride,
            "v_bits": self.v_bits,
        }
        if self.k_history_length is not None:
            payload["k_history_length"] = int(self.k_history_length)
            if self.v_history_length:
                payload["v_history_length"] = int(self.v_history_length)
            version = "v3"
        elif self.v_history_length:
            payload["v_history_length"] = int(self.v_history_length)
            version = "v2"
        else:
            # Preserve the exact signature used by legacy CrossKV-only files.
            version = "v1"
        return f"tempokv-codec-{version}:" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )

    def shared_format_signature(self) -> str:
        """Identity for parameters reusable across active history lengths."""
        payload = {
            "anchor_bits": self.anchor_bits,
            "group_size": self.group_size,
            "k_bits": self.k_bits,
            "k_max_history_length": self.k_max_history_length,
            "reconstruction_dtype": self.reconstruction_dtype,
            "residual_mode": self.residual_mode,
            "scale_dtype": self.scale_dtype,
            "stride": self.stride,
            "v_bits": self.v_bits,
            "v_max_history_length": self.v_max_history_length,
        }
        return "tempokv-codec-v4:" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
