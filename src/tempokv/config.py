from __future__ import annotations

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        for name in ("k_bits", "v_bits", "anchor_bits"):
            value = int(getattr(self, name))
            if value not in (2, 4, 8):
                raise ValueError(f"{name} must be 2, 4, or 8, got {value}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.residual_mode not in {"symmetric", "asymmetric"}:
            raise ValueError("residual_mode must be symmetric or asymmetric")
        supported_dtypes = {"float32", "float16", "bfloat16"}
        if self.scale_dtype not in supported_dtypes:
            raise ValueError("scale_dtype must be float32, float16, or bfloat16")
        if self.reconstruction_dtype not in supported_dtypes:
            raise ValueError(
                "reconstruction_dtype must be float32, float16, or bfloat16"
            )
