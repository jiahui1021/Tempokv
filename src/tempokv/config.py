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
    if payload.get("fit_mode") == "native":
        raise ValueError(
            "native fitting family is no longer supported; use fit_mode=legacy"
        )
    withdrawn = sorted(set(payload) & set(WITHDRAWN_SETTINGS))
    if withdrawn:
        raise ValueError(
            "this codec config uses settings that no longer exist: "
            + "; ".join(
                f"{name} ({WITHDRAWN_SETTINGS[name]})" for name in withdrawn
            )
            + " — remove them and refit if the parameters were shared"
        )
    resolved = dict(payload)
    # There is one repository-wide codec family now.  Keep omitted fields
    # deterministic so hand-written configs cannot silently fall back to the
    # retired native/layer-head identity.
    resolved.setdefault("fit_mode", "legacy")
    resolved.setdefault("k_predictor_mode", "shared")
    resolved.setdefault("cross_kv_weight_mode", "layer_shared")
    return CodecConfig(**resolved)


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
    # None keeps parameter-driven K history selection for predictor modes that
    # support it. Direct K spells out zero explicitly.
    k_history_length: int | None = None
    # Legacy is the repository-wide fitting family.  ``None`` is retained for
    # positional compatibility with older programmatic callers; reference
    # construction resolves it to legacy below.
    cross_matmul_precision: str | None = None
    # ``shared`` pools all layers and heads into [P,P,D].
    # ``direct`` is the zero-history baseline: non-anchor K units are quantized
    # directly and the parameter file contains an empty temporal map.
    k_predictor_mode: str | None = None
    # ``temporal`` predicts V from its own reconstructed history with the same
    # predictor K uses, so V needs no reconstructed K and no CrossKV matrix.
    # ``cross_kv`` keeps the Cross(K) projection. ``innovation`` projects the
    # decoder-visible K temporal innovation and optionally adds V history.
    v_predictor_mode: str = "cross_kv"
    # Select the legacy fixed-point/shared-K and direct temporal-V fitting math.
    fit_mode: str | None = None
    # Production CrossKV always uses layer_shared.  The other values remain
    # available only to the explicit CrossKV sharing ablation.  ``None`` is
    # retained for positional compatibility and resolves to layer_shared for
    # legacy.
    cross_kv_weight_mode: str | None = None

    def __post_init__(self) -> None:
        # Resolve the remaining legacy defaults after the dataclass has kept
        # positional compatibility with older callers.  Reference-shaped
        # construction is the repository default.
        implicit_family = self.fit_mode is None
        if implicit_family:
            legacy_shape = (
                self.anchor_bits == 4
                and self.group_size == 64
                and self.residual_mode == "asymmetric"
                and self.scale_dtype == "bfloat16"
                and self.reconstruction_dtype == "bfloat16"
            )
            resolved_fit_mode = "legacy" if legacy_shape else "native"
        else:
            resolved_fit_mode = self.fit_mode
        if resolved_fit_mode not in {"native", "legacy"}:
            raise ValueError("fit_mode must be native or legacy")
        resolved_k_predictor = self.k_predictor_mode
        if resolved_k_predictor is None:
            resolved_k_predictor = (
                "shared" if resolved_fit_mode == "legacy" else "layer_head"
            )
        resolved_precision = self.cross_matmul_precision
        if resolved_precision is None:
            resolved_precision = "ieee" if resolved_fit_mode == "legacy" else "tf32"
        resolved_cross_mode = self.cross_kv_weight_mode
        if resolved_cross_mode is None:
            resolved_cross_mode = (
                "layer_shared" if resolved_fit_mode == "legacy" else "layer_head"
            )
        object.__setattr__(self, "fit_mode", resolved_fit_mode)
        object.__setattr__(self, "k_predictor_mode", resolved_k_predictor)
        object.__setattr__(self, "cross_matmul_precision", resolved_precision)
        object.__setattr__(self, "cross_kv_weight_mode", resolved_cross_mode)
        if resolved_fit_mode == "legacy" and self.k_history_length is None:
            object.__setattr__(self, "k_history_length", 2)

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
            or int(self.k_history_length) < 0
            or int(self.k_history_length) > 8
        ):
            raise ValueError("k_history_length must be an integer from 0 to 8 or null")
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
        if self.k_predictor_mode not in {"layer_head", "shared", "direct"}:
            raise ValueError(
                "k_predictor_mode must be layer_head, shared, or direct"
            )
        if self.k_predictor_mode == "direct":
            if self.k_history_length != 0:
                raise ValueError(
                    "k_predictor_mode=direct requires k_history_length=0"
                )
        elif self.k_history_length == 0:
            raise ValueError(
                "k_history_length=0 requires k_predictor_mode=direct"
            )
        if self.v_predictor_mode not in {"cross_kv", "temporal", "innovation"}:
            raise ValueError(
                "v_predictor_mode must be cross_kv, temporal, or innovation"
            )
        if self.cross_kv_weight_mode not in {
            "layer_head",
            "layer_shared",
            "head_shared",
            "shared",
        }:
            raise ValueError(
                "cross_kv_weight_mode must be layer_head, layer_shared, "
                "head_shared, or shared"
            )
        # The temporal V predictor reads only its own reconstructed history, so
        # with no history it has nothing to predict from.  Say so here; both
        # fitters would otherwise fail on a zero-length parameter axis.
        if self.v_predictor_mode == "temporal" and self.v_history_length < 1:
            raise ValueError(
                "v_predictor_mode=temporal needs v_history_length >= 1"
            )
        if self.v_predictor_mode == "innovation":
            if self.fit_mode != "legacy":
                raise ValueError("v_predictor_mode=innovation requires fit_mode=legacy")
            if self.k_predictor_mode != "shared":
                raise ValueError(
                    "v_predictor_mode=innovation requires k_predictor_mode=shared"
                )
            if self.cross_kv_weight_mode != "layer_shared":
                raise ValueError(
                    "v_predictor_mode=innovation requires "
                    "cross_kv_weight_mode=layer_shared"
                )
        if (
            self.fit_mode == "legacy"
            and self.v_predictor_mode == "temporal"
            and self.cross_kv_weight_mode != "layer_shared"
        ):
            raise ValueError(
                "temporal V configurations must use "
                "cross_kv_weight_mode=layer_shared"
            )
        if self.fit_mode not in {"native", "legacy"}:
            raise ValueError("fit_mode must be native or legacy")
        if self.cross_kv_weight_mode != "layer_head" and self.fit_mode != "legacy":
            raise ValueError(
                f"cross_kv_weight_mode={self.cross_kv_weight_mode} "
                "requires fit_mode=legacy; native fitting does not support "
                "shared CrossKV weights"
            )
        if (
            self.k_predictor_mode == "shared"
            and self.fit_mode != "legacy"
        ):
            raise ValueError(
                "k_predictor_mode=shared requires fit_mode=legacy"
            )
        if self.fit_mode == "legacy":
            if self.k_predictor_mode not in {"shared", "direct"}:
                raise ValueError(
                    "legacy fitting requires k_predictor_mode=shared or direct"
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
            # K and V are packed and decoded independently.  Legacy keeps its
            # shared predictor and reference arithmetic, but does not require
            # the two cache tensors to use the same residual bit width.
            # anchor_bits and group_size fix the quantizer's bit layout, so the
            # reference implementation is only reproduced at these values.
            # stride is not one of them: it only decides where the encoder cuts
            # units. Self-Forcing and Causal-Forcing use the fixed legacy event
            # protocol at stride 1560; LongCat uses its regular model-shaped
            # trace and records the matching runtime stride in that archive.
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
        # Omitting the default keeps signatures byte-identical to parameters
        # fitted before this experiment switch existed.
        if self.cross_kv_weight_mode != "layer_head":
            payload["cross_kv_weight_mode"] = self.cross_kv_weight_mode
        return "tempokv-codec-v5:" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
