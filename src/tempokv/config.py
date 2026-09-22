from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

# Storing the anchor as INT16 codes costs 16 bits per element *plus* the group
# scale — more than the bfloat16 value it replaces — and is still lossy, so it
# is dominated by simply keeping the tensor.  ``anchor_bits=16`` therefore does
# not mean INT16: it means the anchor skips the quantizer and is stored in
# ``reconstruction_dtype``, which is exactly 16 bits per element and exact.
RAW_ANCHOR_BITS = 16

# The dtypes for which "16 bits" is the honest description of a stored value.
RAW_ANCHOR_DTYPES = frozenset({"bfloat16", "float16"})


# Settings that existed in an earlier format and no longer do.  A hand-written
# config, or a ``runtime_config.json`` stored alongside an older run, would
# otherwise fail with an unexpected-keyword ``TypeError`` that does not say what
# to remove.
WITHDRAWN_SETTINGS = {
    "k_max_history_length": "shared-history K parameters were removed",
    "v_max_history_length": "shared-history V parameters were removed",
}


# Fixed temporal predictors are used symmetrically for K and V.  CrossKV V
# variants add the source projection while retaining the selected fixed V
# history recurrence.
FIXED_TEMPORAL_PREDICTOR_MODES = frozenset({"identity", "taylor", "taylor2"})
FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY = {"identity": 1, "taylor": 2, "taylor2": 3}
FIXED_TEMPORAL_V_PREDICTOR_MODES = FIXED_TEMPORAL_PREDICTOR_MODES
FIXED_CROSS_KV_V_PREDICTOR_MODES = frozenset(
    {f"cross_kv_{mode}" for mode in FIXED_TEMPORAL_PREDICTOR_MODES}
)
FIXED_V_PREDICTOR_MODES = (
    FIXED_TEMPORAL_V_PREDICTOR_MODES | FIXED_CROSS_KV_V_PREDICTOR_MODES
)
TEMPORAL_V_PREDICTOR_MODES = frozenset(
    {"temporal", "direct", *FIXED_TEMPORAL_V_PREDICTOR_MODES}
)
CROSS_KV_V_PREDICTOR_MODES = frozenset(
    {"cross_kv", "innovation", *FIXED_CROSS_KV_V_PREDICTOR_MODES}
)
V_PREDICTOR_MODES = frozenset(
    {*TEMPORAL_V_PREDICTOR_MODES, *CROSS_KV_V_PREDICTOR_MODES}
)


def is_temporal_v_predictor(mode: str) -> bool:
    """Return whether a V mode uses ``TemporalParameters`` at runtime."""
    return mode in TEMPORAL_V_PREDICTOR_MODES


def is_cross_kv_v_predictor(mode: str) -> bool:
    """Return whether a V mode consumes the reconstructed K source."""
    return mode in CROSS_KV_V_PREDICTOR_MODES


def is_fixed_temporal_predictor(mode: str) -> bool:
    """Return whether a predictor uses deterministic temporal coefficients."""
    return mode in FIXED_TEMPORAL_PREDICTOR_MODES


def is_fixed_temporal_v_predictor(mode: str) -> bool:
    """Return whether a V mode has deterministic, non-fitted coefficients."""
    return mode in FIXED_TEMPORAL_V_PREDICTOR_MODES


def is_fixed_v_predictor(mode: str) -> bool:
    """Return whether a V mode has deterministic temporal coefficients."""
    return mode in FIXED_V_PREDICTOR_MODES


def fixed_temporal_predictor_mode(mode: str) -> str | None:
    """Return the base fixed temporal family represented by a V mode."""
    if mode in FIXED_TEMPORAL_PREDICTOR_MODES:
        return mode
    if mode in FIXED_CROSS_KV_V_PREDICTOR_MODES:
        return mode.removeprefix("cross_kv_")
    return None


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
    resolved = dict(payload)
    # Both families are selectable.  A config that does not name one is legacy,
    # because that is what every shipped config and every fitted parameter set
    # uses; the remaining defaults then follow the family rather than being
    # guessed from unrelated settings, so writing ``fit_mode`` is enough to
    # pick a family and nothing else has to be restated.
    resolved.setdefault("fit_mode", "legacy")
    if resolved["fit_mode"] == "legacy":
        resolved.setdefault("k_predictor_mode", "shared")
        resolved.setdefault("cross_kv_weight_mode", "layer_shared")
    else:
        resolved.setdefault("k_predictor_mode", "layer_head")
        resolved.setdefault("cross_kv_weight_mode", "layer_head")
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
    # ``shared`` pools all layers and heads into [P,P,D]; ``layer_head`` fits
    # [L,P,P,H,D] (native, and legacy as a K sharing ablation).  The fixed
    # temporal modes use the shared layout but write analytic coefficients.
    # ``direct`` is the zero-history baseline: non-anchor K units are quantized
    # directly and the parameter file contains an empty temporal map.
    k_predictor_mode: str | None = None
    # ``temporal`` predicts V from its own reconstructed history with the same
    # predictor K uses, so V needs no reconstructed K and no CrossKV matrix.
    # ``identity`` is a fixed one-step hold; ``taylor`` and ``taylor2`` are
    # fixed first- and second-order backward finite-difference extrapolators.
    # ``cross_kv`` keeps the learned Cross(K) projection.  The
    # ``cross_kv_<fixed>`` modes add the same projection to a fixed V-history
    # predictor. ``innovation`` projects the decoder-visible K temporal
    # innovation and optionally adds V history.
    # ``direct`` is the zero-predictor baseline: V units are quantized directly
    # and the parameter file contains an empty temporal map.
    v_predictor_mode: str = "cross_kv"
    # Select the legacy fixed-point/shared-K and direct temporal-V fitting math.
    fit_mode: str | None = None
    # Production CrossKV always uses layer_shared.  The other values remain
    # available only to the explicit CrossKV sharing ablation.  ``None`` is
    # retained for positional compatibility and resolves to layer_shared for
    # legacy.
    cross_kv_weight_mode: str | None = None
    # Which tensor carries the cross dependency.  ``k_to_v`` is the production
    # design: K is predicted from its own temporal history and V is predicted
    # from the reconstructed K of the same span.  ``v_to_k`` mirrors it so V
    # leads and K is projected from the reconstructed V, which exists to test
    # the direction rather than to assume it.  The codec math is identical
    # under the swap, so the mirror is a role exchange and not a second codec.
    predictor_direction: str = "k_to_v"
    # Subtract the Jensen bias (arXiv 2605.26266) from quantized cached
    # attention scores.  It reads the step sizes the codec already stores and
    # changes no stored value, so K_hat, V_hat and every reconstruction metric
    # are identical either way and fitted parameters stay loadable -- which is
    # why it is deliberately absent from format_signature().  What it changes
    # is the attention output, and only for integrations that route their
    # attention through runtime.jensen.  Off by default: on this repository's
    # traces the bias at the production INT4 residual width is roughly two
    # orders of magnitude below the INT2 direct-quantization setting the paper
    # measures, and the end-to-end effect has not been demonstrated here.
    jensen_correction: bool = False

    def __post_init__(self) -> None:
        # Resolve the remaining legacy defaults after the dataclass has kept
        # positional compatibility with older callers.  Reference-shaped
        # construction is the repository default.
        implicit_family = self.fit_mode is None
        if implicit_family:
            # anchor_bits is deliberately absent: the anchor width is a codec
            # setting, not a family marker, so raising it must not silently
            # reinterpret a legacy config as the retired native family.
            legacy_shape = (
                self.group_size == 64
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

        # Coerce the integral settings before anything reads them.  A float
        # that happens to be whole (``4.0`` from JSON) used to survive into the
        # config, reach quantization or a reshape as a float, and produce a
        # different parameter signature from the identical integer config.
        integral = ["k_bits", "v_bits", "anchor_bits", "group_size", "stride",
                    "v_history_length"]
        if self.k_history_length is not None:
            integral.append("k_history_length")
        for name in integral:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be a whole number, got {value!r}")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.jensen_correction, bool):
            raise ValueError("jensen_correction must be a boolean")
        for name in ("k_bits", "v_bits"):
            value = getattr(self, name)
            if not 1 <= value <= 8:
                raise ValueError(f"{name} must be an integer from 1 to 8, got {value}")
        # The anchor is one span in every unit, and it is the only reconstruction
        # the residuals are predicted from, so trading its width against the
        # residual width is a real knob.  1 through 15 are quantized codes; 16
        # is the unquantized anchor described at RAW_ANCHOR_BITS.
        if not 1 <= self.anchor_bits <= RAW_ANCHOR_BITS:
            raise ValueError(
                f"anchor_bits must be an integer from 1 to {RAW_ANCHOR_BITS}, "
                f"got {self.anchor_bits}"
            )
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
        # Checked after the dtype itself is validated, so an unsupported dtype
        # is reported as the dtype problem it is rather than as an anchor
        # problem.
        if (
            self.anchor_bits == RAW_ANCHOR_BITS
            and self.reconstruction_dtype not in RAW_ANCHOR_DTYPES
        ):
            raise ValueError(
                "anchor_bits=16 stores the anchor unquantized in "
                "reconstruction_dtype, so it needs a 16-bit "
                f"reconstruction_dtype, not {self.reconstruction_dtype}"
            )
        if self.cross_matmul_precision not in {"ieee", "tf32"}:
            raise ValueError("cross_matmul_precision must be ieee or tf32")
        if self.k_predictor_mode not in {
            "layer_head",
            "shared",
            "direct",
            *FIXED_TEMPORAL_PREDICTOR_MODES,
        }:
            raise ValueError(
                "k_predictor_mode must be layer_head, shared, identity, taylor, "
                "taylor2, or direct"
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
        k_minimum_history = FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY.get(
            self.k_predictor_mode
        )
        if (
            k_minimum_history is not None
            and self.k_history_length is not None
            and self.k_history_length < k_minimum_history
        ):
            raise ValueError(
                f"k_predictor_mode={self.k_predictor_mode} needs "
                f"k_history_length >= {k_minimum_history}"
            )
        if self.v_predictor_mode not in V_PREDICTOR_MODES:
            raise ValueError(
                "v_predictor_mode must be cross_kv, temporal, identity, taylor, "
                "taylor2, cross_kv_identity, cross_kv_taylor, "
                "cross_kv_taylor2, innovation, or direct"
            )
        if self.v_predictor_mode == "direct" and self.v_history_length:
            raise ValueError(
                "v_predictor_mode=direct requires v_history_length=0 "
                "(pure quantization, no V predictor)"
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
        if self.predictor_direction not in {"k_to_v", "v_to_k"}:
            raise ValueError("predictor_direction must be k_to_v or v_to_k")
        if self.predictor_direction == "v_to_k":
            # The mirror only means something when there is a cross term to
            # point the other way; every other V mode ignores the source
            # tensor entirely, so the flag would silently do nothing.
            if self.v_predictor_mode != "cross_kv":
                raise ValueError(
                    "predictor_direction=v_to_k requires v_predictor_mode=cross_kv"
                )
            # Relaxed: asymmetric K/V widths under the mirror are allowed (the
            # caller owns the interpretation that bit budget no longer matches
            # 1:1 after the role exchange). cross_kv source is still required.
        # A V predictor that reads its own history has nothing to consume when
        # the history is empty.  Fixed Taylor modes also require enough history
        # to express the requested finite-difference order; silently falling
        # back to identity would make the named ablation misleading.
        if is_temporal_v_predictor(self.v_predictor_mode) and (
            self.v_predictor_mode != "direct" and self.v_history_length < 1
        ):
            raise ValueError(
                f"v_predictor_mode={self.v_predictor_mode} needs "
                "v_history_length >= 1"
            )
        minimum_history = FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY.get(
            fixed_temporal_predictor_mode(self.v_predictor_mode)
        )
        if minimum_history is not None and self.v_history_length < minimum_history:
            raise ValueError(
                f"v_predictor_mode={self.v_predictor_mode} needs "
                f"v_history_length >= {minimum_history}"
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
        requires_shared_cross = (
            self.v_predictor_mode in FIXED_CROSS_KV_V_PREDICTOR_MODES
            or self.v_predictor_mode == "innovation"
            or (
                is_temporal_v_predictor(self.v_predictor_mode)
                and self.v_predictor_mode != "direct"
            )
        )
        if (
            self.fit_mode == "legacy"
            and requires_shared_cross
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
            # ``layer_head`` keeps the legacy fitting math (FP64 statistics,
            # ridge toward persistence, IEEE arithmetic) but solves one K
            # predictor per layer and head instead of pooling them.
            if self.k_predictor_mode not in {
                "shared",
                "layer_head",
                "direct",
                *FIXED_TEMPORAL_PREDICTOR_MODES,
            }:
                raise ValueError(
                    "legacy fitting requires k_predictor_mode=shared, layer_head, "
                    "identity, taylor, taylor2, or direct"
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
            # group_size fixes the quantizer's grouping, so the reference
            # implementation is only reproduced at 64.  anchor_bits is not one
            # of them: it selects a width within the same quantizer, the
            # signature records it, and INT4 stays the default, so a wider or
            # narrower anchor is a configuration rather than a second codec.
            # stride is likewise free: it only decides where the encoder cuts
            # units. Self-Forcing and Causal-Forcing use the fixed legacy event
            # protocol at stride 1560; LongCat uses its regular model-shaped
            # trace and records the matching runtime stride in that archive.
            if self.group_size != 64:
                raise ValueError("legacy fitting requires group_size=64")
            if self.residual_mode != "asymmetric":
                raise ValueError("legacy fitting requires asymmetric residuals")
            if (
                self.scale_dtype != "bfloat16"
                or self.reconstruction_dtype != "bfloat16"
            ):
                raise ValueError(
                    "legacy fitting requires bfloat16 scales and reconstruction"
                )

    @property
    def raw_anchor(self) -> bool:
        """Whether the anchor skips the quantizer and is stored as-is."""
        return self.anchor_bits == RAW_ANCHOR_BITS

    @property
    def primary_tensor(self) -> str:
        """The tensor predicted from its own temporal history, "k" or "v"."""
        return "v" if self.predictor_direction == "v_to_k" else "k"

    @property
    def secondary_tensor(self) -> str:
        """The tensor predicted from the reconstructed primary tensor."""
        return "k" if self.predictor_direction == "v_to_k" else "v"

    @property
    def mirrored(self) -> bool:
        """Whether K and V exchange roles relative to the production design."""
        return self.predictor_direction == "v_to_k"

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
        # Same reasoning: the production direction is omitted so every existing
        # parameter archive keeps its signature, while a mirrored set can never
        # be loaded into a forward-direction runtime.
        if self.predictor_direction != "k_to_v":
            payload["predictor_direction"] = self.predictor_direction
        # anchor_bits=16 once meant INT16 codes and now means no quantizer at
        # all, so the width alone no longer identifies the anchor codec.  This
        # marker appears only for the raw anchor, which keeps every quantized
        # parameter set's signature byte-identical while making a set fitted
        # under the old INT16 reading fail to load instead of loading silently.
        if self.raw_anchor:
            payload["anchor_mode"] = "raw"
        return "tempokv-codec-v5:" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
