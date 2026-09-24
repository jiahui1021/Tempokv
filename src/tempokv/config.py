from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

# Storing the anchor as INT16 codes costs 16 bits per element *plus* the group
# scale — more than the bfloat16 value it replaces — and is still lossy, so it
# is dominated by simply keeping the tensor.  ``anchor_bits=16`` therefore does
# not mean INT16: it means the anchor skips the quantizer and is stored in
# ``reconstruction_dtype``, which is exactly 16 bits per element and exact.
RAW_ANCHOR_BITS = 16

# Fixed temporal predictors are used symmetrically for K and V.  CrossKV V
# variants add the source projection while retaining the selected fixed V
# history recurrence.
FIXED_TEMPORAL_PREDICTOR_MODES = frozenset({"identity", "taylor", "taylor2"})
FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY = {"identity": 1, "taylor": 2, "taylor2": 3}
FIXED_CROSS_KV_V_PREDICTOR_MODES = frozenset(
    {f"cross_kv_{mode}" for mode in FIXED_TEMPORAL_PREDICTOR_MODES}
)
TEMPORAL_V_PREDICTOR_MODES = frozenset(
    {"temporal", "direct", *FIXED_TEMPORAL_PREDICTOR_MODES}
)
CROSS_KV_V_PREDICTOR_MODES = frozenset(
    {"cross_kv", "innovation", *FIXED_CROSS_KV_V_PREDICTOR_MODES}
)
V_PREDICTOR_MODES = TEMPORAL_V_PREDICTOR_MODES | CROSS_KV_V_PREDICTOR_MODES
K_PREDICTOR_MODES = frozenset(
    {"layer_head", "shared", "direct", *FIXED_TEMPORAL_PREDICTOR_MODES}
)
CROSS_KV_WEIGHT_MODES = frozenset({"layer_head", "layer_shared", "head_shared", "shared"})
DTYPES = frozenset({"float32", "float16", "bfloat16"})


def is_temporal_v_predictor(mode: str) -> bool:
    """Return whether a V mode uses ``TemporalParameters`` at runtime."""
    return mode in TEMPORAL_V_PREDICTOR_MODES


def is_cross_kv_v_predictor(mode: str) -> bool:
    """Return whether a V mode consumes the reconstructed K source."""
    return mode in CROSS_KV_V_PREDICTOR_MODES


def is_fixed_temporal_predictor(mode: str) -> bool:
    """Return whether a predictor uses deterministic temporal coefficients."""
    return mode in FIXED_TEMPORAL_PREDICTOR_MODES


def is_fixed_v_predictor(mode: str) -> bool:
    """Return whether a V mode has deterministic temporal coefficients."""
    return fixed_temporal_predictor_mode(mode) is not None


def fixed_temporal_predictor_mode(mode: str) -> str | None:
    """Return the base fixed temporal family represented by a V mode."""
    if mode in FIXED_TEMPORAL_PREDICTOR_MODES:
        return mode
    if mode in FIXED_CROSS_KV_V_PREDICTOR_MODES:
        return mode.removeprefix("cross_kv_")
    return None


def config_from_mapping(payload: Mapping[str, Any]) -> "CodecConfig":
    """Build a codec config from parsed JSON.

    A config that does not name a fitting family is legacy, because every
    shipped config and fitted parameter set is; the remaining defaults then
    follow the family, so writing ``fit_mode`` is enough to pick one.
    """
    resolved = dict(payload)
    resolved.setdefault("fit_mode", "legacy")
    if resolved["fit_mode"] == "legacy":
        resolved.setdefault("k_predictor_mode", "shared")
        resolved.setdefault("cross_kv_weight_mode", "layer_shared")
    else:
        resolved.setdefault("k_predictor_mode", "layer_head")
        resolved.setdefault("cross_kv_weight_mode", "layer_head")
    return CodecConfig(**resolved)


def read_config_payload(path: str | Path) -> dict[str, Any]:
    """Read a codec config JSON file, unwrapping an optional ``codec`` object."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload.get("codec", payload))


def load_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> "CodecConfig":
    """Load a codec config file, with individual settings optionally replaced."""
    return config_from_mapping({**read_config_payload(path), **(overrides or {})})


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

        # JSON can spell an integer as ``4.0``; coerce so the signature matches
        # the identical integer config.
        for name in ("k_bits", "v_bits", "anchor_bits", "group_size", "stride",
                     "v_history_length", "k_history_length"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or int(value) != value:
                    raise ValueError(f"{name} must be a whole number, got {value!r}")
                object.__setattr__(self, name, int(value))
        self._validate()

    def _validate(self) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        require(1 <= self.k_bits <= 8 and 1 <= self.v_bits <= 8,
                "k_bits and v_bits must be from 1 to 8")
        # 1 through 15 are quantized anchor codes; 16 is the unquantized anchor
        # described at RAW_ANCHOR_BITS, which needs a 16-bit storage dtype.
        require(1 <= self.anchor_bits <= RAW_ANCHOR_BITS,
                f"anchor_bits must be from 1 to {RAW_ANCHOR_BITS}")
        require(self.group_size > 0 and self.stride > 0,
                "group_size and stride must be positive")
        require(0 <= self.v_history_length <= 5, "v_history_length must be from 0 to 5")
        require(self.k_history_length is None or 0 <= self.k_history_length <= 8,
                "k_history_length must be from 0 to 8 or null")
        require(self.residual_mode in {"symmetric", "asymmetric"},
                "residual_mode must be symmetric or asymmetric")
        require(self.scale_dtype in DTYPES and self.reconstruction_dtype in DTYPES,
                "scale_dtype and reconstruction_dtype must be float32, float16 or bfloat16")
        require(not self.raw_anchor or self.reconstruction_dtype != "float32",
                "anchor_bits=16 stores the anchor unquantized in reconstruction_dtype, "
                "so it needs a 16-bit reconstruction_dtype")
        require(self.cross_matmul_precision in {"ieee", "tf32"},
                "cross_matmul_precision must be ieee or tf32")
        require(self.k_predictor_mode in K_PREDICTOR_MODES,
                f"k_predictor_mode must be one of {sorted(K_PREDICTOR_MODES)}")
        require((self.k_predictor_mode == "direct") == (self.k_history_length == 0),
                "k_predictor_mode=direct and k_history_length=0 go together")
        require(self.v_predictor_mode in V_PREDICTOR_MODES,
                f"v_predictor_mode must be one of {sorted(V_PREDICTOR_MODES)}")
        require(self.v_predictor_mode != "direct" or not self.v_history_length,
                "v_predictor_mode=direct requires v_history_length=0")
        require(self.cross_kv_weight_mode in CROSS_KV_WEIGHT_MODES,
                f"cross_kv_weight_mode must be one of {sorted(CROSS_KV_WEIGHT_MODES)}")
        require(self.predictor_direction in {"k_to_v", "v_to_k"},
                "predictor_direction must be k_to_v or v_to_k")
        # The mirror only means something when there is a cross term to point
        # the other way; every other V mode ignores the source tensor.
        require(self.predictor_direction == "k_to_v" or self.v_predictor_mode == "cross_kv",
                "predictor_direction=v_to_k requires v_predictor_mode=cross_kv")
        # A V predictor that reads its own history has nothing to consume when
        # the history is empty, and a fixed Taylor mode needs enough history for
        # its finite-difference order; silently falling back would make the
        # named ablation misleading.
        require(not is_temporal_v_predictor(self.v_predictor_mode)
                or self.v_predictor_mode == "direct" or self.v_history_length >= 1,
                f"v_predictor_mode={self.v_predictor_mode} needs v_history_length >= 1")
        for role, mode, history in (
            ("k", self.k_predictor_mode, self.k_history_length),
            ("v", fixed_temporal_predictor_mode(self.v_predictor_mode), self.v_history_length),
        ):
            minimum = FIXED_TEMPORAL_PREDICTOR_MIN_HISTORY.get(mode)
            require(minimum is None or history is None or history >= minimum,
                    f"{role}_predictor_mode={mode} needs {role}_history_length >= {minimum}")

        if self.fit_mode == "native":
            require(self.cross_kv_weight_mode == "layer_head",
                    "native fitting only supports cross_kv_weight_mode=layer_head")
            require(self.k_predictor_mode != "shared",
                    "k_predictor_mode=shared requires fit_mode=legacy")
            require(self.v_predictor_mode != "innovation",
                    "v_predictor_mode=innovation requires fit_mode=legacy")
            return
        # Legacy reproduces the reference implementation, which fixes the
        # quantizer grouping and dtypes; bit widths, anchor width and stride
        # stay free.
        require(self.k_history_length is not None,
                "legacy fitting requires an explicit k_history_length")
        require(self.group_size == 64, "legacy fitting requires group_size=64")
        require(self.residual_mode == "asymmetric",
                "legacy fitting requires asymmetric residuals")
        require(self.scale_dtype == "bfloat16" and self.reconstruction_dtype == "bfloat16",
                "legacy fitting requires bfloat16 scales and reconstruction")
        if self.v_predictor_mode == "innovation":
            require(self.k_predictor_mode == "shared",
                    "v_predictor_mode=innovation requires k_predictor_mode=shared")
        # Only CrossKV V reads a per-layer matrix; every other legacy V
        # predictor uses the layer-shared layout.
        if self.v_predictor_mode not in {"cross_kv", "direct"}:
            require(self.cross_kv_weight_mode == "layer_shared",
                    "temporal V configurations must use cross_kv_weight_mode=layer_shared")

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
