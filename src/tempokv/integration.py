from __future__ import annotations

import json
import os
from pathlib import Path

from .codec import TempoKVCodec
from typing import Any, Mapping

from .config import CodecConfig, config_from_mapping
from .parameters import (
    CrossKVParameters,
    TemporalParameters,
    load_parameters,
)
from .runtime import configure_triton_cache


def check_parameters_match_config(
    temporal: TemporalParameters,
    cross_kv: CrossKVParameters,
    config: CodecConfig,
) -> None:
    """Refuse parameters that were fitted for a different codec configuration.

    Every fitted file is stamped with the signature of the config it was fitted
    under, so running INT2-fitted predictors against an INT4 config — the kind
    of mix-up an overwritten parameter directory produces — fails here instead
    of quietly producing wrong numbers.

    Hand-built parameters carry no stamp; those are left alone, since there is
    nothing to compare against.
    """
    expected = config.format_signature()
    for name, parameters in (("K", temporal), ("V", cross_kv)):
        signature = parameters.codec_signature
        if signature is not None and signature != expected:
            raise ValueError(
                f"{name} parameters were fitted for codec {signature!r}, but this "
                f"run is configured as {expected!r}; refit them or point at the "
                "parameter directory that matches this config"
            )
    if temporal.parameter_set_id != cross_kv.parameter_set_id:
        # One stamped and one unstamped file means a freshly fitted parameter
        # was paired with a leftover from an older run. Both being unstamped is
        # a legitimate hand-built pair and compares equal here.
        raise ValueError(
            "K and V parameters come from different fitting runs "
            f"({temporal.parameter_set_id!r} vs {cross_kv.parameter_set_id!r}); "
            "use the k_temporal.npz and v_cross_kv.npz written by the same fit"
        )
    if (
        temporal.coordinate_system is not None
        and cross_kv.coordinate_system is not None
        and temporal.coordinate_system != cross_kv.coordinate_system
    ):
        raise ValueError(
            "K and V parameters use different coordinate systems: "
            f"{temporal.coordinate_system!r} vs {cross_kv.coordinate_system!r}"
        )


def load_codec(
    config_path: str | Path,
    k_parameters_path: str | Path,
    v_parameters_path: str | Path,
    *,
    triton_cache: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> TempoKVCodec:
    """Load a model-ready codec from JSON and fitted parameter files.

    ``overrides`` replaces individual codec settings after the file is read, so
    one committed config can serve every bit width instead of needing a second
    file per combination.  The override participates in the codec signature, so
    a mismatched parameter directory still fails the check below.
    """
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("codec config must be a JSON object")
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload["codec"], dict):
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    if overrides:
        payload = {**payload, **overrides}
    config = config_from_mapping(payload)
    if config.use_triton:
        configure_triton_cache(triton_cache)
    temporal = TemporalParameters.load(k_parameters_path)
    cross_kv = load_parameters(v_parameters_path)
    check_parameters_match_config(temporal, cross_kv, config)
    return TempoKVCodec(temporal, cross_kv, config)


def load_codec_from_env(prefix: str = "TEMPOKV_") -> TempoKVCodec:
    """Load the codec passed to a model by ``run_video_pipeline.py``."""
    required = {
        "CONFIG_PATH": os.environ.get(f"{prefix}CONFIG_PATH"),
        "K_PARAMS_PATH": os.environ.get(f"{prefix}K_PARAMS_PATH"),
        "V_PARAMS_PATH": os.environ.get(f"{prefix}V_PARAMS_PATH"),
    }
    missing = [f"{prefix}{name}" for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"missing environment variables: {', '.join(missing)}")
    return load_codec(
        required["CONFIG_PATH"],
        required["K_PARAMS_PATH"],
        required["V_PARAMS_PATH"],
        triton_cache=os.environ.get(f"{prefix}TRITON_CACHE"),
    )
