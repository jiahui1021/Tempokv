from __future__ import annotations

import os
from pathlib import Path

from typing import Any, Mapping

from .codec import TempoKVCodec
from .config import CodecConfig, load_config
from .parameters import TemporalParameters, load_parameters
from .runtime import configure_triton_cache


# CrossKV layouts each fitting family may deploy.  scripts/common/codec_slug.py
# repeats this table without importing tempokv; keep them in step.
DEPLOYABLE_CROSS_KV_WEIGHT_MODES: dict[str, tuple[str, ...]] = {
    "legacy": ("layer_shared", "layer_head"),
    "native": ("layer_head",),
}


def check_deployable_cross_kv_layout(config: CodecConfig) -> None:
    """Refuse a CrossKV layout that no production parameter set uses."""
    allowed = DEPLOYABLE_CROSS_KV_WEIGHT_MODES[config.fit_mode]
    if config.cross_kv_weight_mode not in allowed:
        raise ValueError(
            f"{config.fit_mode} CrossKV parameters must use "
            f"cross_kv_weight_mode in {list(allowed)}, got "
            f"{config.cross_kv_weight_mode}"
        )


def load_codec(
    config_path: str | Path,
    k_parameters_path: str | Path,
    v_parameters_path: str | Path,
    *,
    triton_cache: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> TempoKVCodec:
    """Load a codec from a JSON config (plus overrides) and fitted parameters.

    The parameter arguments are by role: temporal predictor first, cross
    projection second, which under ``v_to_k`` are V then K.
    """
    config = load_config(config_path, overrides)
    check_deployable_cross_kv_layout(config)
    if config.use_triton:
        configure_triton_cache(triton_cache)
    temporal = TemporalParameters.load(k_parameters_path)
    cross_kv = load_parameters(v_parameters_path)
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
