from __future__ import annotations

import json
import os
from pathlib import Path

from .codec import TempoKVCodec
from .config import CodecConfig
from .parameters import CrossKVParameters, TemporalParameters
from .runtime import configure_triton_cache


def load_codec(
    config_path: str | Path,
    k_parameters_path: str | Path,
    v_parameters_path: str | Path,
    *,
    triton_cache: str | Path | None = None,
) -> TempoKVCodec:
    """Load a model-ready codec from JSON and fitted parameter files."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("codec config must be a JSON object")
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload["codec"], dict):
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    config = CodecConfig(**payload)
    if config.use_triton:
        configure_triton_cache(triton_cache)
    return TempoKVCodec(
        TemporalParameters.load(k_parameters_path),
        CrossKVParameters.load(v_parameters_path),
        config,
    )


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
