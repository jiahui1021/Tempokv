"""Suite B: independent spectral summaries and AR-video diagnostics."""
from .config import SpectralKVConfig, load_config
from .method_spec import MethodSpec, parse_method

__all__ = ["SpectralKVConfig", "load_config", "MethodSpec", "parse_method"]
