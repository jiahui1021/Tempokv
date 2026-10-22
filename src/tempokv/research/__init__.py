"""Independent S1-S3/E1-E7 research infrastructure, disabled by default."""
from .config import ResearchConfig, load_config
from .context import ProbeContext, FrozenAttention
__all__ = ["ResearchConfig", "load_config", "ProbeContext", "FrozenAttention"]
