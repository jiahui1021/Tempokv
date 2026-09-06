"""TempoKV integration for the external Rolling Forcing runtime.

The bridge is imported lazily so the stable ``inference.py --help`` entrypoint
can establish the repository ``src`` path before importing PyTorch/TempoKV.
"""

from typing import Any


def install(*args: Any, **kwargs: Any):
    from .bridge import install as _install

    return _install(*args, **kwargs)


__all__ = ["install"]
