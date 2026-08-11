from __future__ import annotations

import os
import tempfile
from pathlib import Path


PERSISTENT_UNITS_ENV = "TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH"


def persistent_units_per_launch(value: str | int | None = None) -> int:
    """Resolve the recurrent unit batch used by persistent decode kernels."""
    raw = os.environ.get(PERSISTENT_UNITS_ENV, "16") if value is None else value
    try:
        units = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{PERSISTENT_UNITS_ENV} must be an integer from 1 to 16"
        ) from error
    if isinstance(raw, bool) or not 1 <= units <= 16:
        raise ValueError(
            f"{PERSISTENT_UNITS_ENV} must be an integer from 1 to 16"
        )
    return units


def configure_triton_cache(
    root: str | Path | None = None,
    *,
    rank: int | None = None,
) -> Path:
    """Give each rank a Triton cache directory that survives across runs.

    Call this before the first accelerated decode in multi-process jobs.
    Existing ``TRITON_CACHE_DIR`` settings are respected.

    The directory is keyed by rank only.  Keying it by process id as well
    would give every run a cold cache, so each generation would pay the full
    kernel compilation cost again; Triton publishes compiled kernels by
    atomic rename, so concurrent ranks can share one directory safely.
    """
    existing = os.environ.get("TRITON_CACHE_DIR")
    if existing:
        return Path(existing).expanduser().resolve()
    base = Path(root) if root is not None else Path(tempfile.gettempdir()) / "tempokv-triton"
    process_rank = rank
    if process_rank is None:
        process_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    path = (base / f"rank-{process_rank}").expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(path)
    return path
