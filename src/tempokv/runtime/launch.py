from __future__ import annotations

import os
import tempfile
from pathlib import Path


PERSISTENT_UNITS_ENV = "TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH"


def persistent_units_per_launch(value: str | int | None = None) -> int:
    """Resolve the recurrent unit batch used by persistent decode kernels."""
    units = int(os.environ.get(PERSISTENT_UNITS_ENV, "16") if value is None else value)
    if not 1 <= units <= 16:
        raise ValueError(f"{PERSISTENT_UNITS_ENV} must be an integer from 1 to 16")
    return units


def persistent_launch_plan(
    units: int, units_per_launch: int, lead_units: int
) -> list[tuple[int, int, int]]:
    """Split units ``1 .. units-1`` of a span into persistent-kernel launches.

    Each entry is ``(shift_units, start_unit, chunk_units)``.  Units past
    ``lead_units`` are shifted onto one ``start_unit`` so later launches reuse one
    compiled kernel; the per-unit arithmetic is unchanged.
    """
    plan = []
    for start in range(1, units, units_per_launch):
        chunk = min(units_per_launch, units - start)
        if start <= lead_units:
            plan.append((0, start, chunk))
        else:
            plan.append((start - lead_units, lead_units, chunk))
    return plan


def configure_triton_cache(
    root: str | Path | None = None,
    *,
    rank: int | None = None,
) -> Path:
    """Give each rank a persistent Triton cache directory (respects ``TRITON_CACHE_DIR``)."""
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
