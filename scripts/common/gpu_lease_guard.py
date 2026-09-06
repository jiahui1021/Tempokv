#!/usr/bin/env python3
"""Fail closed before a GPU job can run outside its EPIC lease."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys


GPU_UUID = re.compile(r"GPU-[0-9A-Fa-f-]+\Z")


def _parse_devices(raw: str | None, name: str) -> list[str]:
    if not raw or not raw.strip():
        raise RuntimeError(
            f"{name} is required; set it to the current EPIC GPU UUID list "
            "(for example GPU-...,...), never numeric GPU indices"
        )
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise RuntimeError(f"{name} must be a non-empty list of unique GPU UUIDs")
    invalid = [item for item in devices if GPU_UUID.fullmatch(item) is None]
    if invalid:
        raise RuntimeError(
            f"{name} contains non-UUID device(s): {', '.join(invalid)}; "
            "numeric CUDA_VISIBLE_DEVICES values are rejected to prevent "
            "physical/logical lease mix-ups"
        )
    return devices


def _visible_gpu_uuids() -> set[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "nvidia-smi failed; refusing to start an unverified GPU job"
        ) from exc
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _parse_expiry(raw: str | None) -> datetime:
    if not raw or not raw.strip():
        raise RuntimeError(
            "EPIC_GPU_LEASE_EXPIRES_AT is required (ISO-8601 UTC); refusing to "
            "start without a lease expiry guard"
        )
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        expiry = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(
            "EPIC_GPU_LEASE_EXPIRES_AT must be an ISO-8601 timestamp"
        ) from exc
    if expiry.tzinfo is None:
        raise RuntimeError("EPIC_GPU_LEASE_EXPIRES_AT must include a timezone")
    return expiry.astimezone(timezone.utc)


def validate_gpu_lease(
    expected_count: int,
    *,
    environment: dict[str, str] | None = None,
    visible_uuids: set[str] | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Validate the explicit lease attestation used by GPU launchers."""

    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    environment = os.environ if environment is None else environment
    visible = _parse_devices(
        environment.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES"
    )
    attested = _parse_devices(environment.get("EPIC_GPU_UUIDS"), "EPIC_GPU_UUIDS")
    if visible != attested:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must exactly match EPIC_GPU_UUIDS in the same "
            f"lane order; got {visible} vs {attested}"
        )
    if len(visible) != expected_count:
        raise RuntimeError(
            f"lease exposes {len(visible)} GPU(s), but the job requests "
            f"{expected_count}"
        )
    available = _visible_gpu_uuids() if visible_uuids is None else visible_uuids
    missing = [device for device in visible if device not in available]
    if missing:
        raise RuntimeError(
            "leased GPU UUID(s) are not visible on this node: " + ", ".join(missing)
        )
    expiry = _parse_expiry(environment.get("EPIC_GPU_LEASE_EXPIRES_AT"))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    minimum_remaining = int(environment.get("EPIC_GPU_MIN_REMAINING_SECONDS", "900"))
    if minimum_remaining < 0:
        raise RuntimeError("EPIC_GPU_MIN_REMAINING_SECONDS must be non-negative")
    remaining = (expiry - current).total_seconds()
    if remaining < minimum_remaining:
        raise RuntimeError(
            f"EPIC GPU lease expires in {max(0, int(remaining))}s; need at least "
            f"{minimum_remaining}s before starting"
        )
    return visible


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} EXPECTED_GPU_COUNT", file=sys.stderr)
        return 2
    try:
        expected_count = int(sys.argv[1])
        devices = validate_gpu_lease(expected_count)
    except (RuntimeError, ValueError) as exc:
        print(f"GPU lease guard: {exc}", file=sys.stderr)
        return 2
    print("GPU lease verified: " + ",".join(devices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
