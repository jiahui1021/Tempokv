#!/usr/bin/env python3
"""Pre-flight checks for an observation trace (paper section 3.2, step 1).

Two failure modes silently ruin the ladder and neither one raises on its own:

  1. Denoising steps leaking in.  Only the finalized clean write must be
     recorded.  If intermediate steps are captured the frame count comes out
     an integer multiple of the latent frame count, and the ladder then
     measures denoising convergence instead of temporal structure.
  2. Post-RoPE K.  RoPE is a position-dependent rotation, so cross-frame
     prediction on rotated K compares differently-rotated bases; the measured
     predictability drops for a reason that has nothing to do with the signal.
     A rotated channel alternates sign along the frame axis, which V - never
     rotated - does not, so the two are compared here rather than eyeballed.

Exit status is non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_decompose import TraceError, load_trace  # noqa: E402


# Consecutive frames sit 1560 token positions apart, so RoPE turns a rotated
# channel by an effectively arbitrary angle between them and its sign lands
# near a coin flip.  V is never rotated, so it serves as the control: a K that
# is both near coin-flip *and* markedly less stable than V is rotated, while a
# trace that is merely noisy moves both tensors together and is not flagged.
SIGN_FLIP_LIMIT = 0.40
SIGN_FLIP_MARGIN = 0.08


def sign_flip_rate(trace: torch.Tensor, *, channel: int = 0) -> float:
    """Median fraction of adjacent frames that change sign, over (L, S, H)."""
    series = trace[:, :, :, :, channel].to(torch.float32)
    series = series - series.mean(dim=2, keepdim=True)
    flips = (series[:, :, 1:, :] * series[:, :, :-1, :]) < 0
    return float(flips.to(torch.float32).mean(dim=2).median().item())


def lag_one_correlation(trace: torch.Tensor, *, channel: int = 0) -> float:
    """Median lag-1 correlation along frames, over (L, S, H)."""
    series = trace[:, :, :, :, channel].to(torch.float64)
    series = series - series.mean(dim=2, keepdim=True)
    lead, lag = series[:, :, 1:, :], series[:, :, :-1, :]
    numerator = (lead * lag).sum(dim=2)
    denominator = (
        lead.pow(2).sum(dim=2).sqrt() * lag.pow(2).sum(dim=2).sqrt()
    ).clamp_min(1e-12)
    return float((numerator / denominator).median().item())


def check(trace: dict[str, Any], expected_frames: int | None) -> dict[str, Any]:
    keys, values = trace["keys"], trace["values"]
    frames = int(keys.shape[2])
    frame_check: dict[str, Any] = {
        "frames": frames,
        "expected_frames": expected_frames,
        "ok": True,
        "note": "no expected frame count given; frame leakage was not checked",
    }
    if expected_frames is not None:
        multiple = frames % expected_frames == 0 and frames != expected_frames
        frame_check["ok"] = frames == expected_frames
        frame_check["note"] = (
            "frame count matches the latent frame count"
            if frames == expected_frames
            else (
                f"{frames} frames is {frames // expected_frames}x the expected "
                f"{expected_frames}: denoising steps are being recorded"
                if multiple
                else f"{frames} frames does not match the expected {expected_frames}"
            )
        )
    key_flips = sign_flip_rate(keys)
    value_flips = sign_flip_rate(values)
    rope_ok = not (
        key_flips >= SIGN_FLIP_LIMIT and key_flips >= value_flips + SIGN_FLIP_MARGIN
    )
    rope_check = {
        "k_sign_flip_rate": key_flips,
        "v_sign_flip_rate": value_flips,
        "k_lag1_correlation": lag_one_correlation(keys),
        "v_lag1_correlation": lag_one_correlation(values),
        "limit": SIGN_FLIP_LIMIT,
        "margin_over_v": SIGN_FLIP_MARGIN,
        "ok": rope_ok,
        "note": (
            "K alternates sign along frames like a rotated tensor; collect "
            "pre-RoPE K instead"
            if not rope_ok
            else "K varies smoothly along frames, consistent with pre-RoPE K"
        ),
    }
    return {
        "trace": str(trace["path"]),
        "shape": list(keys.shape),
        "events": int(trace["events"]),
        "spatial_indices": trace["token_indices"],
        "frame_count": frame_check,
        "pre_rope": rope_check,
        "ok": bool(frame_check["ok"] and rope_ok),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--expected-frames",
        type=int,
        help="latent frame count the generation should have committed",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        trace = load_trace(args.trace)
    except (OSError, TraceError) as error:
        parser.error(str(error))
    report = check(trace, args.expected_frames)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
