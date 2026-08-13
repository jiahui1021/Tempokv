#!/usr/bin/env python3
"""Report the Self-Forcing runtime breakdown and the Stage 0 decision gates.

Stage 0 of the acceleration plan will not let a kernel be rewritten until the
profile shows that kernel owns a meaningful share of the generation.  This script
reads the per-rank breakdown an instrumented run wrote and answers the two gates
directly: does TempoKV *decode* reach the share that justifies Stages 1 to 3, and
does TempoKV *encode* reach the share that justifies Stage 4.

Record a profile by asking the normal entry point for one:

    PROFILE=1 CHECKPOINT=... WAN_MODEL_DIR=... PROMPTS=... CONFIG=... \\
      bash scripts/Self-Forcing/run_tempokv.sh

then report it:

    python scripts/tools/profile_tempokv_runtime.py results/self_forcing/<label>

Pass ``--baseline`` to compare two runs; later stages need the comparison more
than the absolute numbers, because their stopping conditions are written as
end-to-end speedup against the run they started from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.profiling import (  # noqa: E402
    DECODE_K,
    DECODE_V,
    ENCODE_K,
    ENCODE_V,
    aggregate_by_name,
    format_report,
)

# Stage 0.3 in the plan: below this share, a faster kernel cannot move the
# generation enough to be worth the maintenance it costs.
STAGE_GATE_SHARE = 0.10


def _resolve_profile(target: Path, rank: int) -> dict[str, object]:
    """Accept a run directory, a profile directory, or one rank's JSON file."""
    candidates = (
        target,
        target / f"rank-{rank}.json",
        target / "profile" / f"rank-{rank}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "sections" not in payload:
                raise ValueError(f"not a runtime profile: {candidate}")
            return payload
    raise FileNotFoundError(
        f"no runtime profile for rank {rank} under {target}; run with PROFILE=1"
    )


def _share(collapsed: dict[str, dict[str, object]], *names: str) -> float | None:
    total = 0.0
    seen = False
    for name in names:
        entry = collapsed.get(name)
        if entry is None:
            continue
        share = entry["share_of_generation"]
        if share is None:
            return None
        total += float(share)
        seen = True
    return total if seen else None


def _format_gate(label: str, share: float | None, stages: str) -> str:
    if share is None:
        return f"{label:<22} not measured        -> {stages}: 无法判断（该段未被插桩）"
    verdict = "进入" if share >= STAGE_GATE_SHARE else "跳过"
    return (
        f"{label:<22} {share * 100:>6.2f}% of generation"
        f" -> {stages}: {verdict}"
        f" (门槛 {STAGE_GATE_SHARE * 100:.0f}%)"
    )


def _format_comparison(
    current: dict[str, object], baseline: dict[str, object]
) -> str:
    current_names = aggregate_by_name(current)
    baseline_names = aggregate_by_name(baseline)
    lines = ["", "对比 baseline（正数 = 变快）", ""]
    header = f"{'section':<20}{'baseline s':>13}{'current s':>13}{'delta s':>11}{'speedup':>10}"
    lines.extend((header, "-" * len(header)))
    generation_pair = (
        baseline.get("generation_ms"),
        current.get("generation_ms"),
    )
    rows = sorted((set(current_names) | set(baseline_names)) - {"generation"})
    for name in ("generation", *rows):
        if name == "generation":
            before, after = generation_pair
        else:
            before = (baseline_names.get(name) or {}).get("total_ms")
            after = (current_names.get(name) or {}).get("total_ms")
        if not before or not after:
            continue
        before_s = float(before) / 1000.0
        after_s = float(after) / 1000.0
        lines.append(
            f"{name:<20}{before_s:>13.3f}{after_s:>13.3f}"
            f"{before_s - after_s:>11.3f}{before_s / after_s:>10.3f}x"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="run directory, profile directory, or rank-N.json from a PROFILE=1 run",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="a second profile to compare against, in the same forms as target",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--json",
        type=Path,
        help="also write the collapsed per-section summary here",
    )
    arguments = parser.parse_args(argv)
    try:
        report = _resolve_profile(arguments.target, arguments.rank)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
        parser.error(str(error))

    print(format_report(report))
    collapsed = aggregate_by_name(report)

    print("")
    print("Stage 0 决策门槛")
    print("")
    print(_format_gate("TempoKV decode", _share(collapsed, DECODE_K, DECODE_V), "Stage 1~3"))
    print(_format_gate("TempoKV encode", _share(collapsed, ENCODE_K, ENCODE_V), "Stage 4"))

    if arguments.baseline is not None:
        try:
            baseline = _resolve_profile(arguments.baseline, arguments.rank)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
            parser.error(str(error))
        print(_format_comparison(report, baseline))

    if arguments.json is not None:
        payload = {
            "generation_ms": report.get("generation_ms"),
            "gpu_name": report.get("gpu_name"),
            "wall_seconds": report.get("wall_seconds"),
            "instrumentation": report.get("instrumentation"),
            "sections": sorted(collapsed.values(), key=lambda item: -item["total_ms"]),
        }
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
