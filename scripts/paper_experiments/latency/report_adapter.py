#!/usr/bin/env python3
"""Convert runtime timings into the latency harness's canonical JSON contract.

The ``strict`` format is for paper measurements.  It requires the runtime to
state its own synchronization and exclusion policy; this adapter never invents
those claims.  ``self-forcing-profile`` is a pilot-only bridge for the existing
CUDA-event profiler, whose individual sections are asynchronous.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_experiments.common import (  # noqa: E402
    COMPILATION_SCHEMA,
    RUN_SCHEMA,
    TIMING_KEYS,
    ContractError,
    read_json,
    write_json,
)


def _environment() -> dict[str, Any]:
    required = (
        "PAPER_EXPERIMENT_METHOD",
        "PAPER_EXPERIMENT_PHASE",
        "PAPER_EXPERIMENT_FRAMES",
        "PAPER_EXPERIMENT_REPEAT",
        "PAPER_EXPERIMENT_SESSION_ID",
        "PAPER_EXPERIMENT_SESSION_SCOPE",
        "PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID",
        "PAPER_EXPERIMENT_FROZEN_JSON",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ContractError("missing harness environment: " + ", ".join(missing))
    try:
        frozen = json.loads(os.environ["PAPER_EXPERIMENT_FROZEN_JSON"])
    except json.JSONDecodeError as error:
        raise ContractError("PAPER_EXPERIMENT_FROZEN_JSON is invalid") from error
    return {
        "method": os.environ["PAPER_EXPERIMENT_METHOD"],
        "phase": os.environ["PAPER_EXPERIMENT_PHASE"],
        "frames": int(os.environ["PAPER_EXPERIMENT_FRAMES"]),
        "repeat": int(os.environ["PAPER_EXPERIMENT_REPEAT"]),
        "session_id": os.environ["PAPER_EXPERIMENT_SESSION_ID"],
        "session_scope": os.environ["PAPER_EXPERIMENT_SESSION_SCOPE"],
        "runtime_instance_id": os.environ[
            "PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID"
        ],
        "frozen": frozen,
    }


def strict_report(raw: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "timings_s",
        "synchronization",
        "exclusions",
        "effective_frames",
        "observed_frozen",
        "runtime_instance_id",
    }
    if set(raw) != required:
        raise ContractError(
            f"strict timing input fields must be exactly {sorted(required)}"
        )
    timings = raw["timings_s"]
    if not isinstance(timings, dict) or set(timings) != set(TIMING_KEYS):
        raise ContractError(f"strict timings_s must contain exactly {list(TIMING_KEYS)}")
    if raw["runtime_instance_id"] != context["runtime_instance_id"]:
        raise ContractError(
            "runtime_instance_id differs from the warmup/measure pair assigned "
            "by the coordinator"
        )
    return {
        "schema_version": RUN_SCHEMA,
        "method": context["method"],
        "phase": context["phase"],
        "repeat": context["repeat"],
        "requested_frames": context["frames"],
        "effective_frames": raw["effective_frames"],
        "session_id": context["session_id"],
        "session_scope": context["session_scope"],
        "runtime_instance_id": raw["runtime_instance_id"],
        # This must come from the runtime's environment inspection, not from
        # the expected manifest copied into the adapter environment.  The
        # coordinator compares it against the manifest after conversion.
        "frozen": raw["observed_frozen"],
        "timings_s": timings,
        "synchronization": raw["synchronization"],
        "exclusions": raw["exclusions"],
    }


def _collapsed_profile(profile: Mapping[str, Any]) -> dict[str, float]:
    collapsed: dict[str, float] = {}
    sections = profile.get("sections")
    if not isinstance(sections, list):
        raise ContractError("Self-Forcing profile has no sections list")
    for section in sections:
        if not isinstance(section, dict):
            raise ContractError("Self-Forcing profile section is not an object")
        name = section.get("name")
        total_ms = section.get("total_ms")
        if not isinstance(name, str) or not isinstance(total_ms, (int, float)):
            raise ContractError("Self-Forcing profile section lacks name/total_ms")
        collapsed[name] = collapsed.get(name, 0.0) + float(total_ms) / 1000.0
    return collapsed


def self_forcing_report(
    pipeline: Mapping[str, Any],
    profile: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    e2e = pipeline.get("generation_seconds")
    if not isinstance(e2e, (int, float)) or e2e <= 0:
        raise ContractError("pipeline_report generation_seconds is missing or non-positive")
    if pipeline.get("benchmark_valid") is not True:
        raise ContractError("pipeline_report is not a fresh benchmark run")
    collapsed = _collapsed_profile(profile)
    encode = collapsed.get("encode_k", 0.0) + collapsed.get("encode_v", 0.0)
    decode = collapsed.get("decode_k", 0.0) + collapsed.get("decode_v", 0.0)
    vae = collapsed.get("vae_decode", 0.0)
    other = float(e2e) - encode - decode - vae
    if other < -max(0.005, float(e2e) * 0.01):
        raise ContractError("profile sections exceed pipeline e2e; timing scopes differ")
    other = max(0.0, other)
    return {
        "schema_version": RUN_SCHEMA,
        "method": context["method"],
        "phase": context["phase"],
        "repeat": context["repeat"],
        "requested_frames": context["frames"],
        "effective_frames": context["frames"],
        "session_id": context["session_id"],
        "session_scope": context["session_scope"],
        "runtime_instance_id": context["runtime_instance_id"],
        "frozen": context["frozen"],
        "timings_s": {
            "e2e": float(e2e),
            "cache_encode": encode,
            "cache_decode": decode,
            "vae_decode": vae,
            "denoising_other": other,
        },
        # CUDA events are suitable for attribution but do not satisfy the
        # formal before/after synchronization requirement for each section.
        "synchronization": {
            "api": "cuda_events_async",
            "before_each_segment": False,
            "after_each_segment": False,
        },
        "exclusions": {"model_loading": True, "compilation": True},
    }


def compilation_report(raw: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) != {"one_time_compilation_s", "observed_frozen"}:
        raise ContractError(
            "compilation input must contain one_time_compilation_s and observed_frozen"
        )
    return {
        "schema_version": COMPILATION_SCHEMA,
        "method": context["method"],
        "one_time_compilation_s": raw["one_time_compilation_s"],
        "session_id": context["session_id"],
        "session_scope": context["session_scope"],
        "frozen": raw["observed_frozen"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("strict", "self-forcing-profile", "compilation"), required=True)
    parser.add_argument("--input", type=Path, help="strict timings or compilation JSON")
    parser.add_argument("--pipeline-report", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, help="defaults to PAPER_EXPERIMENT_REPORT")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = _environment()
        output = args.output
        if output is None:
            value = os.environ.get("PAPER_EXPERIMENT_REPORT")
            if not value:
                raise ContractError("--output or PAPER_EXPERIMENT_REPORT is required")
            output = Path(value)
        if args.format == "strict":
            if args.input is None:
                raise ContractError("--format strict requires --input")
            report = strict_report(read_json(args.input), context)
        elif args.format == "compilation":
            if args.input is None:
                raise ContractError("--format compilation requires --input")
            report = compilation_report(read_json(args.input), context)
        else:
            if args.pipeline_report is None or args.profile is None:
                raise ContractError(
                    "--format self-forcing-profile requires --pipeline-report and --profile"
                )
            report = self_forcing_report(
                read_json(args.pipeline_report), read_json(args.profile), context
            )
        write_json(output.expanduser().resolve(), report)
    except (ContractError, OSError, json.JSONDecodeError, ValueError) as error:
        print(f"report adapter rejected input: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
