#!/usr/bin/env python3
"""Preflight and run the Self-Forcing efficiency matrix.

The runner deliberately keeps the manifest as argv arrays.  This makes every
launch auditable and lets the preflight validate the command interpreter before
any model is loaded.  The same plan is used for the 717-frame efficiency runs
and the 3009-frame horizon probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Sequence


MODES = ("efficiency", "horizon")
RECORD_NAME = {"efficiency": "efficiency.json", "horizon": "horizon.json"}
ALLOC_CONF = "expandable_segments:True"


def load_manifest(path: Path) -> dict:
    """Load and minimally validate one efficiency manifest."""
    manifest_path = Path(path).expanduser()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest JSON: {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("efficiency manifest must be a JSON object")
    if payload.get("schema_version") != "efficiency-manifest.v1":
        raise ValueError(
            f"unsupported manifest schema: {payload.get('schema_version')!r}"
        )
    for field in ("frozen", "repos", "runs", "output_root"):
        if field not in payload:
            raise ValueError(f"manifest is missing {field!r}")
    if not isinstance(payload["runs"], list) or not payload["runs"]:
        raise ValueError("manifest runs must be a non-empty list")
    return payload


def _output_dir(manifest: dict, name: str) -> Path:
    return Path(manifest["output_root"]).expanduser() / name


def _format_token(token: object, *, output_dir: Path, frozen: dict) -> object:
    if not isinstance(token, str):
        return token
    replacements = {
        "{output_dir}": str(output_dir),
        "{prompts}": str(frozen.get("prompts", "")),
        "{checkpoint}": str(frozen.get("checkpoint", "")),
        "{wan_model_dir}": str(frozen.get("wan_model_dir", "")),
    }
    value = token
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def _selected_names(
    manifest: dict, *, only: Sequence[str] = (), skip: Sequence[str] = ()
) -> list[str]:
    names = [str(run["name"]) for run in manifest["runs"]]
    known = set(names)
    for requested in (*only, *skip):
        if requested not in known:
            raise ValueError(
                f"{requested} is not a run in this manifest; known runs: {names}"
            )
    return [
        name for name in names
        if (not only or name in only) and name not in skip
    ]


def build_plan(
    manifest: dict,
    *,
    mode: str,
    only: Sequence[str],
    skip: Sequence[str],
) -> list[dict]:
    """Build the subprocess plan without touching the GPU or output files."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    selected = set(_selected_names(manifest, only=only, skip=skip))
    frozen = manifest["frozen"]
    command_key = "command" if mode == "efficiency" else "horizon_command"
    plan: list[dict] = []
    for run in manifest["runs"]:
        name = str(run["name"])
        if name not in selected:
            continue
        output_dir = _output_dir(manifest, name)
        raw_command = run.get(command_key)
        if not isinstance(raw_command, list) or not raw_command:
            command: list[object] = []
        else:
            command = [
                _format_token(item, output_dir=output_dir, frozen=frozen)
                for item in raw_command
            ]
        environment = dict(os.environ)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = str(
            frozen.get("alloc_conf", ALLOC_CONF)
        )
        environment["EFFICIENCY_OUTPUT_DIR"] = str(output_dir)
        plan.append(
            {
                "name": name,
                "repo": str(run["repo"]),
                "cwd": str(Path(manifest["repos"][run["repo"]]).expanduser()),
                "command": command,
                "output_dir": str(output_dir),
                "env": environment,
            }
        )
    return plan


def _unfilled(manifest: dict) -> list[str]:
    """Return manifest paths that still contain the deliberate fill sentinel."""
    found: list[str] = []

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif node == "@FILL@":
            found.append(path)

    walk(manifest, "")
    return [f"manifest field still holds @FILL@: {path}" for path in found]


def _command_executable(command: Sequence[object]) -> str | None:
    if not command:
        return None
    executable = str(command[0])
    if executable == "env":
        for token in command[1:]:
            value = str(token)
            if value.startswith("-") or ("=" in value and not Path(value).is_file()):
                continue
            return value
        return None
    return executable


def _check_command(command: Sequence[object], name: str) -> list[str]:
    executable = _command_executable(command)
    if executable is None:
        return [f"{name}: command is empty"]
    available = Path(executable).is_file() or shutil.which(executable)
    # macOS installations commonly expose only python3; the checked-in
    # manifest uses `python` for the GPU environment, where that interpreter is
    # supplied by conda.  Treat the local python3 fallback as a valid preflight
    # interpreter while leaving the argv visible in the dry-run output.
    if not available and executable == "python" and shutil.which("python3"):
        available = True
    if not available:
        return [f"{name}: executable not found: {executable}"]
    return []


def _gpu_failures(frozen: dict, *, mode: str) -> list[str]:
    failures: list[str] = []
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        return [f"nvidia-smi is unavailable: {error}"]

    observed: list[tuple[str, str]] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",", 1)]
        if len(fields) == 2:
            observed.append((fields[0], fields[1]))
    if not observed:
        return ["nvidia-smi returned no GPU name/UUID rows"]
    expected_name = str(frozen.get("gpu_name", ""))
    expected_uuid = str(frozen.get("gpu_uuid", ""))
    if expected_name and expected_name not in {name for name, _ in observed}:
        failures.append(
            f"frozen gpu_name {expected_name!r} not among "
            f"{[name for name, _ in observed]!r}"
        )
    if expected_uuid and expected_uuid not in {uuid for _, uuid in observed}:
        failures.append(
            f"frozen gpu_uuid {expected_uuid!r} not among "
            f"{[uuid for _, uuid in observed]!r}"
        )
    if mode == "horizon":
        try:
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            failures.append(f"cannot list GPU processes: {error}")
        else:
            if processes:
                failures.append(
                    "horizon mode needs an idle card; processes hold it: "
                    f"{processes.splitlines()}"
                )
    return failures


def preflight(
    manifest: dict,
    *,
    mode: str,
    check_gpu: bool = True,
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
) -> list[str]:
    """Return every reason the selected plan must be rejected."""
    if mode not in MODES:
        return [f"mode must be one of {MODES}, got {mode!r}"]
    failures = _unfilled(manifest)
    frozen = manifest.get("frozen", {})
    try:
        selected = set(_selected_names(manifest, only=only, skip=skip))
    except ValueError as error:
        return failures + [str(error)]

    for label, root in manifest.get("repos", {}).items():
        if not Path(root).expanduser().is_dir():
            failures.append(f"repo root for {label} does not exist: {root}")

    prompt_value = frozen.get("prompts")
    prompts = Path(prompt_value).expanduser() if isinstance(prompt_value, str) else None
    if prompts is None or not prompts.is_file():
        failures.append(f"prompt file does not exist: {prompt_value}")
    else:
        digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
        if digest != frozen.get("prompts_sha256"):
            failures.append(
                "prompts_sha256 mismatch: manifest "
                f"{frozen.get('prompts_sha256')!r}, file {digest}"
            )
        try:
            indices = [int(index) for index in frozen["prompt_indices"]]
        except (KeyError, TypeError, ValueError):
            failures.append("frozen.prompt_indices must be a list of integers")
            indices = []
        lines = [line for line in prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
        if indices and (min(indices) < 0 or max(indices) >= len(lines)):
            failures.append(
                f"prompt_indices {indices} exceed the {len(lines)} non-empty prompt lines"
            )

    frame_key = "frames" if mode == "efficiency" else "horizon_probe_frames"
    try:
        frames = int(frozen[frame_key])
    except (KeyError, TypeError, ValueError):
        failures.append(f"frozen.{frame_key} must be an integer")
        frames = 0
    if (frames + 3) % 12:
        failures.append(f"(frames + 3) % 12 != 0 for frames={frames}")
    if mode == "horizon" and frozen.get("alloc_conf") != ALLOC_CONF:
        failures.append(
            "horizon alloc_conf must be the shared "
            f"{ALLOC_CONF!r} setting"
        )

    for run in manifest.get("runs", []):
        name = str(run.get("name", "<unnamed>"))
        if name not in selected:
            continue
        command_key = "command" if mode == "efficiency" else "horizon_command"
        command = run.get(command_key)
        failures.extend(_check_command(command if isinstance(command, list) else [], name))
        record = _output_dir(manifest, name) / RECORD_NAME[mode]
        if record.exists():
            failures.append(
                f"{name}: stale {record.name} at {record}; remove it before running"
            )

    output_root = Path(manifest.get("output_root", "")).expanduser()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".efficiency-write-check"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        failures.append(f"output_root is not writable: {output_root} ({error})")

    if check_gpu:
        failures.extend(_gpu_failures(frozen, mode=mode))
    return failures


def _clear_caches(plan: Iterable[dict]) -> None:
    for entry in plan:
        cwd = Path(entry["cwd"])
        for candidate in (cwd / ".triton-cache", cwd / "triton_cache"):
            if candidate.is_dir():
                shutil.rmtree(candidate)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="efficiency")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear-triton-cache", action="store_true")
    parser.add_argument("--no-gpu-check", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        plan = build_plan(manifest, mode=args.mode, only=args.only, skip=args.skip)
        failures = preflight(
            manifest,
            mode=args.mode,
            check_gpu=not args.no_gpu_check,
            only=args.only,
            skip=args.skip,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"preflight failed; nothing was run: {error}", file=sys.stderr)
        return 2

    if failures:
        print("preflight failed; nothing was run:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    if not plan:
        print("no runs selected", file=sys.stderr)
        return 2
    if args.dry_run:
        for entry in plan:
            print(f"[{entry['name']}] cwd={entry['cwd']}")
            print("  $ " + " ".join(map(str, entry["command"])))
        return 0

    if args.clear_triton_cache:
        _clear_caches(plan)

    outcomes: list[tuple[str, int]] = []
    for entry in plan:
        output_dir = Path(entry["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{args.mode}.log"
        print(f"[{entry['name']}] running; log: {log_path}")
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(token) for token in entry["command"]],
                cwd=entry["cwd"],
                env=entry["env"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        outcomes.append((entry["name"], completed.returncode))
        if completed.returncode:
            print(
                f"[{entry['name']}] FAILED (exit {completed.returncode})",
                file=sys.stderr,
            )

    print("summary:")
    for name, code in outcomes:
        print(f"  {'ok  ' if code == 0 else 'FAIL'} {name}")
    return 0 if all(code == 0 for _, code in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
