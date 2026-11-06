"""JSON records and content-addressed execution identity (no heavy imports)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_value(value.item())
    return value


def digest(value: Any) -> str:
    raw = json.dumps(json_value(value), sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_value(payload), ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompt_records(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [{"prompt_id": index, "text": line.strip(),
             "sha256": hashlib.sha256(line.strip().encode()).hexdigest()}
            for index, line in enumerate(lines) if line.strip()]


def check_prompt_split(calibration: str | Path, evaluation: str | Path,
                       development: str | Path | None = None) -> dict[str, Any]:
    groups = {"calibration": prompt_records(calibration), "evaluation": prompt_records(evaluation)}
    if development:
        groups["development"] = prompt_records(development)
    for name, rows in groups.items():
        if len({x["sha256"] for x in rows}) != len(rows):
            raise ValueError(f"duplicate prompts in {name}")
        if not rows:
            raise ValueError(f"empty prompt group: {name}")
    names = list(groups)
    for i, name in enumerate(names):
        hashes = {x["sha256"] for x in groups[name]}
        for other in names[i+1:]:
            if hashes & {x["sha256"] for x in groups[other]}:
                raise ValueError(f"prompt leakage: {name} overlaps {other}")
    return {"status": "disjoint", "counts": {k: len(v) for k, v in groups.items()},
            "assurance": "declared_prompt_files; verify_trace_provenance_separately"}


PATH_KEYS = frozenset({
    "output_dir", "codec_config", "base_config", "trace", "calibration_trace",
    "evaluation_trace", "validation_trace", "development_trace", "trace_manifest",
    "checkpoint", "checkpoint_dir", "model_root", "model_config", "prompts",
    "calibration_prompts", "evaluation_prompts", "development_prompts", "k_params",
    "v_params", "parameters_dir", "params_dir", "parameter_file", "research_params",
    "reference", "candidate", "reference_dir", "candidate_dir", "pairs_manifest",
    "cache_events", "cache_metrics", "annotations", "raft_root", "qvg_root",
    "python", "source_manifest", "results_dir", "vbench_prompts", "video", "fit_report",
    "architecture_params", "architecture_parameters", "parameters", "runtime_config",
    "calibration_identity", "evaluation_identity", "validation_identity", "bf16_reference_dir",
    "frames_csv", "video_index_manifest", "reference_metadata", "candidate_metadata",
    "raw_trace", "trace_output_dir",
    "bf16_reference_prompts", "quality_output_dir", "motion_groups",
    "parameter_fit_report",
})


def resolve_paths(value: Any, base: Path = ROOT, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: resolve_paths(v, base, k) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_paths(v, base, key) for v in value]
    if isinstance(value, str) and value and key in PATH_KEYS:
        path = Path(os.path.expandvars(value)).expanduser()
        # Command names such as python3 are executable names, not local paths.
        if key == "python" and "/" not in value:
            return value
        return str(path.resolve() if path.is_absolute() else (base / path).resolve())
    return value


def input_fingerprints(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, k)
        elif isinstance(value, list):
            for part in value:
                visit(part, key)
        elif isinstance(value, str) and key in PATH_KEYS and key != "output_dir":
            path = Path(value)
            if path.is_file():
                if str(path) in records:
                    return
                records[str(path)] = {"path": str(path), "size_bytes": path.stat().st_size,
                                      "sha256": file_sha256(path)}
                if key in {"trace", "calibration_trace", "evaluation_trace", "validation_trace", "development_trace"}:
                    sidecar = path.with_suffix(".identity.json")
                    if sidecar.is_file():
                        visit(str(sidecar), "trace_manifest")
                if key in {"pairs_manifest", "video_index_manifest"}:
                    visit(resolve_paths(read_json(path), path.parent))
    visit(config)
    return list(records.values())


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def code_fingerprint() -> str:
    files = sorted((ROOT / "rebuttal").rglob("*.py"))
    files += [ROOT / "experiments/Self-Forcing/tempokv_adapter/bridge.py",
              ROOT / "experiments/Self-Forcing/tempokv_adapter/trace_bridge.py"]
    return digest({str(path.relative_to(ROOT)): file_sha256(path) for path in files})


def prepare_run(output_dir: Path, config: dict[str, Any], *, experiment: str,
                stage: str, resume: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    # Resume changes execution control, never the scientific run identity.
    identity_config = {k: v for k, v in config.items() if not k.startswith("_") and k != "resume"}
    identity = digest(identity_config)
    inputs = input_fingerprints(config)
    code_sha256 = code_fingerprint()
    if path.exists():
        old = read_json(path)
        if old.get("config_sha256") != identity:
            raise ValueError("output directory belongs to a different configuration; choose a new output_dir")
        # A plan can be promoted to execution. Existing completed outputs need --resume.
        previous = old.get("stages", {}).get(stage)
        if previous and previous.get("status") == "complete" and not resume and stage != "plan":
            raise ValueError(f"stage {stage} already completed; use --resume or a new output directory")
        execution_started = any(name != "plan" for name in old.get("stages", {}))
        if (resume or execution_started) and old.get("inputs") != inputs:
            raise ValueError("input contents changed; refusing to reuse previous results")
        if (resume or execution_started) and old.get("code_sha256") != code_sha256:
            raise ValueError("experiment code changed; choose a new run directory")
        payload = old
        if not execution_started:
            # Plans can name assets that do not yet exist. Freeze their actual
            # bytes/code on promotion instead of retaining an empty snapshot.
            payload.update(inputs=inputs, code_sha256=code_sha256, code_revision=git_revision())
    else:
        payload = {"schema_version": "tempokv.rebuttal.run.v1", "experiment": experiment,
                   "config_sha256": identity, "config": identity_config, "inputs": inputs,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "code_revision": git_revision(), "code_sha256": code_sha256, "stages": {}}
    payload["stages"][stage] = {"status": "running"}
    write_json(path, payload)
    return payload


def finish_run(output_dir: Path, manifest: dict[str, Any], stage: str,
               result: dict[str, Any] | None = None, error: str | None = None) -> None:
    manifest["stages"][stage] = {
        "status": "failed" if error else "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_status": None if result is None else result.get("status"),
        "error": error,
    }
    write_json(output_dir / "manifest.json", manifest)
