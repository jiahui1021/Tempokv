"""Read actual Self-Forcing generation identities before pairing video files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rebuttal.common.video_metrics import _video_index, write_json


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_prompts(path: Path) -> list[str]:
    # Match the bundled TextDataset's line indexing, including whitespace rules.
    prompts = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError(f"prompt file must be nonempty and contain no blank lines: {path}")
    return prompts


def identity_key(record: dict) -> tuple[str, int, int]:
    return str(record["prompt_id"]), int(record["noise_seed"]), int(record["sample_id"])


def video_identities(directory: Path, *, prompts_path: Path | None = None,
                     use_snapshot: bool = True) -> tuple[list[dict], str]:
    """Require a generation summary; never derive identity from a filename.

    Native summaries record prompt index, sample index, and the actual noise
    seed. A snapshot written after generation supplies prompt text. For older
    references, the caller must explicitly identify their original prompt file.
    """
    directory = directory.resolve()
    files = {p.relative_to(directory): p for p in sorted(directory.glob("*.mp4"))}
    if not files:
        raise ValueError(f"no generated videos: {directory}")
    index, source = _video_index(directory, files)
    if source is None:
        raise ValueError(f"generation summary is required to verify video identity: {directory}")
    snapshot = directory / "generation_identity.json"
    saved = None
    if use_snapshot and snapshot.is_file():
        payload = json.loads(snapshot.read_text())
        if payload.get("summary_sha256") != file_digest(Path(source)):
            raise ValueError(f"generation summary changed since identity snapshot: {directory}")
        saved = {}
        for record in payload.get("videos", []):
            key = identity_key(record)
            if key in saved:
                raise ValueError(f"duplicate identity in snapshot: {key}")
            saved[key] = record
    prompts = read_prompts(prompts_path) if prompts_path is not None else None
    if saved is None and prompts is None:
        raise ValueError(f"provide the original prompt file or a generation_identity.json for {directory}")
    records, seen = [], set()
    for relative, observed in index.items():
        record = {"path": str(files[relative]), "prompt_id": str(observed["prompt_id"]),
                  "noise_seed": int(observed["seed"]), "sample_id": int(observed["sample_id"])}
        key = identity_key(record)
        if key in seen:
            raise ValueError(f"duplicate generation identity: {key}")
        seen.add(key)
        prompt = None
        if prompts is not None:
            try:
                prompt_index = int(record["prompt_id"])
            except ValueError as error:
                raise ValueError("native Self-Forcing prompt identities must be integer indices") from error
            if not 0 <= prompt_index < len(prompts):
                raise ValueError(f"generation prompt index is outside the supplied prompt file: {key}")
            prompt = prompts[prompt_index]
        if saved is not None:
            if key not in saved or not isinstance(saved[key].get("prompt"), str):
                raise ValueError(f"identity snapshot does not cover generated video: {key}")
            if prompt is not None and saved[key]["prompt"] != prompt:
                raise ValueError(f"prompt file differs from generation snapshot: {key}")
            prompt = saved[key]["prompt"]
        record["prompt"] = prompt
        records.append(record)
    if saved is not None and set(saved) != seen:
        raise ValueError(f"identity snapshot has a different video cohort: {directory}")
    return records, source


def job_video_identities(job: dict, *, use_snapshot: bool = True) -> tuple[list[dict], str]:
    """Validate native generation records against this job's immutable inputs."""
    command = job["command"]
    prompts_path = Path(command[command.index("--data_path") + 1])
    expected_digest = job.get("input_digests", {}).get("prompts")
    if not expected_digest or file_digest(prompts_path) != expected_digest:
        raise ValueError("generation prompt contents are missing or changed from the job plan")
    records, source = video_identities(Path(job["folder"]) / "videos", prompts_path=prompts_path,
                                       use_snapshot=use_snapshot)
    if len(records) != job["expected_video_count"]:
        raise ValueError(f"incomplete generated video cohort: {job['folder']}")
    if {r["prompt_id"] for r in records} != {str(i) for i in range(job["expected_video_count"])}:
        raise ValueError("generation summary does not cover the planned prompt indices")
    for record in records:
        expected_seed = int(job["seed"]) + int(record["prompt_id"]) * 1_000_003
        if record["noise_seed"] != expected_seed or record["sample_id"] != 0:
            raise ValueError(f"generation summary seed/sample does not match job {job['name']}")
    return records, source


def save_job_identities(job: dict) -> None:
    records, source = job_video_identities(job, use_snapshot=False)
    write_json(Path(job["folder"]) / "videos" / "generation_identity.json", {
        "schema_version": "tempokv.rebuttal.memory.video_identity.v1",
        "identity_scope": "actual_generation_summary_and_verified_job_prompt_file",
        "summary_sha256": file_digest(Path(source)), "summary_source": source,
        "prompts_sha256": job["input_digests"]["prompts"], "videos": records,
    })
