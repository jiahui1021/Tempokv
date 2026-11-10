"""Streaming paired-video fidelity with explicit timing and metric contracts.

Images are BGR uint8, matching the repository QualityBackend's input contract.
PyAV is the default decoder so actual presentation timestamps can be checked.
OpenCV is an explicit alternative with a recorded constant-frame-rate assumption.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IDENTITY_FIELDS = ("model", "prompt_id", "seed", "sample_id", "method", "comparison_id")
TEXT_FIELDS = {"model", "prompt_id", "method", "comparison_id", "variant", "axis", "role", "head"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def read_csv(path: Path) -> list[dict[str, Any]]:
    def value(key: str, text: str) -> Any:
        if text == "":
            return None
        # IDs such as 0_0 and 0001 are opaque strings. Python's int() accepts
        # underscores, so inference would silently change or merge identities.
        if key in TEXT_FIELDS or key.endswith(("_id", "_sha256")) and key not in {"sample_id"}:
            return text
        if text in {"True", "False"}:
            return text == "True"
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    with path.open(newline="", encoding="utf-8") as stream:
        return [{k: value(k, v) for k, v in row.items()} for row in csv.DictReader(stream)]


def video_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def quality_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Settings that determine the stored per-frame fidelity observations."""
    return {"metrics": sorted(config.get("metrics", ["psnr", "ssim", "lpips"])),
            "lpips_network": config.get("lpips_network", "alex"),
            "video_decoder": config.get("video_decoder", "pyav"),
            "require_cfr": config.get("require_cfr", True),
            "timestamp_tolerance_seconds": config.get("timestamp_tolerance_seconds"),
            "pixel_contract": "bgr24_uint8_rgb_ssim_lpips_minus_one_to_one",
            "psnr_contract": "pooled_rgb_pixel_sse_peak_255"}


def _video_index(directory: Path, files: dict[Path, Path], manifest: Path | None = None) -> tuple[dict[Path, dict[str, Any]], str | None]:
    """Map actual file paths to recorded identities without parsing filenames."""
    path = manifest
    if path is None:
        for candidate in (directory / "summary.json", directory.parent / "summary.json"):
            if candidate.is_file():
                path = candidate
                break
    if path is None:
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif "rank_summaries" in payload:
        records = [row for rank in payload["rank_summaries"] for row in rank.get("videos", [])]
    else:
        records = payload.get("videos", [])
    if not records:
        raise ValueError(f"video identity index contains no video records: {path}")
    by_name: dict[str, list[Path]] = {}
    for relative in files:
        by_name.setdefault(relative.name, []).append(relative)
    result = {}
    for record in records:
        filename = record.get("relative_path", record.get("path", record.get("filename")))
        if not filename:
            raise ValueError(f"video index record has no relative_path/path/filename: {path}")
        raw_path = Path(filename)
        if raw_path.is_absolute():
            try:
                relative = raw_path.relative_to(directory.resolve())
            except ValueError:
                relative = None
        else:
            relative = raw_path
        if relative not in files:
            # Stored server paths can change after copying videos. Match the
            # recorded filename only when it identifies exactly one real file.
            matches = by_name.get(raw_path.name, [])
            if len(matches) != 1:
                continue
            relative = matches[0]
        identity = {
            "prompt_id": record.get("prompt_id", record.get("prompt_index")),
            "sample_id": record.get("sample_id", record.get("sample_index")),
            "seed": record.get("noise_seed", record.get("seed")),
        }
        if any(value is None for value in identity.values()):
            raise ValueError(f"video index requires prompt_id/index, sample_id/index, and actual noise_seed/seed: {path}")
        if relative in result:
            raise ValueError(f"duplicate video identity index record for {relative}")
        result[relative] = identity
    missing = files.keys() - result.keys()
    if missing:
        raise ValueError(f"video identity index does not cover files: {sorted(map(str, missing))}")
    return result, str(path)


def resolve_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = config.get("pairs")
    if pairs is None and config.get("pairs_manifest"):
        manifest_path = Path(config["pairs_manifest"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        pairs = payload if isinstance(payload, list) else payload["pairs"]
        for pair in pairs:
            for key in ("reference", "candidate", "cache_events", "cache_metrics", "frames_csv"):
                if pair.get(key) and not Path(pair[key]).is_absolute():
                    pair[key] = str((manifest_path.parent / pair[key]).resolve())
    if pairs is None and config.get("reference_dir") and config.get("candidate_dir"):
        reference, candidate = Path(config["reference_dir"]), Path(config["candidate_dir"])
        if not reference.is_dir() or not candidate.is_dir():
            raise FileNotFoundError("reference_dir and candidate_dir must both exist")
        refs = {p.relative_to(reference): p for p in reference.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS}
        cands = {p.relative_to(candidate): p for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS}
        if refs.keys() != cands.keys():
            raise ValueError(f"video directories do not pair exactly: missing_candidates={sorted(map(str, refs.keys()-cands.keys()))}; missing_references={sorted(map(str,cands.keys()-refs.keys()))}")
        explicit_index = Path(config["video_index_manifest"]) if config.get("video_index_manifest") else None
        ref_index, ref_source = _video_index(reference, refs, explicit_index)
        candidate_index, candidate_source = _video_index(candidate, cands, explicit_index)
        if bool(ref_source) != bool(candidate_source):
            raise ValueError("only one video directory has generation identity records; provide video_index_manifest or both generation summaries")
        pairs = []
        for relative, path in sorted(refs.items()):
            identity = {"prompt_id": relative.with_suffix("").as_posix(), "seed": config.get("seed", 0), "sample_id": 0}
            validation = "filename_stem_and_seed_declared_no_generation_index"
            if ref_source:
                if ref_index[relative] != candidate_index[relative]:
                    raise ValueError(f"paired generation identity differs for {relative}")
                identity = ref_index[relative]
                validation = "explicit_shared_video_index_declared" if explicit_index else "generation_indices_matched_not_full_generation_settings_verified"
            pairs.append({"reference": str(path), "candidate": str(cands[relative]), **identity,
                          "model": config.get("model", "unspecified"), "method": config.get("method", "tempokv"),
                          "identity_validation": validation, "reference_identity_source": ref_source, "candidate_identity_source": candidate_source})
    if not pairs:
        raise ValueError("provide nonempty pairs, pairs_manifest, or matching reference_dir/candidate_dir")
    result, seen, fingerprints = [], set(), {}
    for raw in pairs:
        pair = dict(raw)
        for key in ("reference", "candidate", "prompt_id"):
            if key not in pair:
                raise ValueError(f"video pair is missing {key}")
        pair.setdefault("model", config.get("model", "unspecified"))
        pair.setdefault("method", config.get("method", "tempokv"))
        pair.setdefault("seed", config.get("seed", 0))
        pair.setdefault("sample_id", 0)
        pair["prompt_id"] = str(pair["prompt_id"])
        pair.setdefault("comparison_id", "default")
        identity = tuple(str(pair[k]) for k in IDENTITY_FIELDS)
        if identity in seen:
            raise ValueError(f"duplicate video pair identity: {identity}")
        seen.add(identity)
        # A pair manifest declares matched generation identity. Sidecars, when
        # supplied, substantiate the declaration rather than being inferred.
        sidecars = []
        for key in ("reference_metadata", "candidate_metadata"):
            if pair.get(key):
                sidecars.append(json.loads(Path(pair[key]).read_text(encoding="utf-8")))
        if len(sidecars) == 2:
            for key in config.get("matched_metadata_fields", ["model", "prompt_id", "seed", "frames", "width", "height", "attention_window", "sink_size"]):
                if key not in sidecars[0] or key not in sidecars[1] or sidecars[0][key] != sidecars[1][key]:
                    raise ValueError(f"generation metadata mismatch or missing field: {key}")
        pair["identity_validation"] = "sidecars_matched" if len(sidecars) == 2 else pair.get("identity_validation", "declared_in_pair_manifest")
        # temporal_window persists resolved pairs before evaluation. Capture
        # the bytes here, so downstream reuse can verify the original inputs.
        pair.pop("video_fingerprints", None)
        for role in ("reference", "candidate"):
            path = Path(pair[role]).resolve()
            if path.is_file():
                if path not in fingerprints:
                    fingerprints[path] = video_fingerprint(path)
                pair.setdefault("video_fingerprints", {})[role] = fingerprints[path]
        pair["quality_contract"] = quality_contract(config)
        result.append(pair)
    return result


@dataclass
class VideoFrame:
    index: int
    timestamp: float
    image: Any


class VideoReader:
    def __init__(self, path: Path, decoder: str = "pyav") -> None:
        self.path, self.decoder = Path(path), decoder
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if decoder == "pyav":
            try:
                import av
            except ImportError as exc:
                raise RuntimeError("video decoding requires av; install rebuttal/requirements.txt or explicitly select video_decoder=opencv") from exc
            self.container = av.open(str(self.path))
            self.stream = self.container.streams.video[0]
            rate = self.stream.average_rate or self.stream.guessed_rate
            if rate is None or float(rate) <= 0:
                self.container.close()
                raise ValueError(f"video has no usable frame rate: {self.path}")
            self.fps = float(rate)
            self.width, self.height = self.stream.width, self.stream.height
            self.declared_frames = int(self.stream.frames)
            self.timestamp_policy = "normalized_presentation_timestamps"
        elif decoder == "opencv":
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("video_decoder=opencv requires opencv-python") from exc
            self.cv2, self.capture = cv2, cv2.VideoCapture(str(self.path))
            if not self.capture.isOpened():
                self.capture.release()
                raise ValueError(f"cannot open video: {self.path}")
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
            self.width, self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.declared_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.timestamp_policy = "cfr_frame_index_over_reported_fps_assumption"
            if not math.isfinite(self.fps) or self.fps <= 0:
                self.close()
                raise ValueError(f"invalid video fps: {self.path}")
        else:
            raise ValueError("video_decoder must be pyav or opencv")

    def __iter__(self) -> Iterator[VideoFrame]:
        if self.decoder == "pyav":
            origin = None
            for index, frame in enumerate(self.container.decode(self.stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"missing presentation timestamp at frame {index}: {self.path}")
                time = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = time
                yield VideoFrame(index, time - origin, frame.to_ndarray(format="bgr24"))
        else:
            index = 0
            while True:
                ok, image = self.capture.read()
                if not ok:
                    break
                yield VideoFrame(index, index / self.fps, image)
                index += 1

    def close(self) -> None:
        if hasattr(self, "container"):
            self.container.close()
        if hasattr(self, "capture"):
            self.capture.release()


def psnr(sse: float, count: int) -> float | None:
    if count <= 0:
        raise ValueError("PSNR requires positive pixel count")
    return None if sse == 0 else 10 * math.log10(255 ** 2 * count / sse)


def window_index(timestamp: float, seconds: float) -> int:
    if seconds <= 0 or not math.isfinite(seconds) or timestamp < 0:
        raise ValueError("invalid window duration or negative timestamp")
    return int(math.floor(timestamp / seconds + 1e-9))


def aggregate_frames(rows: Iterable[dict[str, Any]], *, group_by_window: bool = True) -> list[dict[str, Any]]:
    grouped: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        keys = (*IDENTITY_FIELDS, "window_index") if group_by_window else IDENTITY_FIELDS
        group = tuple(row.get(k) for k in keys)
        if group not in grouped:
            grouped[group] = {**{k: row.get(k) for k in keys}, "frames": 0, "sse": 0.0, "pixel_count": 0,
                              "start_seconds": float(row["timestamp"]), "end_seconds": float(row["timestamp"]),
                              "_ssim": [], "_lpips": []}
        entry = grouped[group]
        entry["frames"] += 1
        entry["sse"] += float(row["sse"])
        entry["pixel_count"] += int(row["pixel_count"])
        entry["start_seconds"] = min(entry["start_seconds"], float(row["timestamp"]))
        entry["end_seconds"] = max(entry["end_seconds"], float(row["timestamp"]) + float(row["frame_duration"]))
        for metric in ("ssim", "lpips"):
            if row.get(metric) is not None:
                entry[f"_{metric}"].append(float(row[metric]))
    result = []
    for entry in grouped.values():
        entry["psnr_db"] = psnr(entry["sse"], entry["pixel_count"])
        entry["zero_error"] = entry["sse"] == 0
        entry["mse"] = entry["sse"] / entry["pixel_count"]
        entry["duration_seconds"] = entry["end_seconds"] - entry["start_seconds"]
        for metric in ("ssim", "lpips"):
            values = entry.pop(f"_{metric}")
            entry[metric] = sum(values) / len(values) if values else None
            entry[f"{metric}_frames"] = len(values)
        result.append(entry)
    return result


def make_quality_backend(config: dict[str, Any]) -> Any:
    metrics = config.get("metrics", ["psnr", "ssim", "lpips"])
    if not set(metrics) <= {"psnr", "ssim", "lpips"} or not metrics:
        raise ValueError("metrics must be a nonempty subset of psnr,ssim,lpips")
    if {"ssim", "lpips"} & set(metrics):
        from scripts.evaluation.evaluate_quality import QualityBackend
        # The existing backend deliberately requires its full metric runtime.
        return QualityBackend(device=config.get("device", "auto"), network=config.get("lpips_network", "alex"))
    return None


def iter_pair_metrics(pair: dict[str, Any], config: dict[str, Any], *, backend: Any = None, reader_factory: Callable = VideoReader) -> Iterator[dict[str, Any]]:
    """All frames are checked and evaluated, including partial final windows."""
    import numpy as np
    metrics = config.get("metrics", ["psnr", "ssim", "lpips"])
    if backend is None:
        backend = make_quality_backend(config)
    decoder = config.get("video_decoder", "pyav")
    reference = reader_factory(Path(pair["reference"]), decoder)
    try:
        candidate = reader_factory(Path(pair["candidate"]), decoder)
    except Exception:
        reference.close()
        raise
    frames = 0
    evaluated_frames = 0
    ranges = pair.get("evaluation_frame_ranges")
    if ranges is not None and (not ranges or any(
            len(interval) != 2 or any(type(x) is not int for x in interval) or
            interval[0] < 0 or interval[1] <= interval[0] for interval in ranges)):
        raise ValueError("evaluation_frame_ranges must contain nonempty integer intervals")
    sentinel = object()
    previous_timestamp = None
    try:
        if not math.isclose(reference.fps, candidate.fps, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("paired video fps differs")
        if (reference.width, reference.height) != (candidate.width, candidate.height):
            raise ValueError("paired video dimensions differ")
        expected = pair.get("expected_frames", config.get("expected_frames"))
        duration = 1 / reference.fps
        tolerance = float(config.get("timestamp_tolerance_seconds", max(1e-5, duration * 0.02)))
        for left, right in zip_longest(iter(reference), iter(candidate), fillvalue=sentinel):
            if left is sentinel or right is sentinel:
                raise ValueError(f"paired video frame counts differ after {frames} frames")
            if left.image.shape != right.image.shape:
                raise ValueError(f"paired frame shapes differ at {frames}")
            if abs(left.timestamp - right.timestamp) > tolerance:
                raise ValueError(f"paired timestamps differ at frame {frames}")
            if previous_timestamp is not None:
                delta = left.timestamp - previous_timestamp
                if delta <= 0 or (config.get("require_cfr", True) and abs(delta - duration) > tolerance):
                    raise ValueError(f"nonmonotonic or variable-rate timestamps at frame {frames}")
            previous_timestamp = left.timestamp
            if ranges is not None and not any(start <= frames < end for start, end in ranges):
                frames += 1
                continue
            evaluated_frames += 1
            difference = left.image.astype(np.float64) - right.image.astype(np.float64)
            sse = float(np.square(difference).sum())
            row = {k: pair.get(k) for k in IDENTITY_FIELDS}
            row.update(frame_index=frames, timestamp=left.timestamp, frame_duration=duration, fps=reference.fps,
                       identity_validation=pair.get("identity_validation", "declared_in_pair_manifest"),
                       timestamp_policy=reference.timestamp_policy, window_index=window_index(left.timestamp, float(config.get("window_seconds", 15))),
                       sse=sse, pixel_count=int(difference.size), psnr_db=psnr(sse, int(difference.size)), zero_error=sse == 0,
                       ssim=float(backend.ssim(left.image, right.image)) if "ssim" in metrics else None,
                       lpips=float(backend.lpips(left.image, right.image)) if "lpips" in metrics else None)
            for key in ("ssim", "lpips"):
                if row[key] is not None and not math.isfinite(row[key]):
                    raise ValueError(f"nonfinite {key} at frame {frames}")
            frames += 1
            yield row
        if not frames:
            raise ValueError("paired videos contain no decoded frames")
        if not evaluated_frames:
            raise ValueError("no frames fall within the requested evaluation ranges")
        if expected is not None and frames != int(expected):
            raise ValueError(f"expected {expected} frames, decoded {frames}")
        for reader in (reference, candidate):
            if reader.declared_frames > 0 and reader.declared_frames != frames:
                raise ValueError(f"declared vs decoded frame counts differ: {reader.path}")
    finally:
        reference.close()
        candidate.close()


def pair_key(pair: dict[str, Any]) -> str:
    identity = {k: pair.get(k) for k in IDENTITY_FIELDS}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
