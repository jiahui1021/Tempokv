"""BF16-first difficult-case candidates, explicit annotations and paired clips."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any

from rebuttal.common.video_metrics import VideoReader, aggregate_frames, write_csv, write_json


CATEGORIES = ("large_local_motion", "large_camera_motion", "occlusion_disocclusion", "abrupt_scene_change")


def select_candidates(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    settings = config.get("difficult_cases", {})
    count, duration = int(settings.get("clips_per_category", 3)), float(settings.get("clip_seconds", 3.0))
    if count <= 0 or duration <= 0:
        raise ValueError("difficult_cases clip count and duration must be positive")
    signals = {"large_local_motion": "residual_motion", "large_camera_motion": "global_motion",
               "occlusion_disocclusion": "fb_inconsistent_fraction", "abrupt_scene_change": "cut_score"}
    result = []
    for category, metric in signals.items():
        candidates = [r for r in rows if r.get(metric) is not None and (r.get("is_cut") if category == "abrupt_scene_change" else not r.get("is_cut"))]
        candidates.sort(key=lambda r: float(r[metric]), reverse=True)
        chosen = []
        for row in candidates:
            key = (row["model"], row["prompt_id"], row.get("seed", 0), row.get("sample_id", 0))
            if any(key == entry[0] and abs(row["timestamp"] - entry[1]) < duration for entry in chosen):
                continue
            pair = next(p for p in pairs if (p["model"], p["prompt_id"], p.get("seed", 0), p.get("sample_id", 0)) == key)
            start = max(0.0, row["timestamp"] - duration / 2)
            result.append({"case_id": f"{category}_{len(chosen)}", "category": category, "model": key[0], "prompt_id": key[1],
                           "seed": key[2], "sample_id": key[3], "start_seconds": start, "end_seconds": start + duration,
                           "selection_signal": metric, "selection_value": row[metric], "manual_verified": False,
                           "reference": pair["reference"], "candidate": pair["candidate"], "method": pair.get("method", "tempokv"),
                           "selection_rule": "ranked_BF16_motion_proxy_before_candidate_error_inspection"})
            chosen.append((key, row["timestamp"]))
            if len(chosen) >= count:
                break
    write_json(output_dir / "case_candidates.json", {"cases": result, "warning": "Automatic categories are candidate proxies; manual verification is required for evidence."})
    annotations = settings.get("annotations")
    if annotations:
        payload = json.loads(Path(annotations).read_text(encoding="utf-8"))
        annotated = payload if isinstance(payload, list) else payload["cases"]
        by_id = {r["case_id"]: r for r in result}
        for entry in annotated:
            if entry["case_id"] not in by_id:
                raise ValueError(f"annotation refers to unknown case_id: {entry['case_id']}")
            if entry.get("category", by_id[entry["case_id"]]["category"]) not in CATEGORIES:
                raise ValueError("unknown difficult-case category")
            if "manual_verified" in entry and not isinstance(entry["manual_verified"], bool):
                raise ValueError("manual_verified must be a boolean")
            by_id[entry["case_id"]].update({k: entry[k] for k in ("category", "manual_verified", "notes", "start_seconds", "end_seconds") if k in entry})
            case = by_id[entry["case_id"]]
            if float(case["start_seconds"]) < 0 or float(case["end_seconds"]) <= float(case["start_seconds"]):
                raise ValueError("invalid annotated difficult-case time range")
    return result


def export_paired_clip(case: dict[str, Any], destination: Path, *, decoder: str = "pyav") -> dict[str, Any]:
    import cv2
    import numpy as np
    start, end = float(case["start_seconds"]), float(case["end_seconds"])
    if start < 0 or end <= start:
        raise ValueError("invalid difficult-case time range")
    left, right = VideoReader(Path(case["reference"]), decoder), VideoReader(Path(case["candidate"]), decoder)
    writer, frames = None, 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        for a, b in zip(iter(left), iter(right), strict=True):
            if a.timestamp < start:
                continue
            if a.timestamp >= end:
                break
            if a.image.shape != b.image.shape or abs(a.timestamp - b.timestamp) > 1e-4:
                raise ValueError("difficult-case paired frames differ in shape or time")
            panel = np.concatenate([a.image, b.image], axis=1)
            cv2.putText(panel, "BF16", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, str(case.get("method", "TempoKV")), (a.image.shape[1] + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if writer is None:
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), left.fps, (panel.shape[1], panel.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot create clip {destination}")
                cv2.imwrite(str(destination.with_suffix(".png")), panel)
            writer.write(panel)
            frames += 1
    finally:
        left.close()
        right.close()
        if writer is not None:
            writer.release()
    if not frames:
        raise ValueError("difficult-case clip is empty")
    return {"path": str(destination), "frames": frames, "fps": left.fps, "display_only_reencoded": True}
