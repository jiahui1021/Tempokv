"""Read existing K/V traces while keeping explicit record identities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import torch


@dataclass
class Trace:
    keys: torch.Tensor
    values: torch.Tensor
    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    path: Path
    layer_ids: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.keys.shape)


def device_for(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


@dataclass
class TraceIdentity:
    path: Path
    metadata: dict[str, Any]


def _record_identity(metadata: dict, shape: tuple, sidecar: Path):
    count = int(np.prod(shape[1:-3])) if len(shape) > 4 else 1
    records = [{} for _ in range(count)]
    # Existing trace writers use both per-record arrays and scalar event IDs.
    for field, plural in (("prompt_id", "prompt_ids"), ("prompt_sha256", "prompt_sha256s"),
                          ("seed", "seeds"), ("sample_id", "sample_ids")):
        if plural in metadata:
            values = metadata[plural]
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(f"{plural} must identify every trace record")
            for record, value in zip(records, values, strict=True):
                record[field] = value
        elif field in metadata and not isinstance(metadata[field], (list, dict)):
            for record in records:
                record[field] = metadata[field]
    if sidecar.exists():
        payload = json.loads(sidecar.read_text())
        entries = payload if isinstance(payload, list) else payload.get("records", [])
        if not isinstance(payload, list):
            metadata["identity_metadata"] = {k: v for k, v in payload.items() if k != "records"}
        seen = set()
        for entry in entries:
            begin = int(entry.get("record_index", entry.get("record_start", 0)))
            stop = int(entry.get("record_stop", begin + 1))
            if begin < 0 or stop <= begin or stop > count:
                raise ValueError("invalid identity record interval")
            values = {k: v for k, v in entry.items() if k not in {"record_start", "record_stop", "record_index"}}
            for index in range(begin, stop):
                if index in seen:
                    raise ValueError("overlapping identity record intervals")
                seen.add(index)
                for field in ("prompt_id", "prompt_sha256", "seed", "sample_id"):
                    if field in records[index] and field in values and str(records[index][field]) != str(values[field]):
                        raise ValueError(f"sidecar conflicts with embedded trace identity: {field}")
                records[index].update(values)
        metadata["identity_path"] = str(sidecar.resolve())
    summary = {
        "record_count": count,
        "complete": all(r.get("prompt_id") is not None for r in records),
        "hashes_complete": all(r.get("prompt_sha256") is not None for r in records),
        "prompt_ids": sorted({str(r["prompt_id"]) for r in records if r.get("prompt_id") is not None}),
        "prompt_hashes": sorted({str(r["prompt_sha256"]) for r in records if r.get("prompt_sha256") is not None}),
    }
    metadata["split_identity"] = summary
    metadata["record_identity_complete"] = all(r.get("prompt_id") is not None and r.get("seed") is not None for r in records)
    metadata["original_shape"] = list(shape)
    return records


def _metadata(data):
    result = {}
    for key in data.files:
        if key in {"keys", "values"} or key.startswith(("q_", "k_post")):
            continue
        array = np.asarray(data[key])
        if array.size <= 100000:
            result[key] = array.item() if array.ndim == 0 else array.tolist()
    return result


def load_trace_identity(path: str | Path, *, identity_path: str | Path | None = None) -> TraceIdentity:
    """Check every record's identity without decompressing the KV tensor arrays."""
    import zipfile
    path = Path(path).expanduser().resolve()
    shapes = []
    with zipfile.ZipFile(path) as archive:
        for name in ("keys.npy", "values.npy"):
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported trace NPY format: {version}")
                shapes.append(shape)
    if shapes[0] != shapes[1] or len(shapes[0]) < 4:
        raise ValueError("trace must contain matching [L,...,T,H,D] arrays")
    with np.load(path, allow_pickle=False) as data:
        metadata = _metadata(data)
    sidecar = Path(identity_path) if identity_path else path.with_suffix(".identity.json")
    _record_identity(metadata, shapes[0], sidecar)
    return TraceIdentity(path, metadata)


def load_trace(path: str | Path, *, device: str | torch.device = "cpu",
               identity_path: str | Path | None = None,
               max_records: int | None = None) -> Trace:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        if not {"keys", "values"}.issubset(data.files):
            raise ValueError(f"trace requires keys and values: {path}")
        k, v = np.asarray(data["keys"]), np.asarray(data["values"])
        if k.shape != v.shape or k.ndim < 4:
            raise ValueError("trace must contain matching [L,...,T,H,D] arrays")
        shape = k.shape
        metadata = _metadata(data)
        sidecar = Path(identity_path) if identity_path else path.with_suffix(".identity.json")
        # Build the full split identity before any diagnostic sampling.
        records = _record_identity(metadata, shape, sidecar)
        k, v = (x.reshape(shape[0], -1, *shape[-3:]) for x in (k, v))
        if max_records is not None:
            if isinstance(max_records, bool) or int(max_records) != max_records or max_records <= 0:
                raise ValueError("max_records must be a positive integer")
            k, v = k[:, :int(max_records)], v[:, :int(max_records)]
            records = records[:int(max_records)]
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise ValueError("trace contains non-finite values")
        keys = torch.from_numpy(np.array(k, dtype=np.float32, copy=True)).to(device)
        values = torch.from_numpy(np.array(v, dtype=np.float32, copy=True)).to(device)
    ids = metadata.get("layer_ids", list(range(keys.shape[0])))
    if len(ids) != keys.shape[0]:
        raise ValueError("layer_ids do not match trace layers")
    return Trace(keys, values, metadata, records, path, tuple(int(x) for x in ids))


def check_trace_split(calibration: Trace | TraceIdentity, evaluation: Trace | TraceIdentity,
                      *, allow_in_sample: bool = False) -> dict[str, Any]:
    if calibration.path == evaluation.path and not allow_in_sample:
        raise ValueError("calibration and evaluation trace are identical; explicit allow_in_sample is for diagnostics only")
    def identity(trace):
        if "split_identity" in trace.metadata:
            return trace.metadata["split_identity"]
        records = getattr(trace, "records", [])
        return {"complete": bool(records) and all(r.get("prompt_id") is not None for r in records),
                "hashes_complete": bool(records) and all(r.get("prompt_sha256") is not None for r in records),
                "prompt_ids": [str(r["prompt_id"]) for r in records if r.get("prompt_id") is not None],
                "prompt_hashes": [str(r["prompt_sha256"]) for r in records if r.get("prompt_sha256") is not None]}
    left, right = identity(calibration), identity(evaluation)
    if set(left["prompt_hashes"]) & set(right["prompt_hashes"]) and not allow_in_sample:
        raise ValueError("calibration and evaluation trace prompt hashes overlap")
    basis = "prompt_hashes" if left["hashes_complete"] and right["hashes_complete"] else "prompt_ids"
    first, second = set(left[basis]), set(right[basis])
    if first & second and not allow_in_sample:
        raise ValueError("calibration and evaluation trace prompt identities overlap")
    return {"same_file": calibration.path == evaluation.path,
            "prompt_identity_verified": bool(left["complete"] and right["complete"] and first and second),
            "identity_basis": basis, "scope": "all_trace_records_before_sampling",
            "in_sample_diagnostic": bool(allow_in_sample)}


def convert_identified_events(raw_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Convert one clean-generation event at a time; never pool unrelated prompts."""
    from .manifest import write_json, file_sha256
    from rebuttal.adapters.observer import wan_pixel_interval
    raw_root, output_dir = Path(raw_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for event_dir in sorted({p.parent for p in raw_root.rglob("layer_*.npz")}):
        identity_path = event_dir / "identity.json"
        if not identity_path.exists():
            raise ValueError(f"trace event is missing prompt/seed identity: {event_dir}")
        identity = json.loads(identity_path.read_text())
        if not {"prompt_id", "seed", "fps", "pixel_frames"}.issubset(identity):
            raise ValueError("incomplete clean trace event identity")
        keys, values, layer_ids = [], [], []
        metadata = None
        for file in sorted(event_dir.glob("layer_*.npz")):
            with np.load(file, allow_pickle=False) as data:
                k, v = np.asarray(data["k_pre_rope"]), np.asarray(data["v"])
                if k.shape != v.shape or k.ndim != 5:
                    raise ValueError("raw clean event must have [B,F,P,H,D] arrays")
                keys.append(k.transpose(0, 2, 1, 3, 4).reshape(-1, k.shape[1], *k.shape[-2:]))
                values.append(v.transpose(0, 2, 1, 3, 4).reshape(-1, v.shape[1], *v.shape[-2:]))
                layer_ids.append(int(data["layer_index"]))
                current = {"tokens_start": int(data["tokens_start"]), "tokens_end": int(data["tokens_end"]),
                           "frame_seq_length": int(data["frame_seq_length"]), "token_indices": data["token_indices"].tolist(),
                           "batch_size": k.shape[0], "frames": k.shape[1], "positions": k.shape[2]}
                if metadata is not None and current != metadata:
                    raise ValueError("raw event layers have different temporal/spatial coordinates")
                metadata = current
        begin = metadata["tokens_start"] // metadata["frame_seq_length"]
        path = output_dir / f"{event_dir.parent.name}_{event_dir.name}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite converted trace: {path}")
        np.savez(path, keys=np.stack(keys), values=np.stack(values), layer_ids=np.asarray(layer_ids),
                 stride=np.asarray(metadata["frame_seq_length"]), coordinate_system=np.asarray("pre_rope_normalized"),
                 trace_role=np.asarray("diagnostic"), prompt_id=np.asarray(str(identity["prompt_id"])),
                 seed=np.asarray(identity["seed"]), token_indices=np.asarray(metadata["token_indices"]))
        records = []
        intervals = [list(wan_pixel_interval(begin+frame, int(identity["pixel_frames"]))) for frame in range(metadata["frames"])]
        for batch in range(metadata["batch_size"]):
            records.append({**identity, "record_start": batch*metadata["positions"],
                            "record_stop": (batch+1)*metadata["positions"], "sample_id": batch,
                            "absolute_start_frame": begin, "pixel_intervals": intervals,
                            "token_indices": metadata["token_indices"]})
        write_json(path.with_suffix(".identity.json"), {"records": records, "source": str(event_dir)})
        entries.append({"trace": str(path), "sha256": file_sha256(path), "identity": identity,
                        "shape": [len(keys), *keys[0].shape]})
    if not entries:
        raise ValueError("no raw clean events were found")
    report = {"status": "complete", "traces": entries, "pooling": "one_prompt_event_per_file"}
    write_json(output_dir / "index.json", report)
    return report
