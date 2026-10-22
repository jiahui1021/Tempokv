"""Exclusive, bounded, pickle-free storage of one full attention event."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from .context import FrozenAttention, ProbeContext
from tempokv.spectral_kv.model_spec import ModelSpec
from tempokv.spectral_kv.rope3d import rope_apply


SCHEMA_VERSION = "research.frozen.v1"
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
_STORAGE = {"bfloat16": np.dtype("float32"), "float16": np.dtype("float16"),
            "float32": np.dtype("float32"), "float64": np.dtype("float64")}
_REQUIRED = {"q_post", "k_post", "v", "query_abs_positions", "key_abs_positions", "query_coordinates",
             "key_coordinates", "protected_mask", "eligible_mask", "valid_key_mask", "metadata_json"}
_OPTIONAL = {"k_pre", "pre_valid_mask", "freqs"}


def _limit(max_capture_mb):
    if isinstance(max_capture_mb, bool) or not math.isfinite(max_capture_mb) or max_capture_mb <= 0:
        raise ValueError("max_capture_mb must be finite and positive")
    return int(max_capture_mb * 1024 ** 2)


def _json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value): raise ValueError("metadata must be finite")
        return value
    if isinstance(value, np.generic): return _json(value.item())
    if isinstance(value, (list, tuple)): return [_json(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value): raise ValueError("metadata keys must be strings")
        return {key: _json(item) for key, item in value.items()}
    raise TypeError("metadata must contain JSON scalars, not tensors or model/cache history")


def _coordinates(positions, height, width):
    positions = positions.detach().to(device="cpu", dtype=torch.int64)
    frame_tokens = height * width
    return torch.stack((positions // frame_tokens, (positions % frame_tokens) // width, positions % width), dim=1)


def _validate(window, metadata):
    if not isinstance(window, FrozenAttention): raise TypeError("window must be FrozenAttention")
    context, q, k, v = window.context, window.q_post, window.k_post, window.v
    # FrozenAttention is immutable, but its tensors may have been resized after construction.
    window.__post_init__()
    for name in ("seed", "sample_id", "layer_id", "head_id", "chunk_id", "current_chunk_start", "sink_frames", "protect_recent_frames"):
        value = getattr(context, name)
        if type(value) is not int or value < 0: raise ValueError(f"invalid context {name}")
    for name in ("grid_t", "grid_h", "grid_w"):
        if type(getattr(context, name)) is not int or getattr(context, name) < 1:
            raise ValueError("runtime grid must contain positive integer coordinates")
    for name in ("denoise_step_index", "denoise_timestep"):
        value = getattr(context, name)
        if value is not None and (type(value) is not int or value < 0): raise ValueError(f"invalid context {name}")
    if not isinstance(context.source, str) or not context.source:
        raise ValueError("source must be explicit")
    if q.shape[0] != 1 or min(q.shape[1:]) < 1 or min(k.shape[1:]) < 1:
        raise ValueError("capture one nonempty sample and selected head(s), not a batched history dump")
    dtype = str(q.dtype).removeprefix("torch.")
    if dtype not in _DTYPES: raise ValueError("unsupported original floating dtype")
    for tensor in (q, k, v):
        if not bool(torch.isfinite(tensor).all()): raise ValueError("Q/K/V must contain only finite values")
    if window.k_pre is not None:
        if window.k_pre.dtype != q.dtype or window.k_pre.device != q.device or not bool(torch.isfinite(window.k_pre).all()):
            raise ValueError("pre-RoPE K must be finite and preserve native dtype/device")
    elif window.pre_valid_mask is not None:
        raise ValueError("pre_valid_mask requires captured pre-RoPE K")
    context.__post_init__()
    fs = context.frame_tokens
    positions = context.key_abs_positions.detach().cpu()
    if positions.numel() % fs:
        raise ValueError("partial K/V frames are unsupported; capture the full effective K/V window")
    frames = positions.reshape(-1, fs)
    expected = frames[:, :1] + torch.arange(fs)
    if not torch.equal(frames, expected) or bool((frames[:, 0] % fs != 0).any()):
        raise ValueError("K/V positions must contain all-valid dense frame-major grids without padding")
    end = context.current_chunk_start + context.grid_t * fs
    current = torch.arange(context.current_chunk_start, end)
    actual_current = positions[positions >= context.current_chunk_start]
    if not torch.equal(actual_current, current):
        raise ValueError("partial K/V window: every current-chunk key must be present")
    if bool((context.query_abs_positions >= end).any()):
        raise ValueError("sampled Q positions are outside the runtime current-chunk grid")
    sinks = torch.arange(min(context.sink_frames * fs, end))
    if sinks.numel() and not torch.equal(positions[:sinks.numel()], sinks):
        raise ValueError("full effective window must preserve its declared sink frames")
    for key in ("key_token_count", "full_key_token_count", "expected_key_token_count", "effective_key_tokens"):
        if metadata.get(key) is not None and metadata[key] != k.shape[1]:
            raise ValueError("metadata identifies a partial K/V window")
    if metadata.get("kv_scope", "full_effective_window") != "full_effective_window" or metadata.get("full_effective_kv", True) is not True:
        raise ValueError("only a full effective K/V window can be archived")
    if metadata.get("valid_key_mask", "all_true_no_padding") not in ("all_true_no_padding", "all_true", True):
        raise ValueError("only all-valid dense grids without padding are supported")
    if metadata.get("layout", "frame_major_row_major") != "frame_major_row_major":
        raise ValueError("only frame-major row-major runtime layout is supported")
    if metadata.get("runtime_grid") is not None and metadata["runtime_grid"] != [context.grid_t, context.grid_h, context.grid_w]:
        raise ValueError("metadata runtime grid disagrees with context")
    return dtype


def _rotation(window, metadata, freqs):
    source = None
    theta = None
    if freqs is not None:
        if (not isinstance(freqs, torch.Tensor) or freqs.ndim != 2 or not freqs.is_complex()
                or freqs.dtype not in (torch.complex64, torch.complex128) or freqs.shape[0] < 1
                or freqs.shape[1] != window.q_post.shape[-1] // 2 or not bool(torch.isfinite(freqs).all())):
            raise ValueError("freqs must be a finite complex runtime table [positions, head_dim/2]")
        tolerance = 2e-5 if freqs.dtype == torch.complex64 else 1e-10
        if not torch.allclose(freqs.abs(), torch.ones_like(freqs.real), atol=tolerance, rtol=tolerance):
            raise ValueError("runtime freqs must be unit-magnitude rotations")
        maximum = int(_coordinates(window.context.key_abs_positions, window.context.grid_h, window.context.grid_w).max())
        if freqs.shape[0] <= maximum:
            raise ValueError("runtime freqs do not cover captured grid coordinates")
        source = "runtime_freqs"
    elif metadata.get("rope_default_theta") is not None:
        theta, source = metadata["rope_default_theta"], "default_theta"
    elif metadata.get("rope_source") == "default_theta":
        if metadata.get("rope_theta") is None: raise ValueError("explicit default-theta source requires rope_theta")
        theta, source = metadata["rope_theta"], "default_theta"
    if source is None:
        return {"source": "unavailable", "precision": None, "model_spec": None,
                "reason": "runtime freqs or an explicit default-theta declaration was not captured"}
    precision = metadata.get("rope_precision") or ("float32" if freqs is not None and freqs.dtype == torch.complex64 else "float64")
    if precision == "native":
        if freqs is None: raise ValueError("native RoPE precision requires a captured runtime frequency table")
        precision = "float32" if freqs.dtype == torch.complex64 else "float64"
    precision = {"complex64": "float32", "complex128": "float64"}.get(precision, precision)
    if precision not in {"float32", "float64"}: raise ValueError("unknown RoPE precision")
    d = window.q_post.shape[-1]
    channels = d // 2
    split = metadata.get("rope_split") or (metadata.get("model_spec") or {}).get("rope_split")
    if split is None: split = (2 * (channels - 2 * (channels // 3)), 2 * (channels // 3), 2 * (channels // 3))
    spec = ModelSpec(window.context.layer_id + 1, window.q_post.shape[2], d,
                     (window.context.grid_h, window.context.grid_w), window.context.grid_t,
                     tuple(split), 10000. if theta is None else theta)
    return {"source": source, "precision": precision, "model_spec": asdict(spec)}


def _inspect(path, limit):
    """Validate ZIP sizes and NPY headers before NumPy can allocate any arrays."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)): raise ValueError("duplicate archive members")
            expected = {name + ".npy" for name in _REQUIRED | _OPTIONAL}
            if any(name not in expected for name in names): raise ValueError("unknown frozen archive member")
            if not {name + ".npy" for name in _REQUIRED}.issubset(names): raise ValueError("missing required grid coordinates or frozen arrays")
            if sum(info.file_size for info in infos) > limit:
                raise ValueError("uncompressed frozen archive exceeds max_capture_mb")
            for info in infos:
                with archive.open(info) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                    else: raise ValueError("unsupported NPY header version")
                    if dtype.hasobject or len(shape) > 4 or any(type(size) is not int or size < 0 for size in shape):
                        raise ValueError("unsafe object or unknown array shape in frozen archive")
                    amount = math.prod(shape) * dtype.itemsize
                    if amount > limit or amount + handle.tell() != info.file_size:
                        raise ValueError("declared array shape or size exceeds the bounded member payload")
    except (zipfile.BadZipFile, EOFError, OverflowError) as exc:
        raise ValueError("invalid frozen NPZ archive") from exc


@torch.no_grad()
def save_frozen(path, window: FrozenAttention, *, max_capture_mb=256, metadata=None, freqs=None) -> Path:
    """Atomically write a new one-event NPZ, refusing an existing destination."""
    limit = _limit(max_capture_mb)
    destination = Path(path).expanduser().absolute()
    if os.path.lexists(destination): raise FileExistsError(destination)
    metadata = _json({} if metadata is None else metadata)
    if not isinstance(metadata, dict): raise ValueError("metadata must be a dictionary")
    if not isinstance(window, FrozenAttention): raise TypeError("window must be FrozenAttention")
    if any(tensor.ndim != 4 for tensor in (window.q_post, window.k_post, window.v)):
        raise ValueError("unknown Q/K/V shape; expected [B,Seq,Heads,D]")
    # Reject a large event before finite-value scans or CPU serialization copies.
    # BF16 must be budgeted at its exact FP32 storage size, including sparse pre-K.
    raw_bytes = sum(tensor.numel() * (4 if tensor.dtype == torch.bfloat16 else tensor.element_size())
                    for tensor in (window.q_post, window.k_post, window.v, window.k_pre) if tensor is not None)
    raw_bytes += (window.q_post.shape[1] + window.k_post.shape[1]) * 32 + window.k_post.shape[1] * 3
    if window.pre_valid_mask is not None: raw_bytes += window.pre_valid_mask.numel()
    if isinstance(freqs, torch.Tensor): raw_bytes += freqs.numel() * freqs.element_size()
    if raw_bytes > limit: raise ValueError("float32 serialization and captured arrays exceed max_capture_mb")
    dtype = _validate(window, metadata)
    rope = _rotation(window, metadata, freqs)
    context = {item.name: getattr(window.context, item.name) for item in fields(ProbeContext)
               if item.name not in {"query_abs_positions", "key_abs_positions"}}
    kernel = f"{getattr(window.attention_fn, '__module__', 'unknown')}.{getattr(window.attention_fn, '__qualname__', type(window.attention_fn).__name__)}"
    header = {"schema_version": SCHEMA_VERSION, "context": context, "original_dtype": dtype,
              "runtime_grid": [window.context.grid_t, window.context.grid_h, window.context.grid_w],
              "source": window.context.source, "runtime_kernel": kernel, "attention_backend": "original_native_callback",
              "layout": "frame_major_row_major", "valid_key_mask": "all_true_no_padding",
              "kv_scope": "full_effective_window", "query_scope": "sampled_current_chunk",
              "protection": {"sink_frames": window.context.sink_frames, "protect_recent_frames": window.context.protect_recent_frames, "protect_current_chunk": True},
              "has_k_pre": window.k_pre is not None, "has_pre_valid_mask": window.pre_valid_mask is not None,
              "rope": rope, "metadata": metadata}
    arrays = {"query_abs_positions": window.context.query_abs_positions.detach().cpu().numpy(),
              "key_abs_positions": window.context.key_abs_positions.detach().cpu().numpy(),
              "query_coordinates": _coordinates(window.context.query_abs_positions, window.context.grid_h, window.context.grid_w).numpy(),
              "key_coordinates": _coordinates(window.context.key_abs_positions, window.context.grid_h, window.context.grid_w).numpy(),
              "protected_mask": window.context.protected_mask.detach().cpu().numpy(),
              "eligible_mask": window.context.eligible_mask.detach().cpu().numpy(),
              "valid_key_mask": np.ones(window.k_post.shape[1], dtype=bool),
              "metadata_json": np.array(json.dumps(_json(header), sort_keys=True, allow_nan=False))}
    if window.pre_valid_mask is not None: arrays["pre_valid_mask"] = window.pre_valid_mask.detach().cpu().numpy()
    if freqs is not None: arrays["freqs"] = freqs.detach().cpu().numpy()
    tensor_names = ("q_post", "k_post", "v", "k_pre")
    estimated = sum(value.nbytes for value in arrays.values()) + sum(
        getattr(window, name).numel() * _STORAGE[dtype].itemsize for name in tensor_names if getattr(window, name) is not None)
    if estimated > limit: raise ValueError("float32 serialization and captured arrays exceed max_capture_mb")
    for name in tensor_names:
        tensor = getattr(window, name)
        if tensor is not None:
            arrays[name] = tensor.detach().to(device="cpu", dtype=torch.float32 if dtype == "bfloat16" else tensor.dtype).numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz.tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        _inspect(temporary, limit)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _cpu_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()


def _read_array(archive, name, *, dtype=None, ndim=None):
    value = archive[name]
    if dtype is not None and value.dtype != np.dtype(dtype): raise ValueError(f"invalid dtype for {name}")
    if ndim is not None and value.ndim != ndim: raise ValueError(f"invalid shape for {name}")
    return value


@torch.no_grad()
def load_frozen(path, *, attention_fn=None, device="cpu", max_capture_mb=256) -> tuple[FrozenAttention, dict]:
    """Load bounded arrays without pickle; restore only explicitly captured RoPE."""
    source = Path(path).expanduser().resolve()
    _inspect(source, _limit(max_capture_mb))
    device = torch.device(device)
    if attention_fn is None and device.type != "cpu":
        raise ValueError("non-CPU replay requires an injected runtime attention_fn")
    with np.load(source, allow_pickle=False) as archive:
        text = _read_array(archive, "metadata_json", ndim=0)
        if text.dtype.kind != "U": raise ValueError("metadata_json must be a Unicode JSON scalar")
        try: header = json.loads(str(text.item()))
        except (ValueError, TypeError) as exc: raise ValueError("invalid frozen metadata JSON") from exc
        if not isinstance(header, dict) or header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported frozen schema")
        required_header = {"context", "runtime_grid", "original_dtype", "source", "runtime_kernel", "layout", "valid_key_mask", "kv_scope", "protection", "rope", "metadata", "has_k_pre", "has_pre_valid_mask"}
        if not required_header.issubset(header): raise ValueError("missing runtime grid or frozen metadata")
        dtype_name = header["original_dtype"]
        if dtype_name not in _DTYPES: raise ValueError("unknown original dtype")
        positions = {name: torch.from_numpy(_read_array(archive, name, dtype=np.int64, ndim=1).copy())
                     for name in ("query_abs_positions", "key_abs_positions")}
        scalar_names = {item.name for item in fields(ProbeContext)} - set(positions)
        if not isinstance(header["context"], dict) or set(header["context"]) != scalar_names:
            raise ValueError("missing or unknown context fields")
        context = ProbeContext(**header["context"], **positions)
        tensors = {}
        for name in ("q_post", "k_post", "v", "k_pre"):
            if name not in archive.files:
                if name != "k_pre": raise ValueError(f"missing {name}")
                continue
            array = _read_array(archive, name, dtype=_STORAGE[dtype_name], ndim=4)
            tensors[name] = torch.from_numpy(array.copy()).to(device=device, dtype=_DTYPES[dtype_name])
            if dtype_name == "bfloat16" and not np.array_equal(tensors[name].float().cpu().numpy(), array):
                raise ValueError("stored FP32 values are not exact representations of original BF16")
        pre_mask = torch.from_numpy(_read_array(archive, "pre_valid_mask", dtype=bool, ndim=1).copy()).to(device) if "pre_valid_mask" in archive.files else None
        if bool(header["has_k_pre"]) != ("k_pre" in tensors) or bool(header["has_pre_valid_mask"]) != (pre_mask is not None):
            raise ValueError("pre-RoPE capture declarations disagree with arrays")
        callback = _cpu_sdpa if attention_fn is None else attention_fn
        if not callable(callback): raise TypeError("attention_fn must be callable")
        window = FrozenAttention(context, tensors["q_post"], tensors["k_post"], tensors["v"], callback,
                                 tensors.get("k_pre"), None, pre_mask)
        metadata = header["metadata"]
        if not isinstance(metadata, dict): raise ValueError("invalid user metadata")
        _validate(window, metadata)
        expected_grid = [context.grid_t, context.grid_h, context.grid_w]
        expected_protection = {"sink_frames": context.sink_frames, "protect_recent_frames": context.protect_recent_frames, "protect_current_chunk": True}
        if (header["runtime_grid"] != expected_grid or header["source"] != context.source
                or header["layout"] != "frame_major_row_major" or header["kv_scope"] != "full_effective_window"
                or header["valid_key_mask"] != "all_true_no_padding" or header["protection"] != expected_protection):
            raise ValueError("runtime grid, source, layout or protection metadata is inconsistent")
        for prefix in ("query", "key"):
            coords = _read_array(archive, f"{prefix}_coordinates", dtype=np.int64, ndim=2)
            expected = _coordinates(positions[f"{prefix}_abs_positions"], context.grid_h, context.grid_w).numpy()
            if not np.array_equal(coords, expected): raise ValueError("corrupt frame-major grid coordinates")
        for name, expected in (("protected_mask", context.protected_mask), ("eligible_mask", context.eligible_mask)):
            if not np.array_equal(_read_array(archive, name, dtype=bool, ndim=1), expected.numpy()):
                raise ValueError("protection mask disagrees with absolute positions")
        valid = _read_array(archive, "valid_key_mask", dtype=bool, ndim=1)
        if valid.shape != (window.k_post.shape[1],) or not bool(valid.all()):
            raise ValueError("padding or invalid key positions are unsupported")
        freqs = torch.from_numpy(archive["freqs"].copy()).to(device) if "freqs" in archive.files else None
        rope = header["rope"]
        if not isinstance(rope, dict): raise ValueError("invalid RoPE metadata")
        rotation_source = rope.get("source")
        if rotation_source == "runtime_freqs" and freqs is None: raise ValueError("runtime rotation table is missing")
        if rotation_source not in {"runtime_freqs", "default_theta", "unavailable"}: raise ValueError("unknown rotation source")
        if freqs is not None and rotation_source != "runtime_freqs": raise ValueError("undeclared runtime frequency table")
        if rotation_source != "unavailable":
            try: spec = ModelSpec(**rope["model_spec"])
            except (KeyError, TypeError, ValueError) as exc: raise ValueError("invalid archived runtime ModelSpec") from exc
            if (spec.heads, spec.head_dim, spec.frame_hw, spec.chunk_frames) != (window.q_post.shape[2], window.q_post.shape[-1], (context.grid_h, context.grid_w), context.grid_t):
                raise ValueError("archived ModelSpec disagrees with runtime grid")
            declarations = {"rope_precision": rope.get("precision"), "rope_split": list(spec.rope_split)}
            if rotation_source == "default_theta": declarations["rope_default_theta"] = spec.rope_theta
            restored = _rotation(window, declarations, freqs)
            if restored["precision"] != rope.get("precision"): raise ValueError("invalid rotation precision")
            def rotate_pre(pre, coordinates):
                if not isinstance(coordinates, torch.Tensor) or coordinates.shape != (pre.shape[1], 3):
                    raise ValueError("rotation coordinates must be [tokens,3]")
                return rope_apply(pre, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], spec,
                                  freqs=freqs, precision=rope["precision"])
            window = FrozenAttention(context, window.q_post, window.k_post, window.v, callback,
                                     window.k_pre, rotate_pre, pre_mask)
    result = {**metadata, **header, "archive_path": str(source), "rotation_available": window.rotate_pre is not None,
              "attention_backend": "torch_cpu_sdpa" if attention_fn is None else "injected_runtime_callback"}
    return window, result
