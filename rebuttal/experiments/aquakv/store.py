"""Calibration samples on disk, one Self-Forcing video at a time.

A full BF16 KV trace does not fit in the ``torch.stack`` that upstream
``train_predictors.py`` uses. The normal equations are still the official
ones: each training video is streamed in ``chunk_size`` rows, validation
videos are reconstructed and propagated but never added to ``CXX`` / ``CXY``.
Reconstructions of the previous layer are the only scratch, and they can be
deleted after the next layer is written.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .method import (
    AquaSpec,
    PredictorBank,
    apply_linear,
    flatten_tokens,
    make_quantizers,
    project_layer,
    split_ids,
    unflatten_tokens,
)
from .fit import chunks, fit_streaming


def _bitcast_to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    copied = np.array(array, copy=True)
    if dtype == torch.bfloat16:
        if copied.dtype != np.int16:
            raise ValueError("bf16 memmap is stored as an int16 bitcast")
        return torch.from_numpy(copied).view(torch.bfloat16)
    if dtype == torch.float16:
        return torch.from_numpy(copied.view(np.float16))
    if dtype == torch.float32:
        return torch.from_numpy(copied.astype(np.float32, copy=False))
    raise ValueError(f"unsupported sample dtype {dtype}")


def _tensor_to_bitcast(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        return tensor.detach().contiguous().view(torch.int16).cpu().numpy()
    if tensor.dtype == torch.float16:
        return tensor.detach().contiguous().view(torch.int16).cpu().numpy()
    if tensor.dtype == torch.float32:
        return tensor.detach().contiguous().cpu().numpy()
    raise ValueError(f"unsupported sample dtype {tensor.dtype}")


class ListSampleSource:
    """In-memory videos, each ``(keys, values)`` shaped ``[L, T, H, D]``."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor]],
                 identities: list[dict[str, Any]] | None = None):
        if len(samples) < 1:
            raise ValueError("sample source is empty")
        self.samples = samples
        layers, tokens, heads, dim = samples[0][0].shape
        self._tokens = int(tokens)
        self._shape = (layers, heads, dim)
        for keys, values in samples:
            if keys.shape != values.shape or keys.shape[0] != layers or keys.shape[1] != tokens:
                raise ValueError("every calibration video must share [L, T, H, D]")
        self._identities = identities or [
            {"prompt_index": i, "prompt_sha256": f"synthetic-{i}", "seed": 0} for i in range(len(samples))
        ]
        if len(self._identities) != len(samples):
            raise ValueError("identity count does not match the sample count")

    def __len__(self) -> int:
        return len(self.samples)

    def tokens(self) -> int:
        return self._tokens

    def layer_kv(self, index: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self.samples[index]
        return keys[layer], values[layer]

    def identities(self) -> list[dict[str, Any]]:
        return list(self._identities)


class MemmapSampleSource:
    """Read the collector layout ``rank-*/prompt-*/{meta.json,k.i16,v.i16}``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.dirs = discover_samples(self.root)
        if not self.dirs:
            raise FileNotFoundError(f"no AQUA-KV calibration samples under {self.root}")
        self.metas = [json.loads((path / "meta.json").read_text(encoding="utf-8")) for path in self.dirs]
        first = self.metas[0]
        self._tokens = int(first["frames"]) * int(first["frame_seq_length"])
        self._shape = (int(first["layers"]), int(first["num_heads"]), int(first["head_dim"]))
        seen = set()
        for folder, meta in zip(self.dirs, self.metas):
            shape = (int(meta["layers"]), int(meta["frames"]) * int(meta["frame_seq_length"]),
                     int(meta["num_heads"]), int(meta["head_dim"]))
            if shape[0] != self._shape[0] or shape[1] != self._tokens or shape[2:] != self._shape[1:]:
                raise ValueError("calibration videos do not share one KV shape")
            if meta.get("storage") != "int16_bitcast" or meta.get("dtype") != "bfloat16":
                raise ValueError("calibration memmap must be a bf16 int16 bitcast")
            if meta.get("complete") is not True:
                raise ValueError(f"incomplete calibration sample {folder}")
            identity = (meta["prompt_sha256"], int(meta["seed"]))
            if identity in seen:
                raise ValueError(f"duplicate calibration video {identity}")
            seen.add(identity)

    def __len__(self) -> int:
        return len(self.dirs)

    def tokens(self) -> int:
        return self._tokens

    def layer_kv(self, index: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        meta = self.metas[index]
        shape = (int(meta["layers"]), self._tokens, int(meta["num_heads"]), int(meta["head_dim"]))
        folder = self.dirs[index]
        keys = np.memmap(folder / "k.i16", dtype=np.int16, mode="r", shape=shape)
        values = np.memmap(folder / "v.i16", dtype=np.int16, mode="r", shape=shape)
        return _bitcast_to_tensor(keys[layer], torch.bfloat16), _bitcast_to_tensor(values[layer], torch.bfloat16)

    def identities(self) -> list[dict[str, Any]]:
        return [
            {"prompt_index": int(meta["prompt_index"]), "prompt_sha256": meta["prompt_sha256"],
             "seed": int(meta["seed"])}
            for meta in self.metas
        ]


def discover_samples(root: Path) -> list[Path]:
    nested = sorted(path.parent for path in root.glob("rank-*/prompt-*/meta.json"))
    flat = sorted(path.parent for path in root.glob("prompt-*/meta.json"))
    return nested or flat


def write_sample(folder: str | Path, keys: torch.Tensor, values: torch.Tensor, identity: dict[str, Any],
                 *, frame_seq_length: int) -> Path:
    """Write one video. ``keys`` / ``values`` are ``[L, T, H, D]`` bf16 or cast to it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("sample must be [L, T, H, D]")
    if keys.shape[1] % frame_seq_length:
        raise ValueError("token count must be a whole number of frames")
    stored_k = keys.detach().to(dtype=torch.bfloat16).contiguous()
    stored_v = values.detach().to(dtype=torch.bfloat16).contiguous()
    _tensor_to_bitcast(stored_k).tofile(folder / "k.i16")
    _tensor_to_bitcast(stored_v).tofile(folder / "v.i16")
    meta = {
        "format": "aquakv_self_forcing_kv_v1",
        "prompt_index": int(identity["prompt_index"]),
        "prompt_sha256": identity["prompt_sha256"],
        "seed": int(identity.get("seed", 0)),
        "layers": int(keys.shape[0]),
        "frames": int(keys.shape[1] // frame_seq_length),
        "frame_seq_length": int(frame_seq_length),
        "num_heads": int(keys.shape[2]),
        "head_dim": int(keys.shape[3]),
        "batch": 1,
        "dtype": "bfloat16",
        "storage": "int16_bitcast",
        "layout": "[layer, token, head, dim]",
        "complete": True,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return folder


class _TensorScratch:
    def __init__(self) -> None:
        self._items: dict[tuple[int, int, str], torch.Tensor] = {}

    def write(self, layer: int, index: int, role: str, tensor: torch.Tensor) -> None:
        self._items[(layer, index, role)] = tensor.detach()

    def read(self, layer: int, index: int, role: str) -> torch.Tensor:
        return self._items[(layer, index, role)]


class _MemmapScratch:
    def __init__(self, root: Path, *, tokens: int, heads: int, dim: int, dtype: torch.dtype):
        if dtype != torch.bfloat16:
            raise ValueError("disk scratch stores bf16 as an int16 bitcast")
        self.root = root
        self.tokens = tokens
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.shape = (tokens, heads, dim)
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, layer: int, index: int, role: str) -> Path:
        return self.root / f"layer{layer}_sample{index}_{role}.i16"

    def write(self, layer: int, index: int, role: str, tensor: torch.Tensor) -> None:
        array = _tensor_to_bitcast(tensor.detach().to(dtype=self.dtype).contiguous().reshape(self.shape))
        array.tofile(self._path(layer, index, role))

    def read(self, layer: int, index: int, role: str) -> torch.Tensor:
        array = np.memmap(self._path(layer, index, role), dtype=np.int16, mode="r", shape=self.shape)
        return _bitcast_to_tensor(np.array(array), self.dtype)

    def drop_layer(self, layer: int) -> None:
        for path in self.root.glob(f"layer{layer}_sample*_*.i16"):
            path.unlink()


def _rows(source, scratch, layer: int, role: str, ids: Iterable[int], spec: AquaSpec, *, value: bool):
    for index in ids:
        prev_k = scratch.read(layer - 1, index, "k")
        target_k, target_v = source.layer_kv(index, layer)
        if value:
            current_k = scratch.read(layer, index, "k")
            prev_v = scratch.read(layer - 1, index, "v")
            features = torch.cat([flatten_tokens(current_k), flatten_tokens(prev_v)], dim=-1)
            target = flatten_tokens(target_v)
        else:
            features = flatten_tokens(prev_k)
            target = flatten_tokens(target_k)
        yield from chunks(features, target, spec.chunk_size)


@torch.no_grad()
def calibrate_streaming(source, spec: AquaSpec, *, scratch=None, device: torch.device | None = None,
                        ) -> tuple[PredictorBank, dict[str, Any]]:
    """Same sequential fit as ``method.calibrate``, one video at a time."""
    if len(source) < 2:
        raise ValueError("calibration needs at least one training video and one validation video")
    device = torch.device("cpu") if device is None else torch.device(device)
    sample_k, _ = source.layer_kv(0, 0)
    if sample_k.ndim != 3:
        raise ValueError("layer_kv must return [T, H, D]")
    tokens, heads, dim = sample_k.shape
    if source.tokens() != tokens or (heads, dim) != (spec.num_heads, spec.head_dim):
        raise ValueError("sample source does not match the AQUA spec")
    # Layer count is taken from the first video's store; ListSampleSource checks all.
    probe = source.layer_kv(0, spec.num_layers - 1)
    if probe[0].shape != sample_k.shape:
        raise ValueError("last layer shape does not match layer 0")
    train_ids, valid_ids = split_ids(len(source), spec.valid_nsamples, spec.seed, device)
    residual, first = make_quantizers(spec)
    own_scratch = scratch is None
    if scratch is None:
        scratch = _TensorScratch()
    for index in range(len(source)):
        keys, values = source.layer_kv(index, 0)
        hat_k, _, _ = project_layer(keys.to(dtype=spec.dtype), None, first)
        hat_v, _, _ = project_layer(values.to(dtype=spec.dtype), None, first)
        scratch.write(0, index, "k", hat_k.cpu())
        scratch.write(0, index, "v", hat_v.cpu())
    key_w: dict[int, torch.Tensor] = {}
    key_b: dict[int, torch.Tensor] = {}
    value_w: dict[int, torch.Tensor] = {}
    value_b: dict[int, torch.Tensor] = {}
    channel = spec.channel
    for layer in range(1, spec.num_layers):
        weight, bias = fit_streaming(
            _rows(source, scratch, layer, "k", train_ids, spec, value=False),
            in_features=channel, out_features=channel, reg_rate=spec.percdamp,
            out_dtype=spec.dtype, device=device,
        )
        key_w[layer] = weight
        key_b[layer] = bias
        for index in range(len(source)):
            prev = scratch.read(layer - 1, index, "k").to(device=device)
            pred = unflatten_tokens(apply_linear(weight, bias, flatten_tokens(prev)), heads, dim)
            target, _ = source.layer_kv(index, layer)
            hat, _, _ = project_layer(target.to(device=device, dtype=spec.dtype), pred, residual)
            scratch.write(layer, index, "k", hat.cpu())
        v_weight, v_bias = fit_streaming(
            _rows(source, scratch, layer, "v", train_ids, spec, value=True),
            in_features=channel * 2, out_features=channel, reg_rate=spec.percdamp,
            out_dtype=spec.dtype, device=device,
        )
        value_w[layer] = v_weight
        value_b[layer] = v_bias
        for index in range(len(source)):
            current_k = scratch.read(layer, index, "k").to(device=device)
            prev_v = scratch.read(layer - 1, index, "v").to(device=device)
            features = torch.cat([flatten_tokens(current_k), flatten_tokens(prev_v)], dim=-1)
            pred = unflatten_tokens(apply_linear(v_weight, v_bias, features), heads, dim)
            _, target = source.layer_kv(index, layer)
            hat, _, _ = project_layer(target.to(device=device, dtype=spec.dtype), pred, residual)
            scratch.write(layer, index, "v", hat.cpu())
        if isinstance(scratch, _MemmapScratch):
            scratch.drop_layer(layer - 1)
    bank = PredictorBank(spec, key_w, key_b, value_w, value_b, train_ids, valid_ids)
    report = {"train_indices": train_ids, "valid_indices": valid_ids, "tokens_per_video": int(tokens),
              "videos": len(source), "scratch": "caller" if not own_scratch else "memory"}
    return bank, report


def check_collection_against_prompts(source, calibration_prompts: str | Path,
                                     evaluation_prompts: str | Path) -> dict[str, Any]:
    """Collection records must be the calibration file, and must miss evaluation."""
    from rebuttal.common.manifest import check_prompt_split, prompt_records
    split = check_prompt_split(calibration_prompts, evaluation_prompts)
    calibration = prompt_records(calibration_prompts)
    by_index = {row["prompt_id"]: row["sha256"] for row in calibration}
    evaluation = {row["sha256"] for row in prompt_records(evaluation_prompts)}
    for record in source.identities():
        expected = by_index.get(int(record["prompt_index"]))
        if expected is None:
            raise ValueError(f"collection prompt_index {record['prompt_index']} is outside the calibration file")
        if record["prompt_sha256"] != expected:
            raise ValueError(
                f"collection prompt {record['prompt_index']} hash does not match the calibration file"
            )
        if record["prompt_sha256"] in evaluation:
            raise ValueError("collection leaks an evaluation prompt")
    return split
