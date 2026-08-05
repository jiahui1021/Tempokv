from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().contiguous()


@dataclass(frozen=True, slots=True)
class TemporalParameters:
    """Warm-up-aware multi-history K predictor parameters."""

    history_weights: torch.Tensor
    history_bias: torch.Tensor
    stride: int | None = None
    layer_ids: tuple[int, ...] | None = None
    codec_signature: str | None = None
    parameter_set_id: str | None = None
    coordinate_system: str | None = None

    def __post_init__(self) -> None:
        weights = _tensor(self.history_weights)
        bias = _tensor(self.history_bias)
        if weights.ndim not in (3, 4, 5):
            raise ValueError(
                "history_weights must have shape [P,P,D], [P,P,H,D], "
                "or [L,P,P,H,D]"
            )
        history_axis = 1 if weights.ndim == 5 else 0
        history = int(weights.shape[history_axis])
        if history < 1 or weights.shape[history_axis + 1] != history:
            raise ValueError("history_weights must be square in its first two axes")
        if weights.ndim == 3:
            expected_bias = (history, weights.shape[-1])
        elif weights.ndim == 4:
            expected_bias = (history, weights.shape[-2], weights.shape[-1])
        else:
            expected_bias = (
                weights.shape[0], history, weights.shape[-2], weights.shape[-1]
            )
        if tuple(bias.shape) != expected_bias:
            raise ValueError(f"history_bias must have shape {expected_bias}")
        if self.stride is not None and int(self.stride) <= 0:
            raise ValueError("stride must be positive")
        layer_ids = self.layer_ids
        if weights.ndim == 5:
            if layer_ids is None:
                layer_ids = tuple(range(int(weights.shape[0])))
            else:
                layer_ids = tuple(int(item) for item in layer_ids)
            if len(layer_ids) != weights.shape[0] or len(set(layer_ids)) != len(layer_ids):
                raise ValueError("layer_ids must uniquely identify every parameter layer")
        elif layer_ids is not None:
            raise ValueError("layer_ids are only valid for layered temporal parameters")
        object.__setattr__(self, "history_weights", weights)
        object.__setattr__(self, "history_bias", bias)
        object.__setattr__(self, "layer_ids", layer_ids)
        if self.codec_signature is not None and not str(self.codec_signature):
            raise ValueError("codec_signature must be non-empty")
        if self.parameter_set_id is not None and not str(self.parameter_set_id):
            raise ValueError("parameter_set_id must be non-empty")
        if self.coordinate_system is not None and not str(self.coordinate_system):
            raise ValueError("coordinate_system must be non-empty")

    @property
    def history_length(self) -> int:
        return int(self.history_weights.shape[1] if self.layered else self.history_weights.shape[0])

    @property
    def per_head(self) -> bool:
        return self.history_weights.ndim == 4

    @property
    def layered(self) -> bool:
        return self.history_weights.ndim == 5

    def for_layer(self, layer: int) -> "TemporalParameters":
        if not self.layered:
            if layer not in (0, -1):
                raise IndexError("non-layered temporal parameters only contain layer 0")
            return self
        assert self.layer_ids is not None
        try:
            index = self.layer_ids.index(int(layer))
        except ValueError as error:
            raise IndexError(f"layer {layer} is not present in {self.layer_ids}") from error
        return TemporalParameters(
            self.history_weights[index],
            self.history_bias[index],
            self.stride,
            codec_signature=self.codec_signature,
            parameter_set_id=self.parameter_set_id,
            coordinate_system=self.coordinate_system,
        )

    def to(self, device: torch.device | str) -> "TemporalParameters":
        return TemporalParameters(
            self.history_weights.to(device),
            self.history_bias.to(device),
            self.stride,
            self.layer_ids,
            self.codec_signature,
            self.parameter_set_id,
            self.coordinate_system,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            kind=np.asarray("temporal"),
            history_weights=self.history_weights.cpu().numpy(),
            history_bias=self.history_bias.cpu().numpy(),
            history_length=np.asarray(self.history_length, dtype=np.int32),
            stride=np.asarray(-1 if self.stride is None else self.stride, dtype=np.int32),
            layer_ids=np.asarray(
                [] if self.layer_ids is None else self.layer_ids, dtype=np.int32
            ),
            codec_signature=np.asarray(self.codec_signature or ""),
            parameter_set_id=np.asarray(self.parameter_set_id or ""),
            coordinate_system=np.asarray(self.coordinate_system or ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemporalParameters":
        with np.load(path, allow_pickle=False) as data:
            kind = str(np.asarray(data["kind"]).item())
            if kind != "temporal":
                raise ValueError(f"expected temporal parameters, got {kind!r}")
            stride = int(np.asarray(data["stride"]).item()) if "stride" in data else -1
            layer_ids = (
                tuple(int(item) for item in data["layer_ids"])
                if "layer_ids" in data and data["layer_ids"].size
                else None
            )
            codec_signature = (
                str(np.asarray(data["codec_signature"]).item())
                if "codec_signature" in data
                and str(np.asarray(data["codec_signature"]).item())
                else None
            )
            parameter_set_id = (
                str(np.asarray(data["parameter_set_id"]).item())
                if "parameter_set_id" in data
                and str(np.asarray(data["parameter_set_id"]).item())
                else None
            )
            coordinate_system = (
                str(np.asarray(data["coordinate_system"]).item())
                if "coordinate_system" in data
                and str(np.asarray(data["coordinate_system"]).item())
                else None
            )
            return cls(
                data["history_weights"],
                data["history_bias"],
                None if stride < 0 else stride,
                layer_ids,
                codec_signature,
                parameter_set_id,
                coordinate_system,
            )


@dataclass(frozen=True, slots=True)
class CrossKVParameters:
    """Per-head K-to-V affine projection parameters."""

    weight: torch.Tensor
    bias: torch.Tensor
    stride: int | None = None
    layer_ids: tuple[int, ...] | None = None
    codec_signature: str | None = None
    parameter_set_id: str | None = None
    coordinate_system: str | None = None

    def __post_init__(self) -> None:
        weight = _tensor(self.weight)
        bias = _tensor(self.bias)
        if weight.ndim not in (3, 4) or weight.shape[-1] != weight.shape[-2]:
            raise ValueError("weight must have shape [H,D,D] or [L,H,D,D]")
        expected_bias = tuple(weight.shape[:-2]) + (weight.shape[-1],)
        if tuple(bias.shape) != expected_bias:
            raise ValueError(f"bias must have shape {expected_bias}")
        if self.stride is not None and int(self.stride) <= 0:
            raise ValueError("stride must be positive")
        layer_ids = self.layer_ids
        if weight.ndim == 4:
            if layer_ids is None:
                layer_ids = tuple(range(int(weight.shape[0])))
            else:
                layer_ids = tuple(int(item) for item in layer_ids)
            if len(layer_ids) != weight.shape[0] or len(set(layer_ids)) != len(layer_ids):
                raise ValueError("layer_ids must uniquely identify every parameter layer")
        elif layer_ids is not None:
            raise ValueError("layer_ids are only valid for layered CrossKV parameters")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "layer_ids", layer_ids)
        if self.codec_signature is not None and not str(self.codec_signature):
            raise ValueError("codec_signature must be non-empty")
        if self.parameter_set_id is not None and not str(self.parameter_set_id):
            raise ValueError("parameter_set_id must be non-empty")
        if self.coordinate_system is not None and not str(self.coordinate_system):
            raise ValueError("coordinate_system must be non-empty")

    @property
    def layered(self) -> bool:
        return self.weight.ndim == 4

    def for_layer(self, layer: int) -> "CrossKVParameters":
        if not self.layered:
            if layer not in (0, -1):
                raise IndexError("single-layer CrossKV parameters only contain layer 0")
            return self
        assert self.layer_ids is not None
        try:
            index = self.layer_ids.index(int(layer))
        except ValueError as error:
            raise IndexError(f"layer {layer} is not present in {self.layer_ids}") from error
        return CrossKVParameters(
            self.weight[index],
            self.bias[index],
            self.stride,
            codec_signature=self.codec_signature,
            parameter_set_id=self.parameter_set_id,
            coordinate_system=self.coordinate_system,
        )

    def to(self, device: torch.device | str) -> "CrossKVParameters":
        return CrossKVParameters(
            self.weight.to(device),
            self.bias.to(device),
            self.stride,
            self.layer_ids,
            self.codec_signature,
            self.parameter_set_id,
            self.coordinate_system,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            kind=np.asarray("cross_kv"),
            weight=self.weight.cpu().numpy(),
            bias=self.bias.cpu().numpy(),
            stride=np.asarray(-1 if self.stride is None else self.stride, dtype=np.int32),
            layer_ids=np.asarray(
                [] if self.layer_ids is None else self.layer_ids, dtype=np.int32
            ),
            codec_signature=np.asarray(self.codec_signature or ""),
            parameter_set_id=np.asarray(self.parameter_set_id or ""),
            coordinate_system=np.asarray(self.coordinate_system or ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CrossKVParameters":
        with np.load(path, allow_pickle=False) as data:
            kind = str(np.asarray(data["kind"]).item())
            if kind != "cross_kv":
                raise ValueError(f"expected CrossKV parameters, got {kind!r}")
            stride = int(np.asarray(data["stride"]).item()) if "stride" in data else -1
            layer_ids = (
                tuple(int(item) for item in data["layer_ids"])
                if "layer_ids" in data and data["layer_ids"].size
                else None
            )
            codec_signature = (
                str(np.asarray(data["codec_signature"]).item())
                if "codec_signature" in data
                and str(np.asarray(data["codec_signature"]).item())
                else None
            )
            parameter_set_id = (
                str(np.asarray(data["parameter_set_id"]).item())
                if "parameter_set_id" in data
                and str(np.asarray(data["parameter_set_id"]).item())
                else None
            )
            coordinate_system = (
                str(np.asarray(data["coordinate_system"]).item())
                if "coordinate_system" in data
                and str(np.asarray(data["coordinate_system"]).item())
                else None
            )
            return cls(
                data["weight"],
                data["bias"],
                None if stride < 0 else stride,
                layer_ids,
                codec_signature,
                parameter_set_id,
                coordinate_system,
            )


def load_parameters(path: str | Path) -> TemporalParameters | CrossKVParameters:
    with np.load(path, allow_pickle=False) as data:
        kind: Any = np.asarray(data["kind"]).item()
    if str(kind) == "temporal":
        return TemporalParameters.load(path)
    if str(kind) == "cross_kv":
        return CrossKVParameters.load(path)
    raise ValueError(f"unknown parameter kind {kind!r}")
