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
    maximum_history_length: int | None = None
    active_history_length: int | None = None

    def __post_init__(self) -> None:
        weights = _tensor(self.history_weights)
        bias = _tensor(self.history_bias)
        maximum_history_length = self.maximum_history_length
        active_history_length = self.active_history_length
        if maximum_history_length is not None:
            if (
                isinstance(maximum_history_length, bool)
                or int(maximum_history_length) != maximum_history_length
                or not 1 <= int(maximum_history_length) <= 8
            ):
                raise ValueError(
                    "maximum_history_length must be an integer from 1 to 8"
                )
            maximum_history_length = int(maximum_history_length)
            if weights.ndim not in (2, 3) or tuple(bias.shape) != tuple(weights.shape):
                raise ValueError(
                    "shared temporal weights and bias must have shape [H,D] "
                    "or [L,H,D]"
                )
            if active_history_length is None:
                active_history_length = maximum_history_length
            if (
                isinstance(active_history_length, bool)
                or int(active_history_length) != active_history_length
                or not 1 <= int(active_history_length) <= maximum_history_length
            ):
                raise ValueError(
                    "active_history_length must be between 1 and "
                    "maximum_history_length"
                )
            active_history_length = int(active_history_length)
            layered = weights.ndim == 3
        else:
            if active_history_length is not None:
                raise ValueError(
                    "active_history_length is only valid for shared temporal parameters"
                )
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
            layered = weights.ndim == 5
        if self.stride is not None and int(self.stride) <= 0:
            raise ValueError("stride must be positive")
        layer_ids = self.layer_ids
        if layered:
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
        object.__setattr__(self, "maximum_history_length", maximum_history_length)
        object.__setattr__(self, "active_history_length", active_history_length)
        if self.codec_signature is not None and not str(self.codec_signature):
            raise ValueError("codec_signature must be non-empty")
        if self.parameter_set_id is not None and not str(self.parameter_set_id):
            raise ValueError("parameter_set_id must be non-empty")
        if self.coordinate_system is not None and not str(self.coordinate_system):
            raise ValueError("coordinate_system must be non-empty")

    @property
    def history_length(self) -> int:
        if self.shared_history:
            assert self.active_history_length is not None
            return self.active_history_length
        return int(
            self.history_weights.shape[1]
            if self.layered
            else self.history_weights.shape[0]
        )

    @property
    def max_history_length(self) -> int:
        if self.shared_history:
            assert self.maximum_history_length is not None
            return self.maximum_history_length
        return int(
            self.history_weights.shape[1]
            if self.layered
            else self.history_weights.shape[0]
        )

    @property
    def shared_history(self) -> bool:
        return self.maximum_history_length is not None

    @property
    def history_mode(self) -> str:
        return "shared" if self.shared_history else "independent"

    @property
    def per_head(self) -> bool:
        return True if self.shared_history else self.history_weights.ndim == 4

    @property
    def layered(self) -> bool:
        return (
            self.history_weights.ndim == 3
            if self.shared_history
            else self.history_weights.ndim == 5
        )

    def with_history_length(self, history_length: int | None) -> "TemporalParameters":
        if history_length is None:
            return self
        if not self.shared_history:
            if history_length != self.history_length:
                raise ValueError(
                    "independent temporal parameters require their fitted history length"
                )
            return self
        return TemporalParameters(
            self.history_weights,
            self.history_bias,
            self.stride,
            self.layer_ids,
            self.codec_signature,
            self.parameter_set_id,
            self.coordinate_system,
            self.maximum_history_length,
            history_length,
        )

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
            maximum_history_length=self.maximum_history_length,
            active_history_length=self.active_history_length,
        )

    def to(self, device: torch.device | str) -> "TemporalParameters":
        resolved_device = torch.device(device)
        if (
            self.history_weights.device == resolved_device
            and self.history_bias.device == resolved_device
        ):
            return self
        return TemporalParameters(
            self.history_weights.to(resolved_device),
            self.history_bias.to(resolved_device),
            self.stride,
            self.layer_ids,
            self.codec_signature,
            self.parameter_set_id,
            self.coordinate_system,
            self.maximum_history_length,
            self.active_history_length,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            kind=np.asarray("temporal"),
            history_weights=self.history_weights.cpu().numpy(),
            history_bias=self.history_bias.cpu().numpy(),
            history_length=np.asarray(self.max_history_length, dtype=np.int32),
            history_mode=np.asarray(self.history_mode),
            maximum_history_length=np.asarray(
                self.max_history_length if self.shared_history else -1,
                dtype=np.int32,
            ),
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
            history_mode = (
                str(np.asarray(data["history_mode"]).item())
                if "history_mode" in data
                else "independent"
            )
            if history_mode not in ("independent", "shared"):
                raise ValueError(f"unknown temporal history_mode {history_mode!r}")
            maximum_history_length = None
            if history_mode == "shared":
                maximum_history_length = (
                    int(np.asarray(data["maximum_history_length"]).item())
                    if "maximum_history_length" in data
                    else int(np.asarray(data["history_length"]).item())
                )
            parameters = cls(
                data["history_weights"],
                data["history_bias"],
                None if stride < 0 else stride,
                layer_ids,
                codec_signature,
                parameter_set_id,
                coordinate_system,
                maximum_history_length,
            )
            if "history_length" in data:
                declared_history_length = int(
                    np.asarray(data["history_length"]).item()
                )
                if declared_history_length != parameters.max_history_length:
                    raise ValueError(
                        "history_length does not match the history_weights array"
                    )
            return parameters


@dataclass(frozen=True, slots=True)
class CrossKVParameters:
    """Per-head K/current-V-history projection parameters."""

    weight: torch.Tensor
    bias: torch.Tensor
    stride: int | None = None
    layer_ids: tuple[int, ...] | None = None
    codec_signature: str | None = None
    parameter_set_id: str | None = None
    coordinate_system: str | None = None
    history_weights: torch.Tensor | np.ndarray | None = None
    maximum_history_length: int | None = None
    active_history_length: int | None = None

    def __post_init__(self) -> None:
        weight = _tensor(self.weight)
        bias = _tensor(self.bias)
        if weight.ndim not in (3, 4) or weight.shape[-1] != weight.shape[-2]:
            raise ValueError("weight must have shape [H,D,D] or [L,H,D,D]")
        expected_bias = tuple(weight.shape[:-2]) + (weight.shape[-1],)
        if tuple(bias.shape) != expected_bias:
            raise ValueError(f"bias must have shape {expected_bias}")
        maximum_history_length = self.maximum_history_length
        active_history_length = self.active_history_length
        if self.history_weights is None:
            if maximum_history_length is not None:
                raise ValueError(
                    "shared CrossKV parameters require one history weight map"
                )
            if weight.ndim == 3:
                history_weights = weight.new_empty(
                    (0, weight.shape[0], weight.shape[-2], weight.shape[-1])
                )
            else:
                history_weights = weight.new_empty(
                    (
                        weight.shape[0],
                        0,
                        weight.shape[1],
                        weight.shape[-2],
                        weight.shape[-1],
                    )
                )
        else:
            history_weights = _tensor(self.history_weights)
        if maximum_history_length is not None:
            if (
                isinstance(maximum_history_length, bool)
                or int(maximum_history_length) != maximum_history_length
                or not 0 <= int(maximum_history_length) <= 5
            ):
                raise ValueError(
                    "maximum_history_length must be an integer from 0 to 5"
                )
            maximum_history_length = int(maximum_history_length)
            if tuple(history_weights.shape) != tuple(bias.shape):
                raise ValueError(
                    "shared CrossKV history_weights must have shape [H,D] "
                    "or [L,H,D] matching bias"
                )
            if active_history_length is None:
                active_history_length = maximum_history_length
            if (
                isinstance(active_history_length, bool)
                or int(active_history_length) != active_history_length
                or not 0 <= int(active_history_length) <= maximum_history_length
            ):
                raise ValueError(
                    "active_history_length must be between 0 and "
                    "maximum_history_length"
                )
            active_history_length = int(active_history_length)
        else:
            if active_history_length is not None:
                raise ValueError(
                    "active_history_length is only valid for shared CrossKV parameters"
                )
            if weight.ndim == 3:
                valid_dense_history = (
                    history_weights.ndim == 4
                    and tuple(history_weights.shape[1:]) == tuple(weight.shape)
                )
                valid_diagonal_history = (
                    history_weights.ndim == 3
                    and tuple(history_weights.shape[1:]) == tuple(bias.shape)
                )
                history_axis = 0
            else:
                valid_dense_history = (
                    history_weights.ndim == 5
                    and history_weights.shape[0] == weight.shape[0]
                    and tuple(history_weights.shape[2:]) == tuple(weight.shape[1:])
                )
                valid_diagonal_history = (
                    history_weights.ndim == 4
                    and history_weights.shape[0] == weight.shape[0]
                    and tuple(history_weights.shape[2:]) == tuple(bias.shape[1:])
                )
                history_axis = 1
            if not (valid_dense_history or valid_diagonal_history):
                expected_dense = (
                    f"[P,{weight.shape[0]},{weight.shape[-1]},{weight.shape[-1]}]"
                    if weight.ndim == 3
                    else (
                        f"[{weight.shape[0]},P,{weight.shape[1]},"
                        f"{weight.shape[-1]},{weight.shape[-1]}]"
                    )
                )
                expected_diagonal = (
                    f"[P,{weight.shape[0]},{weight.shape[-1]}]"
                    if weight.ndim == 3
                    else f"[{weight.shape[0]},P,{weight.shape[1]},{weight.shape[-1]}]"
                )
                raise ValueError(
                    "history_weights must have dense shape "
                    f"{expected_dense} or diagonal shape {expected_diagonal}"
                )
            history_length = int(history_weights.shape[history_axis])
            if not 0 <= history_length <= 5:
                raise ValueError(
                    "history_weights must contain between 0 and 5 V history maps"
                )
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
        object.__setattr__(self, "history_weights", history_weights)
        object.__setattr__(self, "layer_ids", layer_ids)
        object.__setattr__(self, "maximum_history_length", maximum_history_length)
        object.__setattr__(self, "active_history_length", active_history_length)
        if self.codec_signature is not None and not str(self.codec_signature):
            raise ValueError("codec_signature must be non-empty")
        if self.parameter_set_id is not None and not str(self.parameter_set_id):
            raise ValueError("parameter_set_id must be non-empty")
        if self.coordinate_system is not None and not str(self.coordinate_system):
            raise ValueError("coordinate_system must be non-empty")

    @property
    def layered(self) -> bool:
        return self.weight.ndim == 4

    @property
    def history_length(self) -> int:
        if self.shared_history:
            assert self.active_history_length is not None
            return self.active_history_length
        return int(
            self.history_weights.shape[1]
            if self.layered
            else self.history_weights.shape[0]
        )

    @property
    def max_history_length(self) -> int:
        if self.shared_history:
            assert self.maximum_history_length is not None
            return self.maximum_history_length
        return int(
            self.history_weights.shape[1]
            if self.layered
            else self.history_weights.shape[0]
        )

    @property
    def shared_history(self) -> bool:
        return self.maximum_history_length is not None

    @property
    def history_mode(self) -> str:
        """Return the V history parameter layout."""
        if self.shared_history:
            return "shared_diagonal"
        dense_dimensions = 5 if self.layered else 4
        return "dense" if self.history_weights.ndim == dense_dimensions else "diagonal"

    @property
    def history_diagonal(self) -> bool:
        """Whether V history uses one scalar coefficient per output channel."""
        return self.history_mode in ("diagonal", "shared_diagonal")

    def with_history_length(self, history_length: int) -> "CrossKVParameters":
        if not self.shared_history:
            if history_length != self.history_length:
                raise ValueError(
                    "independent CrossKV parameters require their fitted history length"
                )
            return self
        return CrossKVParameters(
            self.weight,
            self.bias,
            self.stride,
            self.layer_ids,
            self.codec_signature,
            self.parameter_set_id,
            self.coordinate_system,
            self.history_weights,
            self.maximum_history_length,
            history_length,
        )

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
            history_weights=self.history_weights[index],
            maximum_history_length=self.maximum_history_length,
            active_history_length=self.active_history_length,
        )

    def to(self, device: torch.device | str) -> "CrossKVParameters":
        resolved_device = torch.device(device)
        if (
            self.weight.device == resolved_device
            and self.bias.device == resolved_device
            and self.history_weights.device == resolved_device
        ):
            return self
        return CrossKVParameters(
            self.weight.to(resolved_device),
            self.bias.to(resolved_device),
            self.stride,
            self.layer_ids,
            self.codec_signature,
            self.parameter_set_id,
            self.coordinate_system,
            self.history_weights.to(resolved_device),
            self.maximum_history_length,
            self.active_history_length,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            kind=np.asarray("cross_kv"),
            weight=self.weight.cpu().numpy(),
            bias=self.bias.cpu().numpy(),
            history_weights=self.history_weights.cpu().numpy(),
            history_length=np.asarray(self.max_history_length, dtype=np.int32),
            history_mode=np.asarray(self.history_mode),
            maximum_history_length=np.asarray(
                self.max_history_length if self.shared_history else -1,
                dtype=np.int32,
            ),
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
            history_weights = (
                data["history_weights"] if "history_weights" in data else None
            )
            declared_history_mode = (
                str(np.asarray(data["history_mode"]).item())
                if "history_mode" in data
                else None
            )
            maximum_history_length = None
            if declared_history_mode == "shared_diagonal":
                maximum_history_length = (
                    int(np.asarray(data["maximum_history_length"]).item())
                    if "maximum_history_length" in data
                    else int(np.asarray(data["history_length"]).item())
                )
            parameters = cls(
                data["weight"],
                data["bias"],
                None if stride < 0 else stride,
                layer_ids,
                codec_signature,
                parameter_set_id,
                coordinate_system,
                history_weights,
                maximum_history_length,
            )
            if "history_mode" in data:
                if declared_history_mode != parameters.history_mode:
                    raise ValueError(
                        "history_mode does not match the history_weights array"
                    )
            if history_weights is not None and "history_length" in data:
                declared_history_length = int(
                    np.asarray(data["history_length"]).item()
                )
                if declared_history_length != parameters.max_history_length:
                    raise ValueError(
                        "history_length does not match the history_weights array"
                    )
            return parameters


def load_parameters(path: str | Path) -> TemporalParameters | CrossKVParameters:
    with np.load(path, allow_pickle=False) as data:
        kind: Any = np.asarray(data["kind"]).item()
    if str(kind) == "temporal":
        return TemporalParameters.load(path)
    if str(kind) == "cross_kv":
        return CrossKVParameters.load(path)
    raise ValueError(f"unknown parameter kind {kind!r}")
