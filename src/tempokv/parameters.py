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
    """Warm-up-aware multi-history parameters, including zero-order K."""

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
        if history < 0 or weights.shape[history_axis + 1] != history:
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
        if self.codec_signature is not None and not str(self.codec_signature):
            raise ValueError("codec_signature must be non-empty")
        if self.parameter_set_id is not None and not str(self.parameter_set_id):
            raise ValueError("parameter_set_id must be non-empty")
        if self.coordinate_system is not None and not str(self.coordinate_system):
            raise ValueError("coordinate_system must be non-empty")

    @property
    def history_length(self) -> int:
        return int(
            self.history_weights.shape[1]
            if self.layered
            else self.history_weights.shape[0]
        )

    @property
    def per_head(self) -> bool:
        return self.history_weights.ndim == 4

    @property
    def layered(self) -> bool:
        return self.history_weights.ndim == 5

    def with_history_length(self, history_length: int | None) -> "TemporalParameters":
        """Assert a requested history length matches the fitted one.

        Parameters carry exactly the history they were fitted for, so this only
        checks agreement; it is kept as a method because the codec resolves a
        configured length against whatever the parameter file provides.
        """
        if history_length is None:
            return self
        if history_length != self.history_length:
            raise ValueError(
                "temporal parameters require their fitted history length"
            )
        return self

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
            # Reject a file from the withdrawn shared-history format by name.
            # Its weights are [H,D] rather than [P,P,D], so the shape check
            # below would otherwise report a confusing dimension error instead
            # of saying that the file has to be refitted.
            history_mode = (
                str(np.asarray(data["history_mode"]).item())
                if "history_mode" in data
                else "independent"
            )
            if history_mode != "independent":
                raise ValueError(
                    f"temporal history_mode {history_mode!r} is no longer "
                    "supported; refit the parameters with the current fitter"
                )
            parameters = cls(
                data["history_weights"],
                data["history_bias"],
                None if stride < 0 else stride,
                layer_ids,
                codec_signature,
                parameter_set_id,
                coordinate_system,
            )
            if "history_length" in data:
                declared_history_length = int(
                    np.asarray(data["history_length"]).item()
                )
                if declared_history_length != parameters.history_length:
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

    def __post_init__(self) -> None:
        weight = _tensor(self.weight)
        bias = _tensor(self.bias)
        if weight.ndim not in (3, 4) or weight.shape[-1] != weight.shape[-2]:
            raise ValueError("weight must have shape [H,D,D] or [L,H,D,D]")
        expected_bias = tuple(weight.shape[:-2]) + (weight.shape[-1],)
        if tuple(bias.shape) != expected_bias:
            raise ValueError(f"bias must have shape {expected_bias}")
        if self.history_weights is None:
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
        return int(
            self.history_weights.shape[1]
            if self.layered
            else self.history_weights.shape[0]
        )

    @property
    def history_mode(self) -> str:
        """Return the V history parameter layout."""
        dense_dimensions = 5 if self.layered else 4
        return "dense" if self.history_weights.ndim == dense_dimensions else "diagonal"

    @property
    def history_diagonal(self) -> bool:
        """Whether V history uses one scalar coefficient per output channel."""
        return self.history_mode == "diagonal"

    def with_history_length(self, history_length: int) -> "CrossKVParameters":
        """Assert a requested V history length matches the fitted one."""
        if history_length != self.history_length:
            raise ValueError(
                "CrossKV parameters require their fitted history length"
            )
        return self

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
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            kind=np.asarray("cross_kv"),
            weight=self.weight.cpu().numpy(),
            bias=self.bias.cpu().numpy(),
            history_weights=self.history_weights.cpu().numpy(),
            history_length=np.asarray(self.history_length, dtype=np.int32),
            # dense vs diagonal is a real layout difference the decoder routes
            # on, so it stays recorded and is checked back on load.
            history_mode=np.asarray(self.history_mode),
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
            # Name the withdrawn shared-history layout explicitly: its history
            # weights are [H,D], so the constructor would otherwise reject them
            # as a malformed diagonal map instead of asking for a refit.
            if declared_history_mode == "shared_diagonal":
                raise ValueError(
                    "shared_diagonal V history is no longer supported; refit "
                    "the parameters with the current fitter"
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
            )
            if declared_history_mode is not None:
                if declared_history_mode != parameters.history_mode:
                    raise ValueError(
                        "history_mode does not match the history_weights array"
                    )
            if history_weights is not None and "history_length" in data:
                declared_history_length = int(
                    np.asarray(data["history_length"]).item()
                )
                if declared_history_length != parameters.history_length:
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
