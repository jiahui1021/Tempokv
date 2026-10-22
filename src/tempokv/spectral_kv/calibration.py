"""Streaming affine calibration for suite B's ``KVSKIP_lin`` ablation.

Only CPU float64 sufficient statistics are retained.  The memory requirement
depends on the number of layer/role/head combinations and feature dimension,
not on the number of observed tokens.  Deficient observations produce an
explicit coverage gap, never a guessed transformation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch


Key = tuple[int, str, int]
_ACCUMULATOR_FORMAT = "tempokv.spectral_kv.linear_calibration.v1"
_MAP_FORMAT = "tempokv.spectral_kv.calibration_map.v1"


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ridge(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError("ridge must be a finite positive number")
    return float(value)


def _role(role: str) -> str:
    if role not in ("k", "v"):
        raise ValueError("role must be 'k' or 'v'")
    return role


def _heads(head_ids: Sequence[int] | None, count: int) -> tuple[int, ...]:
    if head_ids is None:
        return tuple(range(count))
    try:
        result = tuple(_integer(h, "head id") for h in head_ids)
    except TypeError as exc:
        raise ValueError("head_ids must be a sequence of nonnegative integers") from exc
    if len(result) != count or len(set(result)) != count:
        raise ValueError("head_ids must contain one unique id for each tensor head")
    return result


def _grid(x: torch.Tensor, *, allow_batch: bool) -> None:
    allowed = (3, 4) if allow_batch else (3,)
    if not isinstance(x, torch.Tensor) or x.ndim not in allowed:
        shapes = "[N,heads,D] or [B,S,heads,D]" if allow_batch else "[N,heads,D]"
        raise ValueError(f"calibration input must have shape {shapes}")
    if not x.is_floating_point() or min(x.shape) < 1:
        raise ValueError("calibration input must have positive dimensions and a real floating-point dtype")


def _scalar(state: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(state[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def _state_keys(state: Mapping[str, Any], required: set[str]) -> None:
    if set(state) != required:
        raise ValueError(f"invalid calibration fields; expected {sorted(required)}")


def _key_arrays(keys: tuple[Key, ...]) -> dict[str, np.ndarray]:
    return {
        "layers": np.asarray([k[0] for k in keys], dtype=np.int64),
        "roles": np.asarray([k[1] for k in keys], dtype="U1"),
        "heads": np.asarray([k[2] for k in keys], dtype=np.int64),
    }


def _read_keys(state: Mapping[str, Any]) -> tuple[Key, ...]:
    layers, roles, heads = (np.asarray(state[n]) for n in ("layers", "roles", "heads"))
    if layers.ndim != 1 or roles.shape != layers.shape or heads.shape != layers.shape:
        raise ValueError("layer, role, and head metadata must be equally sized vectors")
    if layers.dtype.kind not in "iu" or heads.dtype.kind not in "iu" or roles.dtype.kind != "U":
        raise ValueError("invalid layer, role, or head metadata dtype")
    keys = tuple((_integer(l.item(), "layer"), _role(r.item()), _integer(h.item(), "head"))
                 for l, r, h in zip(layers, roles, heads))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate calibration layer/role/head entries")
    return keys


class LinearCalibration:
    """Accumulate ``[x, 1].T @ [x, 1]`` and ``[x, 1].T @ clean``.

    A separate affine transformation is learned for every layer, role and head.
    All entries use the same feature dimension.  Ridge regularizes feature
    weights only; the bias remains unpenalized.  No raw examples are retained.
    """

    def __init__(self, ridge: float = 1e-4):
        self.ridge = _ridge(ridge)
        self.feature_dim = 0
        self._moments: dict[Key, tuple[int, torch.Tensor, torch.Tensor]] = {}

    @property
    def keys(self) -> tuple[Key, ...]:
        return tuple(sorted(self._moments))

    def update(
        self,
        layer: int,
        role: str,
        last: torch.Tensor,
        clean: torch.Tensor,
        *,
        head_ids: Sequence[int] | None = None,
    ) -> None:
        layer, role = _integer(layer, "layer"), _role(role)
        _grid(last, allow_batch=False)
        _grid(clean, allow_batch=False)
        if last.shape != clean.shape:
            raise ValueError("last and clean calibration inputs must have the same shape")
        n, head_count, d = last.shape
        heads = _heads(head_ids, head_count)
        if self.feature_dim not in (0, d):
            raise ValueError(f"feature dimension changed from {self.feature_dim} to {d}")
        x = last.detach().to(device="cpu", dtype=torch.float64)
        y = clean.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("calibration observations must all be finite")
        augmented = torch.cat((x, torch.ones(n, head_count, 1, dtype=torch.float64)), dim=-1)
        gram = torch.einsum("nhi,nhj->hij", augmented, augmented)
        cross = torch.einsum("nhi,nhj->hij", augmented, y)
        if not torch.isfinite(gram).all() or not torch.isfinite(cross).all():
            raise ValueError("calibration observations overflowed the float64 sufficient statistics")
        # Validate the entire batch before changing any state.
        additions = []
        for index, head in enumerate(heads):
            key = (layer, role, head)
            if key in self._moments:
                old_n, old_gram, old_cross = self._moments[key]
                value = (old_n + n, old_gram + gram[index], old_cross + cross[index])
                if not torch.isfinite(value[1]).all() or not torch.isfinite(value[2]).all():
                    raise ValueError("accumulated calibration statistics overflowed float64")
            else:
                value = (n, gram[index].clone(), cross[index].clone())
            additions.append((key, value))
        self.feature_dim = d
        self._moments.update(additions)

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return independent, object-free arrays suitable for ``np.savez``."""
        keys, d = self.keys, self.feature_dim
        state = {
            "format": np.asarray(_ACCUMULATOR_FORMAT),
            "ridge": np.asarray(self.ridge, dtype=np.float64),
            "feature_dim": np.asarray(d, dtype=np.int64),
            **_key_arrays(keys),
            "counts": np.asarray([self._moments[k][0] for k in keys], dtype=np.int64),
            "xtx": np.stack([self._moments[k][1].numpy() for k in keys]) if keys else np.empty((0, d + 1, d + 1), dtype=np.float64),
            "xty": np.stack([self._moments[k][2].numpy() for k in keys]) if keys else np.empty((0, d + 1, d), dtype=np.float64),
        }
        return state

    def merge(self, other: LinearCalibration | Mapping[str, Any]) -> LinearCalibration:
        """Merge compatible sufficient statistics, with no mutation on failure."""
        state = other.state_dict() if isinstance(other, LinearCalibration) else other
        if not isinstance(state, Mapping):
            raise ValueError("merge expects a LinearCalibration or its state_dict")
        required = {"format", "ridge", "feature_dim", "layers", "roles", "heads", "counts", "xtx", "xty"}
        _state_keys(state, required)
        if _scalar(state, "format") != _ACCUMULATOR_FORMAT:
            raise ValueError("unsupported calibration accumulator format")
        ridge = _ridge(_scalar(state, "ridge"))
        d = _integer(_scalar(state, "feature_dim"), "feature_dim")
        if ridge != self.ridge:
            raise ValueError("cannot merge calibration states with different ridge values")
        if self.feature_dim and d and d != self.feature_dim:
            raise ValueError("cannot merge calibration states with different feature dimensions")
        keys = _read_keys(state)
        if keys and d < 1:
            raise ValueError("nonempty calibration state requires a positive feature dimension")
        counts, gram, cross = (np.asarray(state[n]) for n in ("counts", "xtx", "xty"))
        if counts.shape != (len(keys),) or counts.dtype.kind not in "iu" or np.any(counts < 1):
            raise ValueError("calibration sample counts must be positive integers")
        if gram.shape != (len(keys), d + 1, d + 1) or cross.shape != (len(keys), d + 1, d):
            raise ValueError("sufficient-statistic dimensions disagree with calibration metadata")
        if gram.dtype.kind != "f" or cross.dtype.kind != "f" or not np.isfinite(gram).all() or not np.isfinite(cross).all():
            raise ValueError("calibration sufficient statistics must be finite floating-point arrays")
        if not np.allclose(gram, gram.transpose(0, 2, 1), rtol=1e-10, atol=1e-10):
            raise ValueError("calibration normal matrices must be symmetric")
        if keys and not np.array_equal(gram[:, -1, -1], counts):
            raise ValueError("sample counts disagree with the bias normal-matrix entry")
        additions = []
        for index, key in enumerate(keys):
            value = (int(counts[index]), torch.tensor(gram[index], dtype=torch.float64), torch.tensor(cross[index], dtype=torch.float64))
            if key in self._moments:
                old_n, old_gram, old_cross = self._moments[key]
                value = (old_n + value[0], old_gram + value[1], old_cross + value[2])
            if not torch.isfinite(value[1]).all() or not torch.isfinite(value[2]).all():
                raise ValueError("merged calibration statistics overflowed float64")
            additions.append((key, value))
        self.feature_dim = self.feature_dim or d
        self._moments.update(additions)
        return self

    def fit(self) -> CalibrationMap:
        """Fit covered entries; record sample/rank gaps in the returned metadata.

        At least ``D+1`` observations and full rank of the augmented design are
        required.  Ridge improves numerical stability after this coverage
        check; it does not make missing feature directions identifiable.
        """
        d = self.feature_dim
        parameters: dict[Key, tuple[torch.Tensor, torch.Tensor]] = {}
        counts: dict[Key, int] = {}
        skipped = []
        for key in self.keys:
            n, gram, cross = self._moments[key]
            detail = {"layer": key[0], "role": key[1], "head": key[2], "samples": n}
            if n < d + 1:
                skipped.append({**detail, "reason": "insufficient_samples", "required_samples": d + 1})
                continue
            rank = int(torch.linalg.matrix_rank(gram, hermitian=True))
            if rank < d + 1:
                skipped.append({**detail, "reason": "rank_deficient", "rank": rank, "required_rank": d + 1})
                continue
            penalty = torch.eye(d + 1, dtype=torch.float64) * self.ridge
            penalty[-1, -1] = 0.0
            try:
                beta = torch.linalg.solve(gram + penalty, cross)
            except RuntimeError:
                skipped.append({**detail, "reason": "solve_failed"})
                continue
            if not torch.isfinite(beta).all():
                skipped.append({**detail, "reason": "nonfinite_solution"})
                continue
            parameters[key] = (beta[:d], beta[d])
            counts[key] = n
        return CalibrationMap(parameters, feature_dim=d, ridge=self.ridge, counts=counts, skipped=skipped)


class CalibrationMap:
    """Per-layer/head affine maps with explicit coverage and safe NPZ storage."""

    def __init__(
        self,
        parameters: Mapping[Key, tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        feature_dim: int = 0,
        ridge: float = 1e-4,
        counts: Mapping[Key, int] | None = None,
        skipped: Sequence[Mapping[str, Any]] = (),
    ):
        self.feature_dim = _integer(feature_dim, "feature_dim")
        self.ridge = _ridge(ridge)
        self._parameters: dict[Key, tuple[torch.Tensor, torch.Tensor]] = {}
        self._counts: dict[Key, int] = {}
        self._skipped = tuple(dict(record) for record in skipped)
        for raw_key, (weight, bias) in (parameters or {}).items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 3:
                raise ValueError("calibration keys must be (layer, role, head) tuples")
            key = (_integer(raw_key[0], "layer"), _role(raw_key[1]), _integer(raw_key[2], "head"))
            weight = torch.as_tensor(weight).detach().to(device="cpu", dtype=torch.float64).clone()
            bias = torch.as_tensor(bias).detach().to(device="cpu", dtype=torch.float64).clone()
            d = self.feature_dim
            if d < 1 or weight.shape != (d, d) or bias.shape != (d,):
                raise ValueError("calibration weight/bias dimensions disagree with feature_dim")
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError("calibration parameters must be finite")
            count = _integer((counts or {}).get(raw_key, 0), "samples", d + 1)
            self._parameters[key], self._counts[key] = (weight, bias), count
        # Metadata is intentionally JSON compatible and does not expose tensors.
        json.dumps(self._skipped, allow_nan=False)

    @property
    def keys(self) -> tuple[Key, ...]:
        """Sorted fitted ``(layer, role, head)`` coverage, suitable for preflight."""
        return tuple(sorted(self._parameters))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format": _MAP_FORMAT,
            "ridge": self.ridge,
            "feature_dim": self.feature_dim,
            "fitted_entries": len(self._parameters),
            "coverage": [{"layer": k[0], "role": k[1], "head": k[2], "samples": self._counts[k]} for k in self.keys],
            "skipped": [dict(record) for record in self._skipped],
        }

    def apply(
        self,
        layer: int,
        role: str,
        x: torch.Tensor,
        head_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return ``x @ W + bias`` without modifying ``x``.

        Float16/BF16/float32 inputs use float32 arithmetic, including inside an
        enclosing autocast region; float64 input retains float64 arithmetic.
        Output dtype and device always match the input.
        """
        layer, role = _integer(layer, "layer"), _role(role)
        _grid(x, allow_batch=True)
        if not self._parameters:
            raise ValueError("calibration map has no fitted entries; inspect metadata['skipped']")
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"calibration expects feature dimension {self.feature_dim}, got {x.shape[-1]}")
        heads = _heads(head_ids, x.shape[-2])
        keys = tuple((layer, role, head) for head in heads)
        missing = [key for key in keys if key not in self._parameters]
        if missing:
            raise ValueError(f"missing fitted calibration for layer/role/head: {missing}")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        weights = torch.stack([self._parameters[key][0] for key in keys]).to(device=x.device, dtype=dtype)
        biases = torch.stack([self._parameters[key][1] for key in keys]).to(device=x.device, dtype=dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = torch.einsum("...hd,hde->...he", x.to(dtype=dtype), weights) + biases
        return result.to(dtype=x.dtype)

    def state_dict(self) -> dict[str, np.ndarray]:
        keys, d = self.keys, self.feature_dim
        return {
            "format": np.asarray(_MAP_FORMAT),
            "ridge": np.asarray(self.ridge, dtype=np.float64),
            "feature_dim": np.asarray(d, dtype=np.int64),
            **_key_arrays(keys),
            "counts": np.asarray([self._counts[k] for k in keys], dtype=np.int64),
            "weights": np.stack([self._parameters[k][0].numpy() for k in keys]) if keys else np.empty((0, d, d), dtype=np.float64),
            "bias": np.stack([self._parameters[k][1].numpy() for k in keys]) if keys else np.empty((0, d), dtype=np.float64),
            "skipped_json": np.asarray(json.dumps(self._skipped, sort_keys=True, allow_nan=False)),
        }

    def save(self, path: str | Path) -> Path:
        """Create an NPZ exclusively; an existing destination is never replaced."""
        path = Path(path)
        state = self.state_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **state)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CalibrationMap:
        """Load only object-free NPZ arrays (``allow_pickle=False``)."""
        with np.load(Path(path), allow_pickle=False) as archive:
            state = {name: archive[name] for name in archive.files}
        required = {"format", "ridge", "feature_dim", "layers", "roles", "heads", "counts", "weights", "bias", "skipped_json"}
        _state_keys(state, required)
        if _scalar(state, "format") != _MAP_FORMAT:
            raise ValueError("unsupported calibration map format")
        d = _integer(_scalar(state, "feature_dim"), "feature_dim")
        ridge = _ridge(_scalar(state, "ridge"))
        keys = _read_keys(state)
        counts, weights, bias = (np.asarray(state[n]) for n in ("counts", "weights", "bias"))
        if counts.shape != (len(keys),) or counts.dtype.kind not in "iu" or np.any(counts < d + 1):
            raise ValueError("invalid fitted calibration sample counts")
        if weights.shape != (len(keys), d, d) or bias.shape != (len(keys), d):
            raise ValueError("stored calibration shapes disagree with feature metadata")
        if weights.dtype.kind != "f" or bias.dtype.kind != "f":
            raise ValueError("stored calibration parameters must have floating-point dtype")
        try:
            skipped = json.loads(_scalar(state, "skipped_json"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid skipped-entry metadata") from exc
        if not isinstance(skipped, list) or any(not isinstance(record, dict) for record in skipped):
            raise ValueError("skipped-entry metadata must be a list of objects")
        parameters = {key: (torch.from_numpy(weights[i].copy()), torch.from_numpy(bias[i].copy())) for i, key in enumerate(keys)}
        return cls(parameters, feature_dim=d, ridge=ridge,
                   counts={key: int(counts[i]) for i, key in enumerate(keys)}, skipped=skipped)
