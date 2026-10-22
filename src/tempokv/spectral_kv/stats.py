"""Bounded, read-only statistics for the Spectral KV experiment suite.

All public ``update`` methods return plain Python rows suitable for CSV/JSON.
Frequencies are Fourier frequencies in cycles per spatial token.  A radial
radius is normalized by sqrt(0.5), the corner of the 2-D Nyquist square, so
reported radii lie in [0, 1].  DC is explicitly distinguished from low nonzero
frequencies.  Undefined measurements are ``None``, never fabricated zeros.

Only the most recent frames/chunks needed for each comparison are retained.
``state_dict`` exports scalar observations, not raw feature history; it can be
passed directly to ``numpy.savez`` without enabling pickle.  Merging states
merges observations and deliberately does not join temporal histories.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor


_RADIAL_MAX = math.sqrt(0.5)
_IDENTITY_FIELDS = ("sample_id", "layer", "role", "coordinate", "head")


def _plain(value: Any) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Tensor) and value.numel() == 1:
        return _plain(value.item())
    raise TypeError(f"metadata must be scalar or JSON-compatible, got {type(value).__name__}")


def _grid(x: Tensor) -> Tensor:
    if not isinstance(x, Tensor) or x.ndim != 5:
        raise ValueError("expected a tensor with shape [F,H,W,heads,D]")
    if any(n <= 0 for n in x.shape) or not x.is_floating_point():
        raise ValueError("feature dimensions must be nonempty and real floating point")
    return x.detach().to(dtype=torch.float32)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _epsilon(value: float) -> float:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError("eps must be finite and positive")
    return float(value)


def _radius(h: int, w: int, device: torch.device) -> Tensor:
    return torch.sqrt(
        torch.fft.fftfreq(h, device=device)[:, None].square()
        + torch.fft.fftfreq(w, device=device)[None, :].square()
    ) / _RADIAL_MAX


def _band_ids(radius: Tensor, bins: int) -> Tensor:
    return torch.floor(radius * bins).long().clamp(0, bins - 1)


def _band(band: int, bins: int) -> dict[str, Any]:
    return {
        "band": band, "bins": bins,
        "radius_low": band / bins, "radius_high": (band + 1) / bins,
        "frequency_low": band / bins * _RADIAL_MAX,
        "frequency_high": (band + 1) / bins * _RADIAL_MAX,
    }


def _safe_ratio(numerator: float, denominator: float, eps: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator > eps:
        return numerator / denominator
    return 0.0 if numerator <= eps else None


def _cosine(a: Tensor, b: Tensor, eps: float) -> float | None:
    norms = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    dot = float((a.conj() * b).real.sum())
    if norms <= eps or not math.isfinite(norms) or not math.isfinite(dot):
        return None
    return max(-1.0, min(1.0, dot / norms))


def _relative_error(prediction: Tensor, target: Tensor, eps: float) -> float | None:
    return _safe_ratio(float(torch.linalg.vector_norm(prediction - target)),
                       float(torch.linalg.vector_norm(target)), eps)


class _Rows:
    def __init__(self, max_rows: int = 100000) -> None:
        self.max_rows = _positive("max_rows", max_rows)
        self.rows: list[dict[str, Any]] = []
        self.dropped_rows = 0

    def _record(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.rows.extend(rows)
        excess = len(self.rows) - self.max_rows
        if excess > 0:
            del self.rows[:excess]
            self.dropped_rows += excess
        return rows

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "schema": np.asarray("spectral_kv.stats.v1"),
            "statistic": np.asarray(type(self).__name__),
            "rows_json": np.asarray([json.dumps(r, allow_nan=False, sort_keys=True)
                                      for r in self.rows], dtype=np.str_),
            "dropped_rows": np.asarray(self.dropped_rows, dtype=np.int64),
            "max_rows": np.asarray(self.max_rows, dtype=np.int64),
            "config_json": np.asarray(json.dumps({name: getattr(self, name) for name in
                ("bins", "eps", "max_lag", "threshold", "channels", "spatial_bins",
                 "temporal_bins", "roles", "max_streams", "max_stashes") if hasattr(self, name)},
                allow_nan=False, sort_keys=True)),
        }

    def merge(self, other: _Rows | Mapping[str, Any]) -> _Rows:
        if isinstance(other, _Rows):
            if type(other) is not type(self):
                raise TypeError("can only merge the same statistic type")
            rows, dropped = other.rows, other.dropped_rows
        else:
            kind = str(np.asarray(other.get("statistic", type(self).__name__)).item())
            if kind != type(self).__name__:
                raise TypeError("can only merge the same statistic type")
            rows = [json.loads(str(r)) for r in other.get("rows_json", [])]
            dropped = int(np.asarray(other.get("dropped_rows", 0)).item())
        self._record([dict(r) for r in rows])
        self.dropped_rows += dropped
        return self


class OnlineSpectrum(_Rows):
    """Per-frame/head radial power, with separate inclusive and DC-free views."""

    def __init__(self, bins: int = 12, eps: float = 1e-12, *, max_rows: int = 100000):
        super().__init__(max_rows)
        self.bins, self.eps = _positive("bins", bins), _epsilon(eps)

    @torch.no_grad()
    def update(self, layer: int, x: Tensor, *, frame_idx0: int = 0,
               role: str = "k", coordinate: str = "pre", sample_id: int = 0,
               **metadata: Any) -> list[dict[str, Any]]:
        x = _grid(x)
        frames, h, w, heads, _ = x.shape
        finite = torch.isfinite(x).all(dim=(1, 2, 4))
        power = torch.fft.fft2(torch.nan_to_num(x, nan=0, posinf=0, neginf=0),
                               dim=(1, 2), norm="ortho").abs().square().mean(-1)
        finite &= torch.isfinite(power).all(dim=(1, 2))
        radius = _radius(h, w, x.device)
        band_ids = _band_ids(radius, self.bins)
        sorted_radius, order = torch.sort(radius.flatten())
        output = []
        for frame in range(frames):
            for head in range(heads):
                p = power[frame, :, :, head]
                p_without_dc = p.clone()
                p_without_dc[0, 0] = 0
                total, ac_total = float(p.sum()), float(p_without_dc.sum())
                valid = bool(finite[frame, head])

                def quantile_radius(values: Tensor, energy: float) -> float | None:
                    if not valid or energy <= self.eps:
                        return None
                    cumulative = values.flatten()[order].cumsum(0)
                    index = min(int(torch.searchsorted(cumulative, cumulative.new_tensor(energy * 0.95))),
                                cumulative.numel() - 1)
                    return float(sorted_radius[index])

                r95_inc = quantile_radius(p, total)
                r95_exc = quantile_radius(p_without_dc, ac_total)
                base = {**_plain(metadata), "statistic": "spectrum", "sample_id": int(sample_id),
                        "layer": int(layer), "role": str(role), "coordinate": str(coordinate),
                        "frame_idx": int(frame_idx0 + frame), "head": head,
                        "valid": valid, "status": "ok" if valid else "nonfinite_input",
                        "total_power": total if valid else None,
                        "non_dc_power": ac_total if valid else None,
                        "dc_ratio": _safe_ratio(float(p[0, 0]), total, self.eps) if valid else None,
                        "r95": r95_exc, "r95_incl_dc": r95_inc, "r95_excl_dc": r95_exc,
                        "r95_cycles": None if r95_exc is None else r95_exc * _RADIAL_MAX,
                        "radius_normalization": _RADIAL_MAX,
                        "zero_signal": valid and total <= self.eps,
                        "zero_non_dc": valid and ac_total <= self.eps}
                for band in range(self.bins):
                    mask = band_ids == band
                    inc, exc = float(p[mask].sum()), float(p_without_dc[mask].sum())
                    output.append({**base, **_band(band, self.bins),
                                   "frequency_count": int(mask.sum()),
                                   "power_incl_dc": inc if valid else None,
                                   "power_excl_dc": exc if valid else None,
                                   "fraction_incl_dc": _safe_ratio(inc, total, self.eps) if valid else None,
                                   "fraction_excl_dc": _safe_ratio(exc, ac_total, self.eps) if valid else None})
        return self._record(output)


class DistanceDecay(_Rows):
    """Correlate aligned spatial Fourier bands across contiguous frame lags.

    ``channels`` selects the first channels per head deterministically.  The
    ring holds at most ``max_lag`` sampled spectra per active stream.  Missing
    frames or changed shapes reset that stream before any comparison.
    """

    def __init__(self, max_lag: int = 24, bins: int = 8, threshold: float = 0.9,
                 channels: int = 16, eps: float = 1e-12, *, max_rows: int = 100000,
                 max_streams: int = 128):
        super().__init__(max_rows)
        self.max_lag, self.bins = _positive("max_lag", max_lag), _positive("bins", bins)
        self.channels, self.max_streams = _positive("channels", channels), _positive("max_streams", max_streams)
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be in [-1,1]")
        self.threshold, self.eps = float(threshold), _epsilon(eps)
        self._history: OrderedDict[tuple[Any, ...], deque] = OrderedDict()

    @torch.no_grad()
    def update(self, layer: int, x: Tensor, *, frame_idx0: int = 0, role: str = "k",
               sample_id: int = 0, **metadata: Any) -> list[dict[str, Any]]:
        x = _grid(x)
        frames, h, w, heads, _ = x.shape
        coordinate = str(metadata.get("coordinate", "pre"))
        key = (int(sample_id), int(layer), str(role), coordinate)
        history = self._history.pop(key, deque(maxlen=self.max_lag))
        finite = torch.isfinite(x[..., :self.channels]).all(dim=(1, 2, 4)).cpu()
        spectrum = torch.fft.fft2(torch.nan_to_num(x[..., :self.channels], nan=0, posinf=0, neginf=0),
                                  dim=(1, 2), norm="ortho").cpu()
        finite &= torch.isfinite(spectrum).all(dim=(1, 2, 4))
        band_ids = _band_ids(_radius(h, w, torch.device("cpu")), self.bins)
        masks = [band_ids == i for i in range(self.bins)]
        output = []
        for frame in range(frames):
            index, current = int(frame_idx0 + frame), spectrum[frame]
            reset = bool(history and (history[-1][0] + 1 != index or history[-1][1].shape != current.shape))
            if reset:
                history.clear()
            for previous_index, previous, previous_finite in history:
                lag = index - previous_index
                if not 1 <= lag <= self.max_lag:
                    continue
                for head in range(heads):
                    for band, mask in enumerate(masks):
                        count = int(mask.sum())
                        valid_input = bool(finite[frame, head] and previous_finite[head])
                        corr = (_cosine(current[:, :, head][mask].flatten(),
                                        previous[:, :, head][mask].flatten(), self.eps)
                                if valid_input and count else None)
                        output.append({**_plain(metadata), "statistic": "distance_decay",
                                       "sample_id": int(sample_id), "layer": int(layer), "role": str(role),
                                       "coordinate": coordinate, "frame_idx": index,
                                       "previous_frame_idx": previous_index, "head": head, "lag": lag,
                                       **_band(band, self.bins), "frequency_count": count,
                                       "correlation": corr, "corr": corr, "valid": corr is not None,
                                       "status": ("nonfinite_input" if not valid_input else
                                                  "empty_band" if not count else
                                                  "zero_energy" if corr is None else "ok")})
            history.append((index, current.clone(), finite[frame].clone()))
        self._history[key] = history
        while len(self._history) > self.max_streams:
            self._history.popitem(last=False)
        return self._record(output)

    def fit(self, rows: Sequence[Mapping[str, Any]] | None = None):
        return fit_distance_decay(self.rows if rows is None else rows, self.threshold)


def fit_distance_decay(rows: Sequence[Mapping[str, Any]], threshold: float = 0.9
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit a power law to measured contiguous-from-low stable radii.

    Band correlations are averaged over observed frame pairs.  A missing
    nonempty low-frequency band makes the radius unmeasured.  A measured band
    below threshold terminates the stable prefix; high-frequency islands do
    not increase it.  At least two positive radii at distinct lags are needed
    for a fit.  No assumption about a research outcome is made here.
    """
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be in [-1,1]")
    groups: dict[tuple[Any, ...], dict[int, list[Mapping[str, Any]]]] = {}
    for row in rows:
        if "lag" not in row or "band" not in row:
            continue
        key = tuple(row.get(field, "pre" if field == "coordinate" else None) for field in _IDENTITY_FIELDS)
        groups.setdefault(key, {}).setdefault(int(row["lag"]), []).append(row)
    radius_rows, fit_rows = [], []
    for key, by_lag in groups.items():
        identity = dict(zip(_IDENTITY_FIELDS, key))
        measured: list[tuple[int, float]] = []
        for lag, observations in sorted(by_lag.items()):
            bins = max(int(r.get("bins", int(r["band"]) + 1)) for r in observations)
            band_rows: dict[int, list[Mapping[str, Any]]] = {}
            for row in observations:
                band_rows.setdefault(int(row["band"]), []).append(row)
            radius, status, qualifying = 0.0, "ok", 0
            for band in range(bins):
                candidates = band_rows.get(band, [])
                # Empty Fourier annuli contain no frequencies to test.
                if candidates and all(r.get("frequency_count", 1) == 0 for r in candidates):
                    continue
                values = [float(r.get("correlation", r.get("corr"))) for r in candidates
                          if r.get("valid", True) and r.get("correlation", r.get("corr")) is not None
                          and math.isfinite(float(r.get("correlation", r.get("corr"))))]
                if not values:
                    radius, status = None, "missing_band"
                    break
                if float(np.mean(values)) < threshold:
                    break
                radius = max(float(r.get("radius_high", (band + 1) / bins)) for r in candidates)
                qualifying += 1
            valid = radius is not None
            row = {**identity, "statistic": "distance_radius", "lag": lag,
                   "threshold": float(threshold), "rstar": radius, "r_star": radius,
                   "qualifying_bands": qualifying, "valid": valid, "status": status}
            radius_rows.append(row)
            if valid and radius > 0 and lag > 0:
                measured.append((lag, radius))
        if len(measured) < 2:
            fit_rows.append({**identity, "statistic": "distance_powerlaw", "threshold": float(threshold),
                             "a": None, "b": None, "r2": None, "n_lags": len(measured),
                             "valid": False, "status": "insufficient_positive_radii"})
            continue
        log_delta, log_radius = np.log(np.asarray(measured, dtype=float)).T
        slope, intercept = np.polyfit(log_delta, log_radius, 1)
        residual = float(np.square(log_radius - (intercept + slope * log_delta)).sum())
        total = float(np.square(log_radius - log_radius.mean()).sum())
        r2 = 1.0 - residual / total if total > 1e-24 else (1.0 if residual <= 1e-24 else 0.0)
        fit_rows.append({**identity, "statistic": "distance_powerlaw", "threshold": float(threshold),
                         "a": float(np.exp(intercept)), "b": float(-slope), "r2": r2,
                         "n_lags": len(measured), "valid": True, "status": "ok"})
    return radius_rows, fit_rows


class ChunkAxisStats(_Rows):
    """Joint spatial/temporal Fourier bands compared at the same denoising step."""

    def __init__(self, spatial_bins: int = 4, temporal_bins: int = 3, channels: int = 16,
                 eps: float = 1e-12, *, max_rows: int = 100000, max_streams: int = 128):
        super().__init__(max_rows)
        self.spatial_bins = _positive("spatial_bins", spatial_bins)
        self.temporal_bins = _positive("temporal_bins", temporal_bins)
        self.channels, self.max_streams = _positive("channels", channels), _positive("max_streams", max_streams)
        self.eps = _epsilon(eps)
        self._history: OrderedDict[tuple[Any, ...], deque] = OrderedDict()

    @torch.no_grad()
    def update(self, layer: int, x: Tensor, *, chunk_idx: int, step_idx: int,
               sample_id: int = 0, **metadata: Any) -> list[dict[str, Any]]:
        x = _grid(x)
        frames, h, w, heads, _ = x.shape
        key = (int(sample_id), int(layer), int(step_idx), str(metadata.get("role", "feature")),
               str(metadata.get("coordinate", "pre")))
        history = self._history.pop(key, deque(maxlen=2))
        finite = torch.isfinite(x[..., :self.channels]).all(dim=(0, 1, 2, 4)).cpu()
        spectrum = torch.fft.fftn(torch.nan_to_num(x[..., :self.channels], nan=0, posinf=0, neginf=0),
                                  dim=(0, 1, 2), norm="ortho").cpu()
        finite &= torch.isfinite(spectrum).all(dim=(0, 1, 2, 4))
        reset_reason = None
        if history and history[-1][1].shape != spectrum.shape:
            reset_reason = "shape_change"
            history.clear()
        elif history and history[-1][0] + 1 != int(chunk_idx):
            reset_reason = "nonadjacent_chunk"
            history.clear()
        spatial = _band_ids(_radius(h, w, torch.device("cpu")), self.spatial_bins)
        temporal = _band_ids(torch.fft.fftfreq(frames).abs() / 0.5, self.temporal_bins)
        output = []
        for head in range(heads):
            for spatial_band in range(self.spatial_bins):
                for temporal_band in range(self.temporal_bins):
                    mask = (temporal[:, None, None] == temporal_band) & (spatial[None] == spatial_band)
                    current = spectrum[:, :, :, head][mask].flatten()
                    previous = history[-1][1][:, :, :, head][mask].flatten() if history else None
                    valid_inputs = bool(finite[head] and (not history or history[-1][2][head]))
                    cosine = (_cosine(current, previous, self.eps)
                              if previous is not None and valid_inputs and current.numel() else None)
                    reuse = (_relative_error(previous, current, self.eps)
                             if previous is not None and valid_inputs and current.numel() else None)
                    linear = None
                    if len(history) == 2 and valid_inputs and bool(history[-2][2][head]) and current.numel():
                        older = history[-2][1][:, :, :, head][mask].flatten()
                        linear = _relative_error(2 * previous - older, current, self.eps)
                    count = int(mask.sum())
                    status = ("nonfinite_input" if not valid_inputs else "empty_band" if not count else
                              reset_reason or ("warmup" if previous is None else "ok"))
                    output.append({**_plain(metadata), "statistic": "chunk_axis", "sample_id": int(sample_id),
                                   "layer": int(layer), "head": head, "chunk_idx": int(chunk_idx),
                                   "step_idx": int(step_idx), "spatial_band": spatial_band,
                                   "temporal_band": temporal_band, "spatial_bins": self.spatial_bins,
                                   "temporal_bins": self.temporal_bins, "frequency_count": count,
                                   "spatial_radius_low": spatial_band / self.spatial_bins,
                                   "spatial_radius_high": (spatial_band + 1) / self.spatial_bins,
                                   "temporal_frequency_low": temporal_band / self.temporal_bins * 0.5,
                                   "temporal_frequency_high": (temporal_band + 1) / self.temporal_bins * 0.5,
                                   "adjacent_chunk_cosine": cosine, "cosine": cosine,
                                   "reuse_rel_l2": reuse, "linear_rel_l2": linear,
                                   "linear_extrapolation_rel_l2": linear,
                                   "valid": reuse is not None, "status": status,
                                   "linear_valid": linear is not None,
                                   "linear_status": "ok" if linear is not None else "unmeasured"})
        history.append((int(chunk_idx), spectrum.clone(), finite.clone()))
        self._history[key] = history
        while len(self._history) > self.max_streams:
            self._history.popitem(last=False)
        return self._record(output)


class KVRefreshDiff(_Rows):
    """Compare an explicitly stashed last denoising pass with a clean pass.

    No timestep value is used to infer pass identity.  The caller chooses when
    to ``stash`` and when to compare.  Each layer/sample has at most one stash,
    consumed even on a mismatched clean update.  Inputs are copied so that a
    later in-place cache write cannot change the baseline being measured.
    """

    def __init__(self, bins: int = 8, eps: float = 1e-12, *, roles: Sequence[str] = ("k", "v"),
                 max_rows: int = 100000, max_stashes: int = 128):
        super().__init__(max_rows)
        self.bins, self.eps = _positive("bins", bins), _epsilon(eps)
        self.roles = tuple(roles)
        if not self.roles or any(role not in ("k", "v") for role in self.roles):
            raise ValueError("roles must contain k and/or v")
        self.max_stashes = _positive("max_stashes", max_stashes)
        self._stashes: OrderedDict[tuple[int, int], tuple[int, dict[str, Tensor]]] = OrderedDict()

    @torch.no_grad()
    def stash(self, layer: int, k: Tensor, v: Tensor, *, frame_idx0: int,
              sample_id: int = 0, **metadata: Any) -> None:
        k, v = _grid(k), _grid(v)
        if k.shape != v.shape:
            raise ValueError("k and v must have matching shapes")
        key = (int(sample_id), int(layer))
        self._stashes.pop(key, None)
        self._stashes[key] = (int(frame_idx0), {role: value.cpu().clone()
                                              for role, value in (("k", k), ("v", v))
                                              if role in self.roles})
        while len(self._stashes) > self.max_stashes:
            self._stashes.popitem(last=False)

    @torch.no_grad()
    def update(self, layer: int, k: Tensor, v: Tensor, *, frame_idx0: int,
               sample_id: int = 0, **metadata: Any) -> list[dict[str, Any]]:
        k, v = _grid(k), _grid(v)
        if k.shape != v.shape:
            raise ValueError("k and v must have matching shapes")
        stashed = self._stashes.pop((int(sample_id), int(layer)), None)
        frames, h, w, heads, _ = k.shape
        missing = ("missing_stash" if stashed is None else
                   "stash_mismatch" if stashed[0] != int(frame_idx0)
                   or any(value.shape != k.shape for value in stashed[1].values()) else None)
        masks = [_band_ids(_radius(h, w, k.device), self.bins) == band for band in range(self.bins)]
        output = []
        for role, clean in (("k", k), ("v", v)):
            if role not in self.roles:
                continue
            clean_finite = torch.isfinite(clean).all(dim=(1, 2, 4))
            clean_spectrum = torch.fft.fft2(torch.nan_to_num(clean, nan=0, posinf=0, neginf=0),
                                           dim=(1, 2), norm="ortho")
            clean_finite &= torch.isfinite(clean_spectrum).all(dim=(1, 2, 4))
            previous_finite, previous_spectrum = None, None
            if missing is None:
                previous = stashed[1][role].to(clean.device)
                previous_finite = torch.isfinite(previous).all(dim=(1, 2, 4))
                previous_spectrum = torch.fft.fft2(torch.nan_to_num(previous, nan=0, posinf=0, neginf=0),
                                                  dim=(1, 2), norm="ortho")
                previous_finite &= torch.isfinite(previous_spectrum).all(dim=(1, 2, 4))
            for frame in range(frames):
                for head in range(heads):
                    for band, mask in enumerate(masks):
                        count = int(mask.sum())
                        finite = bool(clean_finite[frame, head]) and (
                            previous_finite is None or bool(previous_finite[frame, head]))
                        current = clean_spectrum[frame, :, :, head][mask].flatten()
                        error, previous_energy = None, None
                        current_energy = float(current.abs().square().sum()) if finite else None
                        relative = None
                        if missing is None and finite and count:
                            previous_values = previous_spectrum[frame, :, :, head][mask].flatten()
                            error = float((current - previous_values).abs().square().sum())
                            previous_energy = float(previous_values.abs().square().sum())
                            relative = _safe_ratio(math.sqrt(error), math.sqrt(current_energy), self.eps)
                        status = (missing or ("nonfinite_input" if not finite else "empty_band" if not count
                                              else "zero_reference" if relative is None else "ok"))
                        output.append({**_plain(metadata), "statistic": "refresh_diff", "sample_id": int(sample_id),
                                       "layer": int(layer), "role": role, "frame_idx": int(frame_idx0 + frame),
                                       "head": head, **_band(band, self.bins), "frequency_count": count,
                                       "relative_error": relative, "rel_l2": relative,
                                       "clean_energy": current_energy, "last_denoise_energy": previous_energy,
                                       "error_energy": error, "zero_clean": finite and current_energy <= self.eps,
                                       "valid": relative is not None, "status": status})
        return self._record(output)


__all__ = ["OnlineSpectrum", "DistanceDecay", "ChunkAxisStats", "KVRefreshDiff", "fit_distance_decay"]
