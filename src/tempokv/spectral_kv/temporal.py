"""Bounded joint temporal/spatial Fourier statistics for suite B E1.

The caller normally invokes this collector on clean cache-writing passes.
An explicit ``is_clean_pass=False`` is ignored.  Once two contiguous clean
frames are available, every new frame emits one rolling-window observation.
``window_frames`` records the actual window length during warmup.  Gaps,
shape changes, and nonfinite frames reset that stream rather than connecting
unrelated frames.  Sampling always uses the same first ``channels`` channels
per head at the original spatial positions.

DC removal means the joint (time=0, height=0, width=0) Fourier coefficient.
It does not remove the whole temporal-DC plane: that plane is a measurement
of interest for the chunk/temporal experiments.  Spatial radii are normalized
by sqrt(0.5); temporal frequencies are cycles per observed latent frame.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import json
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .stats import _Rows, _band_ids, _epsilon, _grid, _plain, _positive, _radius, _safe_ratio


class OnlineTemporalSpectrum(_Rows):
    def __init__(self, window: int = 8, spatial_bins: int = 4, temporal_bins: int = 3,
                 channels: int = 16, max_rows: int = 100000, eps: float = 1e-12,
                 *, max_streams: int = 128):
        super().__init__(max_rows)
        self.window = _positive("window", window)
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self.spatial_bins = _positive("spatial_bins", spatial_bins)
        self.temporal_bins = _positive("temporal_bins", temporal_bins)
        self.channels = _positive("channels", channels)
        self.max_streams = _positive("max_streams", max_streams)
        self.eps = _epsilon(eps)
        self._history: OrderedDict[tuple[Any, ...], deque] = OrderedDict()

    def state_dict(self) -> dict[str, np.ndarray]:
        state = super().state_dict()
        config = json.loads(str(state["config_json"].item()))
        config["window"] = self.window
        state["config_json"] = np.asarray(json.dumps(config, allow_nan=False, sort_keys=True))
        return state

    @torch.no_grad()
    def update(self, layer: int, x: Tensor, *, frame_idx0: int = 0, role: str = "k",
               sample_id: int = 0, **metadata: Any) -> list[dict[str, Any]]:
        x = _grid(x)
        if metadata.get("is_clean_pass") is False:
            return []
        coordinate = str(metadata.get("coordinate", "pre"))
        key = (int(sample_id), int(layer), str(role), coordinate)
        history = self._history.pop(key, deque(maxlen=self.window))
        frames, h, w, heads, _ = x.shape
        sampled = x[..., :self.channels].cpu()
        spatial_ids = _band_ids(_radius(h, w, torch.device("cpu")), self.spatial_bins)
        output = []
        for offset in range(frames):
            frame_idx = int(frame_idx0 + offset)
            current = sampled[offset]
            if history and (history[-1][0] + 1 != frame_idx or history[-1][1].shape != current.shape):
                history.clear()
            if not bool(torch.isfinite(current).all()):
                history.clear()
                output.append({**_plain(metadata), "statistic": "temporal_spectrum",
                               "record_kind": "temporal", "sample_id": int(sample_id),
                               "layer": int(layer), "role": str(role), "coordinate": coordinate,
                               "frame_idx": frame_idx, "window_frames": 0, "window_size": self.window,
                               "valid": False, "status": "nonfinite_input", "dc_definition": "joint_t0_h0_w0"})
                continue
            history.append((frame_idx, current.clone()))
            if len(history) < 2:
                continue
            values = torch.stack([entry[1] for entry in history])
            power = torch.fft.fftn(values, dim=(0, 1, 2), norm="ortho").abs().square().mean(-1)
            temporal_ids = _band_ids(torch.fft.fftfreq(len(history)).abs() / 0.5, self.temporal_bins)
            for head in range(heads):
                p = power[..., head]
                without_dc = p.clone()
                without_dc[0, 0, 0] = 0
                finite = bool(torch.isfinite(p).all())
                total = float(p.sum())
                ac_total = float(without_dc.sum())
                base = {**_plain(metadata), "statistic": "temporal_spectrum", "record_kind": "temporal",
                        "sample_id": int(sample_id), "layer": int(layer), "head": head,
                        "role": str(role), "coordinate": coordinate, "frame_idx": frame_idx,
                        "window_start_frame": history[0][0], "window_end_frame": history[-1][0],
                        "window_frames": len(history), "window_size": self.window,
                        "sampled_channels": int(values.shape[-1]), "dc_definition": "joint_t0_h0_w0",
                        "valid": finite, "status": "ok" if finite else "nonfinite_spectrum",
                        "total_power": total if finite else None,
                        "non_dc_power": ac_total if finite else None,
                        "dc_ratio": _safe_ratio(float(p[0, 0, 0]), total, self.eps) if finite else None,
                        "zero_signal": finite and total <= self.eps,
                        "zero_non_dc": finite and ac_total <= self.eps}
                for spatial_band in range(self.spatial_bins):
                    for temporal_band in range(self.temporal_bins):
                        mask = ((temporal_ids[:, None, None] == temporal_band)
                                & (spatial_ids[None] == spatial_band))
                        inc, exc = float(p[mask].sum()), float(without_dc[mask].sum())
                        output.append({**base, "spatial_band": spatial_band, "temporal_band": temporal_band,
                                       "spatial_bins": self.spatial_bins, "temporal_bins": self.temporal_bins,
                                       "spatial_radius_low": spatial_band / self.spatial_bins,
                                       "spatial_radius_high": (spatial_band + 1) / self.spatial_bins,
                                       "temporal_frequency_low": temporal_band / self.temporal_bins * 0.5,
                                       "temporal_frequency_high": (temporal_band + 1) / self.temporal_bins * 0.5,
                                       "frequency_count": int(mask.sum()),
                                       "power_incl_dc": inc if finite else None,
                                       "power_excl_dc": exc if finite else None,
                                       "fraction_incl_dc": _safe_ratio(inc, total, self.eps) if finite else None,
                                       "fraction_excl_dc": _safe_ratio(exc, ac_total, self.eps) if finite else None})
        self._history[key] = history
        while len(self._history) > self.max_streams:
            self._history.popitem(last=False)
        return self._record(output)


__all__ = ["OnlineTemporalSpectrum"]
