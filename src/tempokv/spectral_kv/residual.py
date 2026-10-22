"""E5 compares spectra of supplied features and genuine prediction residuals."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .stats import OnlineSpectrum


class ResidualSpectrum:
    """Wrap ``OnlineSpectrum`` with explicit original/residual source labels.

    The predictor is supplied by the caller; this class never fits or invents
    one.  Per-band rows carry paired r95/DC measurements so comparisons remain
    attributable to the same frame, layer, head, and observed prediction.
    """

    def __init__(self, bins: int = 12, eps: float = 1e-12, *, max_rows: int = 100000):
        self.spectrum = OnlineSpectrum(bins=bins, eps=eps, max_rows=max_rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.spectrum.rows

    @property
    def dropped_rows(self) -> int:
        return self.spectrum.dropped_rows

    @torch.no_grad()
    def update(
        self,
        layer: int,
        original: torch.Tensor,
        predicted: torch.Tensor,
        role: str = "k",
        frame_idx0: int = 0,
        sample_id: int = 0,
        **metadata: Any,
    ) -> list[dict[str, Any]]:
        if (not isinstance(original, torch.Tensor) or not isinstance(predicted, torch.Tensor)
                or original.ndim != 5 or original.shape != predicted.shape):
            raise ValueError("original and predicted must have the same [F,H,W,heads,D] shape")
        if not original.is_floating_point() or not predicted.is_floating_point():
            raise ValueError("original and predicted must be real floating-point tensors")
        if original.device != predicted.device:
            raise ValueError("original and predicted must be on the same device")
        if role not in ("k", "v"):
            raise ValueError("residual role must be 'k' or 'v'")
        metadata = {**metadata, "suite_id": "B_SPECTRAL_KV", "experiment_id": "B_E5_RESIDUAL"}
        metadata.pop("source_kind", None)
        original_rows = self.spectrum.update(layer, original, role=role, frame_idx0=frame_idx0,
            sample_id=sample_id, source_kind="original", **metadata)
        residual = original.float() - predicted.float()
        residual_rows = self.spectrum.update(layer, residual, role=role, frame_idx0=frame_idx0,
            sample_id=sample_id, source_kind="residual", **metadata)
        for observed, difference in zip(original_rows, residual_rows):
            comparison = {}
            for field in ("r95_excl_dc", "r95_incl_dc", "dc_ratio"):
                before, after = observed[field], difference[field]
                comparison[f"original_{field}"] = before
                comparison[f"residual_{field}"] = after
                comparison[f"delta_{field}"] = after - before if before is not None and after is not None else None
            observed.update(comparison)
            difference.update(comparison)
        return original_rows + residual_rows

    def state_dict(self) -> dict[str, np.ndarray]:
        state = self.spectrum.state_dict()
        state["statistic"] = np.asarray("ResidualSpectrum")
        state["suite_id"] = np.asarray("B_SPECTRAL_KV")
        state["experiment_id"] = np.asarray("B_E5_RESIDUAL")
        return state
