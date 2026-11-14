"""Aggregate AQUA-KV / direct / TempoKV metrics with identical formulas and storage accounting."""
from __future__ import annotations

from typing import Any

from rebuttal.common.cache_metrics import finish_metrics


def summarize(rows: list[dict[str, Any]], layer_ids: tuple[int, ...] | None = None) -> dict[str, Any]:
    per_layer, totals = [], {"k": {}, "v": {}, "kv": {}}
    for row in rows:
        entry = {"layer": layer_ids[row["layer_index"]] if layer_ids else row["layer_index"]}
        for role in ("k", "v"):
            entry[role] = finish_metrics(dict(row[role]))
            for target in (totals[role], totals["kv"]):
                for key, value in row[role].items():
                    target[key] = target.get(key, 0) + value
        per_layer.append(entry)
    return {"aggregate": {r: finish_metrics(v) for r, v in totals.items()},
            "aggregate_predicted_layers_only": _predicted(rows), "per_layer": per_layer}


def _predicted(rows):
    out = {}
    for role in ("k", "v"):
        acc: dict[str, float] = {}
        for row in rows[1:]:
            for key, value in row[role].items():
                acc[key] = acc.get(key, 0) + value
        out[role] = finish_metrics(acc)
    return out


def storage(numel_per_role: int, quantizer, predictor_bytes: int = 0) -> dict[str, Any]:
    payload = 2 * numel_per_role * quantizer.bytes_per_element()
    return {"payload_bytes": payload, "predictor_parameter_bytes": predictor_bytes,
            "total_bytes": payload + predictor_bytes,
            "payload_bits_per_element": quantizer.bytes_per_element() * 8,
            "total_bits_per_element": (payload + predictor_bytes) * 8 / (2 * numel_per_role),
            "note": "payload = residual codes + per-group scale (+ zero point when asymmetric); "
                    "predictor parameters reported separately and amortised only in total_*"}


def tempokv_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Project an existing cache_metrics.replay report onto the same metric keys."""
    keep = ("prediction_nmse", "residual_quant_nmse_x", "actual_reconstruction_nmse", "actual_reconstruction_mse")
    return {role: {k: report[role]["all"].get(k) for k in keep} for role in ("k", "v")}
