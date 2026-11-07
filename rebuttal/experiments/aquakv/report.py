"""AquaKV metrics computed from actual HIGGS payload and predictor tensors."""
from __future__ import annotations


def _finish(metrics):
    signal, count = metrics["signal_sse"], metrics["numel"]
    return {**metrics,
            "prediction_nmse": metrics["prediction_sse"]/signal if signal else None,
            "actual_reconstruction_nmse": metrics["error_sse"]/signal if signal else None,
            "actual_reconstruction_mse": metrics["error_sse"]/count if count else None}


def summarize(rows, predictor_bytes=0):
    aggregate = {}
    for role in ("k", "v"):
        totals = {key: sum(row[role][key] for row in rows)
                  for key in ("signal_sse", "prediction_sse", "error_sse", "numel")}
        aggregate[role] = _finish(totals)
    payload = sum(row["payload_bytes"] for row in rows)
    numel = sum(aggregate[r]["numel"] for r in ("k", "v"))
    return {"aggregate": aggregate,
            "per_layer": [{**row, "k": _finish(row["k"]), "v": _finish(row["v"])} for row in rows],
            "storage": {"payload_bytes": payload, "predictor_parameter_bytes": predictor_bytes,
                        "total_bytes": payload+predictor_bytes,
                        "payload_bits_per_element": payload*8/numel,
                        "total_bits_per_element": (payload+predictor_bytes)*8/numel,
                        "note": "actual packed HIGGS indices plus FP32 scales, including Hadamard/grid padding; codebooks and serialization overhead excluded"}}
