"""Average-bit accounting for AQUA-KV and the TempoKV legacy codec.

Both sides count resident KV bytes plus predictor parameters, then divide by
the number of K and V elements. AQUA's packed mode uses ``ceil(log2(n))`` bits
per HIGGS index (the codebook rate). ``uint8`` mode counts the byte the
official ``HiggsQuantizer`` actually stores, which is wider whenever ``n < 256``.
Scales stay float32, one per Hadamard group, as in the official quantizer.
Prefix tokens and the recent buffer stay bf16, matching ``TreatPrefixSeparately``
and ``PredictorHiggsCache``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .method import AquaSpec, make_quantizers
from .official import HiggsHandle


def _bf16_bytes(elements: int) -> int:
    return int(elements) * 2


def region_tokens(tokens: int, prefix: int, recent: int) -> tuple[int, int]:
    if tokens < 0:
        raise ValueError("token count must be non-negative")
    quantized = max(0, int(tokens) - int(prefix) - int(recent))
    return int(tokens) - quantized, quantized


def _code_bytes(handle: HiggsHandle, tokens: int) -> int:
    return int(tokens) * (handle.index_bytes_per_token() + handle.scale_bytes_per_token())


def predictor_count(spec: AquaSpec) -> int:
    channel = spec.channel
    per_layer = (channel * channel + channel) + (2 * channel * channel + channel)
    return per_layer * (spec.num_layers - 1)


def aquakv_storage(spec: AquaSpec, *, tokens: int) -> dict[str, Any]:
    """Bytes for one full rollout after the last block has been finalized."""
    if tokens < 1:
        raise ValueError("tokens must be positive")
    residual, first = make_quantizers(spec)
    exact_tokens, quant_tokens = region_tokens(tokens, spec.prefix_tokens, spec.recent_buffer_tokens)
    channel = spec.channel
    exact_bytes = 0
    code_bytes = 0
    per_layer = []
    for layer in range(spec.num_layers):
        handle = None if layer == 0 and spec.not_quantize_first_layer else (first if layer == 0 else residual)
        if handle is None:
            layer_exact = tokens
            layer_code = 0
            nominal = None
        else:
            layer_exact = exact_tokens
            layer_code = quant_tokens
            nominal = handle.nominal_bits()
        # K and V are stored separately and use the same quantizer.
        exact = _bf16_bytes(layer_exact * channel) * 2
        codes = _code_bytes(handle, layer_code) * 2 if handle is not None else 0
        exact_bytes += exact
        code_bytes += codes
        per_layer.append({
            "layer": layer,
            "exact_tokens": layer_exact,
            "quantized_tokens": layer_code,
            "nominal_bits": nominal,
            "bytes": exact + codes,
        })
    element_size = 2 if spec.activation_dtype != "float32" else 4
    predictor_bytes = predictor_count(spec) * element_size
    elements = 2 * spec.num_layers * tokens * channel
    kv_bytes = exact_bytes + code_bytes
    return {
        "tokens": int(tokens),
        "elements": int(elements),
        "exact_bytes": exact_bytes,
        "code_bytes": code_bytes,
        "kv_bytes": kv_bytes,
        "predictor_parameters": predictor_count(spec),
        "predictor_bytes": predictor_bytes,
        "total_bytes": kv_bytes + predictor_bytes,
        "average_bits_per_element": (kv_bytes + predictor_bytes) * 8 / elements,
        "kv_bits_per_element": kv_bytes * 8 / elements,
        "index_packing": spec.index_packing,
        "per_layer": per_layer,
        "quantizer": {
            "residual": {"edenn_d": spec.edenn_d, "edenn_n": spec.edenn_n,
                         "nominal_bits": residual.nominal_bits(),
                         "codes_per_token": residual.codes_per_token},
            "first_layer": None if first is None else {
                "edenn_d": spec.first_layer_edenn_d, "edenn_n": spec.first_layer_edenn_n,
                "nominal_bits": first.nominal_bits(), "codes_per_token": first.codes_per_token,
                "stored_as": "codes",
                "official_eval_note": "upstream SingleChunkQuantizedCacheWithPredictors dequantizes layer 0 and stores the fp tensor",
            },
        },
    }


def tempokv_storage(*, tokens: int, layers: int, heads: int, dim: int, k_bits: int, v_bits: int,
                    anchor_bits: int, group_size: int, span_frames: int, frame_seq_length: int,
                    scale_bytes: int = 2, zero_point_bytes: int = 1,
                    predictor_bytes: int = 0) -> dict[str, Any]:
    """Legacy asymmetric group quant, plus predictor bytes.

    ``storage_bytes`` on a packed span counts a bf16 scale and a uint8 zero
    point per group. The anchor is one frame per span; when its bit width
    matches the residual, it does not change the average.
    """
    if tokens < 1 or span_frames < 1 or frame_seq_length < 1:
        raise ValueError("invalid TempoKV budget shape")
    if tokens % frame_seq_length:
        raise ValueError("token count must be a whole number of frames")
    frames = tokens // frame_seq_length
    anchor_fraction = min(1.0, (frames + span_frames - 1) // span_frames / frames)

    def rate(bits: int) -> float:
        return float(bits) + (scale_bytes + zero_point_bytes) * 8 / float(group_size)

    def mixed(residual_bits: int) -> float:
        return (1.0 - anchor_fraction) * rate(residual_bits) + anchor_fraction * rate(anchor_bits)

    elements_per_role = layers * tokens * heads * dim
    k_bits_avg = mixed(k_bits)
    v_bits_avg = mixed(v_bits)
    kv_bytes = (k_bits_avg + v_bits_avg) / 8 * elements_per_role
    elements = 2 * elements_per_role
    return {
        "tokens": int(tokens),
        "elements": int(elements),
        "kv_bytes": kv_bytes,
        "predictor_bytes": int(predictor_bytes),
        "total_bytes": kv_bytes + predictor_bytes,
        "average_bits_per_element": (kv_bytes + predictor_bytes) * 8 / elements,
        "k_bits_per_element": k_bits_avg,
        "v_bits_per_element": v_bits_avg,
        "anchor_fraction": anchor_fraction,
        "group_size": int(group_size),
    }


def comparison_rows(spec: AquaSpec, tempokv: dict[str, Any], *, tokens: int) -> dict[str, Any]:
    aqua = aquakv_storage(spec, tokens=tokens)
    tempo = tempokv_storage(
        tokens=tokens, layers=spec.num_layers, heads=spec.num_heads, dim=spec.head_dim, **tempokv,
    )
    bf16_bytes = 2 * spec.num_layers * tokens * spec.channel * 2
    elements = aqua["elements"]
    layer0 = aqua["per_layer"][0]
    # Upstream eval keeps layer 0 as a dequantized fp tensor. That is a debug
    # resident layout, not the codebook the paper budgets. Reported beside the
    # code payload so the two are not mixed.
    fp_layer0 = _bf16_bytes(tokens * spec.channel) * 2
    aqua["official_eval_layer0_fp_bytes"] = fp_layer0
    aqua["official_eval_fp_resident_total_bytes"] = aqua["total_bytes"] - layer0["bytes"] + fp_layer0
    rows = [
        {"method": "bf16", "average_bits_per_element": 16.0, "kv_bytes": bf16_bytes,
         "predictor_bytes": 0, "total_bytes": bf16_bytes, "quality": None},
        {"method": "tempokv", **{k: tempo[k] for k in (
            "average_bits_per_element", "kv_bytes", "predictor_bytes", "total_bytes")},
         "quality": None},
        {"method": "aquakv", **{k: aqua[k] for k in (
            "average_bits_per_element", "kv_bytes", "predictor_bytes", "total_bytes")},
         "quality": None, "index_packing": spec.index_packing},
    ]
    return {
        "elements": elements,
        "rows": rows,
        "bit_gap_aquakv_minus_tempokv": aqua["average_bits_per_element"] - tempo["average_bits_per_element"],
        "aquakv_detail": aqua,
        "tempokv_detail": tempo,
    }


def legacy_tempokv_predictor_bytes(*, k_history: int, v_history: int, heads: int, dim: int,
                                   k_mode: str, cross_mode: str, element_bytes: int = 2) -> int | None:
    """Analytic legacy parameter count when the fitted files are not on disk.

    Shared K is ``[P, P, D]`` plus ``[P, D]``. Layer-shared CrossKV is
    ``[H, D, D]`` plus ``[H, D]``, and Self-Forcing's fast V history is the
    diagonal ``[P, H, D]``. Any other layout returns None: the comparison
    then needs the real parameter files.
    """
    if k_mode != "shared" or cross_mode != "layer_shared":
        return None
    if k_history < 0 or v_history < 0 or heads < 1 or dim < 1:
        raise ValueError("invalid legacy predictor shape")
    k_count = k_history * k_history * dim + k_history * dim
    v_count = heads * dim * dim + heads * dim + v_history * heads * dim
    return (k_count + v_count) * element_bytes


def predictor_file_bytes(paths: list[str | Path] | None) -> int | None:
    if not paths or any(path is None for path in paths):
        return None
    total = 0
    for path in paths:
        file = Path(path)
        if not file.is_file():
            return None
        total += file.stat().st_size
    return total
