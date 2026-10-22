"""A frozen-mask 2x2 quantization/pruning diagnostic against one exact snapshot.

Pruning here changes the attention input only. It does not release the source
cache or allocate a packed quantized cache. Logical packed costs and actual
floating tensor backings are consequently reported separately.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
import math

import torch

from . import attention_oracle as oracle
from . import interventions as ops
from .precision import fake_quantize


def _kept(window, values):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
            raise ValueError("kept_indices must be a one-dimensional integer sequence")
        values = values.detach().cpu().tolist()
    else:
        values = list(values)
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in values):
        raise ValueError("kept_indices must contain integer original-window indices")
    values = [int(index) for index in values]
    if not values or len(set(values)) != len(values) or any(index < 0 or index >= window.k_post.shape[1] for index in values):
        raise ValueError("kept_indices must be nonempty, unique, and within the original window")
    protected = set(window.context.protected_mask.detach().cpu().nonzero().flatten().tolist())
    if not protected.issubset(values):
        raise ValueError("kept_indices must preserve every protected sink/recent/current token")
    return tuple(sorted(values))


def _storage_bytes(*tensors):
    seen = set(); total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            total += storage.nbytes(); seen.add(key)
    return total


def _fake_role(source, eligible, bits, group_size):
    selected = source.index_select(1, eligible)
    quantized = fake_quantize(selected, bits, group_size)
    if bits == 16 or eligible.numel() == 0:
        value = source
    else:
        value = source.clone(); value[:, eligible] = quantized.value
    native = (source.shape[1] - eligible.numel()) * source.shape[0] * source.shape[2] * source.shape[3] * source.element_size()
    # An empty precision cohort needs no packed header or scales.
    payload = quantized.payload_bytes if eligible.numel() else 0
    scales = quantized.scale_bytes if eligible.numel() else 0
    metadata = quantized.metadata_bytes if eligible.numel() else 0
    return value, {"quant_payload_bytes": payload, "native_protected_bytes": native,
                   "payload_bytes": payload + native, "scale_bytes": scales, "metadata_bytes": metadata,
                   "logical_bytes": payload + native + scales + metadata}


def _supplied_pair(window, pair):
    if not isinstance(pair, Mapping):
        raise ValueError("quantized_pair needs a mapping with tensors and the explicit snapshot position contract")
    k, v = pair.get("k_post", pair.get("k")), pair.get("v")
    for tensor in (k, v):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != window.k_post.shape
                or tensor.dtype != window.k_post.dtype or tensor.device != window.k_post.device
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("supplied decoded K/V must be finite and match the snapshot shape/dtype/device")
    positions = pair.get("key_abs_positions")
    if positions is None or pair.get("snapshot_identity") != window.context.identity:
        raise ValueError("supplied decoded K/V require the same snapshot_identity and key_abs_positions")
    positions = torch.as_tensor(positions)
    expected = window.context.key_abs_positions
    if positions.dtype != torch.int64 or positions.shape != expected.shape or not torch.equal(positions.cpu(), expected.cpu()):
        raise ValueError("supplied decoded key_abs_positions differ from the frozen snapshot")
    protected = window.context.protected_mask.nonzero().flatten().to(window.k_post.device)
    if not torch.equal(k.index_select(1, protected), window.k_post.index_select(1, protected)) or not torch.equal(v.index_select(1, protected), window.v.index_select(1, protected)):
        raise ValueError("supplied decoded K/V must preserve protected sink/recent/current values exactly")
    metadata = pair.get("storage_metadata")
    costs = {name: None for name in ("payload_bytes", "scale_bytes", "metadata_bytes", "logical_bytes", "quant_payload_bytes", "native_protected_bytes")}
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("storage_metadata must be a mapping of full-snapshot byte costs")
        for name in costs:
            value = metadata.get(name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("supplied byte counts must be nonnegative integers or unknown")
            costs[name] = value
        pieces = [costs[name] for name in ("payload_bytes", "scale_bytes", "metadata_bytes")]
        if all(piece is not None for piece in pieces):
            total = sum(pieces)
            if costs["logical_bytes"] is not None and costs["logical_bytes"] != total:
                raise ValueError("supplied logical_bytes must equal payload plus scales plus metadata")
            costs["logical_bytes"] = total
    return k, v, costs


def _pearson(x, y):
    x, y = x.detach().double().reshape(-1), y.detach().double().reshape(-1)
    if x.numel() < 2:
        return None, "insufficient_elements"
    valid = torch.isfinite(x) & torch.isfinite(y)
    if not bool(valid.all()):
        return None, "nonfinite_error"
    x, y = x - x.mean(), y - y.mean()
    denominator = float(x.norm() * y.norm())
    if denominator == 0:
        return None, "zero_variance"
    return max(-1., min(1., float((x @ y) / denominator))), "measured"


def _error_correlation(reference, quant, sparse):
    a, b = quant.float() - reference.float(), sparse.float() - reference.float()
    flat, status = _pearson(a, b)
    per_token = []; per_token_status = []
    for batch in range(a.shape[0]):
        for token in range(a.shape[1]):
            value, token_status = _pearson(a[batch, token], b[batch, token])
            per_token.append(value); per_token_status.append(token_status)
    return {"flattened_signed": flat, "status": status, "per_token_signed": per_token,
            "per_token_status": per_token_status, "per_token_shape": list(a.shape[:2]),
            "error_pair": "C_quant_only_minus_A versus B_pruning_only_minus_A",
            "definition": "Pearson correlation of signed output-error components"}


def _ranks(values):
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    result = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and bool(sorted_values[stop] == sorted_values[start]):
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return result


def _selection_result(window, result):
    scores = None
    if isinstance(result, Mapping):
        scores = result.get("scores")
        result = result.get("kept_indices")
    if result is None:
        raise ValueError("selection_fn must return kept_indices or a mapping containing them")
    kept = _kept(window, result)
    if scores is not None:
        scores = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
        if scores.shape != (window.k_post.shape[1],):
            raise ValueError("selection scores must contain one score per original key token")
    return kept, scores


def _selection_consistency(window, k, v, frozen_kept, selection_fn):
    if selection_fn is None:
        return None
    if not callable(selection_fn):
        raise ValueError("selection_fn must be callable")
    baseline, baseline_scores = _selection_result(window, selection_fn(window))
    quantized_window = replace(window, k_post=k, v=v)
    quantized, quant_scores = _selection_result(window, selection_fn(quantized_window))
    eligible = set(window.context.eligible_mask.cpu().nonzero().flatten().tolist())
    a, b = set(baseline) & eligible, set(quantized) & eligible
    union = a | b
    rank = None; rank_status = "scores_not_supplied"
    if baseline_scores is not None and quant_scores is not None:
        indices = torch.tensor(sorted(eligible), dtype=torch.long)
        x, y = baseline_scores[indices], quant_scores[indices]
        finite = torch.isfinite(x) & torch.isfinite(y)
        rank, rank_status = _pearson(_ranks(x[finite]), _ranks(y[finite]))
    return {"baseline_kept_indices": list(baseline), "quantized_scoring_kept_indices": list(quantized),
            "same_selection": baseline == quantized, "eligible_jaccard": len(a & b) / len(union) if union else 1.,
            "eligible_overlap": len(a & b), "eligible_union": len(union), "rank_spearman": rank,
            "rank_status": rank_status, "frozen_mask_matches_baseline_selection": baseline == frozen_kept,
            "joint_mask_frozen": True, "scope": "eligible_history_tokens"}


@torch.no_grad()
def quant_prune_2x2(window, *, kept_indices, k_bits=4, v_bits=4, group_size=64,
                    quantized_pair=None, query_batch_size=64, selection_fn=None) -> dict:
    """Evaluate exact, sparse, quantized, and joint inputs against exact A.

    ``kept_indices`` is an externally computed frozen mask in original token
    coordinates. ``selection_fn(window)`` optionally returns another such mask
    or ``{'kept_indices': ..., 'scores': [one score per original key]}``; its
    quantized scoring result is diagnostic and never replaces the joint mask.

    A supplied decoded pair must be a mapping with k_post (or k), v,
    key_abs_positions and snapshot_identity=context.identity. Optional
    storage_metadata describes the *full* snapshot, including protected native
    tokens. Unknown codec storage remains unknown, rather than inferred from
    the independent fake-quantizer's bit labels.
    """
    if window.context.source != "bf16":
        raise ValueError("the full 2x2 requires an exact BF16 reference snapshot; source=tempokv alone is insufficient")
    exact = ops.identity(window)
    if type(query_batch_size) is not int or query_batch_size < 1:
        raise ValueError("query_batch_size must be a positive integer")
    kept = _kept(window, kept_indices)
    eligible = window.context.eligible_mask.nonzero().flatten().to(window.k_post.device)
    if quantized_pair is None:
        k_quant, k_costs = _fake_role(window.k_post, eligible, k_bits, group_size)
        v_quant, v_costs = _fake_role(window.v, eligible, v_bits, group_size)
        quant_costs = {key: k_costs[key] + v_costs[key] for key in k_costs}
        quant_source = "independent_fake_quant"
    else:
        k_quant, v_quant, quant_costs = _supplied_pair(window, quantized_pair)
        quant_source = "supplied_decoded_snapshot"
    selection = _selection_consistency(window, k_quant, v_quant, kept, selection_fn)
    index = torch.tensor(kept, dtype=torch.long, device=window.k_post.device)
    keep_all = len(kept) == exact.token_count
    sparse_k, sparse_v = (window.k_post, window.v) if keep_all else (window.k_post.index_select(1, index), window.v.index_select(1, index))
    joint_k, joint_v = (k_quant, v_quant) if keep_all else (k_quant.index_select(1, index), v_quant.index_select(1, index))
    native_payload = window.k_post.numel() * window.k_post.element_size() + window.v.numel() * window.v.element_size()
    native_costs = {"payload_bytes": native_payload, "scale_bytes": 0, "metadata_bytes": 0,
                    "logical_bytes": native_payload, "quant_payload_bytes": 0, "native_protected_bytes": native_payload}
    rows = []; outputs = {}; measurements = {}
    cases = (("A", "A_exact", window.k_post, window.v, native_costs, "bf16_reference_snapshot"),
             ("B", "B_pruning_only", sparse_k, sparse_v, native_costs, "bf16_reference_snapshot"),
             ("C", "C_quant_only", k_quant, v_quant, quant_costs, quant_source),
             ("D", "D_joint", joint_k, joint_v, quant_costs, quant_source))
    reference_output = None
    for label, strategy, key, value, costs, source in cases:
        measured = oracle.evaluate(window, k=key, v=value, query_batch_size=query_batch_size)
        if not bool(torch.isfinite(measured["reference_output"]).all()) or not bool(torch.isfinite(measured["output"]).all()):
            raise ValueError("native attention must return finite interaction outputs")
        if reference_output is None:
            reference_output = measured["reference_output"]
        elif not torch.equal(reference_output, measured["reference_output"]):
            raise ValueError("attention reference changed across the frozen 2x2 snapshot")
        error = float(measured["relative_error"])
        if not math.isfinite(error):
            raise ValueError("native attention must have a finite relative interaction error")
        measurements[label] = error; outputs[label] = measured["output"]
        full_k, full_v = (window.k_post, window.v) if label in ("A", "B") else (k_quant, v_quant)
        rows.append({"cell": label, "strategy": strategy, "local_attention_relative_error": error,
                     "actual_retained_tokens": key.shape[1], "actual_retention": key.shape[1] / exact.token_count,
                     "original_tokens": exact.token_count, "eligible_quantized_tokens": int(eligible.numel()) if label in ("C", "D") else 0,
                     "kept_indices": list(kept) if label in ("B", "D") else list(range(exact.token_count)),
                     "source": source, "reference_source": "bf16", "snapshot_identity": window.context.identity,
                     "backend_protocol": measured["backend_protocol"], "comparison_reference": "A_exact_same_snapshot",
                     "k_bits": k_bits if quantized_pair is None and label in ("C", "D") else None,
                     "v_bits": v_bits if quantized_pair is None and label in ("C", "D") else None,
                     **costs, "physical_kv_bytes": _storage_bytes(full_k, full_v),
                     "attention_input_value_bytes": key.numel() * key.element_size() + value.numel() * value.element_size(),
                     "storage_scope": "full_source_snapshot_retained; pruning_changes_attention_input_only",
                     "cache_memory_reduction_claimed": False, "packed_storage_allocated": False,
                     "pruning_kind": "compute_only_frozen_token_gather" if label in ("B", "D") else "none",
                     "metric_scope": "attention_local", "status": "measured"})
    if measurements["A"] != 0.:
        raise ValueError("exact A must be an exact null control")
    return {"rows": rows, "interaction": measurements["D"] - measurements["C"] - measurements["B"],
            "D_minus_C": measurements["D"] - measurements["C"], "B_minus_A": measurements["B"] - measurements["A"],
            "D_minus_B": measurements["D"] - measurements["B"], "C_minus_A": measurements["C"] - measurements["A"],
            "selection_consistency": selection, "error_correlation": _error_correlation(reference_output, outputs["C"], outputs["B"]),
            "outputs": outputs, "frozen_kept_indices": list(kept), "quantization_source": quant_source,
            "snapshot_identity": window.context.identity, "joint_mask_frozen": True,
            "interaction_definition": "D_error - C_error - B_error; A_error=0",
            "interpretation": "algebraic contrast of norm-based errors; nonlinear attention and norms do not imply an additive physical model or a synergy conclusion"}
