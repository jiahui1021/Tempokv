"""Exact residual quantization diagnostics using the shipped codec arithmetic."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable

import torch

from tempokv.codec import anchor_roundtrip, predict_temporal, predict_cross_kv, _quantize
from tempokv.config import CodecConfig, is_temporal_v_predictor
from tempokv.parameters import TemporalParameters, CrossKVParameters, load_parameters
from .manifest import json_value
from .trace import Trace


def _sums(value: torch.Tensor) -> torch.Tensor:
    # [1,H,records,D] -> per-record energies, retaining float64 reduction.
    return value.double().square().sum(dim=(0, 1, 3))


def _ratio(a: float, b: float) -> float | None:
    return a / b if b > 1e-20 else None


def finish_metrics(sums: dict[str, float]) -> dict[str, Any]:
    n = int(sums.get("numel", 0))
    signal = sums.get("signal_sse", 0.0)
    pred = sums.get("prediction_sse", 0.0)
    quant = sums.get("quant_error_sse", 0.0)
    error = sums.get("error_sse", 0.0)
    return {**sums, "numel": n,
            "prediction_mse": pred/n if n else None,
            "prediction_nmse": _ratio(pred, signal),
            "residual_quant_mse": quant/n if n else None,
            "residual_quant_nmse_x": _ratio(quant, signal),
            "residual_quant_nmse_r": _ratio(quant, pred),
            "actual_reconstruction_mse": error/n if n else None,
            "actual_reconstruction_nmse": _ratio(error, signal),
            "zero_signal": signal <= 1e-20,
            "zero_residual": pred <= 1e-20}


def load_pair(k_path: str | Path, v_path: str | Path, config: CodecConfig,
              device: str | torch.device = "cpu") -> tuple[TemporalParameters, Any]:
    from tempokv.codec import TempoKVCodec
    k = TemporalParameters.load(k_path).to(device)
    v = load_parameters(v_path).to(device)
    TempoKVCodec(k, v, config)  # validates signature and paired identities
    return k, v


@torch.inference_mode()
def replay(trace: Trace, config: CodecConfig, temporal: TemporalParameters,
           value_parameters: CrossKVParameters | TemporalParameters, *,
           span_frames: int | None = None, event_path: str | Path | None = None,
           event_stride: int = 1, model: str = "unknown", variant: str = "production",
           return_reconstructed: bool = False,
           event_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Reset history per span and distinguish exact R-DQ(R) from BF16 X-Xhat.

    Every emitted event aggregates only records with an identical explicit
    prompt/seed/time identity; absent identity remains absent, never guessed.
    """
    if config.mirrored:
        raise ValueError("rebuttal diagnostics currently require k_to_v; do not relabel a mirrored replay")
    if event_stride < 1:
        raise ValueError("event_stride must be positive")
    layers, records, frames, heads, dim = trace.keys.shape
    if dim % config.group_size:
        raise ValueError("quantization group_size must divide head_dim")
    span_frames = frames if span_frames is None else int(span_frames)
    if span_frames < 1:
        raise ValueError("span_frames must be positive")
    device = trace.keys.device
    temporal, value_parameters = temporal.to(device), value_parameters.to(device)
    reconstruction_dtype = getattr(torch, config.reconstruction_dtype)
    scale_dtype = getattr(torch, config.scale_dtype)
    totals = {role: {part: {} for part in ("anchor", "warmup", "predicted", "non_anchor", "all")} for role in ("k", "v")}
    rebuilt_all = {role: torch.empty_like(trace.keys) for role in ("k", "v")} if return_reconstructed else None
    groups: dict[str, tuple[dict[str, Any], list[int]]] = {}
    for index, identity in enumerate(trace.records):
        key = json.dumps(json_value(identity), sort_keys=True)
        if key not in groups:
            groups[key] = (identity, [])
        groups[key][1].append(index)
    handle = None
    if event_path:
        path = Path(event_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8")
    event_count = 0
    def emit(role: str, frame: int, span_index: int, offset: int, layer: int,
             current: torch.Tensor, prediction: torch.Tensor,
             residual: torch.Tensor, quantized: torch.Tensor,
             rebuilt: torch.Tensor, scale: torch.Tensor | None) -> None:
        nonlocal event_count
        sums = {"signal_sse": _sums(current), "prediction_sse": _sums(residual),
                "quant_error_sse": _sums(residual-quantized), "error_sse": _sums(current-rebuilt)}
        n_each = heads * dim
        order = temporal.history_length if role == "k" else value_parameters.history_length
        part = "anchor" if offset == 0 else ("warmup" if offset < max(order, 1) else "predicted")
        include = [part, "all"] + (["non_anchor"] if offset else [])
        for name in include:
            target = totals[role][name]
            for metric, value in sums.items():
                target[metric] = target.get(metric, 0.0) + float(value.sum())
            target["numel"] = target.get("numel", 0) + records*n_each
        if frame % event_stride and offset:
            return
        for identity, indices in groups.values():
            selected = torch.as_tensor(indices, device=device, dtype=torch.long)
            values = {name: float(value.index_select(0, selected).sum()) for name, value in sums.items()}
            # Anchor has no predictor or residual quantization objective.
            if offset == 0:
                values["prediction_sse"] = 0.0
                values["quant_error_sse"] = 0.0
            absolute = frame + int(identity.get("absolute_start_frame", 0))
            event = {"model": model, "variant": variant, "metric_scope": "bf16_trace_replay",
                     "prompt_id": identity.get("prompt_id"), "prompt_sha256": identity.get("prompt_sha256"),
                     "seed": identity.get("seed"), "sample_id": identity.get("sample_id"),
                     "layer": trace.layer_ids[layer], "head": "all", "record_indices": indices,
                     "absolute_latent_frame": absolute, "frame_index": frame,
                     "span_index": span_index, "is_anchor": offset == 0, "position_kind": part,
                     "role": role, "numel": len(indices)*n_each, **values}
            intervals = identity.get("pixel_intervals")
            if intervals is not None:
                if frame >= len(intervals):
                    raise ValueError("pixel_intervals does not cover every trace frame")
                interval = intervals[frame]
                event.update({"pixel_start": interval[0], "pixel_end": interval[1]})
                if identity.get("fps"):
                    event.update({"start_seconds": interval[0]/float(identity["fps"]),
                                  "end_seconds": interval[1]/float(identity["fps"])})
            if scale is not None:
                event["scale_mean"] = float(scale.float().mean())
                event["scale_max"] = float(scale.float().max())
            event["residual_abs_max"] = float(residual[:, :, indices].abs().max())
            if handle:
                handle.write(json.dumps(json_value(event), allow_nan=False) + "\n")
            if event_callback:
                event_callback(event)
            event_count += 1
    try:
        for layer in range(layers):
            kp = temporal.for_layer(trace.layer_ids[layer]) if temporal.layered else temporal
            vp = value_parameters.for_layer(trace.layer_ids[layer]) if value_parameters.layered else value_parameters
            for span_index, start in enumerate(range(0, frames, span_frames)):
                kh: list[torch.Tensor] = []
                vh: list[torch.Tensor] = []
                for frame in range(start, min(frames, start+span_frames)):
                    offset = frame-start
                    key = trace.keys[layer, :, frame].permute(1, 0, 2).unsqueeze(0)
                    value = trace.values[layer, :, frame].permute(1, 0, 2).unsqueeze(0)
                    if offset == 0:
                        kr = anchor_roundtrip(key, config).to(reconstruction_dtype).float()
                        vr = anchor_roundtrip(value, config).to(reconstruction_dtype).float()
                        zero = torch.zeros_like(key)
                        emit("k", frame, span_index, offset, layer, key, zero, zero, zero, kr, None)
                        emit("v", frame, span_index, offset, layer, value, zero, zero, zero, vr, None)
                    else:
                        kp_frame = predict_temporal(kh, kp, legacy=config.fit_mode == "legacy") if kp.history_length else torch.zeros_like(key)
                        kres = key.float()-kp_frame
                        _, kscale, _, kquant = _quantize(kres, bits=config.k_bits, group_size=config.group_size,
                                                      mode=config.residual_mode, scale_dtype=scale_dtype)
                        kr = (kp_frame+kquant).to(reconstruction_dtype).float()
                        if is_temporal_v_predictor(config.v_predictor_mode):
                            vp_frame = predict_temporal(vh, vp, legacy=config.fit_mode == "legacy") if vp.history_length else torch.zeros_like(value)
                        else:
                            source = kr-kp_frame if config.v_predictor_mode == "innovation" else kr
                            vp_frame = predict_cross_kv(source, vp, vh, matmul_precision=config.cross_matmul_precision,
                                                        legacy=config.fit_mode == "legacy")
                        vres = value.float()-vp_frame
                        _, vscale, _, vquant = _quantize(vres, bits=config.v_bits, group_size=config.group_size,
                                                      mode=config.residual_mode, scale_dtype=scale_dtype)
                        vr = (vp_frame+vquant).to(reconstruction_dtype).float()
                        emit("k", frame, span_index, offset, layer, key, kp_frame, kres, kquant, kr, kscale)
                        emit("v", frame, span_index, offset, layer, value, vp_frame, vres, vquant, vr, vscale)
                    kh = [kr, *kh[:max(kp.history_length-1, 0)]] if kp.history_length else []
                    vh = [vr, *vh[:max(vp.history_length-1, 0)]] if vp.history_length else []
                    if rebuilt_all is not None:
                        rebuilt_all["k"][layer, :, frame] = kr[0].permute(1, 0, 2)
                        rebuilt_all["v"][layer, :, frame] = vr[0].permute(1, 0, 2)
    finally:
        if handle:
            handle.close()
    report = {"schema_version": "tempokv.rebuttal.cache.v1", "status": "complete",
              "metric_scope": "bf16_trace_replay", "trace": str(trace.path), "shape": list(trace.shape),
              "record_identity_complete": trace.metadata.get("record_identity_complete", False),
              "config": asdict(config), "span_frames": span_frames, "events": event_count,
              "k": {k: finish_metrics(v) for k, v in totals["k"].items()},
              "v": {k: finish_metrics(v) for k, v in totals["v"].items()}}
    # Anchor-only summaries intentionally have no prediction/quantization objective.
    for role in ("k", "v"):
        for field in ("prediction_mse", "prediction_nmse", "residual_quant_mse", "residual_quant_nmse_x", "residual_quant_nmse_r"):
            report[role]["anchor"][field] = None
    if rebuilt_all is not None:
        report["reconstructed"] = rebuilt_all
    return report
