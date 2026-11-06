"""Optional bounded derivative-free quantization-objective fitting baseline.

This is explicitly a local random-search comparison, not a claim about the
best achievable quantization-aware predictor. Every decision uses the real
closed-loop quantizer; no hard-round STE is substituted for that objective.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from rebuttal.common.manifest import write_json, file_sha256


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    if stage == "plan":
        result = {"status": "planned", "method": "bounded_local_random_search_real_quantizer",
                  "required": ["calibration_trace", "validation_trace", "codec_config", "k_params", "v_params"],
                  "steps": int(config.get("optimization_steps", 8)),
                  "selection": "validation_quantization_distortion; test_not_used"}
        write_json(output_dir / "plan.json", result)
        return result
    if stage != "fit":
        raise ValueError("quant_aware_fit supports plan and fit; evaluate exported parameters with quant_calibration")
    import torch
    from tempokv.config import load_config, is_fixed_temporal_predictor, is_fixed_v_predictor
    from tempokv.parameters import TemporalParameters
    from tempokv.fitting import stamp_parameter_identity
    from rebuttal.common.trace import load_trace, device_for, check_trace_split
    from rebuttal.common.cache_metrics import load_pair, replay
    codec = load_config(config["codec_config"], config.get("codec_overrides"))
    if is_fixed_temporal_predictor(codec.k_predictor_mode) or is_fixed_v_predictor(codec.v_predictor_mode):
        raise ValueError("quant-aware fitting requires learned predictors; fixed baselines must remain fixed")
    device = device_for(config.get("device", "auto"))
    calibration = load_trace(config["calibration_trace"], device=device,
                             identity_path=config.get("calibration_identity"), max_records=config.get("max_records"))
    validation = load_trace(config["validation_trace"], device=device,
                            identity_path=config.get("validation_identity"), max_records=config.get("max_records"))
    split = check_trace_split(calibration, validation, allow_in_sample=False)
    if config.get("require_prompt_identity", False) and not split["prompt_identity_verified"]:
        raise ValueError("complete calibration/validation prompt identity is required")
    k, v = load_pair(config["k_params"], config["v_params"], codec, device)
    tensors = [k.history_weights, k.history_bias]
    tensors += [v.history_weights, v.history_bias] if isinstance(v, TemporalParameters) else [v.weight, v.history_weights, v.bias]
    scales = [max(float(t.float().square().mean().sqrt()), 1e-3) if t.numel() else 1.0 for t in tensors]
    steps = int(config.get("optimization_steps", 8))
    radius = float(config.get("perturbation_radius", 0.01))
    if steps < 1 or radius <= 0:
        raise ValueError("optimization_steps and perturbation_radius must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(config.get("optimizer_seed", 0)))
    def score(trace):
        result = replay(trace, codec, k, v, span_frames=config.get("span_frames"), event_stride=trace.shape[-3]+1)
        numerator = sum(result[role]["non_anchor"]["quant_error_sse"] for role in ("k", "v"))
        denominator = sum(result[role]["non_anchor"]["signal_sse"] for role in ("k", "v"))
        if denominator <= 1e-20:
            raise ValueError("zero-signal trace cannot select a quantization objective")
        return numerator / denominator
    def snapshot():
        return [t.clone() for t in tensors]
    def restore(state):
        for target, value in zip(tensors, state, strict=True):
            target.copy_(value)
    started = time.perf_counter()
    baseline_cal, best_val = score(calibration), score(validation)
    current_cal = baseline_cal
    best_state = snapshot()
    history = [{"step": 0, "calibration_loss": current_cal, "validation_loss": best_val, "selected": True}]
    with torch.no_grad():
        for step in range(1, steps+1):
            before = snapshot()
            directions = [(torch.randint(0, 2, t.shape, generator=generator, dtype=torch.int8).float()*2-1).to(device)*scale
                          for t, scale in zip(tensors, scales, strict=True)]
            candidate = None
            candidate_cal = current_cal
            for sign in (1.0, -1.0):
                restore(before)
                for tensor, direction in zip(tensors, directions, strict=True):
                    tensor.add_(direction, alpha=sign*radius)
                loss = score(calibration)
                if loss < candidate_cal:
                    candidate_cal, candidate = loss, snapshot()
            restore(before if candidate is None else candidate)
            current_cal = candidate_cal
            val_loss = score(validation)
            selected = val_loss < best_val
            if selected:
                best_val, best_state = val_loss, snapshot()
            history.append({"step": step, "calibration_loss": current_cal, "validation_loss": val_loss,
                            "calibration_accepted": candidate is not None, "selected": selected})
        restore(best_state)
    k, v = stamp_parameter_identity(k, v, signature=codec.format_signature(), coordinate_system=k.coordinate_system)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_path, v_path = output_dir / "k_temporal.npz", output_dir / ("v_temporal.npz" if isinstance(v, TemporalParameters) else "v_cross_kv.npz")
    k.save(k_path)
    v.save(v_path)
    write_json(output_dir / "codec.json", asdict(codec))
    result = {"status": "complete", "method": "bounded_local_random_search_real_quantizer",
              "best_achievable_claim": False, "selection": "validation_only", "split": split,
              "baseline_calibration_loss": baseline_cal, "selected_validation_loss": best_val,
              "history": history, "fit_seconds": time.perf_counter()-started,
              "codec_config": str(output_dir / "codec.json"), "k_params": str(k_path), "v_params": str(v_path),
              "parameter_set_id": k.parameter_set_id,
              "k_sha256": file_sha256(k_path), "v_sha256": file_sha256(v_path),
              "selection_sources": [{"role": role, "path": str(trace.path), "sha256": file_sha256(trace.path),
                                     "split_identity": trace.metadata["split_identity"]}
                                    for role, trace in (("calibration", calibration), ("validation", validation))]}
    write_json(output_dir / "fit_report.json", result)
    return result
