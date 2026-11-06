"""Fit and replay the two independent architecture studies."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from tempokv.config import load_config
from tempokv.fitting import (fit_predictors, fit_legacy_cross_kv, fit_legacy_temporal_v_alpha,
                            fit_cross_kv)
from tempokv.parameters import TemporalParameters, CrossKVParameters
from rebuttal.adapters.experimental_codec import ExperimentalCodec, ResearchTemporal, ResearchCross, encode_tensor
from .temporal import fit_temporal_architecture, identity_predictor
from .cross_low_rank import fit_cross_architecture


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_trace(path: str | Path, device="cpu"):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "keys" not in data or "values" not in data:
            raise ValueError(f"trace missing keys/values: {path}")
        keys, values = [torch.from_numpy(np.array(data[k], copy=True)).float().to(device) for k in ("keys", "values")]
        info = {}
        for key in data.files:
            if key in {"keys", "values"}:
                continue
            array = data[key]
            if array.size <= 100000 and ("prompt" in key or "seed" in key or "frame" in key or key in {
                "metadata", "record_ids", "sample_ids", "token_indices", "layer_ids", "coordinate_system"}):
                info[key] = array.tolist()
    if keys.shape != values.shape or keys.ndim < 5:
        raise ValueError("architecture traces must have matching [L,...,T,H,D] keys/values")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("non-finite trace")
    original_shape = list(keys.shape)
    shape = (keys.shape[0], -1, *keys.shape[-3:])
    keys, values = keys.reshape(shape), values.reshape(shape)
    return keys, values, {"path": str(path.resolve()), "sha256": digest(path), "original_shape": original_shape,
                          "normalized_shape": list(keys.shape), "identity": info}


def _prompt_set(info):
    identity = info["identity"]
    for key in ("prompts", "prompt_texts", "prompt_hashes", "prompt_ids", "record_prompt_indices"):
        if key not in identity:
            continue
        value = identity[key]
        if isinstance(value, list):
            return {json.dumps(x, sort_keys=True) for x in value}, key
    return None, None


def validate_split(calibration, evaluation, *, allow_in_sample=False, require_prompt_identity=False):
    if calibration["sha256"] == evaluation["sha256"] and not allow_in_sample:
        raise ValueError("calibration and evaluation traces are identical; provide held-out trace or explicit allow_in_sample")
    cal, ck = _prompt_set(calibration); eva, ek = _prompt_set(evaluation)
    overlap = sorted(cal & eva) if cal is not None and eva is not None and ck == ek else None
    if overlap and not allow_in_sample:
        raise ValueError("calibration/evaluation prompt identities overlap")
    if require_prompt_identity and overlap is None:
        raise ValueError("trace prompt identity is unavailable; cannot verify formal train/test split")
    return {"in_sample_explicitly_allowed": bool(allow_in_sample), "prompt_identity_keys": [ck, ek],
            "prompt_overlap_count": None if overlap is None else len(overlap),
            "status": "verified_prompt_disjoint" if overlap == [] else
                      ("explicit_in_sample" if allow_in_sample else "file_distinct_prompt_identity_unavailable")}


def reconstruct_keys(keys, predictor, config):
    cfg = replace(config, stride=1)
    rebuilt = torch.empty_like(keys)
    for layer in range(keys.shape[0]):
        _, value = encode_tensor(keys[layer].permute(0, 2, 1, 3), predictor, cfg, bits=cfg.k_bits, layer=layer)
        rebuilt[layer] = value.permute(0, 2, 1, 3).float()
    return rebuilt


@torch.no_grad()
def refit_production_cross(keys, values, baseline, config):
    """Retain production V family, refitting it for a changed reconstructed K."""
    if config.fit_mode == "legacy":
        result = fit_legacy_cross_kv(keys, values, ridge=0.1, stride=config.stride,
                                    weight_mode=config.cross_kv_weight_mode)
        if config.v_history_length:
            result = fit_legacy_temporal_v_alpha(keys, values, result, config, rounds=2, ridge=0.0,
                                                 gain_limit=1.0, fit_history_source="reconstructed")
        return result
    result = fit_cross_kv(keys[..., 1:, :, :], values[..., 1:, :, :], ridge=0.1, stride=config.stride)
    if config.v_history_length:
        from tempokv.fitting import _reconstruct_cross_kv_trace
        for _ in range(2):
            history = _reconstruct_cross_kv_trace(keys, values, result, config)
            result = fit_cross_kv(keys, values, reconstructed_values=history,
                                  history_length=config.v_history_length, history_mode="diagonal",
                                  ridge=0.1, stride=config.stride)
    return result


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _settings(config):
    axis = config.get("axis", "temporal")
    if axis not in {"temporal", "cross"}:
        raise ValueError("architecture axis must be temporal or cross")
    variants = config.get("variants", ["identity", "scalar_linear", "channel_linear", "dense_linear"] if axis == "temporal"
                          else ["diagonal", "low_rank", "dense"])
    allowed = {"identity", "scalar_linear", "channel_linear", "dense_linear"} if axis == "temporal" else {
        "diagonal", "low_rank", "low_rank_1_8", "low_rank_1_4", "low_rank_1_2", "dense"}
    if not variants or len(set(variants)) != len(variants) or set(variants) - allowed:
        raise ValueError(f"invalid architecture variants for {axis}: {variants}")
    # A production control through the same reference codec is always retained.
    control = "channel_linear" if axis == "temporal" else "dense"
    variants = list(variants)
    if control not in variants:
        variants.append(control)
    return axis, variants


@torch.no_grad()
def fit(config: dict, output_dir: Path):
    axis, variants = _settings(config)
    device = config.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    keys, values, source = load_trace(config["calibration_trace"], device)
    codec_config = load_config(config.get("codec_config", "configs/legacy.json"), config.get("codec_overrides"))
    if codec_config.mirrored or codec_config.v_predictor_mode != "cross_kv":
        raise ValueError("architecture studies require production k_to_v cross_kv")
    if config.get("history_length") is not None:
        codec_config = replace(codec_config, k_history_length=int(config["history_length"]))
    if codec_config.group_size > keys.shape[-1] or keys.shape[-1] % codec_config.group_size:
        raise ValueError("codec group_size must divide trace head dimension")
    if codec_config.cross_matmul_precision != "ieee" and torch.device(device).type == "cuda":
        # The comparison intentionally has one reference GEMM policy.
        codec_config = replace(codec_config, cross_matmul_precision="ieee")
    kwargs = {"config": codec_config, "iterations": int(config.get("fit_iterations", 2)),
              "temporal_ridge": float(config.get("temporal_ridge", 1.0 if codec_config.fit_mode == "legacy" else 0.0)),
              "cross_ridge": float(config.get("cross_ridge", 0.1)), "v_history_ridge": 0.0,
              "v_history_mode": "diagonal", "max_gain": float(config.get("max_gain", 1.0)),
              "fit_history_source": "reconstructed"}
    baseline_start = time.perf_counter()
    if config.get("k_params") or config.get("v_params"):
        if not config.get("k_params") or not config.get("v_params"):
            raise ValueError("k_params and v_params must be supplied together")
        production_k = TemporalParameters.load(config["k_params"]).to(device)
        production_v = CrossKVParameters.load(config["v_params"]).to(device)
        from tempokv.codec import check_parameters_match_config
        check_parameters_match_config(production_k, production_v, codec_config)
        source["initial_parameter_files"] = {k: {"path": str(Path(config[k]).resolve()), "sha256": digest(Path(config[k]))}
                                              for k in ("k_params", "v_params")}
    else:
        production_k, production_v = fit_predictors(keys, values, **kwargs)
    if not isinstance(production_v, CrossKVParameters):
        raise ValueError("architecture study requires dense CrossKV production parameters")
    layers, _, _, heads, dimension = keys.shape
    baseline_k = ResearchTemporal.from_production(production_k, layers, heads)
    baseline_v = ResearchCross.from_production(production_v, codec_config.cross_kv_weight_mode, layers, heads)
    reconstructed_k = reconstruct_keys(keys, baseline_k, codec_config)
    baseline_fit_seconds = time.perf_counter() - baseline_start
    rows, models = [], []
    for variant in variants:
        if axis == "cross" and variant.startswith("low_rank"):
            divisor = {"low_rank_1_8": 8, "low_rank_1_4": 4, "low_rank_1_2": 2}.get(variant)
            ranks = [max(1, dimension // divisor)] if divisor else config.get("ranks", [max(1, dimension // 8), max(1, dimension // 4), max(1, dimension // 2)])
            ranks = list(dict.fromkeys(int(r) for r in ranks))
        else:
            ranks = [None]
        for rank in ranks:
            start = time.perf_counter(); log = []
            temporal, cross = baseline_k, baseline_v
            name = variant if rank is None else f"low_rank_r{rank}"
            if axis == "temporal" and variant != "channel_linear":
                initial = identity_predictor(layers, heads, dimension, baseline_k.history_length, baseline_k.sharing,
                                             device=device, kind=variant)
                temporal, log = fit_temporal_architecture(keys, initial, codec_config,
                    rounds=int(config.get("fit_iterations", 2)), ridge=float(config.get("ridge", 1.0)),
                    max_gain=float(config.get("max_gain", 1.0)))
                changed_keys = reconstruct_keys(keys, temporal, codec_config)
                refit_v = refit_production_cross(changed_keys, values, production_v, codec_config)
                cross = ResearchCross.from_production(refit_v, codec_config.cross_kv_weight_mode, layers, heads)
            elif axis == "cross" and variant != "dense":
                cross, log = fit_cross_architecture(reconstructed_k, values, baseline_v, codec_config,
                    kind="low_rank" if rank is not None else variant, rank=rank,
                    rounds=int(config.get("fit_iterations", 2)), ridge=float(config.get("ridge", 0.1)),
                    als_steps=int(config.get("als_steps", 30)))
            metadata = {"axis": axis, "variant": name, "rank": rank, "calibration": source,
                        "history_source": "reconstructed", "fit_controls": {k: config[k] for k in (
                            "fit_iterations", "als_steps", "ridge", "temporal_ridge", "cross_ridge", "max_gain") if k in config},
                        "cross_history_policy": "production_refit_for_changed_k" if axis == "temporal" else "fixed_production_coefficients",
                        "optimization": log}
            codec = ExperimentalCodec(temporal, cross, codec_config, metadata)
            path = output_dir / "params" / axis / f"{name}.npz"
            codec.save(path)
            additional_fit_seconds = time.perf_counter() - start
            row = {"axis": axis, "variant": name, "rank": rank, "parameters": str(path.resolve()),
                   "parameter_bytes": codec.parameter_bytes(), "fit_seconds": baseline_fit_seconds + additional_fit_seconds,
                   "baseline_preparation_seconds": baseline_fit_seconds, "additional_fit_seconds": additional_fit_seconds,
                   "production_reference_control": variant == ("channel_linear" if axis == "temporal" else "dense"),
                   "decoder_backend": codec.backend, "temporal_sharing": temporal.sharing, "cross_sharing": cross.sharing,
                   "generation": {"codec_factory": "rebuttal.adapters.experimental_codec:load_codec_factory",
                                  "architecture_params": str(path.resolve()), "incremental": False,
                                  "decoder_backend": codec.backend}}
            rows.append(row); models.append((row, codec))
    report = {"schema": "tempokv.rebuttal.architecture.fit.v1", "axis": axis, "status": "complete", "calibration": source,
              "variants": rows, "fit_source_only": True}
    _write(output_dir / "fit.json", report)
    return report


def generation_plan(config: dict, output_dir: Path):
    """Build actual per-variant generation jobs without requiring model weights."""
    from rebuttal.adapters.generation import build_job
    axis, variants = _settings(config)
    fit_report_path = Path(config.get("fit_report", output_dir / "fit.json"))
    missing = []
    if fit_report_path.is_file():
        fit_report = json.loads(fit_report_path.read_text())
        if fit_report.get("axis") != axis:
            raise ValueError("generation fit report belongs to another architecture axis")
        rows = fit_report["variants"]
    else:
        rows = []
        missing.append(f"architecture fit report missing: {fit_report_path}; run stage=fit first")
        for key in ("checkpoint", "model_root", "prompts"):
            if not config.get(key):
                missing.append(f"generation requires config.{key}")
    jobs = []
    for row in rows:
        path = str(Path(row["parameters"]).resolve())
        cfg = {**config, "research_params": path, "torch_decoder": True,
               "decoder_backend": "pytorch_research_reference", "incremental": False}
        try:
            job = build_job(cfg, output_dir / "runs" / axis / row["variant"],
                            mode="tempokv", name=f"architecture_{axis}_{row['variant']}")
            jobs.append(job)
            missing.extend(f"missing input: {item}" for item in job.required_files if not Path(item).is_file())
            missing.extend(f"missing directory: {item}" for item in job.required_directories if not Path(item).is_dir())
        except ValueError as error:
            missing.append(f"{row['variant']}: {error}")
    result = {"schema": "tempokv.rebuttal.architecture.generation.v1", "axis": axis,
              "status": "needs_inputs" if missing else "planned", "fit_report": str(fit_report_path.resolve()),
              "variants": [x["variant"] for x in rows] if rows else variants,
              "production_reference_control": "channel_linear" if axis == "temporal" else "dense",
              "jobs": [job.record() for job in jobs], "missing": list(dict.fromkeys(missing)),
              "decoder_backend": "pytorch_research_reference", "incremental": False}
    _write(output_dir / "generation_plan.json", result)
    return result, jobs


@torch.no_grad()
def replay_codec(codec: ExperimentalCodec, keys: torch.Tensor, values: torch.Tensor):
    cfg = replace(codec.config, stride=1)
    totals = {role: {"prediction_sse": 0.0, "quantization_sse": 0.0, "reconstruction_sse": 0.0,
                     "target_energy": 0.0, "count": 0, "all_reconstruction_sse": 0.0,
                     "all_target_energy": 0.0, "all_count": 0} for role in ("k", "v")}
    events = []
    storage_bytes = 0
    for layer in range(keys.shape[0]):
        kinput, vinput = [x[layer].permute(0, 2, 1, 3) for x in (keys, values)]
        kp, kr, krecords = encode_tensor(kinput, codec.temporal, cfg, bits=cfg.k_bits, layer=layer, diagnostics=True)
        vp, vr, vrecords = encode_tensor(vinput, codec.v_parameters, cfg, bits=cfg.v_bits, layer=layer, source=kr, diagnostics=True)
        # Verify packed-state decode, not merely an encoder-side approximation.
        if not torch.equal(kr, kp.decode()) or not torch.equal(vr, vp.decode(source=kr)):
            raise RuntimeError("experimental encoder/decoder mismatch")
        storage_bytes += kp.storage_bytes() + vp.storage_bytes()
        for role, original, rebuilt, records in (("k", kinput, kr, krecords), ("v", vinput, vr, vrecords)):
            summary = totals[role]
            for record in records:
                for key in ("prediction_sse", "quantization_sse", "reconstruction_sse", "target_energy", "count"):
                    summary[key] += record[key]
                events.append({"layer": layer, "role": role, "latent_frame": record["offset"],
                               "metric_scope": "bf16_trace_replay", "sample_coverage": "all_normalized_trace_streams",
                               **record})
            summary["all_reconstruction_sse"] += float((original.float() - rebuilt.float()).double().square().sum())
            summary["all_target_energy"] += float(original.double().square().sum())
            summary["all_count"] += original.numel()
    for summary in totals.values():
        denominator = summary["target_energy"]
        for key in ("prediction", "quantization", "reconstruction"):
            summary[f"{key}_nmse"] = summary[f"{key}_sse"] / denominator if denominator else None
        summary["all_reconstruction_nmse"] = summary["all_reconstruction_sse"] / summary["all_target_energy"] if summary["all_target_energy"] else None
    return {"k": totals["k"], "v": totals["v"], "storage_bytes": storage_bytes,
            "parameter_bytes": codec.parameter_bytes(), "frame_count": keys.shape[-3],
            "normalized_stream_count": keys.shape[1]}, events


def run(config: dict, stage: str, output_dir: Path) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    axis, variants = _settings(config)
    if stage == "plan":
        report = {"schema": "tempokv.rebuttal.architecture.plan.v1", "axis": axis, "variants": variants,
                  "ranks": config.get("ranks", "D/8,D/4,D/2"), "stages": ["plan", "fit", "replay", "generate", "evaluate", "summarize"],
                  "required_fit_inputs": ["calibration_trace", "codec_config"],
                  "required_evaluation_inputs": ["evaluation_trace", "architecture_parameters_or_fit_report"],
                  "generation_interface": "ExperimentalCodec.load(path); ResearchPackedTensor.decode(source=...)",
                  "generation_backend": "pytorch_research_reference", "generation_incremental": False}
        report["generation"], _ = generation_plan(config, output_dir)
        _write(output_dir / "plan.json", report); return report
    if stage == "fit":
        return fit(config, output_dir)
    if stage == "generate":
        from rebuttal.adapters.generation import execute_jobs
        plan, jobs = generation_plan(config, output_dir)
        if plan["missing"]:
            raise ValueError("; ".join(plan["missing"]))
        return execute_jobs(jobs, resume=bool(config.get("_resume", False)))
    if stage == "summarize":
        path = output_dir / "reconstruction.json"
        if not path.is_file():
            path = output_dir / "fit.json"
        if not path.is_file():
            raise ValueError("no architecture fit/evaluation output exists")
        return json.loads(path.read_text())
    if stage not in {"replay", "evaluate"}:
        raise ValueError(f"architecture stage {stage!r} is unsupported; generation uses the common model adapter")
    source = config.get("evaluation_trace", config.get("trace"))
    if not source:
        raise ValueError("evaluation_trace is required for architecture replay")
    keys, values, evaluation = load_trace(source, config.get("device", "cpu"))
    parameters = config.get("architecture_parameters", config.get("parameters"))
    if parameters:
        if isinstance(parameters, str):
            parameters = [parameters]
        rows = [{"parameters": path} for path in parameters]
    else:
        fit_report_path = Path(config.get("fit_report", output_dir / "fit.json"))
        if not fit_report_path.is_file():
            if config.get("fit_report"):
                raise FileNotFoundError(fit_report_path)
            fit(config, output_dir)
        rows = json.loads(fit_report_path.read_text())["variants"]
    results, events = [], []
    for row in rows:
        codec = ExperimentalCodec.load(row["parameters"], device=keys.device)
        if codec.metadata.get("axis") != axis:
            raise ValueError("loaded architecture parameters belong to another ablation axis")
        split = validate_split(codec.metadata["calibration"], evaluation,
                               allow_in_sample=bool(config.get("allow_in_sample", False)),
                               require_prompt_identity=bool(config.get("require_prompt_identity", False)))
        if (codec.temporal.layers, codec.temporal.heads, codec.temporal.bias.shape[-1]) != (keys.shape[0], keys.shape[-2], keys.shape[-1]):
            raise ValueError("evaluation model shape differs from fitted model")
        start = time.perf_counter(); metrics, records = replay_codec(codec, keys, values)
        identity = {"variant": codec.metadata["variant"], "axis": axis}
        results.append({**row, **identity, **metrics, "split_validation": split,
                        "replay_seconds": time.perf_counter() - start, "metric_scope": "bf16_trace_replay"})
        events.extend({**identity, **x} for x in records)
    report = {"schema": "tempokv.rebuttal.architecture.reconstruction.v1", "status": "complete", "axis": axis,
              "evaluation": evaluation, "variants": results}
    _write(output_dir / "reconstruction.json", report)
    with (output_dir / "cache_events.jsonl").open("w") as handle:
        for event in events:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
    from .report import write_table
    write_table(output_dir / "summary.csv", results)
    return report
