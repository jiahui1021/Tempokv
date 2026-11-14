"""Experiment 7: AQUA-KV cross-layer prediction on video K/V traces vs TempoKV.

Stages: plan | fit (calibration only) | evaluate (frozen predictor on heldout) .
See README.md in this directory for formulas and the adaptation caveats.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rebuttal.common.manifest import write_json


def quantizer_from(config: dict[str, Any]):
    from .model import Quantizer
    q = dict(config.get("quantizer") or {})
    if config.get("codec_config"):
        from tempokv.config import load_config
        codec = load_config(config["codec_config"], config.get("codec_overrides"))
        if codec.k_bits != codec.v_bits and "bits" not in q:
            raise ValueError("codec k_bits != v_bits; set quantizer.bits explicitly")
        base = {"bits": codec.k_bits, "group_size": codec.group_size, "mode": codec.residual_mode,
                "scale_dtype": codec.scale_dtype, "reconstruction_dtype": codec.reconstruction_dtype}
        mismatch = {k: (q[k], base[k]) for k in q if k in base and q[k] != base[k]}
        if mismatch and config.get("tempokv"):
            raise ValueError(f"quantizer differs from TempoKV codec, comparison not controlled: {mismatch}")
        q = {**base, **q}
    if "bits" not in q:
        raise ValueError("need codec_config or quantizer.bits")
    return Quantizer(**q)


def _load(config, key, device):
    from rebuttal.common.trace import load_trace
    return load_trace(config[key], device=device, identity_path=config.get(key.replace("trace", "identity")),
                      max_records=config.get("max_records"))


def _check_layers(trace):
    ids = trace.layer_ids
    if any(b - a != 1 for a, b in zip(ids, ids[1:])):
        raise ValueError(f"AQUA-KV needs consecutive transformer layers; trace has layer_ids={list(ids)}")


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if stage == "plan":
        plan = {"status": "planned", "experiment": "aquakv_lite",
                "method": "AQUA-KV (arXiv 2501.19392) cross-layer ridge predictor + residual quantization, video adaptation",
                "stages": ["fit", "evaluate"],
                "missing": [k for k in ("calibration_trace", "evaluation_trace", "codec_config") if not config.get(k)],
                "comparison": ["direct_quantization", "aquakv", "tempokv (needs tempokv.k_params/v_params)"]}
        write_json(output_dir / "plan.json", plan)
        return plan
    import torch
    from rebuttal.common.trace import device_for
    from .model import AquaKVPredictor, encode, direct
    from .report import summarize, storage, tempokv_summary
    device = device_for(config.get("device", "auto"))
    quant = quantizer_from(config)
    pdtype = config.get("parameter_dtype", "bfloat16")
    params_path = Path(config.get("predictor_path") or output_dir / "aquakv_predictor.pt")
    if stage == "fit":
        cal = _load(config, "calibration_trace", device)
        _check_layers(cal)
        predictor, rows, _, _ = encode(cal.keys, cal.values, quant, fit=True,
                                       ridge_value=float(config.get("ridge", 1e-2)),
                                       bias=bool(config.get("bias", True)),
                                       fit_token_stride=int(config.get("fit_token_stride", 1)))
        params_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**predictor.state_dict(), "layer_ids": list(cal.layer_ids),
                    "quantizer": quant.__dict__, "calibration_trace": str(cal.path)}, params_path)
        result = {"status": "complete", "split": "calibration_in_sample", "trace": str(cal.path),
                  "shape": list(cal.shape), "predictor_path": str(params_path),
                  "parameter_count": predictor.parameter_count(),
                  "parameter_bytes": predictor.parameter_bytes(pdtype), "parameter_dtype": pdtype,
                  "quantizer": quant.__dict__, "aquakv": summarize(rows, cal.layer_ids)}
        write_json(output_dir / "fit.json", result)
        return result
    if stage != "evaluate":
        raise ValueError("aquakv_lite supports plan, fit, evaluate")
    ev = _load(config, "evaluation_trace", device)
    _check_layers(ev)
    state = torch.load(params_path, map_location=device)
    if state.get("calibration_trace") == str(ev.path) and not config.get("allow_in_sample"):
        raise ValueError("evaluation trace equals calibration trace; set allow_in_sample for diagnostics only")
    if state.get("quantizer") != quant.__dict__:
        raise ValueError("predictor was fitted with a different quantizer")
    if list(state.get("layer_ids", [])) != list(ev.layer_ids):
        raise ValueError("predictor layer_ids do not match evaluation trace")
    if config.get("calibration_trace"):
        from rebuttal.common.trace import check_trace_split
        check_trace_split(_load(config, "calibration_trace", "cpu"), ev,
                          allow_in_sample=bool(config.get("allow_in_sample")))
    predictor = AquaKVPredictor.from_state_dict(state)
    _, rows, _, _ = encode(ev.keys, ev.values, quant, predictor=predictor)
    numel = ev.keys.numel()
    result = {"status": "complete", "split": "heldout", "trace": str(ev.path), "shape": list(ev.shape),
              "metric_scope": "trace_replay_nmse_no_video_quality", "quantizer": quant.__dict__,
              "aquakv": {**summarize(rows, ev.layer_ids),
                         "storage": storage(numel, quant, predictor.parameter_bytes(pdtype))},
              "direct": {**summarize(direct(ev.keys, ev.values, quant), ev.layer_ids),
                         "storage": storage(numel, quant)}}
    if config.get("teacher_forcing_diagnostic"):
        _, tf, _, _ = encode(ev.keys, ev.values, quant, predictor=predictor, teacher_forcing=True)
        result["aquakv_teacher_forcing_diagnostic"] = summarize(tf, ev.layer_ids)["aggregate"]
    tk = config.get("tempokv") or {}
    if tk.get("k_params") and tk.get("v_params"):
        from tempokv.config import load_config
        from rebuttal.common.cache_metrics import load_pair, replay
        codec = load_config(config["codec_config"], config.get("codec_overrides"))
        k, v = load_pair(tk["k_params"], tk["v_params"], codec, device)
        rep = replay(ev, codec, k, v, span_frames=tk.get("span_frames"), model=config.get("model", "unknown"))
        write_json(output_dir / "tempokv_replay.json", rep)
        sizes = sum(Path(p).stat().st_size for p in (tk["k_params"], tk["v_params"]))
        result["tempokv"] = {**tempokv_summary(rep),
                             "storage_note": "TempoKV payload follows its codec (anchors at anchor_bits); "
                                             "parameter file bytes reported as stored on disk",
                             "parameter_file_bytes": sizes}
    else:
        result["tempokv"] = {"status": "not_run", "reason": "tempokv.k_params / tempokv.v_params not given"}
    write_json(output_dir / "evaluate.json", result)
    return result
