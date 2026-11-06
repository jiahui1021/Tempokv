"""Measure the prediction objective against exact residual quantization loss."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rebuttal.common.manifest import read_json, write_json


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    from rebuttal.experiments.previous_frame import run as predictors
    variants = config.get("variants", ["direct", "identity", "production"])
    bits = config.get("bits", [2, 4])
    if not bits or any(isinstance(b, bool) or int(b) != b or not 1 <= b <= 8 for b in bits):
        raise ValueError("bits must be a nonempty list of integers in [1,8]")
    if config.get("k_params") or config.get("v_params"):
        if not config.get("k_params") or not config.get("v_params"):
            raise ValueError("supply both k_params and v_params")
        if len(bits) != 1 or variants != ["production"]:
            raise ValueError("pre-fitted parameters require variants=['production'] and one matching bit width")
        if stage in {"evaluate", "replay"}:
            from tempokv.config import load_config
            from rebuttal.common.trace import load_trace, device_for
            from rebuttal.common.cache_metrics import load_pair, replay
            codec = load_config(config["codec_config"], config.get("codec_overrides"))
            if (codec.k_bits, codec.v_bits) != (bits[0], bits[0]):
                raise ValueError("pre-fitted parameter bit width does not match requested diagnostic bits")
            device = device_for(config.get("device", "auto"))
            trace = load_trace(config.get("evaluation_trace") or config["trace"], device=device,
                               identity_path=config.get("evaluation_identity") or config.get("trace_manifest"),
                               max_records=config.get("max_records"))
            k, v = load_pair(config["k_params"], config["v_params"], codec, device)
            report = replay(trace, codec, k, v, span_frames=config.get("span_frames"),
                            event_path=output_dir / "cache_events.jsonl", model=config.get("model", "unknown"))
            write_json(output_dir / "summary.json", report)
            return report
    reports = {}
    inner_stage = "replay" if stage == "evaluate" else stage
    for bit in bits:
        item = {**config, "variants": variants,
                "codec_overrides": {**config.get("codec_overrides", {}), "k_bits": bit, "v_bits": bit}}
        reports[f"int{bit}"] = predictors(item, inner_stage, output_dir / f"int{bit}")
    result = {"status": "planned" if stage == "plan" else "complete", "experiment": "quant_calibration",
              "objective": "R=X-P; L_quant=||R-D(Q(R))||^2; actual storage loss measured separately",
              "by_bits": reports, "optional_quant_aware_fit_executed": False}
    write_json(output_dir / f"{stage}_summary.json", result)
    if inner_stage == "replay":
        _plot(reports, output_dir)
    return result


def _plot(reports: dict[str, Any], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for bit, report in reports.items():
        for variant, item in report.get("variants", {}).items():
            for role in ("k", "v"):
                block = item[role]["non_anchor"]
                rows.append({"bits": bit, "variant": variant, "role": role,
                             "prediction_mse": block["prediction_mse"],
                             "residual_quant_mse": block["residual_quant_mse"],
                             "actual_reconstruction_nmse": block["actual_reconstruction_nmse"]})
    write_json(output_dir / "objective_points.json", rows)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for axis, role in zip(axes, ("k", "v")):
        for row in rows:
            if row["role"] == role and row["prediction_mse"] is not None:
                axis.scatter(row["prediction_mse"], row["residual_quant_mse"])
                axis.annotate(f"{row['bits']} {row['variant']}", (row["prediction_mse"], row["residual_quant_mse"]), fontsize=7)
        axis.set(xlabel="Prediction MSE", ylabel="Residual quantization MSE", title=role.upper())
    fig.savefig(output_dir / "prediction_vs_quantization.png", dpi=180)
    plt.close(fig)
