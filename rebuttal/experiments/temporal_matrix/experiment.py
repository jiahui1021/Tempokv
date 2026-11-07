"""Matched real-trace temporal-only fitting and quantized reconstruction."""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
import time

import torch

from rebuttal.common.manifest import file_sha256, read_json, write_json
from rebuttal.experiments.block_dplr.experiment import load_trace
from .model import fit, replay, parameter_bytes, model_state, load_model


def codec_from(config):
    from tempokv.config import CodecConfig
    codec = CodecConfig(**config["codec"])
    if codec.fit_mode != "native" or codec.v_predictor_mode != "temporal":
        raise ValueError("temporal-only comparison requires native arithmetic and no CrossKV")
    return codec


def finish_metrics(rows):
    sums = {key: sum(row[key] for row in rows) for key in
            ("signal_sse", "prediction_sse", "quantization_sse", "error_sse", "numel")}
    denominator = sums["signal_sse"]
    return {**sums, "prediction_nmse": sums["prediction_sse"]/denominator,
            "residual_quantization_nmse": sums["quantization_sse"]/denominator,
            "reconstruction_nmse": sums["error_sse"]/denominator}


def fit_all(config, output):
    destination = output/"temporal_predictors.pt"
    if destination.exists():
        if config.get("_resume") and (output/"fit.json").is_file():
            report = read_json(output/"fit.json")
            if report["predictor_sha256"] != file_sha256(destination):
                raise ValueError("temporal predictors changed after fitting")
            return report
        raise FileExistsError("temporal predictors already exist; use a new output directory")
    keys, values, source = load_trace(config)
    codec = codec_from(config)
    ids = source["calibration_record_ids"]
    controls = {"rounds": int(config.get("fit_rounds", 2)), "ridge": float(config.get("ridge", 1.0)),
                "max_gain": float(config.get("max_gain", 1.0))}
    models = {f"{role}_{kind}": [] for role in ("k", "v") for kind in ("per_channel", "dense")}
    diagnostics = {name: [] for name in models}
    started = time.perf_counter()
    for layer, layer_id in enumerate(source["layer_ids"]):
        layer_started = time.perf_counter()
        for role, tensor, history, bits in (("k", keys, codec.k_history_length, codec.k_bits),
                                            ("v", values, codec.v_history_length, codec.v_bits)):
            calibration = tensor[layer, ids]
            for name, kind in (("per_channel", "channel_linear"), ("dense", "dense_linear")):
                model, logs = fit(calibration, codec, kind=kind, history=history, bits=bits, **controls)
                key = f"{role}_{name}"
                models[key].append(model_state(model))
                diagnostics[key].append({"layer": layer_id, "parameter_bytes": parameter_bytes(model), "optimization": logs})
        print(f"Temporal layer {layer_id} done ({time.perf_counter()-layer_started:.2f}s), elapsed {time.perf_counter()-started:.1f}s", flush=True)
    state = {"schema": "tempokv.temporal_channel_dense.v1", "source": source, "codec": asdict(codec),
             "controls": controls, "models": models, "cross_kv_enabled": False,
             "sharing": "per layer/head; dense only mixes channels within one head"}
    torch.save(state, destination)
    report = {"status": "complete", "source": source, "codec": asdict(codec), "controls": controls,
              "predictor_path": str(destination), "predictor_sha256": file_sha256(destination),
              "fit_seconds": time.perf_counter()-started, "cross_kv_enabled": False,
              "diagnostics": diagnostics}
    write_json(output/"fit.json", report)
    return report


def evaluate(config, output):
    fit_report = read_json(output/"fit.json")
    path = output/"temporal_predictors.pt"
    if file_sha256(path) != fit_report["predictor_sha256"]:
        raise ValueError("temporal predictor artifact changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != "tempokv.temporal_channel_dense.v1" or state["cross_kv_enabled"]:
        raise ValueError("expected pure temporal predictor artifact")
    keys, values, source = load_trace(config)
    codec = codec_from(config)
    if source != state["source"] or asdict(codec) != state["codec"]:
        raise ValueError("temporal evaluation source/codec differs from fitted declaration")
    ids, results = source["evaluation_record_ids"], []
    for role, tensor, history, bits in (("k", keys, codec.k_history_length, codec.k_bits),
                                       ("v", values, codec.v_history_length, codec.v_bits)):
        for variant in ("per_channel", "dense"):
            name = f"{role}_{variant}"
            per_layer, bytes_total, per_frame = [], 0, []
            for layer, layer_id in enumerate(source["layer_ids"]):
                model = load_model(state["models"][name][layer], tensor.device)
                _, frames = replay(tensor[layer, ids], model, codec, bits=bits)
                per_layer.append({"layer": layer_id, **finish_metrics(frames)})
                per_frame.extend({"layer": layer_id, **row} for row in frames)
                bytes_total += parameter_bytes(model)
            result = {"name": name, "role": role, "variant": variant, "history_length": history,
                      "parameter_bytes": bytes_total, "parameter_mib": bytes_total/2**20,
                      **finish_metrics(per_layer), "per_layer": per_layer, "per_frame": per_frame}
            results.append(result)
            print(f"{name}: reconstruction NMSE={result['reconstruction_nmse']:.9f}, prediction NMSE={result['prediction_nmse']:.6f}, temporal={result['parameter_mib']:.6f} MiB", flush=True)
    for role in ("k", "v"):
        baseline = next(row for row in results if row["name"] == f"{role}_per_channel")
        for row in results:
            if row["role"] == role:
                row["relative_reconstruction_change"] = row["reconstruction_nmse"]/baseline["reconstruction_nmse"]-1
                row["relative_prediction_change"] = row["prediction_nmse"]/baseline["prediction_nmse"]-1
                row["parameter_ratio"] = row["parameter_bytes"]/baseline["parameter_bytes"]
    report = {"status": "complete", "experiment": "per_channel_vs_dense_temporal", "model": "Self-Forcing",
              "source": source, "codec": asdict(codec), "cross_kv_enabled": False,
              "sharing": state["sharing"], "fit": {key: fit_report[key] for key in ("controls", "fit_seconds", "predictor_sha256")},
              "metric": "pooled SSE(reconstructed-target)/SSE(target), frame>=1; own quantized causal temporal history",
              "parameter_scope": "FP32 temporal weights including warm-up row allocation plus per-channel bias; no CrossKV weights",
              "results": results}
    write_json(output/"reconstruction.json", report)
    fields = ["name", "role", "history_length", "parameter_mib", "prediction_nmse", "reconstruction_nmse",
              "relative_reconstruction_change", "relative_prediction_change", "parameter_ratio"]
    with (output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    write_summary(report, output)
    return report


def write_summary(report, output):
    source = report["source"]
    lines = ["# Per-channel temporal vs Dense temporal：真实 Self-Forcing 重建", "",
             "只比较 temporal；K 和 V 分别使用自己的历史，所有行 CrossKV 均关闭。Dense 只在同一 head 内混合通道，不混合 heads/layers。", "",
             f"数据：`{source['path']}`，形状 `{source['shape']}`，SHA256 `{source['sha256']}`。",
             f"校准 {len(source['calibration_record_ids'])} 条、测试 {len(source['evaluation_record_ids'])} 条记录；原始文件缺少 prompt 身份，未核实 prompt 级独立。", "",
             "K 使用 KH2，V 使用 VH1；同为 4-bit residual、4-bit anchor、group64、BF16 重建。两类参数均按 layer/head 独立拟合。", "",
             f"拟合控制：`{report['fit']['controls']}`。共用 head-level ridge 强度、persistence prior、逐输出 incoming L1 gain cap、warm-up 和重建历史；没有修改 CrossKV 矩阵或重新拟合 CrossKV。", "",
             "| 对象 | temporal | 历史帧数 | 参数 MiB | 预测 NMSE ↓ | 重建 NMSE ↓ | 相对 per-channel 重建变化 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["results"]:
        lines.append(f"| {row['role'].upper()} | {row['variant']} | {row['history_length']} | {row['parameter_mib']:.6f} | {row['prediction_nmse']:.9f} | {row['reconstruction_nmse']:.9f} | {row['relative_reconstruction_change']*100:+.4f}% |")
    lines += ["", "首帧 anchor 不计入主指标；总体按 SSE 总和之比汇总，没有平均逐层 NMSE。", "",
              "参数大小按实际 FP32 weights+bias 计算，包括已分配的 warm-up 系数槽，不包含量化 payload。只有缓存重建测试，没有视频质量或生成速度结果。", ""]
    (output/"summary.md").write_text("\n".join(lines))


def run(config, stage, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("threads", 4)))
    if stage == "plan":
        codec = codec_from(config)
        result = {"status": "planned", "experiment": "temporal_matrix", "cross_kv_enabled": False,
                  "roles": {"k": codec.k_history_length, "v": codec.v_history_length},
                  "variants": ["per_channel", "dense"], "missing": [] if config.get("trace") else ["trace"]}
        write_json(output/"plan.json", result); return result
    if stage == "fit": return fit_all(config, output)
    if stage in {"pipeline", "evaluate"}:
        if not (output/"fit.json").is_file(): fit_all(config, output)
        return evaluate(config, output)
    if stage == "replay": return evaluate(config, output)
    if stage == "summarize":
        report = read_json(output/"reconstruction.json"); write_summary(report, output); return report
    raise ValueError("temporal_matrix supports plan/fit/replay/evaluate/pipeline/summarize")
