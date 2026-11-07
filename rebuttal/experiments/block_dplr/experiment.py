"""Real-trace matched CrossKV study with frozen K, V history and intercept."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from rebuttal.common.manifest import file_sha256, read_json, write_json
from .model import BlockDPLR, RegressionStatistics, replay_layer

PAPER = "https://arxiv.org/html/2610.12514v1#S4.SS2"


def load_trace(config):
    from rebuttal.common.trace import device_for
    path = Path(config["trace"])
    device = device_for(config.get("device", "cpu"))
    with np.load(path, allow_pickle=False) as data:
        coordinate = str(data["coordinate_system"].item()) if "coordinate_system" in data else None
        if coordinate != "pre_rope_normalized":
            raise ValueError("block-DPLR study requires normalized pre-RoPE K")
        keys, values = torch.from_numpy(data["keys"]), torch.from_numpy(data["values"])
        if keys.shape != values.shape or keys.ndim != 5 or not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError("trace must contain finite matching [L,records,frames,H,D]")
        layer_ids = data["layer_ids"].tolist() if "layer_ids" in data else list(range(keys.shape[0]))
    ids = config.get("layer_ids") or layer_ids
    positions = [layer_ids.index(layer) for layer in ids]
    keys, values = keys[positions].to(device), values[positions].to(device)
    step = int(config.get("holdout_step", 5))
    if step < 3 or keys.shape[1] < step or keys.shape[2] < 2:
        raise ValueError("insufficient trace records/frames for heldout reconstruction")
    evaluation = list(range(step-1, keys.shape[1], step))
    selected = set(evaluation)
    calibration = [i for i in range(keys.shape[1]) if i not in selected]
    source = {"path": str(path.resolve()), "sha256": file_sha256(path), "shape": list(keys.shape),
              "layer_ids": ids, "coordinate_system": coordinate, "calibration_record_ids": calibration,
              "evaluation_record_ids": evaluation, "prompt_disjointness": "unknown_no_prompt_identity_in_archive"}
    old = config.get("previous_report")
    if old and Path(old).is_file():
        previous = read_json(old)
        source["previous_trace_sha256"] = previous["trace"]["sha256"]
        source["same_trace_as_previous_report"] = source["sha256"] == source["previous_trace_sha256"]
    if config.get("expected_trace_sha256") and source["sha256"] != config["expected_trace_sha256"]:
        raise ValueError("trace differs from expected SHA256; refusing to relabel experiments")
    return keys, values, source


def codec_from(config):
    from tempokv.config import CodecConfig
    codec = CodecConfig(**config["codec"])
    if codec.fit_mode != "native" or codec.cross_kv_weight_mode != "layer_head" or codec.v_predictor_mode != "cross_kv":
        raise ValueError("this study matches native per-layer/head CrossKV only")
    return codec


def _metrics(rows):
    total = {key: sum(row[key] for row in rows) for key in ("signal_sse", "error_sse", "prediction_sse", "numel")}
    return {**total, "v_reconstruction_nmse": total["error_sse"]/total["signal_sse"],
            "prediction_nmse": total["prediction_sse"]/total["signal_sse"]}


def fit(config, output):
    from tempokv.fitting import fit_predictors, _reconstruct_temporal_trace, _reconstruct_cross_kv_trace
    path = output/"block_dplr_predictors.pt"
    if path.exists():
        if config.get("_resume") and (output/"fit.json").is_file():
            result = read_json(output/"fit.json")
            if result["predictor_sha256"] != file_sha256(path):
                raise ValueError("DPLR predictors changed since fitting")
            return result
        raise FileExistsError("DPLR output already exists; use a new directory or --resume")
    keys, values, source = load_trace(config)
    codec = codec_from(config)
    ids = source["calibration_record_ids"]
    cal_k, cal_v = keys[:, ids], values[:, ids]
    del keys, values
    started = time.perf_counter()
    print(f"Fitting fresh dense/K/history baseline on {len(ids)} calibration records", flush=True)
    kp, vp = fit_predictors(cal_k, cal_v, config=codec, iterations=int(config.get("baseline_iterations", 2)),
                            temporal_ridge=float(config.get("temporal_ridge", 0)),
                            cross_ridge=float(config.get("baseline_cross_ridge", 0)),
                            v_history_ridge=float(config.get("v_history_ridge", 0)),
                            v_history_mode="diagonal", layer_ids=tuple(source["layer_ids"]),
                            coordinate_system="pre_rope_normalized")
    kp.save(output/"k_temporal.npz")
    vp.save(output/"v_cross_kv.npz")
    reconstructed_k = _reconstruct_temporal_trace(cal_k, kp, codec)
    reconstructed_v = _reconstruct_cross_kv_trace(reconstructed_k, cal_v, vp, codec)
    del cal_k
    channels = cal_v.shape[-2]*cal_v.shape[-1]
    ranks = [int(rank) for rank in config.get("ranks", [8, 32, 128])]
    if len(set(ranks)) != len(ranks) or any(not 0 < rank <= channels for rank in ranks):
        raise ValueError("DPLR ranks must be unique positive global ranks <= H*D")
    rounds, ridge = int(config.get("rounds", 3)), float(config.get("ridge", 1e-3))
    variants = {"dense_source": [], "block_diagonal_matched_ridge": []}
    variants.update({f"block_dplr_r{rank}": [] for rank in ranks})
    diagnostics = {key: [] for key in variants}
    for layer in range(cal_v.shape[0]):
        layer_started = time.perf_counter()
        target = cal_v[layer, :, 1:].float().clone()
        target.sub_(vp.bias[layer])
        for lag in range(1, vp.history_length+1):
            target[:, lag-1:].sub_(reconstructed_v[layer, :, :cal_v.shape[2]-lag].float()*vp.history_weights[layer, lag-1])
        x = reconstructed_k[layer, :, 1:].reshape(-1, *cal_v.shape[-2:]).float()
        y = target.reshape_as(x)
        stats = RegressionStatistics(x, y, ridge=ridge)
        empty_a = torch.empty(channels, 0, device=x.device)
        empty_b = torch.empty(0, channels, device=x.device)
        variants["dense_source"].append(BlockDPLR(vp.weight[layer], empty_a, empty_b).state())
        for rank in [0, *ranks]:
            model, logs = stats.fit(rank, rounds=rounds)
            name = "block_diagonal_matched_ridge" if rank == 0 else f"block_dplr_r{rank}"
            variants[name].append(model.state())
            diagnostics[name].append({"layer": source["layer_ids"][layer], "optimization": logs,
                                      "matrix_bytes": model.storage_bytes()})
        del stats, target, x, y
        print(f"Layer {source['layer_ids'][layer]} done ({time.perf_counter()-layer_started:.2f}s), elapsed {time.perf_counter()-started:.1f}s", flush=True)
    state = {"schema": "tempokv.block_dplr.same_layer_matrix.v1", "source": source,
             "codec": asdict(codec), "variants": variants, "ridge": ridge, "rounds": rounds,
             "paper": PAPER, "adaptation": "same_layer_K_to_V_matrix_only_with_frozen_baseline_V_history_and_intercept",
             "calibration_history": "decoder-visible reconstruction from fresh calibration-only dense baseline"}
    torch.save(state, path)
    result = {"status": "complete", "source": source, "codec": asdict(codec), "ridge": ridge, "rounds": rounds,
              "paper": PAPER, "adaptation": state["adaptation"], "calibration_history": state["calibration_history"],
              "predictor_path": str(path), "predictor_sha256": file_sha256(path), "fit_seconds": time.perf_counter()-started,
              "diagnostics": diagnostics,
              "baseline_parameters": [{"path": str(p), "sha256": file_sha256(p)} for p in (output/"k_temporal.npz", output/"v_cross_kv.npz")]}
    write_json(output/"fit.json", result)
    return result


def evaluate(config, output):
    from tempokv.parameters import TemporalParameters, CrossKVParameters
    from tempokv.fitting import _reconstruct_temporal_trace, _reconstruct_cross_kv_trace
    fit_report = read_json(output/"fit.json")
    if file_sha256(output/"block_dplr_predictors.pt") != fit_report["predictor_sha256"]:
        raise ValueError("DPLR predictor artifact changed")
    for artifact in fit_report["baseline_parameters"]:
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("frozen K/history baseline changed")
    state = torch.load(output/"block_dplr_predictors.pt", map_location=config.get("device", "cpu"), weights_only=True)
    keys, values, source = load_trace(config)
    if state["source"] != source:
        raise ValueError("DPLR evaluation source/split differs from calibration declaration")
    codec = codec_from(config)
    if state["codec"] != asdict(codec):
        raise ValueError("DPLR evaluation codec differs from calibration")
    ids = source["evaluation_record_ids"]
    keys, values = keys[:, ids], values[:, ids]
    kp = TemporalParameters.load(output/"k_temporal.npz").to(keys.device)
    vp = CrossKVParameters.load(output/"v_cross_kv.npz").to(keys.device)
    reconstructed_k = _reconstruct_temporal_trace(keys, kp, codec)
    results = []
    block_nmse = source_nmse = None
    for name, layers in state["variants"].items():
        started = time.perf_counter()
        layer_rows, matrix_bytes = [], 0
        for layer, model_state in enumerate(layers):
            model = BlockDPLR.from_state(model_state, keys.device)
            metrics = replay_layer(reconstructed_k[layer], values[layer], model, vp.bias[layer], vp.history_weights[layer], codec)
            layer_rows.append({"layer": source["layer_ids"][layer], **metrics,
                               "v_reconstruction_nmse": metrics["error_sse"]/metrics["signal_sse"]})
            matrix_bytes += model.storage_bytes()
        metrics = _metrics(layer_rows)
        if name == "dense_source":
            # Independent reference path checks the custom replay arithmetic.
            original = _reconstruct_cross_kv_trace(reconstructed_k, values, vp, codec)
            expected = (values[:, :, 1:].double()-original[:, :, 1:].double()).square().sum().item()
            if not np.isclose(metrics["error_sse"], expected, rtol=1e-12):
                raise RuntimeError("DPLR replay arithmetic does not reproduce the production dense baseline")
            source_nmse = metrics["v_reconstruction_nmse"]
        if name == "block_diagonal_matched_ridge":
            block_nmse = metrics["v_reconstruction_nmse"]
        result = {"name": name, "global_rank": int(layers[0]["a"].shape[1]), "matrix_bytes": matrix_bytes,
                  "matrix_mib": matrix_bytes/2**20, **metrics, "per_layer": layer_rows,
                  "replay_seconds": time.perf_counter()-started}
        results.append(result)
        print(f"{name}: V NMSE={result['v_reconstruction_nmse']:.9f}, matrix={result['matrix_mib']:.4f} MiB", flush=True)
    for result in results:
        result["relative_nmse_change_vs_dense_source"] = result["v_reconstruction_nmse"]/source_nmse-1
        result["relative_nmse_change_vs_matched_block"] = result["v_reconstruction_nmse"]/block_nmse-1
    report = {"status": "complete", "experiment": "coral_style_block_dplr_crosskv_reconstruction",
              "model": "Self-Forcing", "source": source, "codec": asdict(codec), "paper": PAPER,
              "scope": {"prediction_input": "same-layer reconstructed K", "changed": "K_to_V_matrix",
                        "fixed": ["K temporal predictor", "V history coefficients", "V intercept", "residual quantizer"],
                        "full_CORAL_cross_layer_codec": False, "video_generation": False,
                        "stored_factor_dtype": "float32", "full_head_blocks": True,
                        "parameter_tradeoff": "block-DPLR adds cross-head factors to the existing per-head dense map"},
              "metric": "pooled SSE(Vhat-V)/SSE(V), frame>=1, quantized closed-loop V history",
              "fit": {key: fit_report[key] for key in ("ridge", "rounds", "fit_seconds", "calibration_history", "predictor_sha256")},
              "results": results}
    write_json(output/"reconstruction.json", report)
    with (output/"summary.csv").open("w", newline="") as stream:
        fields = ["name", "global_rank", "matrix_mib", "v_reconstruction_nmse", "prediction_nmse",
                  "relative_nmse_change_vs_dense_source", "relative_nmse_change_vs_matched_block"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in results)
    write_summary(report, output)
    return report


def write_summary(report, output):
    source = report["source"]
    lines = ["# Self-Forcing CrossKV：CORAL-style block-DPLR 实测", "",
             "本次只迁移 CORAL 的矩阵结构到现有同层 CrossKV，不代表完整 CORAL-KV 跨层编解码器。", "",
             f"真实 trace：`{source['path']}`；形状 `{source['shape']}`；SHA256 `{source['sha256']}`。",
             f"拟合 {len(source['calibration_record_ids'])} 条记录、测试 {len(source['evaluation_record_ids'])} 条记录；未记录 prompt 身份。", "",
             "旧表原始文件缺失；当前 trace 与旧表不同，本表各行在当前同一份 trace 上重新测试，不与旧表混算。", "",
             "CORAL 结构是每个 head 内完整矩阵加跨 head 全局低秩项。本实验原有 dense 已按 head 分开，因此 block-DPLR 相对它会增加矩阵大小。", "",
             "三轮交替 headwise ridge / exact reduced-rank regression，ridge=1e-3。拟合输入只来自 calibration；V history/bias 固定，拟合的 history 输入来自 calibration baseline 重建。", "",
             "K4/V4，anchor 4-bit，group64，BF16 重建；参数 FP32。指标为 frame>=1 的闭环 V 重建 NMSE。没有视频生成质量或生成加速结果。", "",
             "| 方法 | 全局 rank | 矩阵 MiB | V 重建 NMSE ↓ | 相对原 dense 变化 | 相对同 ridge block 变化 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["results"]:
        lines.append(f"| {row['name']} | {row['global_rank']} | {row['matrix_mib']:.6f} | {row['v_reconstruction_nmse']:.9f} | {row['relative_nmse_change_vs_dense_source']*100:+.4f}% | {row['relative_nmse_change_vs_matched_block']*100:+.4f}% |")
    lines += ["", f"[CORAL-KV Algorithm 1 / §4.2]({PAPER})", "",
              "矩阵大小包含 block 与两侧全局低秩因子；不包含固定的 V bias/history、K 参数与 KV payload。", ""]
    (output/"summary.md").write_text("\n".join(lines))


def run(config, stage, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("threads", 4)))
    if stage == "plan":
        result = {"status": "planned", "experiment": "block_dplr", "ranks": config.get("ranks", [8, 32, 128]),
                  "stages": ["fit", "replay"], "missing": [] if config.get("trace") else ["trace"],
                  "matrix_structure": "head-aligned full block diagonal + global cross-head low rank", "paper": PAPER}
        write_json(output/"plan.json", result); return result
    if stage == "fit":
        return fit(config, output)
    if stage in {"pipeline", "evaluate"}:
        if not (output/"fit.json").is_file():
            fit(config, output)
        return evaluate(config, output)
    if stage == "replay":
        return evaluate(config, output)
    if stage == "summarize":
        report = read_json(output/"reconstruction.json")
        write_summary(report, output); return report
    raise ValueError("block_dplr supports plan/fit/replay/evaluate/pipeline/summarize")
