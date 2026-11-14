"""Cluster commands for the Self-Forcing AQUA-KV comparison.

Nothing here launches a GPU job. ``plan`` prints the commands; the cluster
runs them. Quality numbers are not invented.
"""
from __future__ import annotations

from typing import Any

from .budget import (
    aquakv_storage,
    comparison_rows,
    legacy_tempokv_predictor_bytes,
    predictor_file_bytes,
)
from .method import AquaSpec


def collection_bytes(spec: AquaSpec, tokens: int, videos: int) -> int:
    """One bf16 K and one bf16 V memmap per video (int16 bitcast, same size)."""
    return int(videos) * 2 * spec.num_layers * int(tokens) * spec.channel * 2


def scratch_bytes(spec: AquaSpec, tokens: int, videos: int) -> int:
    """Previous-layer K and V reconstructions for every calibration video."""
    return int(videos) * 2 * int(tokens) * spec.channel * 2


def _shell(args: list[str], *, env: list[str] | None = None) -> str:
    parts = list(env or [])
    parts.extend(args)
    return " ".join(parts)


def cluster_commands(config: dict[str, Any], spec: AquaSpec, *, tokens: int, videos: int,
                     repo: str = ".") -> dict[str, Any]:
    frames = int(config.get("frames", 1401))
    seed = int(config.get("seed", 0))
    gpus = int(config.get("num_gpus", 1))
    prompts_cal = config.get("calibration_prompts", "assets/calibration/cal_prompts.txt")
    prompts_eval = config.get("evaluation_prompts", "assets/calibration/heldout_prompts.txt")
    spec_path = config.get("aquakv_spec", "rebuttal/configs/aquakv_official.json")
    codec = config.get("codec_config", "configs/legacy.json")
    out = config.get("output_dir") or "rebuttal/outputs/aquakv_official"
    common = [
        "--checkpoint", "${CHECKPOINT:?set CHECKPOINT to the Self-Forcing weight}",
        "--frames", str(frames),
        "--samples", "1",
        "--seed", str(seed),
        "--num-gpus", str(gpus),
        "--sink-size", "0",
    ]
    env = ["FORCING_WAN_MODEL_DIR=${WAN_MODEL_DIR:?set WAN_MODEL_DIR}",
           "python", f"{repo}/experiments/Self-Forcing/tempokv_adapter/generate.py"]
    timing = _shell([*env, "--mode", "bf16", *common,
                     "--prompts", str(prompts_cal),
                     "--output-dir", f"{out}/timing_one_prompt",
                     "--start-index", "0", "--end-index", "1"])
    collect = _shell([*env, "--mode", "aquakv", "--aquakv-collect",
                      "--aquakv-spec", str(spec_path),
                      "--aquakv-trace-dir", f"{out}/collection",
                      *common, "--prompts", str(prompts_cal),
                      "--output-dir", f"{out}/collection_run"])
    calibrate = _shell([
        "python", f"{repo}/rebuttal/run.py",
        "--experiment", "aquakv_official", "--stage", "calibrate",
        "--config", str(spec_path),
        "--set", f"collection_dir={out}/collection",
        "--output-dir", f"{out}/calibrate",
    ])
    generate = {
        "bf16": _shell([*env, "--mode", "bf16", *common,
                        "--prompts", str(prompts_eval),
                        "--output-dir", f"{out}/videos/bf16"]),
        "tempokv": _shell([*env, "--mode", "tempokv",
                           "--config", str(codec),
                           "--k-params", "${TEMPOKV_K_PARAMS:?set TEMPOKV_K_PARAMS}",
                           "--v-params", "${TEMPOKV_V_PARAMS:?set TEMPOKV_V_PARAMS}",
                           *common, "--prompts", str(prompts_eval),
                           "--output-dir", f"{out}/videos/tempokv"]),
        "aquakv": _shell([*env, "--mode", "aquakv",
                          "--aquakv-spec", str(spec_path),
                          "--aquakv-predictors", f"{out}/calibrate/aquakv_predictors.pt",
                          *common, "--prompts", str(prompts_eval),
                          "--output-dir", f"{out}/videos/aquakv"]),
    }
    evaluate = _shell([
        "python", f"{repo}/rebuttal/run.py",
        "--experiment", "aquakv_official", "--stage", "evaluate",
        "--config", str(spec_path),
        "--set", f"bf16_video_dir={out}/videos/bf16/videos",
        "--set", f"tempokv_video_dir={out}/videos/tempokv/videos",
        "--set", f"aquakv_video_dir={out}/videos/aquakv/videos",
        "--output-dir", f"{out}/evaluate",
    ])
    disk = collection_bytes(spec, tokens, videos)
    return {
        "timing_one_prompt": timing,
        "collect_calibration": collect,
        "calibrate": calibrate,
        "generate": generate,
        "evaluate": evaluate,
        "expected_compute": {
            "calibration_videos": videos,
            "evaluation_videos_per_arm": videos,
            "arms": ["bf16", "tempokv", "aquakv"],
            "full_rollouts": videos + 3 * videos,
            "latent_frames": int(config.get("latent_frames", 351)),
            "denoising_forwards_per_block": "4 denoise + 1 clean refresh",
            "blocks_per_rollout": "latent_frames / 3",
            "bf16_kv_gib_from_experiment_matrix": 93.99490356445312,
            "gpu_class": "BF16 full history is about 94 GiB of KV before weights and workspace; use an H200-class 141 GiB GPU. This bridge does not offload the cache.",
            "collection_disk_bytes": disk,
            "collection_disk_gib": disk / (1024 ** 3),
            "calibration_scratch_bytes_one_layer": scratch_bytes(spec, tokens, videos),
            "predictor_bytes": aquakv_storage(spec, tokens=tokens)["predictor_bytes"],
            "wall_clock": "Time the timing_one_prompt command on the target GPU, then scale by full_rollouts. Add the calibrate job (fp32 normal equations, one layer of scratch). Do not treat any number in this file as a measurement.",
            "quality": None,
        },
    }


def comparison_from_config(config: dict[str, Any], spec: AquaSpec, *, tokens: int,
                           reference_spec: AquaSpec | None = None) -> dict[str, Any]:
    from tempokv.config import load_config
    codec = load_config(config["codec_config"], config.get("codec_overrides"))
    scale_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}[codec.scale_dtype]
    zero_point_bytes = 1 if codec.residual_mode == "asymmetric" else 0
    files = predictor_file_bytes([config.get("k_params"), config.get("v_params")])
    if files is None:
        analytic = legacy_tempokv_predictor_bytes(
            k_history=int(codec.k_history_length or 0),
            v_history=int(codec.v_history_length),
            heads=spec.num_heads, dim=spec.head_dim,
            k_mode=str(codec.k_predictor_mode),
            cross_mode=str(codec.cross_kv_weight_mode),
        )
        predictor_bytes = 0 if analytic is None else analytic
        predictor_source = "analytic_legacy" if analytic is not None else "unknown_layout_counted_as_zero"
    else:
        predictor_bytes = files
        predictor_source = "parameter_files"
    payload = dict(
        k_bits=int(codec.k_bits), v_bits=int(codec.v_bits), anchor_bits=int(codec.anchor_bits),
        group_size=int(codec.group_size),
        span_frames=int(config.get("span_frames") or 24),
        frame_seq_length=int(config.get("frame_seq_length") or codec.stride),
        scale_bytes=scale_bytes, zero_point_bytes=zero_point_bytes,
        predictor_bytes=int(predictor_bytes),
    )
    table = comparison_rows(spec, payload, tokens=tokens)
    table["tempokv_predictor_source"] = predictor_source
    table["quality"] = None
    if reference_spec is not None:
        table["official_readme_2bit_not_bit_matched"] = comparison_rows(reference_spec, payload, tokens=tokens)["rows"][-1]
    return table
