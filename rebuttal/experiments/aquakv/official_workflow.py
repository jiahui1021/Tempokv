"""Faithful AQUA-KV on Self-Forcing: plan, collect, calibrate, generate, evaluate.

``plan`` only writes the bit table and the cluster commands. It does not
invent PSNR, SSIM, or LPIPS. ``calibrate`` reads BF16 KV memmaps collected by
``--mode aquakv --aquakv-collect`` and fits the official sequential predictors.
``generate`` / ``evaluate`` record commands or, when the video directories
already exist, call ``scripts/evaluation/evaluate_quality.py``.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from rebuttal.common.manifest import write_json

from .commands import cluster_commands, comparison_from_config
from .method import AquaSpec, state_dict
from .store import MemmapSampleSource, _MemmapScratch, calibrate_streaming, check_collection_against_prompts


def spec_from_config(config: dict[str, Any], **overrides: Any) -> AquaSpec:
    payload = dict(
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        head_dim=int(config["head_dim"]),
        hadamard_groupsize=int(config.get("hadamard_groupsize", 1024)),
        edenn_d=int(config.get("edenn_d", 2)),
        edenn_n=int(config.get("edenn_n", 64)),
        first_layer_edenn_d=int(config.get("first_layer_edenn_d", 2)),
        first_layer_edenn_n=int(config.get("first_layer_edenn_n", 256)),
        not_quantize_first_layer=bool(config.get("not_quantize_first_layer", False)),
        prefix_tokens=int(config.get("prefix_tokens", 4)),
        recent_buffer_tokens=int(config.get("recent_buffer_tokens", 128)),
        percdamp=float(config.get("percdamp", 1e-3)),
        valid_nsamples=int(config.get("valid_nsamples", 4)),
        seed=int(config.get("calibration_seed", config.get("seed", 0))),
        activation_dtype=str(config.get("activation_dtype", "bfloat16")),
        index_packing=str(config.get("index_packing", "packed")),
        chunk_size=int(config.get("chunk_size", 4096)),
        higgs_chunk_size=int(config.get("higgs_chunk_size", 64)),
    )
    payload.update(overrides)
    return AquaSpec(**payload)


def _tokens(config: dict[str, Any]) -> int:
    frames = int(config.get("latent_frames", 351))
    frame_seq = int(config.get("frame_seq_length", 1560))
    return frames * frame_seq


def _videos(config: dict[str, Any]) -> int:
    return int(config.get("expected_prompt_count", 32))


def _table(config: dict[str, Any], spec: AquaSpec) -> dict[str, Any]:
    reference = spec_from_config(config, edenn_n=16, edenn_d=2)
    return comparison_from_config(config, spec, tokens=_tokens(config), reference_spec=reference)


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage not in {"plan", "collect", "calibrate", "fit", "generate", "evaluate"}:
        raise ValueError("aquakv supports plan, collect, calibrate, generate, evaluate")
    spec = spec_from_config(config)
    if stage == "plan":
        table = _table(config, spec)
        commands = cluster_commands(
            {**config, "output_dir": str(output_dir), "aquakv_spec": config.get("aquakv_spec", "rebuttal/configs/aquakv_official.json")},
            spec, tokens=_tokens(config), videos=_videos(config),
        )
        plan = {
            "status": "planned",
            "experiment": "aquakv_official",
            "official_commit": "e744183f2f3a4a90381f20a6975456871fd1ae29",
            "quantizer": "HiggsQuantizer",
            "spec": spec.to_dict(),
            "comparison": table,
            "commands": commands,
            "quality": None,
            "stages": ["collect", "calibrate", "generate", "evaluate"],
            "note": "EDEN2-64 is the closest official grid to legacy TempoKV k4/v4. The bit gap is reported; the codebook is not retuned.",
        }
        write_json(output_dir / "plan.json", plan)
        return plan
    if stage == "collect":
        commands = cluster_commands(config, spec, tokens=_tokens(config), videos=_videos(config))
        result = {"status": "commands_only", "quality": None,
                  "command": commands["collect_calibration"],
                  "reason": "KV collection is a BF16 Self-Forcing rollout; this process does not generate it."}
        write_json(output_dir / "collect.json", result)
        return result
    if stage in {"calibrate", "fit"}:
        collection = config.get("collection_dir")
        if not collection:
            raise FileNotFoundError("calibrate needs collection_dir from --mode aquakv --aquakv-collect")
        source = MemmapSampleSource(collection)
        check_collection_against_prompts(
            source, config["calibration_prompts"], config["evaluation_prompts"],
        )
        if source.tokens() != _tokens(config) and not config.get("allow_short_collection"):
            raise ValueError(
                f"collection has {source.tokens()} tokens, config expects {_tokens(config)}; "
                "set allow_short_collection only for a reduced debug rollout"
            )
        scratch_dir = Path(config.get("scratch_dir") or output_dir / "scratch")
        scratch = _MemmapScratch(
            scratch_dir, tokens=source.tokens(), heads=spec.num_heads, dim=spec.head_dim, dtype=spec.dtype,
        )
        if spec.dtype != torch.bfloat16:
            raise ValueError("disk calibration scratch is bf16; set activation_dtype=bfloat16")
        bank, report = calibrate_streaming(source, spec, scratch=scratch)
        path = Path(config.get("predictors_path") or output_dir / "aquakv_predictors.pt")
        torch.save(state_dict(bank, extra={"collection_dir": str(collection), "fit_report": report}), path)
        result = {"status": "complete", "quality": None, "predictor_path": str(path),
                  "parameter_count": bank.parameter_count(), "parameter_bytes": bank.parameter_bytes(),
                  "train_indices": report["train_indices"], "valid_indices": report["valid_indices"],
                  "videos": report["videos"], "comparison": _table(config, spec)}
        write_json(output_dir / "calibrate.json", result)
        return result
    if stage == "generate":
        commands = cluster_commands(config, spec, tokens=_tokens(config), videos=_videos(config))
        result = {"status": "commands_only", "quality": None, "commands": commands["generate"],
                  "reason": "generation runs on the cluster via generate.py; no videos were created here."}
        write_json(output_dir / "generate.json", result)
        return result
    return _evaluate(config, spec, output_dir)


def _evaluate(config: dict[str, Any], spec: AquaSpec, output_dir: Path) -> dict[str, Any]:
    table = _table(config, spec)
    reference = config.get("bf16_video_dir")
    quality: dict[str, Any] = {}
    script = Path(__file__).resolve().parents[3] / "scripts" / "evaluation" / "evaluate_quality.py"
    for name in ("tempokv", "aquakv"):
        candidate = config.get(f"{name}_video_dir")
        if not reference or not candidate or not Path(reference).is_dir() or not Path(candidate).is_dir():
            quality[name] = None
            continue
        if not any(Path(reference).rglob("*.mp4")) or not any(Path(candidate).rglob("*.mp4")):
            quality[name] = None
            continue
        report_path = output_dir / f"quality_{name}.json"
        completed = subprocess.run(
            [sys.executable, str(script), str(reference), str(candidate),
             "--output", str(report_path), "--no-fid"],
            check=False, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            quality[name] = {"status": "failed", "returncode": completed.returncode,
                             "stderr": completed.stderr[-2000:]}
            continue
        from rebuttal.common.manifest import read_json
        quality[name] = read_json(report_path)
    ran = any(isinstance(item, dict) and "psnr_db" in item for item in quality.values())
    result = {
        "status": "complete" if ran else "not_run",
        "quality": quality if ran else None,
        "comparison": table,
        "reason": None if ran else "video directories are absent; quality metrics were not computed",
    }
    for row in result["comparison"]["rows"]:
        if row["method"] == "bf16":
            row["quality"] = None
        elif row["method"] in quality and isinstance(quality[row["method"]], dict) and "psnr_db" in quality[row["method"]]:
            row["quality"] = {"psnr_db": quality[row["method"]]["psnr_db"],
                              "ssim": quality[row["method"]]["ssim"],
                              "lpips": quality[row["method"]]["lpips"]}
    write_json(output_dir / "evaluate.json", result)
    return result
