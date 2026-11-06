#!/usr/bin/env python3
"""Isolated Self-Forcing worker for research codecs and identified clean traces."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


def _load_bridge(filename: str):
    path = ROOT / "experiments/Self-Forcing/tempokv_adapter" / filename
    spec = importlib.util.spec_from_file_location(f"rebuttal_worker_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def research_span_frames(config: dict) -> int:
    """Require a complete batch span before the final three-frame model block."""
    frames = int(config.get("frames", 1401))
    if frames < 1 or (frames + 3) % 12:
        raise ValueError("Self-Forcing frames must satisfy (frames+3)%12 == 0")
    latest_finalized = (frames + 3) // 4 - 3
    span = int(config.get("span_frames", min(24, latest_finalized)))
    if span < 2 or span > latest_finalized:
        raise ValueError("research batch codec requires 2 <= span_frames <= latent_frames - 3; "
                         "otherwise no compressed span is consumed during generation")
    return span


def require_codec_activity(before: dict, after: dict) -> dict:
    """Reject successful generation that never used both compressed cache roles."""
    counts = {key: int(after.get(key, 0)) - int(before.get(key, 0))
              for key in ("encode_k", "encode_v", "decode_k", "decode_v")}
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"research codec did not encode and decode both K/V during generation: {counts}")
    return counts


def install(config: dict, runtime_root: Path):
    """Install before importing inference; returns the actual pipeline subclass."""
    import torch
    from rebuttal.common.manifest import prompt_records, write_json
    observer = None
    frames = int(config.get("frames", 1401))
    if (frames+3)%12:
        raise ValueError("Self-Forcing frames must satisfy (frames+3)%12 == 0")
    latent_frames = (frames+3)//4
    mode = config.get("mode", "tempokv")
    os.environ["TEMPOKV_HORIZON_FRAMES"] = str(latent_frames)
    os.environ["TEMPOKV_NUM_FRAMES"] = str(latent_frames)
    span_frames = research_span_frames(config) if config.get("research_params") else int(config.get("span_frames", latent_frames))
    os.environ["TEMPOKV_SPAN_FRAMES"] = str(span_frames)
    os.environ["TEMPOKV_BF16_LAG_FRAMES"] = str(config.get("bf16_lag_frames", 18))
    if mode == "tempokv":
        def factory():
            nonlocal observer
            if config.get("research_params"):
                from rebuttal.adapters.experimental_codec import ExperimentalCodec
                base = ExperimentalCodec.load(config["research_params"])
            else:
                from tempokv.integration import load_codec
                base = load_codec(config["codec_config"], config["k_params"], config["v_params"],
                                  overrides={"use_triton": not config.get("torch_decoder", False), "strict_triton": False})
            if config.get("observer"):
                from rebuttal.adapters.observer import ObservedCodec, ReconstructionObserver
                settings = config["observer"] if isinstance(config["observer"], dict) else {}
                rank = int(os.environ.get("RANK", "0"))
                hashes = {row["prompt_id"]: row["sha256"] for row in prompt_records(config["prompts"])}
                observer = ReconstructionObserver(Path(config["output_dir"]) / f"cache_events.rank-{rank}.jsonl",
                                                 model="self_forcing", frames=frames,
                                                 tokens_per_frame=int(settings.get("tokens_per_frame", 8)), prompt_hashes=hashes)
                return ObservedCodec(base, observer)
            return base
        if config.get("research_params"):
            os.environ["TEMPOKV_INCREMENTAL"] = "0"
        base_class = _load_bridge("bridge.py").install(runtime_root, codec_factory=factory)
    elif mode == "bf16_trace":
        os.environ["TEMPOKV_TRACE_DIR"] = str(Path(config["output_dir"]) / "raw_trace")
        os.environ["TEMPOKV_TRACE_TOKENS_PER_FRAME"] = str(config.get("trace_tokens_per_frame", 8))
        os.environ["TEMPOKV_TRACE_EVENT_FRAMES"] = "0"
        base_class = _load_bridge("trace_bridge.py").install(runtime_root, preserve_video_decoder=True)
    elif mode == "bf16":
        base_class = importlib.import_module("pipeline.causal_inference").CausalInferencePipeline
    else:
        raise ValueError(f"unsupported research worker mode: {mode}")
    hashes = {row["prompt_id"]: row["sha256"] for row in prompt_records(config["prompts"])}

    class RebuttalPipeline(base_class):
        def inference(self, *args, **kwargs):
            research = bool(config.get("research_params"))
            before = dict(self.tempokv_codec.activity_counts) if research else None
            result = super().inference(*args, **kwargs)
            if research:
                counts = require_codec_activity(before, self.tempokv_codec.activity_counts)
                rank = int(os.environ.get("RANK", "0"))
                path = Path(config["output_dir"]) / f"codec_activity.rank-{rank}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as handle:
                    handle.write(json.dumps({**getattr(self, "_rebuttal_identity", {}),
                                             "span_frames": span_frames, "counts": counts,
                                             "status": "verified"}) + "\n")
            return result

        def set_research_context(self, *, prompt_index, seed):
            self._rebuttal_identity = {"prompt_id": int(prompt_index), "seed": int(seed),
                                      "prompt_sha256": hashes.get(int(prompt_index)), "model": "self_forcing"}
            active_observer = getattr(self.tempokv_codec, "observer", None) if hasattr(self, "tempokv_codec") else None
            if active_observer is not None:
                active_observer.set_context(prompt_id=int(prompt_index), seed=int(seed))

        def _write_trace_event(self):
            if mode != "bf16_trace":
                return super()._write_trace_event()
            first = self._trace_event_index
            result = super()._write_trace_event()
            rank = int(os.environ.get("RANK", "0"))
            rank_dir = Path(os.environ["TEMPOKV_TRACE_DIR"]) / f"rank_{rank:03d}"
            for path in rank_dir.glob("event_*"):
                index = int(path.name.split("_")[1])
                if index >= first:
                    write_json(path / "identity.json", {**getattr(self, "_rebuttal_identity", {}),
                                                       "pixel_frames": frames, "fps": 16,
                                                       "mapping": "wan_first_latent_1_pixel_then_4",
                                                       "source_scope": "bf16_clean_generation_with_video"})
            return result

    module = importlib.import_module("pipeline.causal_inference")
    module.CausalInferencePipeline = RebuttalPipeline
    importlib.import_module("pipeline").CausalInferencePipeline = RebuttalPipeline
    return RebuttalPipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Self-Forcing generation requires CUDA; no CPU mock is used for experimental results")
    runtime = ROOT / "experiments/Self-Forcing"
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / "tempokv_adapter"))
    os.environ["FORCING_WAN_MODEL_DIR"] = str(config["model_root"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    install(config, runtime)
    frames = int(config.get("frames", 1401))
    output = Path(config["output_dir"]) / "videos"
    output.mkdir(parents=True, exist_ok=True)
    model_config = config.get("model_config") or runtime / "configs/self_forcing_dmd.yaml"
    arguments = [str(runtime / "inference.py"), "--config_path", str(model_config),
                 "--checkpoint_path", str(config["checkpoint"]), "--data_path", str(config["prompts"]),
                 "--output_folder", str(output), "--num_output_frames", str((frames+3)//4),
                 "--seed", str(config.get("seed", 0)), "--num_samples", str(config.get("samples", 1)),
                 "--start_index", str(config.get("start_index", 0)), "--end_index", str(config.get("end_index", -1)),
                 "--local_attn_size", str(config.get("local_attn_size", (frames+3)//4)),
                 "--sink_size", str(config.get("sink_size", 0)), "--save_with_index"]
    if config.get("use_ema", False):
        arguments.append("--use_ema")
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = arguments
        os.chdir(runtime)
        runpy.run_path(str(runtime / "inference.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
