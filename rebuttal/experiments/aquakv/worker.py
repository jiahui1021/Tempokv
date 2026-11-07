#!/usr/bin/env python3
"""Isolated Self-Forcing execution for independent AquaKV and BF16 calibration."""
from __future__ import annotations

import argparse
from functools import partial
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


class CacheSlot:
    """Reset-compatible placeholder; allocates no native K/V tensor."""
    def clear(self):
        pass


def require_activity(report):
    counts = report["counts"]
    if any(counts.get(key, 0) <= 0 for key in ("encode_k", "encode_v", "decode_k", "decode_v")):
        raise RuntimeError(f"AquaKV generation did not consume compressed K/V: {counts}; increase frames or reduce recent_buffer_frames")
    return counts


def install(config, runtime_root):
    import numpy as np
    import torch
    from rebuttal.common.manifest import file_sha256, prompt_records, write_json
    from .runtime import AquaKVCache, rotate_pre_rope
    from .predictors import load_predictor
    from .quantizer import Quantizer, HiggsConfig
    module = importlib.import_module("pipeline.causal_inference")
    package = importlib.import_module("pipeline")
    base = module.CausalInferencePipeline
    mode = config.get("mode", "aquakv")
    if mode not in {"aquakv", "bf16", "collect"}:
        raise ValueError("unsupported independent AquaKV worker mode")
    hashes = {r["prompt_id"]: r["sha256"] for r in prompt_records(config["prompts"])}
    runtime_settings = config.get("cache", {})

    class AquaPipeline(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._incremental_tempokv = mode in {"aquakv", "collect"}
            self.aquakv_cache = None
            if mode == "aquakv":
                state = torch.load(config["predictor_path"], map_location="cpu", weights_only=True)
                if state["parameter_dtype"] != "bfloat16":
                    raise ValueError("Self-Forcing requires AquaKV calibrated in bfloat16")
                if state["quantizer"] != config.get("higgs", {}):
                    if state["quantizer"] != Quantizer(self.num_heads*self.head_dim, HiggsConfig(**config.get("higgs", {}))).settings():
                        raise ValueError("generation HIGGS settings differ from calibration")
                if state["layer_ids"] != list(range(self.num_transformer_blocks)):
                    raise ValueError("AquaKV calibration must cover every Self-Forcing layer in order")
                self.aquakv_predictor = load_predictor(state, next(self.generator.parameters()).device)
                self.aquakv_quantizer = Quantizer(self.num_heads*self.head_dim, HiggsConfig(**state["quantizer"]))
            if mode == "collect":
                for layer, block in enumerate(self.generator.model.blocks):
                    block.self_attn.norm_k.register_forward_hook(self._capture(layer, "k"))
                    block.self_attn.v.register_forward_hook(self._capture(layer, "v"))

        def _initialize_kv_cache(self, batch_size, dtype, device, target_frames=None):
            if mode != "aquakv":
                return super()._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device, target_frames=target_frames)
            self.aquakv_predictor.to(device=device)
            self.aquakv_cache = AquaKVCache(self.aquakv_predictor, self.aquakv_quantizer,
                frame_seqlen=self.frame_seq_length, prefix_frames=int(runtime_settings.get("prefix_frames", 1)),
                recent_buffer_frames=int(runtime_settings.get("recent_buffer_frames", 3)),
                local_attn_size=self.local_attn_size, sink_size=int(config.get("sink_size", 0)),
                rope_precision=config.get("rope_precision", "native"))
            self.kv_cache1 = []
            for layer in range(self.num_transformer_blocks):
                cache = {"k": CacheSlot(), "v": CacheSlot(), "global_end_index": 0, "local_end_index": 0,
                         "aquakv_handler": partial(self._attend, layer)}
                self.kv_cache1.append(cache)

        def _attend(self, layer, q, k, v, **kwargs):
            result = self.aquakv_cache.attend(layer, q, k, v, rotate=rotate_pre_rope, **kwargs)
            cache = self.kv_cache1[layer]
            cache["global_end_index"] = int(kwargs["current_start"])+k.shape[1]
            cache["local_end_index"] = sum(c.end-c.start for c in self.aquakv_cache.chunks)+k.shape[1]
            return result

        def reset_quantization_state(self):
            self._aquakv_clean = False
            self._capture_block = {}
            self._capture_sequence = {}
            if self.aquakv_cache is not None:
                self.aquakv_cache.reset()

        def note_clean_refresh(self):
            self._aquakv_clean = True

        def quantize_kv_cache(self, start, end, max_tokens):
            if mode == "bf16":
                return
            if not self._aquakv_clean:
                raise RuntimeError("AquaKV must commit only the pipeline-announced clean refresh")
            if mode == "aquakv":
                self.aquakv_cache.commit_clean(start, end)
            else:
                if len(self._capture_block) != self.num_transformer_blocks:
                    raise RuntimeError("clean trace did not capture every transformer layer")
                for layer, tensors in self._capture_block.items():
                    if set(tensors) != {"k", "v"}:
                        raise RuntimeError("clean trace is missing K or V")
                    sequence = self._capture_sequence.setdefault(layer, {"k": [], "v": []})
                    for role in ("k", "v"):
                        sequence[role].append(tensors[role])
                self._capture_block = {}
            self._aquakv_clean = False

        def _capture(self, layer, role):
            def hook(module, args, output):
                if not self._aquakv_clean:
                    return
                batch, tokens, channels = output.shape
                positions = min(int(config.get("trace_tokens_per_frame", 8)), self.frame_seq_length)
                indices = torch.linspace(0, self.frame_seq_length-1, positions, device=output.device).round().long().unique()
                self._capture_indices = indices.cpu().numpy()
                sampled = output.detach().reshape(batch, -1, self.frame_seq_length, self.num_heads, self.head_dim)
                sampled = sampled.index_select(2, indices).transpose(1, 2).float().cpu().numpy()
                self._capture_block.setdefault(layer, {})[role] = sampled
            return hook

        def set_research_context(self, *, prompt_index, seed):
            self._aquakv_identity = {"prompt_id": int(prompt_index), "prompt_sha256": hashes[int(prompt_index)], "seed": int(seed)}

        def flush_incremental_tail(self):
            if self.aquakv_cache is not None and runtime_settings.get("flush_tail", True):
                self.aquakv_cache.flush()

        def _resident_kv_bytes(self):
            if self.aquakv_cache is None:
                return super()._resident_kv_bytes()
            from .runtime import owned_bytes
            tensors = self.aquakv_cache.tensors()
            tokens = sum(c.end-c.start for c in self.aquakv_cache.chunks)
            if self.aquakv_cache.active is not None:
                tokens += self.aquakv_cache.active.end-self.aquakv_cache.active.start
            batch = tensors[0].shape[0] if tensors else 1
            # Packed states flatten batch/token axes, so use the inference batch.
            batch = getattr(self, "_aquakv_batch", batch)
            return owned_bytes(tensors), batch*tokens*self.num_heads*self.head_dim*4*self.num_transformer_blocks

        def inference(self, *args, **kwargs):
            noise = kwargs.get("noise", args[0] if args else None)
            self._aquakv_batch = noise.shape[0]
            result = super().inference(*args, **kwargs)
            output = Path(config["output_dir"])
            rank = int(os.environ.get("RANK", "0"))
            identity = self._aquakv_identity
            if mode == "collect":
                layers = sorted(self._capture_sequence)
                keys, values = [], []
                for layer in layers:
                    for role, target in (("k", keys), ("v", values)):
                        array = np.concatenate(self._capture_sequence[layer][role], axis=2)
                        target.append(array.reshape(-1, *array.shape[2:]))
                path = output / "traces" / f"prompt_{identity['prompt_id']:06d}.rank_{rank}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError(f"calibration trace already exists: {path}")
                np.savez(path, keys=np.stack(keys), values=np.stack(values), layer_ids=np.asarray(layers),
                         **{k: np.asarray(v) for k, v in identity.items()},
                         sample_ids=np.repeat(np.arange(self._aquakv_batch), len(self._capture_indices)),
                         token_indices=self._capture_indices, coordinate_system=np.asarray("pre_rope_normalized"),
                         trace_role=np.asarray(config.get("trace_role", "calibration")))
            if mode == "aquakv":
                report = self.aquakv_cache.report()
                require_activity(report)
                self.summary["last_kv_cache_memory"] = report
                record = {"status": "verified", **identity, **report,
                          "predictor_sha256": file_sha256(config["predictor_path"]), "higgs": config.get("higgs", {})}
                output.mkdir(parents=True, exist_ok=True)
                with (output / f"aquakv_activity.rank-{rank}.jsonl").open("a") as handle:
                    handle.write(json.dumps(record)+"\n")
            return result

    module.CausalInferencePipeline = package.CausalInferencePipeline = AquaPipeline
    return AquaPipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Self-Forcing generation requires CUDA and local model weights")
    runtime = ROOT / "experiments/Self-Forcing"
    sys.path.insert(0, str(runtime))
    os.environ["FORCING_WAN_MODEL_DIR"] = str(config["model_root"])
    os.environ["SELF_FORCING_CACHE_ROPE_PRECISION"] = config.get("rope_precision", "native")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    # Run as a module so the vendored AQUA-KV package has a stable import name.
    install(config, runtime)
    frames = int(config.get("frames", 1401))
    arguments = [str(runtime / "inference.py"), "--config_path", str(config.get("model_config") or runtime / "configs/self_forcing_dmd.yaml"),
                 "--checkpoint_path", str(config["checkpoint"]), "--data_path", str(config["prompts"]),
                 "--output_folder", str(Path(config["output_dir"]) / "videos"), "--num_output_frames", str((frames+3)//4),
                 "--seed", str(config.get("seed", 0)), "--num_samples", str(config.get("samples", 1)),
                 "--start_index", str(config.get("start_index", 0)), "--end_index", str(config.get("end_index", -1)),
                 "--local_attn_size", str(config.get("local_attn_size", -1)),
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
