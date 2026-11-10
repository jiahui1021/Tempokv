"""Run BF16, real QVG, or TempoKV in the same Self-Forcing generator.

This deliberately uses an independent-span, batch-compression lifecycle.
It is a separate budget experiment, not a replacement for the production
incremental schedule.  No model implementation is copied into this module.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from rebuttal.experiments.memory.budget_manager import BudgetManager, ReleasableKVCache


def install(runtime_root: str | Path, config: dict) -> type:
    runtime_root = Path(runtime_root).expanduser().resolve()
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    module = importlib.import_module("pipeline.causal_inference")
    package = importlib.import_module("pipeline")
    loaded_path = getattr(module, "__file__", None)
    if loaded_path and not Path(loaded_path).resolve().is_relative_to(runtime_root):
        raise RuntimeError("a different pipeline checkout is imported; use a fresh worker")
    base = module.CausalInferencePipeline
    settings = dict(config)

    class BudgetCausalInferencePipeline(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Generation-time KV residency is unchanged. Completed pixels need
            # no GPU storage and otherwise make long-video concatenation OOM.
            from rebuttal.experiments.memory.vae_output import install_cpu_video_output
            install_cpu_video_output(self.vae.model)
            self._budget_method = settings.get("method", "tempokv")
            if self._budget_method == "qvg_prq_real_codec":
                self._budget_method = "qvg"
            if self._budget_method not in ("bf16", "qvg", "tempokv"):
                raise ValueError("budget method must be bf16/qvg/tempokv")
            self._budget_history = int(settings["history_frames"])
            self._budget_span = int(settings.get("span_frames", 12))
            self._budget_lag = int(settings.get("bf16_lag_frames", 18))
            if self._budget_span < 2 or self._budget_span % self.num_frame_per_block:
                raise ValueError("budget span must align to model blocks and contain an anchor/residual")
            if self._budget_lag < 0:
                raise ValueError("bf16_lag_frames cannot be negative")
            if self._budget_method != "bf16" and self._budget_history < self._budget_span + self._budget_lag + self.num_frame_per_block:
                raise ValueError("history too short for the independent-span/lag schedule")
            if self.local_attn_size != self._budget_history:
                raise ValueError("native local_attn_size must equal history_frames")
            for block in self.generator.model.blocks:
                if getattr(block.self_attn, "sink_size", 0):
                    raise ValueError("first fixed-budget protocol requires sink=0")
                block.self_attn.max_attention_size = self._budget_history * self.frame_seq_length
            self._uses_pre_rope_cache = True
            # This flag requests the base pipeline's per-clean-block hook.  The
            # codec schedule below is batch independent-span, not incremental.
            self._incremental_tempokv = True
            self._budget_codec = None
            self._budget_qvg = None
            if self._budget_method == "tempokv":
                from tempokv.integration import load_codec
                self._budget_codec = load_codec(settings["codec_config"], settings["k_params"],
                                                settings["v_params"], overrides=settings.get("codec_overrides"))
                c = self._budget_codec.config
                if c.stride != self.frame_seq_length:
                    raise ValueError("TempoKV stride must match the model's tokens per frame")
                if c.reconstruction_dtype != "bfloat16" or c.jensen_correction:
                    raise ValueError("budget comparison requires BF16 reconstruction and Jensen disabled")
                if self._budget_codec.k_channel_mean:
                    raise ValueError("mean-split is outside the fixed-budget comparison protocol")
            elif self._budget_method == "qvg":
                from rebuttal.adapters.qvg import QVGBackend
                self._budget_qvg = QVGBackend(settings["qvg_root"], **settings.get("qvg", {}))
            self._budget_manager = None
            self._budget_finalized = 0
            self._budget_next_span = 0
            self.generator.register_forward_pre_hook(self._budget_before_forward, with_kwargs=True)
            self.generator.register_forward_hook(self._budget_after_forward, with_kwargs=True)

        def _budget_decode(self, state, source_tensor=None, output_tensor=None, output_layout="BSHD"):
            if self._budget_qvg is not None:
                return self._budget_qvg.decode_tensor(state, source_tensor=source_tensor,
                    output_tensor=output_tensor, output_layout=output_layout)
            from tempokv.codec import decode_tensor
            return decode_tensor(state, source=source_tensor, output_tensor=output_tensor,
                output_layout=output_layout, use_triton=self._budget_codec.config.use_triton,
                strict_triton=self._budget_codec.config.strict_triton)

        def _initialize_kv_cache(self, batch_size, dtype, device, target_frames=None):
            count = int(target_frames or self._budget_history)
            caches = []
            for _ in range(self.num_transformer_blocks):
                layer = {"global_end_index": 0, "local_end_index": 0}
                for role in ("k", "v"):
                    cache = ReleasableKVCache(batch_size, self.frame_seq_length, self.num_heads,
                        self.head_dim, count, dtype, device, layout="BSHD", cache_role=role.upper(),
                        quantized_decoder=self._budget_decode)
                    cache.rope_precision = settings.get("rope_precision", "complex64")
                    cache.tempokv_predictor_direction = (self._budget_codec.config.predictor_direction
                                                       if self._budget_codec is not None else "k_to_v")
                    layer[role] = cache
                caches.append(layer)
            self.kv_cache1 = caches

        def _budget_event(self, event):
            path = settings.get("memory_events")
            if path:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                sample = self.summary.get("budget_sample", 0)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"sample_index": sample, **event}) + "\n")

        def reset_quantization_state(self):
            if self._budget_manager is not None:
                self._budget_manager.close()
            self._budget_finalized = self._budget_next_span = 0
            self.summary["budget_sample"] = int(self.summary.get("budget_sample", -1)) + 1
            params = (None if self._budget_codec is None else
                      (self._budget_codec.temporal, self._budget_codec.v_parameters,
                       self._budget_codec._layer_cache))
            self._budget_manager = BudgetManager(self.kv_cache1,
                budget_bytes=settings.get("budget_bytes"), history_frames=self._budget_history,
                tokens_per_frame=self.frame_seq_length, parameters=params, event_sink=self._budget_event)
            for layer in self.kv_cache1:
                for role in ("k", "v"):
                    layer[role].after_write = lambda: self._budget_manager.snapshot("after_write")

        def _budget_before_forward(self, generator, args, kwargs):
            del generator
            value = kwargs.get("noisy_image_or_video")
            if value is None and args:
                value = args[0]
            if value is None:
                raise ValueError("generator pass does not identify its incoming video block")
            start_tokens = int(kwargs.get("current_start", 0))
            if start_tokens % self.frame_seq_length:
                raise ValueError("generator current_start is not frame aligned")
            start = start_tokens // self.frame_seq_length
            self._budget_manager.prepare(start + int(value.shape[1]), start)

        def _budget_after_forward(self, generator, args, kwargs, result):
            del generator, args, kwargs, result
            self._budget_manager.snapshot("after_forward")

        def quantize_kv_cache(self, tokens_to_quantize_start, tokens_to_quantize_end, max_tokens_to_quantize):
            del max_tokens_to_quantize
            start = int(tokens_to_quantize_start) // self.frame_seq_length
            end = int(tokens_to_quantize_end) // self.frame_seq_length
            if start != self._budget_finalized:
                raise ValueError("clean blocks must finalize in absolute order")
            self._budget_finalized = end
            if self._budget_method == "bf16":
                self._budget_manager.snapshot("clean_bf16")
                return
            eligible = max(0, end - self._budget_lag)
            from tempokv.runtime.cache import ChunkState
            while self._budget_next_span + self._budget_span <= eligible:
                first = self._budget_next_span
                last = first + self._budget_span
                self._budget_next_span = last
                if any(layer[role].chunk_state[i] != ChunkState.BF16
                       for layer in self.kv_cache1 for role in ("k", "v") for i in range(first, last)):
                    raise RuntimeError("a not-yet-encoded span was evicted; history/span/lag are incompatible")
                a, b = first * self.frame_seq_length, last * self.frame_seq_length
                with torch.no_grad():
                    for li, layer in enumerate(self.kv_cache1):
                        key = layer["k"].read_bhsd_independent(a, b)
                        value = layer["v"].read_bhsd_independent(a, b)
                        if self._budget_codec is not None:
                            pair = self._budget_codec.encode(key, value, layer=li)
                            pk, pv = pair.key, pair.value
                        else:
                            pk, pv = self._budget_qvg.encode(key, value, layer=li)
                        layer["k"].store_quantized(a, b, pk)
                        layer["v"].store_quantized(a, b, pv)
                        del key, value, pk, pv
                        self._budget_manager.snapshot("after_encode_layer")
            self._budget_manager.snapshot("after_encode")

        def flush_incremental_tail(self):
            # Preserve the generation-time state.  A post-generation flush
            # would hide the dense tail from this budget experiment.
            self._budget_manager.snapshot("generation_complete_without_tail_flush")

        def _resident_kv_bytes(self):
            if self._budget_manager is None:
                return 0, 0
            snapshot = self._budget_manager.snapshot("resident_sample")
            return snapshot["kv_owned_bytes"], self._budget_history * self.num_transformer_blocks * 2 * self.num_heads * self.head_dim * self.frame_seq_length * 2

        def inference(self, *args, **kwargs):
            result = super().inference(*args, **kwargs)
            report = self._budget_manager.snapshot("inference_complete")
            report.update(method=self._budget_method, history_frames=self._budget_history,
                          span_frames=self._budget_span, bf16_lag_frames=self._budget_lag,
                          schedule="batch_independent_spans", generator="bundled_self_forcing",
                          rope_precision=settings.get("rope_precision", "complex64"))
            self.summary["fixed_memory"] = report
            path = settings.get("memory_report")
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return result

    module.CausalInferencePipeline = BudgetCausalInferencePipeline
    package.CausalInferencePipeline = BudgetCausalInferencePipeline
    return BudgetCausalInferencePipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-memory video generation requires CUDA and model weights")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.reset_peak_memory_stats()
    runtime = ROOT / "experiments" / "Self-Forcing"
    if config.get("model_root"):
        os.environ["FORCING_WAN_MODEL_DIR"] = str(config["model_root"])
    install(runtime, config)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(runtime / "inference.py"), *arguments]
    os.chdir(runtime)
    try:
        runpy.run_path(str(runtime / "inference.py"), run_name="__main__")
    finally:
        torch.cuda.synchronize()
        report = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "scope": "whole_device_worker_including_model_and_workspace"}
        target = Path(config.get("memory_report", args.config.with_suffix(".report.json")))
        target.with_name(target.stem + ".device.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
