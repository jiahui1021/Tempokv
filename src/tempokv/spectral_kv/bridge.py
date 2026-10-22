"""Explicit suite B installation for Self-Forcing and Causal-Forcing."""
from __future__ import annotations

from collections import Counter
import importlib
import os
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .context import RunContext
from .probe import SpectralKVProbe


def _arg(args, kwargs, name, index, default=None):
    return kwargs[name] if name in kwargs else args[index] if len(args) > index else default


def install(runtime_root, runtime="self_forcing"):
    root = Path(runtime_root).resolve()
    missing = [key for key in ("SPECTRAL_KV_CONFIG", "SPECTRAL_KV_OUTPUT_DIR") if not os.environ.get(key)]
    if missing:
        raise ValueError(f"suite B requires {', '.join(missing)}")
    config = load_config(os.environ["SPECTRAL_KV_CONFIG"])
    output = Path(os.environ["SPECTRAL_KV_OUTPUT_DIR"]).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    prefix = {"self_forcing": "pipeline", "causal_forcing": "causal_forcing.pipeline"}.get(runtime)
    if prefix is None:
        raise ValueError("suite B currently supports Self-Forcing and Causal-Forcing")
    package, module = importlib.import_module(prefix), importlib.import_module(prefix + ".causal_inference")
    base = module.CausalInferencePipeline
    if getattr(base, "_spectral_kv_installed", False):
        return base
    if getattr(base, "_spectral_installed", False):
        raise ValueError("suite A and suite B must run separately")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    class SpectralKVPipeline(base):
        _spectral_kv_installed = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if getattr(self, "_uses_pre_rope_cache", False):
                raise ValueError("suite B requires native BF16; quantizer composition is not supported")
            blocks = self.generator.model.blocks
            if config.layers and max(config.layers) >= len(blocks):
                raise ValueError("suite B layer selection exceeds runtime model")
            self.spectral_probe = SpectralKVProbe(config, output, rank=int(os.environ.get("RANK", "0")), layers=len(blocks),
                metadata={"git_commit": commit, "runtime": runtime, "model": os.environ.get("SPECTRAL_KV_MODEL", "unknown"),
                          "source": "runtime_native_bf16", "generation_seed_contract": "seed + prompt_index * 1000003"})
            if self.spectral_probe.method.skip == "lin":
                coverage = set(self.spectral_probe.calibration.keys)
                required = {(layer, role, head) for layer, block in enumerate(blocks)
                            for role in ("k", "v") for head in range(block.self_attn.num_heads)}
                if not required.issubset(coverage):
                    raise ValueError("KVSKIP_lin calibration does not cover every runtime layer/head/role")
            self._spectral_context = {}
            self._b_clean_pending = False
            self._b_calls, self._b_starts = Counter(), {}
            self._b_step_count = len(self.denoising_step_list)
            for layer, block in enumerate(blocks):
                block.self_attn.spectral_layer_idx = layer
                block.self_attn.spectral_probe = self.spectral_probe
                block.register_forward_hook(lambda _module, _args, output, index=layer: self.spectral_probe.on_block(index, output))
            self.generator.register_forward_pre_hook(self._before_generator, with_kwargs=True)
            original_forward = self.generator.forward
            def forward(*call_args, **call_kwargs):
                phase = self.spectral_probe.phase
                device = _arg(call_args, call_kwargs, "noisy_image_or_video", 0).device
                if phase == "clean" and self.spectral_probe.method.skip:
                    cache = _arg(call_args, call_kwargs, "kv_cache", 3)
                    self.spectral_probe.skip_clean(cache)
                    return None  # The pipeline ignores this announced clean-forward result.
                with self.spectral_probe.sink.time("generator_clean" if phase == "clean" else "generator_denoise", device):
                    return original_forward(*call_args, **call_kwargs)
            self.generator.forward = forward
            original_decode = self.vae.decode_to_pixel
            def decode(*decode_args, **decode_kwargs):
                if not self.spectral_probe.active:
                    return original_decode(*decode_args, **decode_kwargs)
                device = decode_args[0].device if decode_args else None
                with self.spectral_probe.sink.time("vae", device):
                    return original_decode(*decode_args, **decode_kwargs)
            self.vae.decode_to_pixel = decode

        def set_spectral_context(self, *, prompt_id, **metadata):
            self._spectral_context = {"prompt_id": prompt_id, **metadata}

        def note_clean_refresh(self):
            super().note_clean_refresh()
            self._b_clean_pending = True

        def note_ephemeral_block(self, denoise_steps):
            parent = getattr(super(), "note_ephemeral_block", None)
            if parent is not None:
                parent(denoise_steps)
            self._b_step_count = int(denoise_steps)

        def _before_generator(self, _module, args, kwargs):
            start = int(_arg(args, kwargs, "current_start", 5, 0))
            if start not in self._b_starts:
                self._b_starts[start] = len(self._b_starts)
            chunk = self._b_starts[start]
            clean = self._b_clean_pending
            self._b_clean_pending = False
            step = self._b_calls[start]
            if not clean:
                self._b_calls[start] += 1
            if runtime == "causal_forcing" and chunk == 0 and getattr(self, "denoising_step_list_first_chunk", None) is not None:
                steps = len(self.denoising_step_list_first_chunk)
            else:
                steps = self._b_step_count
            self.spectral_probe.set_call(RunContext(chunk, -1 if clean else step, clean,
                                                   self.spectral_probe.context.frame_idx0,
                                                   not clean and step == steps - 1))

        def inference(self, *args, **kwargs):
            noise = _arg(args, kwargs, "noise", 0)
            if noise is None or _arg(args, kwargs, "initial_latent", 2) is not None:
                raise ValueError("suite B currently requires text-to-video noise without initial_latent")
            self._b_calls.clear(); self._b_starts.clear(); self._b_clean_pending = False
            self._b_step_count = len(self.denoising_step_list)
            context = dict(self._spectral_context)
            prompt = context.pop("prompt_id", 0)
            self.spectral_probe.reset(prompt_id=prompt, batch_size=noise.shape[0], metadata={**context,
                "local_attn_size": getattr(self.generator.model, "local_attn_size", "unknown"),
                "sink_size": getattr(self.generator.model.blocks[0].self_attn, "sink_size", 0),
                "latent_frames": noise.shape[1]})
            try:
                result = super().inference(*args, **kwargs)
            except BaseException as exc:
                self.spectral_probe.flush(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            else:
                self.spectral_probe.flush()
                return result

    module.CausalInferencePipeline = package.CausalInferencePipeline = SpectralKVPipeline
    return SpectralKVPipeline
