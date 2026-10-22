"""Install an optional spectral probe on the native, contiguous BF16 runtime."""
from __future__ import annotations

from collections import Counter
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from tempokv.spectral import SpectralProbe, load_config


def _argument(args, kwargs, name, index, default=None):
    return kwargs[name] if name in kwargs else args[index] if len(args) > index else default


def install(runtime_root: str | Path) -> type:
    runtime_root = Path(runtime_root).resolve()
    missing = [name for name in ("SPECTRAL_CONFIG", "SPECTRAL_OUTPUT_DIR") if not os.environ.get(name)]
    if missing:
        raise ValueError(f"spectral mode requires {', '.join(missing)}")
    config = load_config(os.environ["SPECTRAL_CONFIG"])
    output_dir = Path(os.environ["SPECTRAL_OUTPUT_DIR"]).expanduser().resolve()
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    package = importlib.import_module("pipeline")
    module = importlib.import_module("pipeline.causal_inference")
    base = module.CausalInferencePipeline
    if getattr(base, "_spectral_installed", False):
        return base
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime_root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    class SpectralCausalInferencePipeline(base):
        _spectral_installed = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if getattr(self, "_uses_pre_rope_cache", False):
                raise ValueError("spectral mode requires native BF16; pre-RoPE/quantized caches are unsupported")
            blocks = self.generator.model.blocks
            if max(config.sampling.layers) >= len(blocks):
                raise ValueError("configured spectral layer is outside the runtime layer count")
            if any(block.self_attn.sink_size != 0 for block in blocks):
                raise ValueError("spectral mode requires SINK_SIZE=0; no implicit sink position mapping")
            if any(max(config.sampling.heads) >= block.self_attn.num_heads for block in blocks):
                raise ValueError("configured spectral head is outside the runtime head count")
            self.spectral_probe = SpectralProbe(
                config, output_dir, rank=int(os.environ.get("RANK", "0")),
                metadata={"git_commit": commit, "model": os.environ.get("SPECTRAL_MODEL", "unknown"),
                          "runtime": str(runtime_root), "source": "native_bf16"})
            self._spectral_context = {}
            self._spectral_clean_pending = False
            self._spectral_denoise_calls = Counter()
            for layer, block in enumerate(blocks):
                block.self_attn.spectral_layer_idx = layer
                block.self_attn.spectral_probe = self.spectral_probe
            self.generator.register_forward_pre_hook(self._spectral_generator_start, with_kwargs=True)

        def set_spectral_context(self, *, prompt_id, **metadata):
            self._spectral_context = {"prompt_id": prompt_id, **metadata}

        def note_clean_refresh(self) -> None:
            super().note_clean_refresh()
            self._spectral_clean_pending = True

        def _spectral_generator_start(self, _module, args, kwargs):
            # WanDiffusionWrapper.forward current_start is argument 5; callers
            # normally pass it by keyword. No timestep is consulted.
            current_start = int(_argument(args, kwargs, "current_start", 5, 0))
            phase = "clean" if self._spectral_clean_pending else "denoise"
            self._spectral_clean_pending = False
            denoise_call = self._spectral_denoise_calls[current_start]
            if phase == "denoise":
                self._spectral_denoise_calls[current_start] += 1
            self.spectral_probe.set_call(phase=phase, denoise_call=denoise_call)

        def inference(self, *args: Any, **kwargs: Any):
            noise = _argument(args, kwargs, "noise", 0)
            initial_latent = _argument(args, kwargs, "initial_latent", 2)
            if initial_latent is not None:
                raise ValueError("spectral pilot supports text-to-video only; initial context lacks clean announcements")
            if noise is None:
                raise ValueError("spectral inference requires noise")
            self._spectral_clean_pending = False
            self._spectral_denoise_calls.clear()
            context = dict(self._spectral_context)
            prompt_id = context.pop("prompt_id", self.spectral_probe._sequence)
            self.spectral_probe.reset(prompt_id=prompt_id, batch_size=noise.shape[0], metadata=context)
            try:
                result = super().inference(*args, **kwargs)
            except BaseException as exc:
                self.spectral_probe.flush(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            else:
                self.spectral_probe.flush()
                return result

    SpectralCausalInferencePipeline.__name__ = base.__name__
    module.CausalInferencePipeline = SpectralCausalInferencePipeline
    package.CausalInferencePipeline = SpectralCausalInferencePipeline
    return SpectralCausalInferencePipeline
