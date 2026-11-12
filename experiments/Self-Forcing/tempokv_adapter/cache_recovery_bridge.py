"""Explicit worker mode for the paired cache-recovery pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tempokv.research.cache_recovery import RecoveryRollout, RecoverySettings
from tempokv.research.io import atomic_json


def install():
    import pipeline as package
    from pipeline import causal_inference as module

    launch = json.loads(Path(os.environ["RESEARCH_CACHE_RECOVERY_LAUNCH"]).read_text())
    settings = RecoverySettings(**launch["settings"])
    output = Path(launch["output_dir"])
    base = module.CausalInferencePipeline

    class CacheRecoveryPipeline(base):
        def set_research_context(self, *, prompt_index, seed):
            self.recovery_context = {"prompt_index": prompt_index, "seed": seed}

        def inference(
            self, noise, text_prompts, initial_latent=None, return_latents=False,
            profile=False, low_memory=False, decode_output=True,
        ):
            if initial_latent is not None or profile or low_memory or decode_output or not return_latents:
                raise ValueError("cache recovery runs non-offloaded latent-only text-to-video inference")
            context = self.recovery_context
            path = output / f"prompt-{context['prompt_index']}-seed-{context['seed']}.json"
            if path.exists():
                raise FileExistsError(path)
            print(f"[cache-recovery] {path.stem} bits={settings.bits}", flush=True)
            rollout = RecoveryRollout(self, super().inference, settings, noise, text_prompts)
            evidence, reference = rollout.run()
            atomic_json(path, {
                **evidence, **context, "status": "completed",
                "data_origin": "runtime_bf16",
                "model": "Self-Forcing/Wan",
                "runtime": launch["runtime"],
                "commit_sha": launch["environment"]["commit_sha"],
                "rope_precision": os.environ["SELF_FORCING_CACHE_ROPE_PRECISION"],
                "gpu_name": torch.cuda.get_device_name(noise.device),
            })
            print(f"[cache-recovery] completed {path.name}", flush=True)
            return None, reference

    module.CausalInferencePipeline = package.CausalInferencePipeline = CacheRecoveryPipeline
