"""Research-only installer with explicit opt-in and detachable hook handles."""
from __future__ import annotations
from collections import Counter
import importlib
import os
from pathlib import Path
import sys
from tempokv.research.config import load_config
from tempokv.research.probe import ResearchProbe


def argument(args, kwargs, name, index, default=None):
    return kwargs[name] if name in kwargs else args[index] if len(args) > index else default


def install(runtime_root):
    if os.environ.get("RESEARCH_ENABLE") != "1": return None
    value = os.environ.get("RESEARCH_CONFIG_PATH")
    if not value: raise ValueError("RESEARCH_ENABLE=1 requires RESEARCH_CONFIG_PATH")
    config = load_config(value)
    runtime_root = Path(runtime_root).resolve()
    if str(runtime_root) not in sys.path: sys.path.insert(0, str(runtime_root))
    package, module = importlib.import_module("pipeline"), importlib.import_module("pipeline.causal_inference")
    base = module.CausalInferencePipeline
    if getattr(base, "_research_installed", False): return base
    if getattr(base, "_spectral_installed", False) or getattr(base, "_spectral_kv_installed", False):
        raise ValueError("research.v1 and legacy A/B probes must run separately")

    class ResearchPipeline(base):
        _research_installed = True
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            blocks = self.generator.model.blocks
            if max(config.selection.layers) >= len(blocks): raise ValueError("research layer selection exceeds runtime")
            if any(max(config.selection.heads) >= block.self_attn.num_heads for block in blocks): raise ValueError("research head selection exceeds runtime")
            if max(config.selection.denoise_step_indices) >= len(self.denoising_step_list): raise ValueError("denoise step index exceeds actual schedule")
            chunks = (config.latent_frames + self.num_frame_per_block - 1) // self.num_frame_per_block
            if max(config.selection.chunk_indices) >= chunks: raise ValueError("selected chunk outside actual rollout")
            if config.intervention.paired_rollout:
                if config.intervention.target_chunk>=chunks or config.intervention.target_step>=len(self.denoising_step_list):
                    raise ValueError("paired target outside the actual chunk/denoise schedule")
                if any(len(policy)!=len(self.denoising_step_list) for policy in config.intervention.e6_step_policies):
                    raise ValueError("E6 policy must cover the actual denoise schedule")
            self.research_probe = ResearchProbe(config, rank=int(os.environ.get("RANK", 0)),
                model_metadata={"model": "Self-Forcing/Wan", "checkpoint": os.environ.get("RESEARCH_MODEL", "unknown"),
                                "runtime": "self_forcing", "layer_count":len(blocks),"head_count":blocks[0].self_attn.num_heads,
                                "window":config.local_attn_size,"sink_size":config.sink_size})
            if bool(getattr(self,"_uses_pre_rope_cache",False)) != (config.source=="tempokv"):
                raise ValueError("research source differs from the installed BF16/TempoKV backend")
            self._research_detached=False
            self._research_context = {}; self._research_clean = False; self._research_counts = Counter(); self._research_starts = {}
            for layer, block in enumerate(blocks):
                block.self_attn.research_probe = self.research_probe
                block.self_attn.research_layer_idx = layer
            self._research_handles = [self.generator.register_forward_pre_hook(self._start, with_kwargs=True),
                                      self.generator.register_forward_hook(self._end, with_kwargs=True)]

        def set_research_context(self, *, prompt_index, seed, **metadata): self._research_context = {"prompt_index": prompt_index, "seed": seed, **metadata}
        def note_clean_refresh(self):
            super().note_clean_refresh(); self._research_clean = True
        def _start(self, _module, args, kwargs):
            if not self.research_probe.active: return
            start = int(argument(args, kwargs, "current_start", 5, 0))
            if start not in self._research_starts: self._research_starts[start] = len(self._research_starts)
            clean = self._research_clean; self._research_clean = False
            step = self._research_counts[start]
            if not clean: self._research_counts[start] += 1
            timestep = argument(args, kwargs, "timestep", 2)
            timestep_value = int(timestep.flatten()[0]) if timestep is not None else None
            self.research_probe.set_call(chunk_id=self._research_starts[start], phase="clean_refresh" if clean else "denoise",
                                         step_index=None if clean else step, timestep=timestep_value)
            controller=getattr(self.research_probe,"counterfactual",None)
            if controller is not None and controller.recorder is not None:
                controller.recorder.record_rng("before_forward",(self._research_starts[start],"clean_refresh" if clean else "denoise",None if clean else step))
        def _end(self, _module, args, kwargs, output):
            if self.research_probe.active and isinstance(output, tuple) and len(output) >= 2:
                self.research_probe.after_model_output({"chunk_id": self.research_probe.chunk_id, "step_index": self.research_probe.step_index,
                    "phase": self.research_probe.phase}, output[1])
                controller=getattr(self.research_probe,"counterfactual",None)
                if controller is not None and controller.recorder is not None:
                    probe=self.research_probe
                    controller.recorder.record_rng("after_forward",(probe.chunk_id,probe.phase,probe.step_index))
                    if probe.phase=="denoise":
                        controller.recorder.record_output(probe.chunk_id,probe.step_index,output[1])
                        if probe.grids:
                            (frames,h,w),_,_=next(iter(probe.grids.values()))
                            controller.recorder.record_chunk(probe.chunk_id,probe.current_start//(h*w),frames)
        def inference(self, *args, **kwargs):
            if self._research_detached:return super().inference(*args,**kwargs)
            if argument(args, kwargs, "initial_latent", 2) is not None: raise ValueError("core research currently supports text-to-video without initial latent")
            if config.intervention.paired_rollout:
                if config.source!="bf16":raise ValueError("paired mechanism experiments require BF16 reference; TempoKV E7 observation is separate")
                from tempokv.research.pipeline_replay import run_paired_pipeline
                return run_paired_pipeline(self,super().inference,args,kwargs,config)
            self._research_clean = False; self._research_counts.clear(); self._research_starts.clear()
            self.research_probe.begin(origin="runtime_"+config.source,**self._research_context)
            try: result = super().inference(*args, **kwargs)
            except BaseException as exc:
                self.research_probe.finish(status="failed", extra={"error": f"{type(exc).__name__}: {exc}"}); raise
            self.research_probe.finish(); return result
        def detach_research(self):
            for handle in self._research_handles: handle.remove()
            for block in self.generator.model.blocks: block.self_attn.research_probe = None
            self._research_handles.clear()
            self._research_detached=True

    module.CausalInferencePipeline = package.CausalInferencePipeline = ResearchPipeline
    return ResearchPipeline
