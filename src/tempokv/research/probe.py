"""Research observer. Only an explicitly installed instance can touch tensors."""
from __future__ import annotations
from dataclasses import replace
import time
import torch
from tempokv.spectral.transforms import grid_shape
from .context import ProbeContext, native_positions
from .io import RunStore

class ResearchProbe:
    def __init__(self, config, *, rank=0, model_metadata=None):
        self.config, self.rank, self.model_metadata = config, rank, dict(model_metadata or {})
        self.active = False
        self.phase = "denoise"; self.chunk_id = 0; self.step_index = 0; self.timestep = None
        self.events = 0
        self.grids = {}

    def begin(self, *, prompt_index=0, seed=None, origin="runtime_bf16", metadata=None):
        if self.active: raise RuntimeError("finish the previous research sample")
        self.store = RunStore(self.config, rank=self.rank, prompt_index=prompt_index, seed=seed, origin=origin,
                              metadata={**self.model_metadata, **(metadata or {})})
        self.prompt_index, self.seed = prompt_index, self.config.seed if seed is None else seed
        self.active, self.events = True, 0
        self.analysis_ms = 0.; self.attention_ms = 0.; self.windows = [];self.started=time.perf_counter()
        return self.store.root

    def set_call(self, *, chunk_id, phase, step_index=None, timestep=None):
        if phase not in {"denoise", "clean_refresh"}: raise ValueError("phase must come from an explicit pipeline event")
        self.chunk_id, self.phase, self.step_index, self.timestep = chunk_id, phase, step_index, timestep

    def selected(self, layer):
        return (self.active and layer in self.config.selection.layers and self.chunk_id in self.config.selection.chunk_indices
                and self.phase == "denoise" and self.step_index in self.config.selection.denoise_step_indices)

    @torch.no_grad()
    def on_projected_native(self, layer, *, q_pre, k_pre, v, grid_sizes, current_start, freqs=None):
        if not self.active: return
        shape = grid_shape(k_pre, grid_sizes)
        if q_pre.shape != k_pre.shape or v.shape != k_pre.shape: raise ValueError("projection Q/K/V shape mismatch")
        self.grids[layer] = (shape, grid_sizes, freqs)

    def context(self, layer, head, sample, q, k, metadata, query_indices=None):
        f, h, w = grid_shape(q, metadata["grid_sizes"])
        positions = native_positions(metadata, k.shape[1])
        query_positions = torch.arange(int(metadata["current_start"]), int(metadata["current_start"]) + q.shape[1], dtype=torch.int64)
        if query_indices is not None: query_positions = query_positions[query_indices.cpu()]
        return ProbeContext(self.prompt_index, self.seed, sample, layer, head, self.chunk_id, self.phase,
            self.step_index if self.phase == "denoise" else None, self.timestep,
            int(metadata["current_start"]), f, h, w, query_positions, positions,
            int(metadata.get("sink_frames", 0)), self.config.sampling.protect_recent_latent_frames, self.config.source)

    @torch.no_grad()
    def on_attention_native(self, layer, *, q_post, k_post, v, metadata, attention_fn):
        if not self.selected(layer) or self.events >= self.config.max_events: return None
        started = time.perf_counter()
        context = self.context(layer, self.config.selection.heads[0], 0, q_post, k_post, metadata)
        self.windows.append({**context.row(), "key_abs_positions": context.key_abs_positions.tolist(),
                             "valid_key_mask": "all_true_no_padding", "dtype": str(q_post.dtype),
                             "kernel": attention_fn.__module__, "rope_precision": metadata.get("rope_precision")})
        self.analysis_ms += (time.perf_counter() - started) * 1000
        t = time.perf_counter(); result = attention_fn(q_post, k_post, v); self.attention_ms += (time.perf_counter()-t)*1000
        self.after_attention(context, result)
        self.events += 1
        return result

    def after_attention(self, context, out):
        if out.ndim != 4: raise ValueError("attention output must be [B,Q,heads,D]")

    def after_model_output(self, context, denoised_pred):
        if not isinstance(denoised_pred, torch.Tensor): raise ValueError("model output must be an explicit denoised prediction")

    def finish(self, *, status="completed", extra=None):
        if not self.active: return None
        try: return self.store.close(status=status, extra={"effective_windows": self.windows, "analysis_ms": self.analysis_ms,
            "attention_host_ms": self.attention_ms, "events_observed": self.events, **(extra or {})})
        finally: self.active = False; self.grids.clear()

# M1 logic is attached below to keep the observation contract above explicit.
def _begin_diagnostics(self):
    from itertools import product
    from .capture import CleanBank, stable_seed
    self.bank=CleanBank(self.config)
    combinations=list(product(self.config.selection.chunk_indices,self.config.selection.denoise_step_indices,
                              self.config.selection.layers,self.config.selection.heads))
    self.schedule=set(sorted(combinations,key=lambda x:stable_seed(self.seed,"event_schedule",x))[:self.config.max_events])
    self.skipped_memory=0;self.last_local={};self.kernel_timings=[];self.oracle_attention_ms=0.;self.production_attention_calls=0
    self.physical_by_layer={};self.peak_resident_kv_bytes=0


@torch.no_grad()
def _capture_clean(self,layer,*,q_pre,k_pre,v,grid_sizes,current_start,freqs=None):
    from .transforms import spatial_layout
    from .context import ProbeContext
    started=time.perf_counter()
    shape=grid_shape(k_pre,grid_sizes)
    if q_pre.shape!=k_pre.shape or v.shape!=k_pre.shape:raise ValueError("projection shapes differ")
    self.grids[layer]=(shape,grid_sizes,freqs);self.current_start=int(current_start)
    if not self.active or self.phase!="clean_refresh" or layer not in self.config.selection.layers:return
    heads=self.config.selection.heads
    if max(heads)>=k_pre.shape[2]:raise ValueError("configured head outside runtime")
    f,h,w=shape;fs=h*w;frame0=int(current_start)//fs
    if frame0>=self.config.sampling.max_clean_frames:return
    # Full spatial grid, selected heads; never use sparse trace tokens for FFT2.
    estimated=(k_pre.numel()+v.numel())//k_pre.shape[2]*len(heads)*k_pre.element_size()
    if self.bank.capture_bytes+estimated>self.config.sampling.max_capture_mb*1024**2:
        self.skipped_memory+=1
        self.store.write("S1",{"prompt_index":self.prompt_index,"seed":self.seed,"layer":layer,"phase":"clean_refresh",
                               "status":"not_measured","reason":"capture_memory_budget"})
        return
    dense={role:spatial_layout(x,grid_sizes)[...,list(heads),:].detach().cpu() for role,x in (("k",k_pre),("v",v))}
    for sample in range(k_pre.shape[0]):
        for fi in range(min(f,self.config.sampling.max_clean_frames-frame0)):
            for hi,head in enumerate(heads):
                records=self.bank.add(sample=sample,layer=layer,head=head,frame=frame0+fi,
                    k=dense["k"][sample,fi,:,:,hi],v=dense["v"][sample,fi,:,:,hi],prompt=self.prompt_index,seed=self.seed)
                if "S1" in self.config.experiments:
                    base={"prompt_index":self.prompt_index,"seed":self.seed,"sample_id":sample,"layer":layer,"head":head,
                          "chunk_index":self.chunk_id,"phase":"clean_refresh","denoise_step_index":None,
                          "denoise_timestep":self.timestep,"current_chunk_start":int(current_start),"grid_t":f,"grid_h":h,"grid_w":w,
                          "valid_token_mask":"all_true_no_padding"}
                    for row in records:self.store.write("S1",{**base,**row})
    self.analysis_ms+=(time.perf_counter()-started)*1000


def _freeze_window(self,layer,head,sample,q,k,v,metadata,query_indices=None):
    from .context import FrozenAttention
    from tempokv.spectral_kv.model_spec import ModelSpec
    from tempokv.spectral_kv.rope3d import rope_apply
    ctx=self.context(layer,head,sample,q,k,metadata,query_indices)
    candidates=self.bank.candidates(ctx)
    query=q[sample:sample+1,:,head:head+1]
    if query_indices is not None:query=query.index_select(1,query_indices.to(q.device))
    key=k[sample:sample+1,:,head:head+1];value=v[sample:sample+1,:,head:head+1]
    exact=None;valid=None
    if candidates and (set(self.config.experiments)&{"E3","S3"} or self.config.output.save_dense_kv):
        exact=torch.zeros_like(key);valid=torch.zeros(key.shape[1],dtype=torch.bool,device=key.device)
        for entry,indices in candidates:
            index=torch.as_tensor(indices,device=key.device);exact[:,index]=entry.k_pre.to(key.device,key.dtype);valid[index]=True
    dim=key.shape[-1];channels=dim//2
    split=(2*(channels-2*(channels//3)),2*(channels//3),2*(channels//3))
    spec=ModelSpec(max(layer+1,1),1,dim,(ctx.grid_h,ctx.grid_w),ctx.grid_t,split)
    precision="float32" if metadata.get("rope_precision") in {"complex64","float32"} else "float64"
    freqs=self.grids[layer][2]
    def rotate(pre,coordinates):return rope_apply(pre,*[coordinates[:,i] for i in range(3)],spec,freqs=freqs,precision=precision)
    def timed_attention(qq,kk,vv):
        began=time.perf_counter()
        devices=[qq.device.index if qq.device.index is not None else torch.cuda.current_device()] if qq.device.type=="cuda" else []
        # Extra analysis calls must not consume the production noise stream,
        # including a backend that internally requests a random generator.
        with torch.random.fork_rng(devices=devices):
            if qq.device.type=="cuda":
                before,after=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);before.record()
                output=self._production_attention(qq,kk,vv);after.record();after.synchronize();duration=before.elapsed_time(after)
            else:output=self._production_attention(qq,kk,vv);duration=(time.perf_counter()-began)*1000
        self.oracle_attention_ms+=duration
        return output
    return FrozenAttention(ctx,query,key,value,timed_attention,exact,rotate,valid),candidates


@torch.no_grad()
def _diagnostic_attention(self,layer,*,q_post,k_post,v,metadata,attention_fn):
    from .capture import stable_seed
    from .diagnostics import write_local
    if not self.active:return None
    if "physical_kv_bytes" in metadata:
        self.physical_by_layer[layer]=int(metadata["physical_kv_bytes"])
        self.peak_resident_kv_bytes=max(self.peak_resident_kv_bytes,sum(self.physical_by_layer.values()))
    scheduled=(self.chunk_id,self.step_index,layer)
    if self.phase!="denoise" or not any(x[:3]==scheduled for x in self.schedule):return None
    self._production_attention=attention_fn
    if self.config.max_gpu_memory_gb is not None and q_post.device.type=="cuda":
        estimated=6*k_post[:, :, :1].numel()*k_post.element_size()
        if torch.cuda.memory_allocated(q_post.device)+estimated>self.config.max_gpu_memory_gb*1024**3:
            self.skipped_memory+=1
            return None
    started=time.perf_counter()
    for sample in range(q_post.shape[0]):
        for head in self.config.selection.heads:
            if (*scheduled,head) not in self.schedule:continue
            if head>=q_post.shape[2]:raise ValueError("head outside actual model")
            rng=__import__("numpy").random.default_rng(stable_seed(self.seed,self.prompt_index,sample,*scheduled,head,"queries"))
            selected=sorted(rng.choice(q_post.shape[1],min(q_post.shape[1],self.config.sampling.queries_per_head),replace=False))
            indices=torch.tensor(selected,dtype=torch.long,device=q_post.device)
            window,candidates=_freeze_window(self,layer,head,sample,q_post,k_post,v,metadata,indices)
            self.last_local[layer,head]=write_local(window,candidates,self.store,self.config)
            if set(self.config.experiments)&{"S3","E7"}:
                from .compression import run_compression
                run_compression(window,candidates,self.last_local[layer,head],self.store,self.config)
            if self.config.output.save_dense_kv:
                from .frozen_io import save_frozen
                path=self.store.root/"snapshots"/(window.context.identity+".npz")
                save_frozen(path,window,max_capture_mb=self.config.sampling.max_capture_mb,freqs=self.grids[layer][2],
                    metadata={"config_signature":self.config.signature,"data_origin":self.store.origin,"rope_precision":metadata.get("rope_precision","native")})
            self.windows.append({**window.context.row(),"key_abs_positions":window.context.key_abs_positions.tolist(),
                                 "dtype":str(q_post.dtype),"valid_key_mask":"all_true_no_padding","reference_kernel":attention_fn.__module__})
            self.events+=1
    self.analysis_ms+=(time.perf_counter()-started)*1000
    began=time.perf_counter()
    if q_post.device.type=="cuda":
        before,after=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);before.record()
        output=attention_fn(q_post,k_post,v);after.record();after.synchronize();duration=before.elapsed_time(after)
    else:output=attention_fn(q_post,k_post,v);duration=(time.perf_counter()-began)*1000
    self.attention_ms+=duration;self.production_attention_calls+=1
    return output  # Same production callback, same complete Q/K/V; exactly one real call.

_original_begin=ResearchProbe.begin
_original_finish=ResearchProbe.finish

def _begin(self,**kwargs):
    path=_original_begin(self,**kwargs);_begin_diagnostics(self);return path

def _finish(self,*,status="completed",extra=None):
    if not self.active:return None
    extra={"skipped_memory_events":self.skipped_memory,"capture_reservoir_bytes":self.bank.capture_bytes,"analysis_oracle_attention_ms":self.oracle_attention_ms,
           "production_attention_measured_calls":self.production_attention_calls,
           "peak_resident_kv_bytes":self.peak_resident_kv_bytes if self.physical_by_layer else None,"physical_kv_layer_ids":sorted(self.physical_by_layer),
           "peak_gpu_allocated_bytes_process":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
           "complete_inference_wall_seconds":time.perf_counter()-self.started,
           "capture_dropped_blocks":self.bank.dropped,"event_schedule":sorted(self.schedule),
           "timing_contract":"analysis pass; attention_ms uses CPU elapsed or synchronized GPU event time; no production speedup inferred",**(extra or {})}
    try:return _original_finish(self,status=status,extra=extra)
    finally:self.bank.blocks.clear();self.bank.trajectories.clear();self.last_local.clear()

ResearchProbe.begin=_begin
ResearchProbe.finish=_finish
ResearchProbe.on_projected_native=_capture_clean
ResearchProbe.on_attention_native=_diagnostic_attention
ResearchProbe.freeze_window=_freeze_window

# M2 replay policy is independent from read-only local diagnostics.
_m1_attention=ResearchProbe.on_attention_native

@torch.no_grad()
def _replay_attention(self,layer,*,q_post,k_post,v,metadata,attention_fn):
    controller=getattr(self,"counterfactual",None)
    if controller is None:return _m1_attention(self,layer,q_post=q_post,k_post=k_post,v=v,metadata=metadata,attention_fn=attention_fn)
    baseline=None
    if controller.case is None:
        baseline=_m1_attention(self,layer,q_post=q_post,k_post=k_post,v=v,metadata=metadata,attention_fn=attention_fn)
    if not self.active or not controller.wants(self,layer):return baseline
    if baseline is None:baseline=attention_fn(q_post,k_post,v)
    return controller.on_attention(self,layer,q_post,k_post,v,metadata,attention_fn,baseline)

ResearchProbe.on_attention_native=_replay_attention
