"""Paired inference adapter: replay original pipeline, not reference cache states."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import torch
from .interaction_rollout import make_controller
from .io import atomic_json
from .rollout import RolloutOutput,paired_replay


def _encode_video(path,frames,fps):
    """Encode real RGB frames; stream to system ffmpeg when it is available."""
    executable=shutil.which("ffmpeg")
    if executable:
        height,width=frames.shape[1:3]
        # yuv444p also accepts odd fixture dimensions; real Wan dimensions are
        # even and use the ordinary yuv420p playback format.
        pixel_format="yuv420p" if height%2==0 and width%2==0 else "yuv444p"
        process=subprocess.Popen([executable,"-nostdin","-hide_banner","-loglevel","error",
            "-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","pipe:0",
            "-an","-c:v","libx264","-pix_fmt",pixel_format,"-y",str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for frame in frames:
                process.stdin.write(memoryview(frame.contiguous().numpy()).cast("B"))
            process.stdin.close();process.stdin=None
            _,errors=process.communicate()
        except BaseException:
            process.kill();process.communicate();raise
        if process.returncode:raise RuntimeError(f"ffmpeg video encoding failed: {errors.decode(errors='replace')}")
        return "ffmpeg/libx264"
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("paired video saving needs system ffmpeg or imageio with an FFmpeg backend") from exc
    iio.imwrite(path,frames.numpy(),fps=fps,codec="libx264")
    return "imageio/libx264"


def _save_case_video(path,video,*,writer=None,fps=16):
    """Encode into an owned temporary file and publish without replacement."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    if not isinstance(video,torch.Tensor) or video.ndim!=5 or video.shape[0]!=1 or video.shape[2]!=3:
        raise ValueError("paired video saving expects normalized [1,T,3,H,W] pixels")
    if min(video.shape)<1 or not torch.isfinite(video).all():raise ValueError("paired video pixels must be finite and nonempty")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary_file=tempfile.NamedTemporaryFile(dir=path.parent,suffix=".mp4",delete=False)
    temporary=Path(temporary_file.name);temporary_file.close()
    began=time.perf_counter()
    try:
        frames=video[0].detach().permute(0,2,3,1).float().mul(255).round().clamp(0,255).to(device="cpu",dtype=torch.uint8).contiguous()
        backend=(writer or _encode_video)(temporary,frames,fps)
        if not temporary.is_file() or temporary.stat().st_size==0:raise RuntimeError("video encoder did not create a nonempty file")
        with temporary.open("rb") as handle:os.fsync(handle.fileno())
        digest=hashlib.sha256()
        with temporary.open("rb") as handle:
            for block in iter(lambda:handle.read(1024**2),b""):digest.update(block)
        os.link(temporary,path)
        return {"path":str(path.resolve()),"sha256":digest.hexdigest(),"bytes":path.stat().st_size,
                "frames":frames.shape[0],"height":frames.shape[1],"width":frames.shape[2],"fps":fps,
                "encoder":backend or "injected_writer","encode_and_persist_seconds":time.perf_counter()-began}
    finally:temporary.unlink(missing_ok=True)


def _synchronize(tensor):
    if tensor.device.type=="cuda":torch.cuda.synchronize(tensor.device)


def run_paired_pipeline(pipeline,base_inference,args,kwargs,config,*,video_writer=None):
    probe=pipeline.research_probe
    noise=kwargs.get("noise",args[0] if args else None)
    if noise is None or noise.shape[0]!=1:raise ValueError("paired replay requires one sample per prompt/seed")
    initial=noise.detach().clone()
    desired=kwargs.get("return_latents",args[3] if len(args)>3 else False)
    controller=make_controller(config);probe.counterfactual=controller
    context=dict(pipeline._research_context)
    save_videos=config.output.save_videos
    session_id=uuid.uuid4().hex
    video_records=[]
    video_index=Path(config.output.root)/"paired-videos"/f"{session_id}.json" if save_videos else None
    def persist_index(status,error=None):
        if video_index is not None:
            atomic_json(video_index,{"schema_version":"research.paired_videos.v1","paired_session_id":session_id,
                "status":status,"error":error,"source":config.source,"prompt_context":context,
                "videos":video_records,"cost_contract":"inference, VAE decode and encoding are separate research costs; no production speedup claim"})
    def run(case,recorder):
        if probe.active:probe.finish(extra={"replay_case":controller.case.case_id if controller.case else "reference"})
        pipeline._research_clean=False;pipeline._research_counts.clear();pipeline._research_starts.clear()
        origin="runtime_"+config.source+("_reference" if case is None else "_counterfactual")
        probe.begin(origin=origin,metadata={"paired_phase":"reference" if case is None else "branch",
                                           "case_id":case.case_id if case else None},**context)
        controller.setup_pass(case,recorder)
        call_args=list(args);call_kwargs=dict(kwargs)
        if call_args:call_args[0]=initial.clone();call_kwargs.pop("noise",None)
        else:call_kwargs["noise"]=initial.clone()
        if len(call_args)>3:call_args[3]=True;call_kwargs.pop("return_latents",None)
        else:call_kwargs["return_latents"]=True
        profile=call_kwargs.get("profile",call_args[4] if len(call_args)>4 else False)
        # Ordinary paired runs decode here so the VAE cost can be recorded
        # separately. Preserve the base pipeline's explicit profiling behavior.
        decode_in_base=bool(save_videos and profile)
        if len(call_args)>6:call_args[6]=decode_in_base;call_kwargs.pop("decode_output",None)
        else:call_kwargs["decode_output"]=decode_in_base
        _synchronize(initial);started=time.perf_counter()
        value=base_inference(*call_args,**call_kwargs)
        if not isinstance(value,tuple) or len(value)<2 or not isinstance(value[1],torch.Tensor):raise ValueError("paired pipeline requires explicit final clean latents")
        _synchronize(value[1]);inference_seconds=time.perf_counter()-started
        if save_videos:
            decode_seconds=None
            if decode_in_base:video=value[0]
            else:
                started=time.perf_counter()
                video=pipeline.vae.decode_to_pixel(value[1],use_cache=False)
                video.mul_(.5).add_(.5).clamp_(0,1)
                _synchronize(video);decode_seconds=time.perf_counter()-started
                value=(video,*value[1:])
            case_id=case.case_id if case is not None else "reference"
            label="reference" if case is None else "case-"+re.sub(r"[^A-Za-z0-9_.-]","_",case_id)[:64]+"-"+hashlib.sha256(case_id.encode()).hexdigest()[:12]
            folder=probe.store.root/"videos"/label
            folder.mkdir(parents=True,exist_ok=False)
            artifact=_save_case_video(folder/"sample-0.mp4",video,writer=video_writer)
            record={"schema_version":"research.paired_video.v1","paired_session_id":session_id,"case_id":case_id,
                "paired_phase":"reference" if case is None else "branch","origin":origin,"prompt_context":context,
                "inference_seconds":inference_seconds,"decode_seconds":decode_seconds,
                "decode_timing_scope":"included_in_base_profile" if decode_in_base else "VAE_decode_only",
                "config_signature":getattr(config,"signature",None),"video":artifact}
            manifest=folder/"manifest.json";atomic_json(manifest,record)
            video_records.append({**record,"manifest":str(manifest.resolve())})
        return RolloutOutput(value[1],value)
    def on_branch(case,trace,output,rows):controller.record_branch(probe,case,trace,output,rows)
    count=len(pipeline.denoising_step_list);target=config.intervention.target_chunk
    try:
        result=paired_replay(run,controller.cases,max_capture_mb=config.sampling.max_capture_mb,
            horizons=config.intervention.horizon_chunks,target_chunk=target,record_steps={(target,s) for s in range(count)},on_branch=on_branch)
        controller.finalize(probe)
        persist_index("completed")
        probe.finish(extra={"paired_branches":result.branch_summaries,"unrun_branch_ids":controller.unrun,
                            "reference_frozen_targets":len(controller.targets),"paired_rng":"per_call_and_final_matched",
                            "paired_video_index":str(video_index.resolve()) if video_index is not None else None,
                            "persisted_video_cases":[record["case_id"] for record in video_records],
                            "quality_metrics":"not_run; final latent differences are reference consistency, not quality"})
        final=result.last_output.payload
        return final if desired else final[0]
    except BaseException as exc:
        if video_index is not None and not video_index.exists():persist_index("failed",f"{type(exc).__name__}: {exc}")
        probe.finish(status="failed",extra={"error":f"{type(exc).__name__}: {exc}"});raise
    finally:probe.counterfactual=None
