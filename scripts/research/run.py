#!/usr/bin/env python3
"""Independent S1-S3/E1-E7 runner with model-free dry-run and CPU verification."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from tempokv.research.config import ResearchConfig,load_config
from tempokv.research.io import atomic_json,environment


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--experiments")
    parser.add_argument("--sampling-config",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--max-events",type=int)
    parser.add_argument("--max-gpu-memory-gb",type=float)
    parser.add_argument("--max-prompts",type=int,default=2)
    parser.add_argument("--seeds",default=None)
    parser.add_argument("--checkpoint",type=Path,default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--model-root",type=Path,default=os.environ.get("FORCING_WAN_MODEL_DIR") or os.environ.get("WAN_MODEL_DIR"))
    parser.add_argument("--prompts",type=Path,default=os.environ.get("PROMPTS"))
    parser.add_argument("--codec-config",type=Path)
    parser.add_argument("--k-params",type=Path);parser.add_argument("--v-params",type=Path)
    parser.add_argument("--num-gpus",type=int,default=1)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--frozen-input",type=Path,help="bounded full-window research.frozen.v1 snapshot; CPU local diagnostics")
    parser.add_argument("--e5-features",type=Path,help="local declared entity/state feature and annotation bundle")
    parser.add_argument("--e5-evidence",type=Path,help="real paired evidence and explicit exploration rationale for long E5 generation")
    args=parser.parse_args(argv);config=load_config(args.config)
    data=config.to_dict()
    if args.experiments:data["experiments"]=args.experiments.split(",")
    if args.sampling_config:
        extra=json.loads(args.sampling_config.read_text());unknown=set(extra)-{"selection","sampling"}
        if unknown:parser.error("sampling-config may contain only selection and sampling")
        for key,value in extra.items():data[key].update(value)
    if args.output_dir:data["output"]["root"]=str(args.output_dir.resolve())
    else:data["output"]["root"]=str(Path(data["output"]["root"]).expanduser().resolve())
    if args.max_events is not None:data["max_events"]=args.max_events
    if args.max_gpu_memory_gb is not None:data["max_gpu_memory_gb"]=args.max_gpu_memory_gb
    config=ResearchConfig.from_dict(data)
    seeds=[int(x) for x in args.seeds.split(",")] if args.seeds else [config.seed]
    if not seeds or any(seed<0 for seed in seeds) or args.max_prompts<1 or args.num_gpus<1:parser.error("invalid seed/prompt/GPU count")
    plan={"schema_version":"research.v1","config":config.to_dict(),"config_signature":config.signature,
          "environment":environment(),"max_prompts":args.max_prompts,"seeds":seeds,
          "enable_contract":"RESEARCH_ENABLE=1 only inside this explicitly selected runner",
          "reference":"native production callback on complete effective K/V window",
          "milestones":{"M0":"audit_and_read_only_identity","M1":"S1_S2_E1_local_E3_E4",
              "M2":"paired_E1_clean_E2_E6","M3":"S3_frozen_and_paired_E7","M4":"E5_causal_archives_and_declared_feature_evaluation"},
          "runtime_inputs":{key:str(getattr(args,key).expanduser().resolve()) if getattr(args,key) else "not provided" for key in ("checkpoint","model_root","prompts")},
          "model_resolution":{"model_root":str(args.model_root.expanduser().resolve()) if args.model_root else None,
                              "runtime_environment_variable":"FORCING_WAN_MODEL_DIR"},
          "gpu_generation":"not_run"}
    if args.dry_run:
        print(json.dumps(plan,indent=2,ensure_ascii=False));return 0
    if args.e5_features:
        from tempokv.research.memory_evaluation import evaluate_bundle
        print(evaluate_bundle(config,args.e5_features));return 0
    if args.synthetic and args.frozen_input:parser.error("choose synthetic or frozen input")
    if args.frozen_input:
        from tempokv.research.offline import run_offline
        print(run_offline(config,args.frozen_input));return 0
    if args.synthetic:
        from tempokv.research.synthetic import run_synthetic
        results=[run_synthetic(replace(config,seed=seed)) for seed in seeds]
        result=results[0] if len(results)==1 else [str(path) for path in results]
        if 'E5' in config.experiments:
            from tempokv.research.memory_evaluation import synthetic_feature_checks
            print(json.dumps({'replay':str(result),'feature_controls':str(synthetic_feature_checks(config))}));return 0
        print(result);return 0
    if 'E5' in config.experiments:
        from tempokv.research.memory_scenarios import heavy_benchmark_gate
        gate=heavy_benchmark_gate(args.e5_evidence) if args.e5_evidence else {'allowed':False,'reasons':['real paired evidence and explicit exploration rationale are absent']}
        if not gate['allowed']:parser.error('E5 long generation evidence gate: '+json.dumps(gate,ensure_ascii=False))
    import torch
    if not torch.cuda.is_available():parser.error("CUDA unavailable: use --synthetic or --dry-run; real generation not run")
    for key in ("checkpoint","prompts"):
        value=getattr(args,key)
        if value is None or not value.is_file():parser.error(f"{key} needs an existing file")
    if args.model_root is None or not args.model_root.is_dir():parser.error("model-root needs an existing local Wan model")
    root=Path(config.output.root)/"launches"/uuid.uuid4().hex[:12];root.mkdir(parents=True)
    config_path=root/"config.json";atomic_json(config_path,config.to_dict());atomic_json(root/"launch_manifest.json",plan)
    runtime=ROOT/"experiments/Self-Forcing"
    for seed in seeds:
        output=root/f"seed-{seed}"
        env={**os.environ,"RESEARCH_ENABLE":"1","RESEARCH_CONFIG_PATH":str(config_path.resolve()),
             "RESEARCH_MODEL":str(args.checkpoint.resolve()),"WAN_MODEL_DIR":str(args.model_root.resolve()),
             "FORCING_WAN_MODEL_DIR":str(args.model_root.resolve()),
             "SELF_FORCING_CACHE_ROPE_PRECISION":"complex64"}
        mode="bf16"
        if config.source=="tempokv":
            if any(getattr(args,key) is None or not getattr(args,key).is_file() for key in ("codec_config","k_params","v_params")):
                parser.error("source=tempokv E7 requires existing --codec-config/--k-params/--v-params")
            mode="tempokv";env.update({"TEMPOKV_CONFIG_PATH":str(args.codec_config.resolve()),"TEMPOKV_K_PARAMS_PATH":str(args.k_params.resolve()),
                "TEMPOKV_V_PARAMS_PATH":str(args.v_params.resolve()),"TEMPOKV_NUM_FRAMES":str(config.latent_frames),"TEMPOKV_HORIZON_FRAMES":str(config.latent_frames)})
        else:
            env={key:value for key,value in env.items() if not key.startswith("TEMPOKV_")}
        command=[sys.executable,"-m","torch.distributed.run","--standalone",f"--nproc_per_node={args.num_gpus}",str(runtime/"tempokv_adapter/worker.py"),"--mode",mode,"--",
            "--config_path",str(runtime/"configs/self_forcing_dmd.yaml"),"--checkpoint_path",str(args.checkpoint.resolve()),
            "--data_path",str(args.prompts.resolve()),"--output_folder",str(output),"--num_output_frames",str(config.latent_frames),
            "--num_samples","1","--seed",str(seed),"--start_index","0","--end_index",str(args.max_prompts),"--save_with_index","--use_ema",
            "--local_attn_size",str(config.local_attn_size),"--sink_size",str(config.sink_size)]
        if not config.output.save_videos:command.append("--skip_video_save")
        result=subprocess.run(command,cwd=runtime,env=env,check=False)
        if result.returncode:return result.returncode
    return 0

if __name__=="__main__":raise SystemExit(main())
