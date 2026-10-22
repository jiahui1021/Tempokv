from pathlib import Path
import csv
import json
import pytest
from tempokv.research.config import ResearchConfig
from tempokv.research.synthetic_rollout import run_synthetic_paired


def config(tmp_path,experiments=("E1","E2"),**intervention):
    return ResearchConfig.from_dict({"experiments":list(experiments),"frames":45,"local_attn_size":6,"max_events":2,
        "selection":{"layers":[0],"heads":[0,1],"chunk_indices":[1,2],"denoise_step_indices":[0,1]},
        "sampling":{"queries_per_head":3,"spatial_block_hw":[1,1],"protect_recent_latent_frames":0,"max_history_blocks_per_event":3,"max_capture_mb":16,"max_pairs":1},
        "spectral":{"temporal_windows":[2],"temporal_positions":2},"compression":{"bits":[2,4]},
        "intervention":{"paired_rollout":True,"target_layer":0,"target_head":0,"target_chunk":1,"target_step":1,
            "kind":"gaussian","role":"v","strengths":[.1],"horizon_chunks":[1,2],"max_branches":3,**intervention},
        "output":{"root":str(tmp_path)}})

def records(root,experiment):
    result=[]
    for path in Path(root).rglob(f"{experiment}/events-*.csv"):
        with path.open(newline="") as handle:result.extend(csv.DictReader(handle))
    return result

def test_hooked_paired_surrogate_null_and_single_event_propagation(tmp_path):
    run_synthetic_paired(config(tmp_path))
    manifests=[json.loads(path.read_text()) for path in tmp_path.rglob("manifest.json")]
    assert all(m["data_origin"].startswith("synthetic") for m in manifests)
    e1=[r for r in records(tmp_path,"E1") if r["clean_latent_relative_error"]]
    assert e1 and any(float(r["clean_latent_relative_error"])==0 for r in e1 if r["strategy"]=="none")
    changed=[r for r in e1 if r["strategy"]=="gaussian"]
    assert changed and float(changed[0]["local_attention_relative_error"])>0
    assert float(changed[0]["model_output_relative_error"])>0 and float(changed[0]["clean_latent_relative_error"])>0
    future=[r for r in records(tmp_path,"E2") if r["strategy"]=="gaussian" and r["clean_latent_relative_error"]]
    assert {int(r["horizon_chunks"]) for r in future}=={0,1,2}
    assert all(float(r["clean_latent_relative_error"])>0 for r in future)
    summaries=[json.loads(path.read_text()) for path in tmp_path.rglob("summary.json")]
    branches=[b for s in summaries for b in s.get("paired_branches",[])]
    assert branches and all(b["paired_rng"] for b in branches)
    assert all(hit["hits"]==1 for b in branches for hit in b["hits"])


def test_real_e6_complete_schedules_keep_equal_cost_and_match_null(tmp_path):
    cfg=config(tmp_path,("E6",),kind="none",max_branches=8,e6_step_policies=[[1,1],[0,1],[.5,.5]])
    run_synthetic_paired(cfg)
    rows=records(tmp_path,"E6")
    policies=[r for r in rows if r["strategy"]=="complete_step_policy"]
    assert len(policies)==3
    details=[json.loads(r["details_json"]) for r in policies]
    full=next((r,d) for r,d in zip(policies,details) if d["retentions"]==[1,1])
    assert float(full[0]["clean_latent_relative_error"])==0
    budgeted=[d["total_token_accesses"] for d in details if d["retentions"]!=[1,1]]
    assert len(set(budgeted))==1
    assert {r["denoise_step_index"] for r in rows if r["strategy"]=="single_step"}=={"0","1"}


def test_zero_strength_stays_bitwise_equal_after_clean_refresh(tmp_path):
    run_synthetic_paired(config(tmp_path,strengths=[0]))
    changed=[r for r in records(tmp_path,"E1") if r["strategy"]=="gaussian" and r["clean_latent_relative_error"]]
    assert changed and all(float(r["clean_latent_relative_error"])==0 and float(r["local_attention_relative_error"])==0 for r in changed)


def test_installed_real_attention_bridge_replays_base_pipeline_cache(tmp_path,monkeypatch):
    import ast,importlib.util,sys,types
    import torch
    root=Path(__file__).resolve().parents[2]
    path=root/"experiments/Self-Forcing/wan/modules/causal_model.py"
    parsed=ast.parse(path.read_text());names={"CausalWanSelfAttention","_cache_index","_set_cache_index","causal_rope_apply"}
    nodes=[n for n in parsed.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    spec=importlib.util.spec_from_file_location("research_bridge_native_attention",path.with_name("attention.py"));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    from tempokv.runtime.rope import causal_rope_apply_long_input
    scope={"torch":torch,"nn":torch.nn,"math":__import__("math"),"attention":mod.attention,"causal_rope_apply_long_input":causal_rope_apply_long_input}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec"),scope)
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.self_attn=scope["CausalWanSelfAttention"](16,2,local_attn_size=6,qk_norm=False).to(torch.bfloat16)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.blocks=torch.nn.ModuleList([Block()]);self.local_attn_size=6
    class Generator(torch.nn.Module):
        def __init__(self):super().__init__();self.model=Model()
        def forward(self,noisy_image_or_video,conditional_dict,timestep,kv_cache=None,crossattn_cache=None,current_start=0):
            x=noisy_image_or_video.reshape(1,6,16).to(torch.bfloat16)
            freqs=torch.polar(torch.ones(64,4,dtype=torch.float64),torch.arange(64,dtype=torch.float64)[:,None]*torch.tensor([1,.01,1,1],dtype=torch.float64))
            out=self.model.blocks[0].self_attn(x,seq_lens=None,grid_sizes=torch.tensor([[1,2,3]]),freqs=freqs,block_mask=None,kv_cache=kv_cache[0],current_start=current_start)
            pred=.55*noisy_image_or_video+.25*out.float().reshape_as(noisy_image_or_video)
            return pred,pred
    class Base:
        def __init__(self):self.generator=Generator();self.num_frame_per_block=1;self.denoising_step_list=torch.tensor([900,0]);self._uses_pre_rope_cache=False
        def note_clean_refresh(self):pass
        def inference(self,noise,text_prompts,initial_latent=None,return_latents=False,decode_output=True):
            cache={"k":torch.zeros(1,36,2,8,dtype=torch.bfloat16),"v":torch.zeros(1,36,2,8,dtype=torch.bfloat16),"global_end_index":0,"local_end_index":0,"cache_backend":"bf16"}
            outputs=[]
            for chunk in range(noise.shape[1]):
                value=noise[:,chunk:chunk+1].clone()
                for step in range(2):
                    _,pred=self.generator(value,{},torch.tensor(step),kv_cache=[cache],current_start=chunk*6)
                    if step==0:value=pred+.05*torch.randn_like(pred)
                outputs.append(pred);self.note_clean_refresh();self.generator(pred,{},torch.tensor(44),kv_cache=[cache],current_start=chunk*6)
            return torch.zeros(0),torch.cat(outputs,dim=1)
    package,module=types.ModuleType("pipeline"),types.ModuleType("pipeline.causal_inference")
    package.CausalInferencePipeline=module.CausalInferencePipeline=Base
    monkeypatch.setitem(sys.modules,"pipeline",package);monkeypatch.setitem(sys.modules,"pipeline.causal_inference",module)
    cfg=config(tmp_path,horizon_chunks=[1,2]);cfgpath=tmp_path/"config.json";cfgpath.write_text(json.dumps(cfg.to_dict()))
    monkeypatch.setenv("RESEARCH_ENABLE","1");monkeypatch.setenv("RESEARCH_CONFIG_PATH",str(cfgpath))
    bridgepath=root/"experiments/Self-Forcing/tempokv_adapter/research_bridge.py"
    spec=importlib.util.spec_from_file_location("research_actual_bridge",bridgepath);bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    pipe=bridge.install(root/"experiments/Self-Forcing")();pipe.set_research_context(prompt_index=3,seed=5)
    torch.manual_seed(5)
    with torch.no_grad():result=pipe.inference(torch.randn(1,12,2,3,2,8),["synthetic"],return_latents=True)
    assert result[1].shape==(1,12,2,3,2,8)
    output=[r for r in records(tmp_path,"E1") if r["strategy"]=="gaussian" and r["clean_latent_relative_error"]]
    assert output and float(output[0]["local_attention_relative_error"])>0
    pipe.detach_research();assert not pipe._research_handles and pipe.generator.model.blocks[0].self_attn.research_probe is None
