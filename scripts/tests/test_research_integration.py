from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import torch
import pytest
from tempokv.research.config import ResearchConfig
from tempokv.research.context import ProbeContext, native_positions
from tempokv.research.io import RunStore, atomic_json
from tempokv.research.probe import ResearchProbe

ROOT=Path(__file__).resolve().parents[2]

def toy_config(tmp_path, **updates):
    data={"selection":{"layers":[0],"heads":[0],"chunk_indices":[0,1,2,3],"denoise_step_indices":[0]},
          "sampling":{"protect_recent_latent_frames":0},"output":{"root":str(tmp_path)},**updates}
    return ResearchConfig.from_dict(data)

def test_context_protection_and_sink_rolling():
    positions=native_positions({"current_start":60,"local_start_index":30,"attention_start":0,"frame_seqlen":15,
        "sink_frames":1,"contiguous":False,"absolute_positions":list(range(15))+list(range(45,75))},45)
    assert positions.tolist()==list(range(15))+list(range(45,75))
    ctx=ProbeContext(0,0,0,0,0,4,"denoise",0,0,60,1,3,5,torch.arange(60,75),positions,1,0)
    assert ctx.eligible_mask.nonzero().flatten().tolist()==list(range(15,30))
    assert ctx.row()["protected_token_count"]==30
    with pytest.raises(ValueError): native_positions({"current_start":60,"local_start_index":30,"attention_start":0,"frame_seqlen":15,"sink_frames":1},45)

def test_manifest_atomic_resume_and_legacy_ids(tmp_path):
    cfg=toy_config(tmp_path)
    store=RunStore(cfg,origin="synthetic_m0")
    store.write("E1",{"event_id":"null-control","status":"measured","local_attention_relative_error":0.0})
    root=store.close()
    assert json.loads((root/"summary.json").read_text())["completed_experiments"]==[]
    assert "E1" in json.loads((root/"summary.json").read_text())["synthetic_completed_experiments"]
    resumed=RunStore(cfg,origin="synthetic_m0",resume=root)
    resumed.write("E1",{"event_id":"null-control"}); assert not resumed.counts
    resumed.write("E1",{"event_id":"new-event","status":"not_measured"}); resumed.close()
    assert len(list((root/"E1").glob("events-*.csv")))==2
    with pytest.raises(FileExistsError): atomic_json(root/"manifest.json",{})
    with pytest.raises(ValueError): ResearchConfig.from_dict({"experiments":["B_E1_SPECTRUM"]})

@pytest.mark.parametrize("enabled",[False,True])
def test_real_native_attention_readonly_identity_and_rng(tmp_path,enabled):
    path=ROOT/"experiments/Self-Forcing/wan/modules/causal_model.py"
    parsed=ast.parse(path.read_text()); wanted={"CausalWanSelfAttention","_cache_index","_set_cache_index"}
    nodes=[n for n in parsed.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    spec=importlib.util.spec_from_file_location("research_native_attention",path.with_name("attention.py")); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def rope(x,*args,**kwargs):
        sign=x.new_ones(x.shape[-1]);sign[::2]=-1;return x*sign
    scope={"torch":torch,"nn":torch.nn,"math":__import__("math"),"attention":module.attention,"causal_rope_apply":rope,"causal_rope_apply_long_input":rope}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec"),scope)
    cls=scope["CausalWanSelfAttention"]
    torch.manual_seed(4);base=cls(16,2,local_attn_size=3,sink_size=1,qk_norm=False).to(torch.bfloat16)
    candidate=cls(16,2,local_attn_size=3,sink_size=1,qk_norm=False).to(torch.bfloat16);candidate.load_state_dict(base.state_dict())
    probe=ResearchProbe(toy_config(tmp_path));probe.begin(origin="synthetic_m0")
    if enabled:candidate.research_probe=probe;candidate.research_layer_idx=0
    def cache():return {"k":torch.zeros(1,45,2,8,dtype=torch.bfloat16),"v":torch.zeros(1,45,2,8,dtype=torch.bfloat16),"global_end_index":0,"local_end_index":0,"cache_backend":"bf16"}
    caches=[cache(),cache()]
    with torch.no_grad():
        for frame in range(4):
            x=torch.randn(1,15,16).to(torch.bfloat16);state=torch.get_rng_state().clone();probe.set_call(chunk_id=frame,phase="denoise",step_index=0,timestep=0)
            outputs=[layer(x,seq_lens=None,grid_sizes=torch.tensor([[1,3,5]]),freqs=None,block_mask=None,kv_cache=c,current_start=frame*15) for layer,c in zip((base,candidate),caches)]
            torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0);assert torch.equal(state,torch.get_rng_state())
            for role in ("k","v"):torch.testing.assert_close(caches[0][role],caches[1][role],rtol=0,atol=0)
    root=probe.finish();manifest=json.loads((root/"manifest.json").read_text())
    assert manifest["schema_version"]=="research.v1"
    assert probe.events==(4 if enabled else 0)

def test_install_requires_explicit_enable(monkeypatch):
    path=ROOT/"experiments/Self-Forcing/tempokv_adapter/research_bridge.py"
    spec=importlib.util.spec_from_file_location("research_bridge_gate",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.delenv("RESEARCH_ENABLE",raising=False)
    assert m.install(path.parents[1]) is None
    monkeypatch.setenv("RESEARCH_ENABLE","1");monkeypatch.delenv("RESEARCH_CONFIG_PATH",raising=False)
    with pytest.raises(ValueError,match="RESEARCH_CONFIG_PATH"):m.install(path.parents[1])
