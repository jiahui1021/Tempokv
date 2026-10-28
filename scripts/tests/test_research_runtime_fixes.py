"""CPU regression checks for runtime paths, immutable E5 events and paired videos."""
from __future__ import annotations

import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tempokv.research.config import ResearchConfig
from tempokv.research.io import RunStore, environment
from tempokv.research.memory_evaluation import evaluate_bundle
from tempokv.research import pipeline_replay
from tempokv.research.rollout import ReplayCase


ROOT=Path(__file__).resolve().parents[2]


def _script(relative,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _model_inputs(tmp_path):
    checkpoint=tmp_path/"checkpoint.pt";checkpoint.touch()
    prompts=tmp_path/"prompts.txt";prompts.write_text("fixture\n")
    model=tmp_path/"selected-model";model.mkdir()
    return checkpoint,prompts,model


def _wan_path():
    path=ROOT/"experiments/Self-Forcing/utils/wan_wrapper.py"
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_wan_model_path")
    scope={"os":os,"Path":Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec"),scope)
    return scope["_wan_model_path"]()


def test_research_model_root_reaches_actual_wan_resolver_and_launch_manifest(tmp_path,monkeypatch):
    runner=_script("scripts/research/run.py","research_fixed_runner")
    checkpoint,prompts,model=_model_inputs(tmp_path);captured=[]
    monkeypatch.setenv("FORCING_WAN_MODEL_DIR","/wrong/inherited-model")
    monkeypatch.setattr(runner,"environment",lambda:{})
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(runner.subprocess,"run",lambda command,**kwargs:captured.append((command,kwargs)) or SimpleNamespace(returncode=0))
    assert runner.main(["--config",str(ROOT/"scripts/research/configs/m2_self_forcing.yaml"),
        "--checkpoint",str(checkpoint),"--prompts",str(prompts),"--model-root",str(model),"--output-dir",str(tmp_path/"out")])==0
    child=captured[0][1]["env"]
    assert child["WAN_MODEL_DIR"]==child["FORCING_WAN_MODEL_DIR"]==str(model.resolve())
    with monkeypatch.context() as context:
        context.setenv("FORCING_WAN_MODEL_DIR",child["FORCING_WAN_MODEL_DIR"])
        assert _wan_path()==str(model.resolve())
    manifest=json.loads(next((tmp_path/"out").rglob("launch_manifest.json")).read_text())
    assert manifest["model_resolution"]["model_root"]==manifest["runtime_inputs"]["model_root"]==str(model.resolve())


@pytest.mark.parametrize("runtime",["self_forcing","causal_forcing"])
def test_suite_b_model_root_contract_and_immutable_launch_manifest(tmp_path,monkeypatch,runtime):
    runner=_script("scripts/paper_experiments/spectral/run.py",f"suite_b_fixed_{runtime}")
    checkpoint,prompts,model=_model_inputs(tmp_path);captured=[]
    monkeypatch.setenv("FORCING_WAN_MODEL_DIR","/wrong/inherited-model")
    monkeypatch.setattr(runner.subprocess,"run",lambda command,**kwargs:captured.append((command,kwargs)) or SimpleNamespace(returncode=0))
    assert runner.main(["--config",str(ROOT/"scripts/paper_experiments/spectral/configs/B_E1_SPECTRUM.yaml"),
        "--runtime",runtime,"--checkpoint",str(checkpoint),"--prompts",str(prompts),"--model-root",str(model),"--output",str(tmp_path/"out")])==0
    command,options=captured[0]
    if runtime=="self_forcing":
        assert options["env"]["FORCING_WAN_MODEL_DIR"]==str(model.resolve())
    else:
        assert command[command.index("--model-root")+1]==str(model.resolve())
        # Verify the CF adapter passes the explicit path on to its actual API.
        source=(ROOT/"experiments/Causal-Forcing/causal_forcing/tempokv_adapter/generate.py").read_text()
        assert '"--model_root",' in source
    manifest=json.loads(next((tmp_path/"out"/"launches").glob("*.json")).read_text())
    assert manifest["model_resolution"]["model_root"]==str(model.resolve())
    assert manifest["model_resolution"]["contract"]==("FORCING_WAN_MODEL_DIR" if runtime=="self_forcing" else "--model_root")


def test_runtime_environment_manifest_uses_the_actual_model_variable(tmp_path,monkeypatch):
    monkeypatch.setenv("FORCING_WAN_MODEL_DIR",str(tmp_path))
    monkeypatch.setenv("WAN_MODEL_DIR","/stale/alias")
    assert environment()["resolved_model_root"]==str(tmp_path.resolve())


def _e5_config(tmp_path):
    return ResearchConfig.from_dict({"experiments":["E5"],"output":{"root":str(tmp_path)}})


def _feature_bundle(path,extra=False):
    reference=np.tile([1.,0.,1.,0.],(15,1)).tolist()
    def event(sequence,entity,first=0,disappear=2,reappear=5):
        return {"sequence_id":sequence,"entity_id":entity,"first_frame":first,"disappearance_frame":disappear,
                "reappearance_frame":reappear,"feature_indices":[0,1]}
    annotations=[event("sequence-a","entity-a"),event("sequence-b","entity-a"),event("sequence-a","entity-b"),
                 event("sequence-a","entity-a",7,9,12)]
    if extra:annotations.append(event("sequence-c","entity-a"))
    path.write_text(json.dumps({"reference":reference,"candidate":reference,"annotations":annotations}))
    return path


def _rows(root):
    rows=[]
    for path in sorted(Path(root).glob("E5/events-*.csv")):
        with path.open(newline="") as handle:rows.extend(csv.DictReader(handle))
    return rows


def test_e5_sequences_entities_and_repeated_visits_all_have_unique_persisted_ids(tmp_path):
    config=_e5_config(tmp_path/"reports")
    root=evaluate_bundle(config,_feature_bundle(tmp_path/"features.json"))
    rows=_rows(root)
    assert len(rows)==4 and len({row["event_id"] for row in rows})==4
    details=[json.loads(row["details_json"]) for row in rows]
    assert len({(row["sequence_id"],row["entity_id"],row["first_frame"],row["reappearance_frame"]) for row in details})==4


def test_e5_resume_rebuilds_unique_counts_and_keeps_prior_parts_and_summaries(tmp_path):
    config=_e5_config(tmp_path/"reports");bundle=_feature_bundle(tmp_path/"features.json")
    root=evaluate_bundle(config,bundle)
    original={path:path.read_bytes() for path in root.rglob("*") if path.is_file()}
    evaluate_bundle(config,bundle,resume=root)
    assert len(_rows(root))==4
    second=_feature_bundle(tmp_path/"features-with-new-sequence.json",extra=True)
    evaluate_bundle(config,second,resume=root)
    assert len(_rows(root))==5
    summaries=[json.loads(path.read_text()) for path in root.glob("summary-*.json")]
    assert any(summary["row_counts"]=={"E5":5} and summary["row_counts_this_session"]=={"E5":1} for summary in summaries)
    for path,content in original.items():assert path.read_bytes()==content


def test_legacy_feature_id_is_readable_and_semantically_deduplicated_on_resume(tmp_path):
    config=_e5_config(tmp_path)
    row={"strategy":"features","sequence_id":"s","entity_id":"e","first_frame":0,
         "disappearance_frame":2,"reappearance_frame":5,"feature_indices":[0,1]}
    store=RunStore(config,origin="synthetic_e5_features")
    store.write("E5",{**row,"event_id":"legacy-feature-id"});root=store.close()
    resumed=RunStore(config,origin="synthetic_e5_features",resume=root)
    assert "legacy-feature-id" in resumed.seen and not resumed.counts
    resumed.write("E5",row)
    resumed.write("E5",{**row,"first_frame":7,"disappearance_frame":9,"reappearance_frame":12})
    resumed.close()
    rows=_rows(root)
    assert len(rows)==2 and "legacy-feature-id" in {saved["event_id"] for saved in rows}
    assert json.loads(resumed.summary_path.read_text())["row_counts"]=={"E5":2}


def test_nonfeature_event_ids_keep_the_exact_legacy_hash(tmp_path):
    config=ResearchConfig.from_dict({"experiments":["E1"],"output":{"root":str(tmp_path)}})
    from tempokv.research import io
    raw={"schema_version":"research.v1","experiment_id":"E1","data_origin":"synthetic_fixture","status":"measured",
         "layer":0,"head":1,"strategy":"drop"}
    legacy={name:raw.get(name) for name in io._IDENTITY_LABELS}
    legacy["config_signature"]=config.signature
    legacy["details_labels"]={name:None for name in io._DETAIL_LABELS}
    expected=hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()[:32]
    store=RunStore(config,origin="synthetic_fixture");store.write("E1",{"layer":0,"head":1,"strategy":"drop"});root=store.close()
    with next((root/"E1").glob("*.csv")).open(newline="") as handle:assert next(csv.DictReader(handle))["event_id"]==expected


def _paired_fixture(tmp_path,monkeypatch,*,save_videos,writer=None,positional=False):
    class Probe:
        def __init__(self):self.active=False;self.index=0;self.finished=[]
        def begin(self,**kwargs):
            self.active=True;self.index+=1
            self.store=SimpleNamespace(root=tmp_path/f"run-{self.index}");self.store.root.mkdir(parents=True)
        def finish(self,**kwargs):self.active=False;self.finished.append(kwargs)
    class Controller:
        def __init__(self):self.case=None;self.targets={};self.unrun=[]
        def setup_pass(self,case,recorder):self.case,self.recorder=case,recorder
        def cases(self,trace):return [ReplayCase("branch-A",()),ReplayCase("branch B/2",())]
        def record_branch(self,*args):pass
        def finalize(self,*args):pass
    class VAE:
        def __init__(self):self.calls=0
        def decode_to_pixel(self,latents,use_cache=False):
            assert use_cache is False;self.calls+=1;return latents.clone()
    controller=Controller();probe=Probe();vae=VAE();calls=[]
    pipeline=SimpleNamespace(research_probe=probe,vae=vae,_research_context={"prompt_index":3,"seed":7},
        _research_counts=Counter(),_research_starts={},denoising_step_list=[900])
    config=SimpleNamespace(source="bf16",signature="fixture-v1",output=SimpleNamespace(root=str(tmp_path),save_videos=save_videos),
        sampling=SimpleNamespace(max_capture_mb=8),intervention=SimpleNamespace(target_chunk=0,horizon_chunks=(0,)))
    def base(noise,text_prompts,initial_latent=None,return_latents=False,profile=False,low_memory=False,decode_output=True):
        assert decode_output is False and return_latents is True
        label=controller.case.case_id if controller.case else "reference"
        calls.append(label);controller.recorder.record_chunk(0,0,3)
        value={"reference":-.8,"branch-A":-.2,"branch B/2":.6}[label]
        return None,torch.full_like(noise,value)
    monkeypatch.setattr(pipeline_replay,"make_controller",lambda config:controller)
    noise=torch.zeros(1,3,3,4,6)
    args=(noise,["fixture"],None,True,False,False,True) if positional else ()
    kwargs={} if positional else {"noise":noise,"text_prompts":["fixture"],"return_latents":True}
    returned=pipeline_replay.run_paired_pipeline(pipeline,base,args,kwargs,config,video_writer=writer)
    return returned,probe,vae,calls


def test_default_paired_run_neither_decodes_nor_creates_video_artifacts(tmp_path,monkeypatch):
    def forbidden(*args):raise AssertionError("video writer called while disabled")
    returned,probe,vae,calls=_paired_fixture(tmp_path,monkeypatch,save_videos=False,writer=forbidden)
    assert returned[0] is None and vae.calls==0 and len(calls)==3
    assert not (tmp_path/"paired-videos").exists() and not list(tmp_path.rglob("*.mp4"))


def test_injected_writer_checks_each_case_scheduling_and_positional_api(tmp_path,monkeypatch):
    encoded=[]
    def writer(path,frames,fps):
        encoded.append((frames.clone(),fps));path.write_bytes(b"explicit-test-writer")
        return "injected_test_writer_no_codec_claim"
    returned,probe,vae,calls=_paired_fixture(tmp_path,monkeypatch,save_videos=True,writer=writer,positional=True)
    assert calls==["reference","branch-A","branch B/2"] and vae.calls==len(encoded)==3
    torch.testing.assert_close(returned[0],torch.full_like(returned[0],.8))
    index=json.loads(next((tmp_path/"paired-videos").glob("*.json")).read_text())
    assert index["status"]=="completed" and [row["case_id"] for row in index["videos"]]==calls
    assert len({row["video"]["path"] for row in index["videos"]})==3
    for record in index["videos"]:
        assert record["decode_seconds"] is not None and record["inference_seconds"]>=0
        manifest=json.loads(Path(record["manifest"]).read_text())
        assert manifest["case_id"]==record["case_id"] and manifest["video"]["encoder"]=="injected_test_writer_no_codec_claim"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),reason="system ffmpeg/ffprobe unavailable; injected-writer scheduling is tested separately")
def test_real_cpu_h264_encoding_and_decoding_for_reference_and_every_branch(tmp_path,monkeypatch):
    returned,probe,vae,calls=_paired_fixture(tmp_path,monkeypatch,save_videos=True)
    index=json.loads(next((tmp_path/"paired-videos").glob("*.json")).read_text())
    levels=[]
    for record in index["videos"]:
        path=Path(record["video"]["path"])
        info=json.loads(subprocess.check_output([shutil.which("ffprobe"),"-v","error","-select_streams","v:0",
            "-show_entries","stream=codec_name,width,height,nb_frames","-of","json",str(path)],text=True))["streams"][0]
        assert info["codec_name"]=="h264" and (info["width"],info["height"],int(info["nb_frames"]))==(6,4,3)
        decoded=subprocess.check_output([shutil.which("ffmpeg"),"-v","error","-i",str(path),"-f","rawvideo","-pix_fmt","rgb24","pipe:1"])
        pixels=np.frombuffer(decoded,dtype=np.uint8).reshape(3,4,6,3)
        levels.append(float(pixels.mean()))
        assert record["video"]["encoder"]=="ffmpeg/libx264"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record["video"]["sha256"]
    assert levels==pytest.approx([25.5,102.,204.],abs=4)
    assert len(calls)==vae.calls==3 and returned[0].shape==(1,3,3,4,6)


def test_video_publication_never_overwrites_and_cleans_failed_temporary_file(tmp_path):
    path=tmp_path/"video.mp4";video=torch.zeros(1,2,3,2,2);calls=[]
    def writer(path,frames,fps):calls.append(path);path.write_bytes(b"first-owned-file");return "fixture"
    pipeline_replay._save_case_video(path,video,writer=writer)
    original=path.read_bytes()
    with pytest.raises(FileExistsError):pipeline_replay._save_case_video(path,video,writer=writer)
    assert path.read_bytes()==original and len(calls)==1
    def empty(path,frames,fps):return "empty-fixture"
    with pytest.raises(RuntimeError,match="nonempty"):pipeline_replay._save_case_video(tmp_path/"failed.mp4",video,writer=empty)
    assert not (tmp_path/"failed.mp4").exists() and list(tmp_path.glob("*.mp4"))==[path]
