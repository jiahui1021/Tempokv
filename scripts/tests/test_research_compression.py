"""M3 exported records use exact budgets and one frozen 2x2 mask."""
import json
import pytest
from tempokv.research.compression import run_compression
from tempokv.research.diagnostics import write_local
from tempokv.research.io import RunStore
from test_research_diagnostics import fixture_window,small_config,read_rows


def test_compression_records_and_exact_budget_cohorts(tmp_path):
    window,candidates=fixture_window();cfg=small_config(tmp_path,("S3","E7"))
    store=RunStore(cfg,origin="synthetic_m3_cpu")
    information=write_local(window,candidates,store,cfg)
    run_compression(window,candidates,information,store,cfg)
    rows=read_rows(store.close())
    assert set(rows)=={"S3","E7"}
    for retention in (1.,*cfg.compression.retention):
        selected=[row for row in rows['S3'] if float(row['retention'])==retention]
        assert len(selected)==len(cfg.compression.policies)
        assert len({row['actual_retained_tokens'] for row in selected})==1
        assert len({row['logical_bytes'] for row in selected})==1
        for row in selected:
            assert {0,1,10,11}<=set(row['kept_indices'])
            if retention==1:assert float(row['local_attention_relative_error'])==0
    for retention in (1.,*cfg.compression.retention):
        for kb,vb in ((4,4),*cfg.compression.mixed_bits):
            pair={row['strategy']:row for row in rows['E7'] if float(row['retention'])==retention and int(row['k_bits'])==kb and int(row['v_bits'])==vb}
            assert set(pair)=={'A_exact','B_pruning_only','C_quant_only','D_joint'}
            a,b,c,d=[pair[name] for name in ('A_exact','B_pruning_only','C_quant_only','D_joint')]
            assert b['actual_retained_tokens']==d['actual_retained_tokens']
            assert a['logical_bytes']==b['logical_bytes']
            assert c['logical_bytes']==d['logical_bytes']
            assert float(a['local_attention_relative_error'])==0
            assert d['mask_protocol']=='frozen_BF16_mask_for_B_and_D'
            assert d['selection_consistency']['frozen_mask_matches_baseline_selection']
            assert d['selection_consistency']['rank_status'] in ('measured','zero_variance')
            assert d['interaction']==pytest.approx(float(d['local_attention_relative_error'])-float(c['local_attention_relative_error'])-float(b['local_attention_relative_error']))


def test_frozen_offline_cli_restores_partial_pre_capture_and_preserves_origin(tmp_path):
    import subprocess,sys
    from dataclasses import replace
    from test_research_frozen_io import example
    from tempokv.research.frozen_io import save_frozen
    from tempokv.research.offline import run_offline
    from tempokv.research.config import ResearchConfig
    window,freqs,metadata,_=example()
    window=replace(window,context=replace(window.context,protect_recent_frames=0),
        q_post=window.q_post[:,:,:1],k_post=window.k_post[:,:,:1],v=window.v[:,:,:1],k_pre=window.k_pre[:,:,:1])
    path=save_frozen(tmp_path/'event.npz',window,freqs=freqs,metadata={**metadata,'data_origin':'synthetic_m3_cpu'})
    cfg=ResearchConfig.from_dict({'experiments':['S3','E7'],'selection':{'layers':[7],'heads':[3],'chunk_indices':[4],'denoise_step_indices':[1]},
        'sampling':{'spatial_block_hw':[1,2],'max_capture_mb':16,'protect_recent_latent_frames':0},
        'compression':{'retention':[.5],'mixed_bits':[[4,2]],'policies':['attention','frequency','uniform','oracle']},'output':{'root':str(tmp_path/'out')}})
    root=run_offline(cfg,path);rows=read_rows(root)
    assert all(row['data_origin']=='synthetic_offline' for group in rows.values() for row in group)
    assert any(int(row.get('candidate_tokens',0))==6 for row in rows['S3'])
    cfgpath=tmp_path/'config.json';cfgpath.write_text(json.dumps(cfg.to_dict()))
    result=subprocess.run([sys.executable,'scripts/research/run.py','--config',str(cfgpath),'--frozen-input',str(path)],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((root/'summary.json').read_text());assert summary['completed_experiments']==[]


def test_invalid_fake_quant_precision_rejected():
    from tempokv.research.config import ResearchConfig
    for change in ({'bits':[12]},{'mixed_bits':[[4,9]]},{'mixed_bits':[[4]]}):
        with pytest.raises(ValueError):ResearchConfig.from_dict({'compression':change})
