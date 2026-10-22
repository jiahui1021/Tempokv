"""Real hooked CPU replay for E5/E7, with own history and paired noise."""
import csv,json
from pathlib import Path
import pytest
from tempokv.research.config import load_config,ResearchConfig
from tempokv.research.synthetic_rollout import run_synthetic_paired
from tempokv.research.memory_rollout import token_mass
import torch


def load_all(root,exp):
    rows=[]
    for path in Path(root).rglob(f'{exp}/events-*.csv'):
        with path.open(newline='') as h:
            for row in csv.DictReader(h):rows.append({**row,**json.loads(row['details_json'])})
    return rows


def test_memory_replay_all_policies_have_equal_accesses_and_exact_full_control(tmp_path):
    data=load_config('scripts/research/configs/synthetic_m4.yaml').to_dict();data['output']['root']=str(tmp_path)
    root=run_synthetic_paired(ResearchConfig.from_dict(data));rows=load_all(tmp_path,'E5')
    assert root.exists()
    control=[r for r in rows if r['strategy']=='full_window' and r.get('clean_latent_relative_error') not in ('',None)]
    assert control and all(float(r['clean_latent_relative_error'])==0 for r in control)
    events={}
    for row in rows:
        if row['metric_scope']!='causal_history_attention_access':continue
        key=tuple(row[x] for x in ('layer','head','chunk_index','denoise_step_index'))
        events.setdefault(key,[]).append(row)
    assert events
    for group in events.values():
        assert {r['strategy'] for r in group}=={'recent_only','uniform_archive','attention_archive','diversity_archive','oracle_archive'}
        assert len({r['actual_retained_tokens'] for r in group})==1
        assert len({r['logical_bytes'] for r in group})==1
        for r in group:assert r['paired_rng']=='matched_per_call_and_final_state'
    assert all(r.get('feature_consistency_status','not_measured').startswith('not_measured') for r in rows)


def test_paired_e7_measures_actual_four_cells_and_clean_interaction(tmp_path):
    data=load_config('scripts/research/configs/synthetic_e7_paired.yaml').to_dict();data['output']['root']=str(tmp_path)
    run_synthetic_paired(ResearchConfig.from_dict(data));rows=load_all(tmp_path,'E7')
    measured=[r for r in rows if r['metric_scope']=='paired_generation' and r['horizon_chunks']=='0']
    cells={r['strategy']:r for r in measured}
    assert {'A_exact','B_pruning_only','C_quant_only','D_joint','paired_clean_interaction'}<=set(cells)
    assert float(cells['A_exact']['clean_latent_relative_error'])==0
    assert cells['B_pruning_only']['actual_retained_tokens']==cells['D_joint']['actual_retained_tokens']
    expected=float(cells['D_joint']['clean_latent_relative_error'])-float(cells['C_quant_only']['clean_latent_relative_error'])-float(cells['B_pruning_only']['clean_latent_relative_error'])
    assert cells['paired_clean_interaction']['interaction']==pytest.approx(expected)


def test_streamed_token_mass_matches_dense_full_softmax():
    g=torch.Generator().manual_seed(43);q=torch.randn(1,7,1,8,generator=g);k=torch.randn(1,13,1,8,generator=g)
    expected=torch.softmax(torch.einsum('bqhd,bkhd->bhqk',q,k)/8**.5,-1).mean((0,1,2))
    torch.testing.assert_close(token_mass(q,k,3),expected,atol=2e-8,rtol=1e-6)


def test_feature_cli_and_known_null_flip_fixtures_remain_synthetic(tmp_path):
    import subprocess,sys
    from tempokv.research.memory_evaluation import synthetic_feature_checks
    from tempokv.research.memory_scenarios import make_scenario
    cfg=load_config('scripts/research/configs/synthetic_m4.yaml');data=cfg.to_dict();data['output']['root']=str(tmp_path)
    root=synthetic_feature_checks(ResearchConfig.from_dict(data));rows=load_all(root,'E5')
    assert len(rows)==16
    assert all(r['data_origin']=='synthetic_e5_features' for r in rows)
    assert all(float(r['reappearance_score'])==pytest.approx(1. if r['strategy']=='null' else -1.) for r in rows)
    scenario=make_scenario('interaction_persistence')
    bundle=tmp_path/'features.json';bundle.write_text(json.dumps({'reference':scenario['reference_features'].tolist(),
        'candidate':scenario['reference_features'].tolist(),'annotations':scenario['annotations'],
        'source_metadata':{'extractor':'synthetic_fixture'}}))
    config=tmp_path/'config.json';config.write_text(json.dumps(data))
    result=subprocess.run([sys.executable,'scripts/research/run.py','--config',str(config),'--e5-features',str(bundle)],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((Path(result.stdout.strip())/'summary.json').read_text())
    assert summary['feature_summary']['exploratory'] and summary['feature_summary']['quality_status']=='unknown'
    assert summary['completed_experiments']==[]
