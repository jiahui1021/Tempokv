"""Local feature-based E5 reports, separated from generated-video claims."""
from .io import RunStore,atomic_json
from .memory_metrics import evaluate_reappearance,load_reappearance_bundle


def evaluate_bundle(config,path,*,resume=None):
    if set(config.experiments)!={'E5'}:raise ValueError('entity/state feature reports require a separate E5 configuration')
    bundle=load_reappearance_bundle(path)
    result=evaluate_reappearance(**bundle)
    # A supplied feature bundle is its own evidence cohort; it is never promoted
    # to a model-generation run by the file name or a claimed extractor name.
    store=RunStore(config,origin='supplied_video_features',metadata={'feature_bundle':str(path),'feature_metadata':bundle.get('source_metadata')},resume=resume)
    for row in result['rows']:
        store.write('E5',{**row,'strategy':'supplied_entity_state_features','policy_id':row.get('entity_id'),
            'reappearance_score':row.get('matched_reappearance_identity_cosine')})
    return store.close(extra={'feature_summary':result['summary'],'real_generation':'not_run_by_feature_evaluator'})


def synthetic_feature_checks(config):
    """Known null and identity-change controls; no archive-method result."""
    from .memory_scenarios import SCENARIO_NAMES,make_scenario
    store=RunStore(config,origin='synthetic_e5_features',metadata={'model':'known_feature_fixture','checkpoint':'none'})
    for name in SCENARIO_NAMES:
        scenario=make_scenario(name,seed=config.seed)
        for matched,record in (('reappearance',scenario),('static_control',scenario['control'])):
            reference=record['reference_features']
            for control in ('null','identity_flip'):
                candidate=reference.copy()
                if control=='identity_flip':
                    for event in record['annotations']:
                        candidate[event['reappearance_frame'],event['feature_indices']]*=-1
                result=evaluate_reappearance(reference,candidate,record['annotations'],source_metadata={'extractor':'synthetic_known_entity_state_coordinates','data_origin':'synthetic_e5_features'})
                for row in result['rows']:
                    store.write('E5',{**row,'strategy':control,'scenario_id':name,'policy_id':matched,
                        'reappearance_score':row.get('matched_reappearance_identity_cosine'),
                        'interpretation':'feature_evaluator_sanity_control; not_archive_effectiveness'})
        atomic_json(store.root/'scenarios'/f'{name}.json',{
            'scenario_id':name,'prompt':scenario['prompt'],'annotations':scenario['annotations'],
            'matched_control_prompt':scenario['control']['prompt'],'data_origin':'synthetic_e5_features',
            'generated_video_status':'not_run'})
    return store.close(extra={'real_generation':'not_run','evidence_level':'synthetic_numerical_verification_only'})
