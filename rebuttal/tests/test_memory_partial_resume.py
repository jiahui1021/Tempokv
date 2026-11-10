import json
from types import SimpleNamespace
import pytest
from rebuttal.experiments.memory.generation import execute_jobs


@pytest.mark.parametrize('matching_identity',[True,False])
def test_failed_job_resume_preserves_videos_and_original_log(tmp_path,monkeypatch,matching_identity):
    folder=tmp_path/'output';(folder/'videos').mkdir(parents=True)
    video=folder/'videos/0_0.mp4';video.write_bytes(b'existing-result')
    log=folder/'run.log';log.write_text('original failure evidence\n')
    for name in ('checkpoint.pt','runtime.yaml','prompts.txt'):
        (tmp_path/name).touch()
    job=dict(folder=str(folder),name='partial',identity='same-settings',
        config_path=str(folder/'worker_config.json'),
        worker_config=dict(model_root=str(tmp_path)),command=['python',
        '--checkpoint_path',str(tmp_path/'checkpoint.pt'),'--config_path',str(tmp_path/'runtime.yaml'),
        '--data_path',str(tmp_path/'prompts.txt')])
    (folder/'job.json').write_text(json.dumps(dict(identity='same-settings' if matching_identity else 'different',status='failed')))
    commands=[]
    def failed_process(command,**kwargs):
        commands.append(command);kwargs['stdout'].write('retry failure evidence\n')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr('rebuttal.experiments.memory.generation.subprocess.run',failed_process)
    with pytest.raises(RuntimeError,match='generation failed' if matching_identity else 'Partial videos preserved'):
        execute_jobs(dict(jobs=[job],missing_configuration=[]),resume=True)
    assert video.read_bytes()==b'existing-result'
    assert log.read_text().startswith('original failure evidence\n')
    if matching_identity:
        assert '--skip_existing' in commands[0]
        assert log.read_text().endswith('retry failure evidence\n')
    else:
        assert not commands
