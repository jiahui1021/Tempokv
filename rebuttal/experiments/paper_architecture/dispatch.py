"""Finite CPU fit and exclusive-GPU measurement job; never preempts generation."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--python',required=True)
    p.add_argument('--gpu',default='3');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**kwargs):
        save(root/'job_status.json',dict(stage=stage,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    command=[a.python,str(root/'run.py'),'--config',str(root/'config.json'),'--output',str(root),'--stage']
    for stage in ['plan','fit']:
        status(stage)
        with (root/f'{stage}.log').open('ab') as log:
            code=subprocess.run(command+[stage],env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:status('failed',failed_stage=stage,returncode=code);return code
    # The GPU is shared with the user's ongoing experiment queue. A clean
    # process list and two consecutive idle observations are required.
    status('waiting_for_exclusive_gpu',physical_gpu=a.gpu)
    consecutive=0
    while consecutive<2:
        text=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        memory,utilization=[int(v.strip()) for v in text.split(',')]
        consecutive=consecutive+1 if memory<512 and utilization==0 else 0
        status('waiting_for_exclusive_gpu',physical_gpu=a.gpu,memory_mib=memory,utilization=utilization)
        if consecutive<2:time.sleep(30)
    env['CUDA_VISIBLE_DEVICES']=a.gpu
    status('measuring',physical_gpu=a.gpu)
    with (root/'measure.log').open('ab') as log:
        code=subprocess.run(command+['measure','--device','cuda:0'],env=env,cwd=root,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
    status('complete' if code==0 else 'failed',failed_stage=None if code==0 else 'measure',returncode=code)
    return code


if __name__=='__main__':raise SystemExit(main())
