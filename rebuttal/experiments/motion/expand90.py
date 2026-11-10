"""Generate MovieGen50:90 paired videos, then finish the90-video motion study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.adapters.generation import build_job,execute_jobs
from rebuttal.common.manifest import read_json,write_json


def main(args):
    source=Path(args.source).resolve();out=Path(args.output).resolve()
    config=dict(read_json(source/'manifest.json')['config'])
    config.update(start_index=50,end_index=90,num_gpus=2,skip_existing=True,output_dir=str(out))
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='running',phase='generation',pid=os.getpid(),target_groups={'low':30,'medium':30,'high':30},reuse_video_ids=list(range(50)),new_video_ids=list(range(50,90))))
    def generate(mode,gpus):
        job=build_job(config,out/'runs'/mode,mode=mode,name=f'motion90_{mode}_50_90')
        job.environment.update(CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        return execute_jobs([job],resume=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(generate,'bf16','0,1'),pool.submit(generate,'tempokv','2,3')]
        for future in futures:future.result()
    write_json(out/'status.json',dict(status='running',phase='analysis',pid=os.getpid()))
    subprocess.run([sys.executable,str(ROOT/'rebuttal/experiments/motion/expand90_finish.py'),
                    '--source',str(source),'--output',str(out)],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
