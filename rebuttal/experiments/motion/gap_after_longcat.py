"""Insert the user-prioritized LongCat expansion ahead of supplemental generation."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json


def main(args):
    longcat=Path(args.longcat);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    while not (longcat/'gpu_release.json').exists():
        state=read_json(longcat/'status.json') if (longcat/'status.json').exists() else {}
        write_json(out/'status.json',dict(status='waiting',phase='after_user_prioritized_longcat8',pid=os.getpid(),
            longcat_phase=state.get('phase'),longcat_status=state.get('status')))
        time.sleep(15)
    assert read_json(longcat/'gpu_release.json')['status']=='complete'
    os.execv(sys.executable,[sys.executable,str(ROOT/'rebuttal/experiments/motion/gap_overlap.py'),
        '--parent',args.parent,'--output',args.output])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('longcat','parent','output'):parser.add_argument('--'+field,required=True)
    main(parser.parse_args())
