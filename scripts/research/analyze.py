#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from tempokv.research.analysis import analyze

def main(argv=None):
    p=argparse.ArgumentParser(description="Rebuild research.v1 tables/figures without rerunning generation")
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path)
    a=p.parse_args(argv);print(analyze(a.input,a.output_dir));return 0
if __name__=="__main__":raise SystemExit(main())
