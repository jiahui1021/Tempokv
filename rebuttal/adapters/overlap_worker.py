"""Existing generator with completed VAE pixels staged on CPU for overlap."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.adapters import worker
from rebuttal.experiments.memory.vae_output import install_cpu_video_output

original_install=worker.install
def install(config,runtime):
    cls=original_install(config,runtime)
    original_init=cls.__init__
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        install_cpu_video_output(self.vae.model)
    cls.__init__=initialize
    return cls

if __name__=='__main__':
    worker.install=install
    raise SystemExit(worker.main())
