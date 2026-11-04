"""Only training and validation enter the complete MIMO fit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.compositional_completion_20261031.harness import worker_run,read,write

def run(folder,job,config):
    from experiments.completion_development_verified_20261031 import study
    return worker_run(study,folder,job,config)

if __name__=='__main__':run(Path(sys.argv[1]),read(sys.argv[2]),read(sys.argv[3]))
