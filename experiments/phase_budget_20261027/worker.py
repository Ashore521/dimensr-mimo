"""The archived physical-data worker with only phased timing enabled."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.slump_physical_20261026 import worker,data

if __name__=='__main__':
    config=data.read(sys.argv[3])
    if config['time_allocation']!='phased':raise ValueError('This frozen arm requires phased timing.')
    worker.run(Path(sys.argv[1]),data.read(sys.argv[2]),config)
