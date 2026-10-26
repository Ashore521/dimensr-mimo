"""The same physical worker and phased config, with exact grammar acceleration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.slump_physical_20261026 import worker,data
if __name__=='__main__':
    settings=data.read(sys.argv[3])
    if settings['time_allocation']!='phased':raise ValueError('Expected the fixed phased configuration.')
    worker.run(Path(sys.argv[1]),data.read(sys.argv[2]),settings)
