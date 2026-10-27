"""Same observed-data worker, with the newly frozen mutation-unit repair."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.compound_search_20261027.worker import run,read

if __name__=='__main__':
    run(Path(sys.argv[1]),read(Path(sys.argv[2])),read(Path(sys.argv[3])))
