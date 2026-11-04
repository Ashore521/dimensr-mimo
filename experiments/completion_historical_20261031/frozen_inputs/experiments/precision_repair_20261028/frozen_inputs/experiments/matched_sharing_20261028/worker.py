"""Run the existing training-only worker with one explicit sharing difference."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.compound_search_20261027 import worker as base
from experiments.native_api_20261026.run import read, write


def run(folder, job, config):
    if job['method'] not in ('random', 'matched'):
        raise ValueError('Outside the frozen sharing comparison.')
    base.run(folder, job, dict(config, sharing_strategy=job['method']))


if __name__ == '__main__':
    run(Path(sys.argv[1]), read(sys.argv[2]), read(sys.argv[3]))
