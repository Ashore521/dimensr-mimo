"""Independent fixed development completion batch; no automatic next stage."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.compositional_completion_20261031 import harness
from experiments.completion_development_verified_20261031 import gate
from experiments.completion_development_verified_20261031 import archive_io
from experiments.completion_development_verified_20261031 import worker
STAGE='development'

def budget(): return harness.read(HERE/'BUDGET.json')['start_unix']+LIMIT

def check_candidate_lock(): return harness.check_candidate_lock(sys.modules[__name__])
def check_stage_gate(): return gate.check(sys.modules[__name__])
def matrix(): return harness.matrix(sys.modules[__name__])
def configuration(): return harness.configuration(sys.modules[__name__])
def verify_files(files,root=None): return archive_io.verify_files(files,ROOT if root is None else root)
def preflight(): return harness.preflight(sys.modules[__name__])
def freeze_inputs(): return gate.freeze_inputs(sys.modules[__name__])
def freeze(): return harness.freeze(sys.modules[__name__])
def train(): return harness.train(sys.modules[__name__])

harness.install(sys.modules[__name__],worker)

if __name__=='__main__':
    command=sys.argv[1]
    if command=='run':
        preflight();freeze();train()
        from experiments.completion_development_verified_20261031.evaluate import run
        run()
    elif command in ('preflight','freeze','train'):globals()[command]()
    else:raise ValueError('Use preflight, freeze, train or run.')
