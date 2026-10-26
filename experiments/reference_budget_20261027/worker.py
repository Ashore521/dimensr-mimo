"""Audit every generated system before the unchanged fitting evaluator."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import in_bounds
from experiments.slump_physical_20261026 import worker,data

if __name__=='__main__':
    folder=Path(sys.argv[1]);settings=data.read(sys.argv[3])
    if settings['reference_budget']!='expanded':raise ValueError('Expected expanded references.')
    counts=dict(neural_proposals=0,other_proposals=0,invalid_expanded_systems=0)
    original=SystemGrammar.generate
    def checked(self,rng,policy=None):
        result=original(self,rng,policy)
        counts['neural_proposals' if policy is not None else 'other_proposals']+=1
        if not in_bounds(result[0],self.cfg):
            counts['invalid_expanded_systems']+=1
            raise ValueError('Generated system violates expanded bounds.')
        return result
    SystemGrammar.generate=checked
    worker.run(folder,data.read(sys.argv[2]),settings)
    data.write(folder/'GENERATION_CHECK.json',dict(**counts,passed=counts['neural_proposals']>0 and counts['invalid_expanded_systems']==0))
