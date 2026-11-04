from dataclasses import replace
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import SearchConfig
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import Builder
from dimensr_mimo.evaluation import Evaluator
from dimensr_mimo._single.units import ZERO,unit
from experiments.shared_discovery_20261026.seeds import typed_seeds,nonlinear_seeds,original_seeds


def test_original_seed_prefix_and_enabled_operator_bounds():
    cfg=SearchConfig(max_nodes=15,max_depth=6)
    g=SystemGrammar((ZERO,ZERO),(ZERO,ZERO),(ZERO,),cfg).plain[0]
    old=list(original_seeds(g,np.random.default_rng(3)))
    new=list(typed_seeds(g,np.random.default_rng(3)))
    assert new[:len(old)]==old and len(new)>len(old)
    extra=list(nonlinear_seeds(g))
    assert len(extra)==8
    assert all(t.size<=15 and t.depth<=6 for t in extra)
    no_trig=SystemGrammar((ZERO,),(ZERO,), (ZERO,),replace(cfg,operators=('add','mul','square'))).plain[0]
    assert list(nonlinear_seeds(no_trig))==[]


def test_dimensional_input_requires_inverse_unit_parameter():
    length=unit([1,0,0,0,0,0,0]);inverse=unit([-1,0,0,0,0,0,0])
    cfg=SearchConfig(max_nodes=15,max_depth=6)
    absent=SystemGrammar((length,),(ZERO,), (ZERO,),cfg).plain[0]
    assert list(nonlinear_seeds(absent))==[]
    present=SystemGrammar((length,),(ZERO,), (ZERO,inverse),cfg).plain[0]
    trees=list(nonlinear_seeds(present))
    assert len(trees)==4
    for tree in trees:present.trace(tree)


def test_existing_fitter_can_fit_new_atoms_without_more_budget():
    cfg=SearchConfig(max_nodes=15,max_depth=6,constant_restarts=1,constant_max_nfev=25)
    g=SystemGrammar((ZERO,),(ZERO,ZERO),(ZERO,),cfg).plain[0]
    atoms=[t for t in nonlinear_seeds(g) if t.token in ('sin','cos')]
    assert len(atoms)==2
    b=Builder();b.roots=[b.tree(t) for t in atoms];system,warm=b.finish()
    X=np.linspace(-.8,.8,70)[:,None]
    Y=np.column_stack((np.sin(1.7*X[:,0]),np.cos(1.7*X[:,0])))
    c=Evaluator(X,Y,cfg,input_bounds=[[-.8,.8]])(system,warm)
    assert c is not None and max(c.errors)<1e-7
    np.testing.assert_allclose(c.constants,[1.7,1.7],atol=1e-6)
