import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from dimensr_mimo import DimenSRMIMORegressor, SearchConfig, structure
from dimensr_mimo._single.units import ZERO, unit, scale
from dimensr_mimo.generation import SystemGrammar


def problem():
    length = unit([1,0,0,0,0,0,0])
    time = unit([0,0,1,0,0,0,0])
    cfg = SearchConfig(max_nodes=15, max_depth=6, operators=('add','mul','div','square'))
    units = ((time,length), (length,scale(length,2)), (ZERO,scale(time,-1)))
    X = np.random.default_rng(97).uniform([.2,.4], [1.1,1.4], (28,2))
    a,b = X.T
    Y = np.column_stack((1.4*b/(.8+(1.2*a)**2), .6*b*b/(.8+(1.2*a)**2)))
    return cfg,units,X,Y


def test_implicit_unit_legal_rational_initial_values_and_limits():
    cfg,units,X,Y = problem()
    for j,g in enumerate(SystemGrammar(*units,cfg).plain):
        seeds,stats = structure.structured_seeds(g,X,Y[:,j],12,compound=True,implicit=True)
        assert 0 < stats['implicit_denominators'] <= 64
        assert stats['implicit_linear_subproblems'] > 0 and stats['implicit_valid_proposals'] > 0
        assert stats['implicit_retained_indices'] and len(seeds)<=12
        errors = []
        for tree,c in seeds:
            g.trace(tree)
            assert tree.size<=cfg.max_nodes and tree.depth<=cfg.max_depth
            errors.append(np.sqrt(np.mean((tree.evaluate(X,c)-Y[:,j])**2))/Y[:,j].std())
        best = int(np.argmin(errors))
        assert errors[best]<1e-10 and best in stats['implicit_retained_indices']


def test_disabled_implicit_matches_frozen_compound_bytes_behavior():
    path = Path(__file__).resolve().parents[1]/'experiments/balanced_closure_20261028/source/src/dimensr_mimo/structure.py'
    spec = importlib.util.spec_from_file_location('dimensr_mimo._frozen_compound',path)
    old = importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg,units,X,Y = problem();g = SystemGrammar(*units,cfg).plain[0]
    before,counts = old.structured_seeds(g,X,Y[:,0],12,compound=True)
    after,new_counts = structure.structured_seeds(g,X,Y[:,0],12,compound=True)
    assert counts==new_counts and [t for t,c in before]==[t for t,c in after]
    for (_,a),(_,b) in zip(before,after):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('offset', ['zero_at_sample','nonfinite'])
def test_implicit_rejects_invalid_fitted_denominator(monkeypatch,offset):
    original = np.linalg.lstsq
    def solve(*args,**kwargs):
        result = original(*args,**kwargs)
        local = sys._getframe(1).f_locals
        if local.get('fit_offset',False):
            c = result[0].copy();index = local['base'].shape[1]-1
            c[index] = (-local['denominator'][2][0]*local['norms'][index]
                        if offset=='zero_at_sample' else np.inf)
            result = (c,)+result[1:]
        return result
    monkeypatch.setattr(np.linalg,'lstsq',solve)
    cfg,units,X,Y = problem();g = SystemGrammar(*units,cfg).plain[0]
    with np.errstate(all='ignore'):
        seeds,stats = structure.structured_seeds(g,X,Y[:,0],12,compound=True,implicit=True)
    assert stats['implicit_denominator_rejections']>0
    assert stats['implicit_retained_indices']==[]
    assert all(np.isfinite(t.evaluate(X,c)).all() for t,c in seeds)


def test_implicit_seeds_receive_new_joint_evaluations_and_reload(tmp_path):
    cfg,units,X,Y = problem()
    cfg.initialization='implicit';cfg.linear_seeds=False
    cfg.population_size=6;cfg.epochs=1;cfg.gp_generations=1;cfg.selection_generations=2
    cfg.max_evaluations=80;cfg.constant_max_nfev=8;cfg.constant_restarts=1
    cfg.polish_candidates=0;cfg.min_candidates=2
    model = DimenSRMIMORegressor(cfg,*units).fit(X,Y,X_val=X[::3],Y_val=Y[::3],
        input_bounds=[[.2,1.1],[.4,1.4]])
    row = next(r for r in model.history_ if r['phase']=='initialization')
    assert 0 < row['implicit_seed_valid'] <= row['implicit_seed_new_evaluations'] <= row['implicit_seed_jobs']
    assert model.diagnostics_['evaluations']<=cfg.max_evaluations
    assert all(s['implicit_retained_indices'] for s in model.diagnostics_['initialization_diagnostics'])
    model.save(tmp_path/'implicit.json')
    loaded = DimenSRMIMORegressor.load(tmp_path/'implicit.json')
    np.testing.assert_array_equal(model.predict(X),loaded.predict(X))
    assert loaded.config.initialization=='implicit'
