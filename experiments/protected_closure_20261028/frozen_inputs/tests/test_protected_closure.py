from copy import deepcopy
from dataclasses import asdict, replace
import json
import time
import numpy as np
import pytest

from dimensr_mimo import DimenSRMIMORegressor, SearchConfig
from dimensr_mimo.acceptance import select
from dimensr_mimo.evaluation import Evaluator
from dimensr_mimo.graph import Builder, System, Vertex
from dimensr_mimo.selection import FIELDS, digest
from dimensr_mimo.sharing import protected_closure
from dimensr_mimo.structure import operator
from dimensr_mimo.syntax import Node


def limits():
    return dict(nrmse=[.03]*2, shared_graph_size=24, independent_parameters=6,
        expanded_total_nodes=30, maximum_output_depth=6)


def test_closure_joint_fit_protection_and_original_states():
    X = np.linspace(.1, .8, 40)[:, None]
    for same in (True, False):
        b = Builder()
        for op in (('sin', 'cos') if same else ('mul', 'mul')):
            tree = operator('mul', Node('c0'), Node('x0'))
            b.roots.append(b.tree(operator(op, tree) if same else tree, [1.7 if same else len(b.roots)+1.]))
        graph, warm = b.finish()
        Y = graph.evaluate(X, warm)
        evaluator = Evaluator(X, Y, SearchConfig(max_evaluations=60, constant_restarts=1))
        parent = evaluator(graph, warm)
        before = deepcopy(parent.to_dict())
        pool, report = protected_closure([parent], evaluator)
        assert pool[0].to_dict() == parent.to_dict() == before
        assert report['new_evaluations'] > 0 and report['valid'] > 0
        assert evaluator.calls <= 60 and report['new_evaluations'] <= 48
        if same:
            assert len(pool) > 1 and report['protected'] > 0
            assert all(c.system.size < parent.system.size and np.all(c.errors < 1e-8) for c in pool[1:])
        else:
            assert pool == [parent] and report['protection_rejections'] > 0
        evaluator.deadline = time.monotonic()-1
        stopped, record = protected_closure([parent], evaluator)
        assert stopped == [parent] and record['new_evaluations'] == 0
        evaluator.deadline = None
        evaluator.cfg.max_evaluations = evaluator.calls
        assert protected_closure([parent], evaluator)[1]['new_evaluations'] == 0


def selection_source():
    large = System((Vertex('x0'), Vertex('c0', parameter=0), Vertex('mul', (0, 1))), (2, 2))
    small = System((Vertex('x0'),), (0, 0))
    candidates = []
    for graph in (large, small):
        candidates.append(dict(system=graph.to_dict(), constants=[1.] if graph.n_parameters else [],
            errors=[.01]*2, validation_errors=[.01]*2, complexity=float(graph.size),
            rademacher=0., rademacher_complexity=0., violation=0., score=.01))
    config = SearchConfig(sharing_strategy='closure', final_selection='precision', acceptance=limits())
    source = dict(n_outputs=2, config=asdict(config), candidates=candidates,
        feature_units=[[0]*7], target_units=[[0]*7]*2, constant_units=[[0]*7], reference_train=[1., 1.],
        diagnostics=dict(sharing_closure=dict(base_candidate_count=1,
            base_candidate_sha256=digest([{k:candidates[0][k] for k in FIELDS}]))))
    rehash(source)
    return source


def rehash(source):
    candidates = source['candidates']
    source['diagnostics']['frozen_candidate_sha256'] = digest([{k:c[k] for k in FIELDS} for c in candidates])
    anchor = min(candidates, key=lambda c:(c['violation'], c['score'], c['complexity']))
    source['best'] = {k:deepcopy(anchor[k]) for k in FIELDS}


@pytest.mark.parametrize('field', ['errors', 'validation_errors'])
def test_closure_selection_retains_base_when_one_output_degrades(field):
    source = selection_source()
    source['candidates'][1][field] = [.009, .015]
    rehash(source)
    result = select(source, limits())
    assert result['post_selection']['candidate_index'] == 0
    assert result['post_selection']['base_candidate_index'] == 0
    assert result['post_selection']['status'] == 'accepted'
    source['candidates'][1][field] = [.0105, .0105]
    rehash(source)
    assert select(source, limits())['post_selection']['candidate_index'] == 1
    source['diagnostics']['sharing_closure']['base_candidate_count'] = 2
    with pytest.raises(ValueError, match='base pool'):
        select(source, limits())


def test_closure_unmet_gates_fall_back_to_unchanged_base():
    source = selection_source()
    result = select(source, dict(limits(), shared_graph_size=0))
    assert result['post_selection']['status'] == 'not_met'
    assert result['post_selection']['fallback_method'] == 'base_precision'
    assert result['post_selection']['candidate_index'] == 0
    for config in (SearchConfig(sharing_strategy='closure'),
                   SearchConfig(sharing_strategy='closure', final_selection='precision', acceptance=limits(), use_sharing=False)):
        with pytest.raises(ValueError, match='Closure'):
            config.validate()


def test_closure_base_matches_random_and_model_reload_checks_provenance(tmp_path):
    X = np.linspace(.2, 1.2, 28)[:, None]
    Y = np.column_stack((np.sin(1.3*X[:, 0]), np.cos(1.3*X[:, 0])))
    cfg = SearchConfig(population_size=6, epochs=2, gp_generations=1, selection_generations=2,
        max_evaluations=100, min_candidates=5, max_nodes=9, constant_max_nfev=8,
        constant_restarts=1, polish_candidates=1, polish_max_nfev=12, random_state=1601,
        final_selection='precision', acceptance=limits())
    def fit(config):
        return DimenSRMIMORegressor(config).fit(X, Y, X_val=X[::3], Y_val=Y[::3], input_bounds=[[.2, 1.2]])
    base, extended = fit(cfg), fit(replace(cfg, sharing_strategy='closure'))
    count = extended.diagnostics_['sharing_closure']['base_candidate_count']
    assert base.candidates_ == extended.candidates_[:count]
    assert extended.diagnostics_['sharing_closure']['base_search_fingerprint'] == base.search_fingerprint_
    for name in ('operations', 'changed', 'novel', 'successful'):
        assert base.diagnostics_[name] == extended.diagnostics_[name]
    assert extended.post_selection_['base_choice_sha256'] == digest(base.best_.to_dict())
    path = tmp_path/'closure.json'
    extended.save(path)
    loaded = DimenSRMIMORegressor.load(path)
    assert json.loads(json.dumps(loaded.to_dict())) == json.loads(json.dumps(extended.to_dict()))
    np.testing.assert_array_equal(loaded.predict(X), extended.predict(X))
    tampered = json.loads(path.read_text())
    tampered['diagnostics']['sharing_closure']['base_candidate_sha256'] = 'changed'
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='base pool'):
        DimenSRMIMORegressor.load(path)


@pytest.mark.parametrize('token', ['log', 'square'])
def test_validation_invalid_extension_does_not_discard_valid_base(monkeypatch, token):
    from dimensr_mimo import sharing
    from dimensr_mimo.evaluation import Candidate
    def extension(pool, evaluator):
        graph = System((Vertex('x0'), Vertex(token, (0,))), (1, 1))
        error = np.sqrt(np.mean(((graph.evaluate(evaluator.X, [])-evaluator.Y)/evaluator.scales)**2, axis=0))
        extra = Candidate(graph, np.array([]), error, 2., 0., 0.)
        return list(pool)+[extra], dict(base_candidate_count=len(pool), base_candidate_sha256=digest([c.to_dict() for c in pool]),
            new_evaluations=0, protected=1)
    monkeypatch.setattr(sharing, 'protected_closure', extension)
    X = np.linspace(.2, 1.2, 16)[:, None]; Y = np.column_stack((X[:, 0], 2*X[:, 0]))
    cfg = SearchConfig(sharing_strategy='closure', final_selection='precision', acceptance=limits(),
        operators=('add',), population_size=4, epochs=1, gp_generations=1, selection_generations=2,
        max_evaluations=40, min_candidates=3, polish_candidates=0, use_rnn=False)
    validation = -X if token=='log' else np.full_like(X, 1e100)
    targets = np.column_stack((validation[:, 0], 2*validation[:, 0]))
    model = DimenSRMIMORegressor(cfg).fit(X, Y, X_val=validation, Y_val=targets, input_bounds=[[.2, 1.2]])
    record = model.diagnostics_['sharing_closure']
    assert len(record['validation_rejections']) == 1 and record['validation_retained_extensions'] == 0
    assert len(model.candidates_) == record['base_candidate_count']
    assert record['full_training_pool_sha256'] != model.diagnostics_['frozen_candidate_sha256']


def test_closure_stops_at_48_new_evaluations_and_global_budget(monkeypatch):
    from dimensr_mimo import sharing
    from dimensr_mimo.evaluation import Candidate
    def graph(i):
        return System((Vertex('x'+str(i)),), (0, 0))
    def candidate(system):
        return Candidate(system, np.array([]), np.zeros(2), 1., 0., 0.)
    parents = [candidate(graph(i)) for i in range(4)]
    # Stub only the proposal source to expose more than 48 unique scheduler jobs.
    monkeypatch.setattr(sharing, 'matched_proposals', lambda system, constants:
        [(graph(4+16*int(system.nodes[0].token[1:])+j), np.array([])) for j in range(16)])
    class Evaluation:
        def __init__(self, limit):
            self.cfg = SearchConfig(max_evaluations=limit)
            self.cache = {c.system:c for c in parents}
            self.calls = 4
            self.X, self.scales = np.ones((3, 80)), np.ones(2)
        def expired(self):
            return False
        def __call__(self, system, warm):
            self.calls += 1
            self.cache[system] = candidate(system)
            return self.cache[system]
    for limit, expected in ((100, 48), (5, 1)):
        evaluator = Evaluation(limit)
        pool, report = protected_closure(parents, evaluator)
        assert report['new_evaluations'] == expected and len(pool) == 4+expected
        assert evaluator.calls == 4+expected
