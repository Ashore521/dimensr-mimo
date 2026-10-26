from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/guarded_selection_20261026/selector.py'
spec = importlib.util.spec_from_file_location('guarded_selector_test', PATH)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def c(errors, complexity, score, violation=0.):
    return dict(validation_errors=errors, complexity=complexity, score=score, violation=violation,
                system={'id': score}, constants=[], errors=errors, rademacher=0., rademacher_complexity=0.)


def test_output_guard_rejects_lower_average_with_damaged_output():
    pool = [c([1., 1.], 2., .1), c([0., 1.5], 1., .2), c([.8, .9], 3., .3)]
    frozen = deepcopy(pool)
    assert selector.choose(pool, 'no_penalty')['candidate_index'] == 1
    result = selector.choose(pool, 'guarded')
    assert result['anchor_index'] == 0 and result['candidate_index'] == 2
    assert result['chosen_within_bounds'] and pool == frozen


def test_near_equal_accuracy_prefers_simple_and_preserves_violation_tier():
    pool = [c([1., 1.], 5., .1), c([.8, .8], 9., .2), c([.805, .805], 2., .3),
            c([.7, .7], 1., .4, violation=.1)]
    assert selector.choose(pool, 'guarded')['candidate_index'] == 2
    assert selector.choose(pool, 'no_penalty')['candidate_index'] == 1


def test_zero_error_tolerance_and_anchor_fallback():
    pool = [c([0., 0.], 2., .1), c([0., 0.], 8., .2), c([0., 1e-7], 1., .3)]
    assert selector.choose(pool, 'guarded')['candidate_index'] == 0
    pool = [c([.2, .2], 4., .1), c([.1, .4], 1., .2)]
    assert selector.choose(pool, 'guarded')['candidate_index'] == 0
    assert selector.choose(pool, 'guarded')['protected_count'] == 1


def test_invalid_metrics_and_policy_rejected():
    with pytest.raises(ValueError):
        selector.choose([c([float('nan'), 0.], 1., .1)], 'guarded')
    with pytest.raises(ValueError):
        selector.choose([c([0., 0.], 1., .1)], 'guarded', dict(selector.POLICY, degradation_absolute=-1))


def test_shared_search_matrix_and_training_firewall():
    import sys
    root=PATH.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    from experiments.guarded_selection_20261026 import study, cases, worker
    jobs=study.matrix()
    assert len(jobs)==80 and sum(j['seconds'] for j in jobs)==3000
    assert sum(j['seconds']==60 for j in jobs)==40
    assert set(cases.SEEDS).isdisjoint({0,1,17,101,102,103,104,105})
    assert study.legacy.HERE==root/'experiments/guarded_selection_20261026'
    assert worker.base_worker.generate is cases.training_data
    with pytest.raises(ValueError,match='cannot access test'):
        worker.base_worker.generate('shared_parameter','test',201)


def test_failed_search_retains_derived_failure_without_new_fit(tmp_path,monkeypatch):
    import sys
    root=PATH.parents[2]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from experiments.guarded_selection_20261026 import worker
    source=tmp_path/'shared_parameter_s201_original';source.mkdir()
    worker.write(source/'resources.json',{'status':'system_timeout'})
    folder=tmp_path/'shared_parameter_s201_guarded';folder.mkdir()
    monkeypatch.setattr(worker.base_worker,'run',lambda *a:pytest.fail('Derived choice must never fit.'))
    job=dict(case='shared_parameter',seed=201,method='guarded',seconds=15)
    worker.run(folder,job,{})
    result=worker.read(folder/'result.json')
    assert result['status']=='source_search_failed' and result['models']==[]
    assert result['search_reused'] and result['source_search']==source.name
