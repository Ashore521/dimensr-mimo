from experiments.matched_sharing_20261028 import study, worker, evaluate
from dimensr_mimo import SearchConfig
import pytest


def test_matched_worker_changes_only_the_declared_option(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(worker.base, 'run', lambda folder, job, config:seen.append((job, config)))
    for method in ('random', 'matched'):
        worker.run(tmp_path, dict(method=method), dict(max_evaluations=1200))
    assert seen[0][1] == dict(max_evaluations=1200, sharing_strategy='random')
    assert seen[1][1] == dict(max_evaluations=1200, sharing_strategy='matched')
    with pytest.raises(ValueError, match='frozen'):
        worker.run(tmp_path, dict(method='typo'), {})


def test_full_matrix_and_failed_cell_remain_in_adoption_denominator():
    jobs = study.matrix()
    assert len(jobs) == 34 and len({study.name(j) for j in jobs}) == 34
    assert all(sum(j['method'] == method for j in jobs) == 17 for method in study.METHODS)
    assert study.configuration()['initialization'] == 'compound' and study.LIMIT == 2880
    SearchConfig(**study.configuration()).validate()
    assert study.configuration()['wall_seconds'] == 52
    scores = [dict(job=j, status='complete', both_accepted=True, mean_clean_nrmse=.001,
        clean_nrmse=[.001, .001], structure=dict(shared_graph_size=20)) for j in jobs]
    verified = [dict(job=j, verified=True, matched_share=dict(operations=1, changed=1, novel=1, successful=1))
                for j in jobs]
    assert evaluate.summarize(scores, verified)['adoption_pass']
    scores[0] = dict(job=jobs[0], status='system_timeout')
    verified[0] = dict(job=jobs[0], verified=False)
    result = evaluate.summarize(scores, verified)
    assert not result['adoption_pass'] and not result['technical_pass']
    assert result['expected_outputs'] == 34 and result['protected_outputs'] == 32
    assert sum(row['expected'] for row in result['methods'].values()) == 34
