"""Protocol guard tests; these do not launch performance experiments."""
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.second_round_20261026 import cases, study, worker


def test_second_round_matrix_and_same_search_budget():
    jobs = study.matrix()
    assert len(jobs) == 60 and sum(j['seconds'] for j in jobs) == 3600
    assert {j['seed'] for j in jobs}.isdisjoint({0, 1, 17})
    base = study.read(ROOT / 'experiments/first_round/FREEZE.json')['config']
    configured = study.configs(base, {'config_overrides': {'selection_complexity_weight': 0.0}})
    assert configured['original'] == configured['independent'] == base
    assert {k for k in base if configured['proposed'][k] != base[k]} == {'selection_complexity_weight'}
    with pytest.raises(ValueError):
        study.configs(base, {'config_overrides': {'max_evaluations': 3}})


def test_training_worker_rejects_test_access():
    assert worker.previous_worker.generate is cases.training_data
    with pytest.raises(ValueError, match='cannot request test'):
        worker.previous_worker.generate('shared_parameter', 'test', 101)
    X, Y = cases.training_data('shared_parameter', 'train', 101)
    assert X.shape == Y.shape == (96, 2)


def test_frozen_drift_and_selected_pool_identity(tmp_path):
    file = tmp_path / 'pinned.txt'
    file.write_text('frozen')
    pins = {'pinned.txt': study.sha(file)}
    study.verify_files(pins, tmp_path)
    file.write_text('changed')
    with pytest.raises(RuntimeError, match='drift'):
        study.verify_files(pins, tmp_path)
    data = {'best': {'constants': [1]}, 'candidates': [{'constants': [2]}, {'constants': [1], 'score': 0}]}
    assert worker.selection_record(data)['candidate_index'] == 1
    data['candidates'].append(data['candidates'][1])
    with pytest.raises(RuntimeError, match='uniquely'):
        worker.selection_record(data)


def test_reused_supervisor_cannot_write_first_round():
    assert study.legacy.HERE == ROOT / 'experiments/second_round_20261026'
    assert study.legacy.budget is study.budget
    assert study.legacy.mechanism is study.mechanism
    assert study.legacy.generate is cases.generate


def test_evaluate_requires_full_freeze_before_test_data(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'HERE', tmp_path)
    monkeypatch.setattr(study, 'budget', lambda: float('inf'))
    study.write(tmp_path / 'FREEZE.json', {'files': {}})
    called = []
    monkeypatch.setattr(study.legacy, 'evaluate', lambda: called.append('test'))
    with pytest.raises(FileNotFoundError):
        study.evaluate()
    assert not called


def test_freeze_includes_diagnostics_and_counts_hashing_time(tmp_path, monkeypatch):
    here = tmp_path / 'experiments/second_round_20261026'
    here.mkdir(parents=True)
    first = here.parent / 'first_round'
    first.mkdir()
    base = study.read(ROOT / 'experiments/first_round/FREEZE.json')['config']
    study.write(first / 'FREEZE.json', {'config': base})
    study.write(here / 'NEW_CHANGE.json', {'config_overrides': {'selection_complexity_weight': 0.0}})
    for path in ('PROTOCOL.md', 'study.py', 'diagnostics/result.json', 'diagnostics/nested/trace.jsonl'):
        target = here / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}')
    for path in ('pyproject.toml', 'docs/environment.txt', 'tests/test_second_round.py'):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('fixture')
    monkeypatch.setattr(study, 'HERE', here)
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study.platform, 'platform', lambda: 'mock-platform')
    monkeypatch.setattr(study.subprocess, 'check_output', lambda args, **kw: b'' if 'ls-files' in args else b'commit')
    ticks = iter([100., 115.])
    monkeypatch.setattr(study.time, 'time', lambda: next(ticks))
    real_sha = study.sha
    def hash_after_budget_started(path):
        assert study.read(here / 'BUDGET.json')['start_unix'] == 100.
        return real_sha(path)
    monkeypatch.setattr(study, 'sha', hash_after_budget_started)
    study.freeze()
    frozen = study.read(here / 'FREEZE.json')
    assert frozen['frozen_unix'] == 115. and frozen['budget']['start_unix'] == 100.
    assert all(str((here / path).relative_to(tmp_path).as_posix()) in frozen['files']
               for path in ('diagnostics/result.json', 'diagnostics/nested/trace.jsonl'))


def test_train_freezes_each_cell_and_refuses_unverified_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'HERE', tmp_path)
    job = {'case': 'partial', 'seed': 101, 'method': 'original', 'seconds': 60}
    folder = tmp_path / 'runs/partial_s101_original'
    study.write(tmp_path / 'FREEZE.json', {'files': {}, 'jobs': [job], 'configs': {'original': {}}})
    calls = []
    def fake_run(*args):
        calls.append(args)
        folder.mkdir(parents=True)
        study.write(folder / 'job.json', job)
        study.write(folder / 'resources.json', {'status': 'complete'})
        study.write(folder / 'result.json', {'model': 'frozen'})
    monkeypatch.setattr(study.legacy, 'run_job', fake_run)
    study.train()
    assert (folder / 'CELL_FROZEN.json').exists() and len(calls) == 1
    (tmp_path / 'ALL_MODELS_FROZEN.json').unlink()
    study.train()
    assert len(calls) == 1
    (tmp_path / 'ALL_MODELS_FROZEN.json').unlink()
    study.write(folder / 'result.json', {'model': 'changed'})
    with pytest.raises(RuntimeError, match='drift'):
        study.train()
    (folder / 'CELL_FROZEN.json').unlink()
    with pytest.raises(RuntimeError, match='incident review'):
        study.train()
    assert study.read(folder / 'INTERRUPTION.json')['automatic_retry'] is False
    assert len(calls) == 1


def test_expired_budget_never_enters_scoring_or_generates_test(monkeypatch):
    monkeypatch.setattr(study, 'budget', lambda: 100.)
    monkeypatch.setattr(study.time, 'time', lambda: 101.)
    called = []
    monkeypatch.setattr(study.legacy, 'evaluate', lambda: called.append('score'))
    monkeypatch.setattr(study.legacy, 'generate', lambda *args: called.append('test'))
    with pytest.raises(RuntimeError, match='expired before test scoring'):
        study.evaluate()
    assert not called


@pytest.mark.parametrize('expires_during_generation', [True, False])
def test_scoring_checks_each_test_generation_and_retains_overrun(tmp_path, monkeypatch, expires_during_generation):
    monkeypatch.setattr(study, 'HERE', tmp_path)
    monkeypatch.setattr(study, 'budget', lambda: 100.)
    clock = [99.]
    monkeypatch.setattr(study.time, 'time', lambda: clock[0])
    study.write(tmp_path / 'FREEZE.json', {'files': {}, 'jobs': []})
    study.write(tmp_path / 'ALL_MODELS_FROZEN.json',
                {'files': {}, 'freeze_sha256': study.sha(tmp_path / 'FREEZE.json')})
    generated = []
    original = lambda *args: generated.append(args)
    monkeypatch.setattr(study.legacy, 'generate', original)
    def scoring():
        clock[0] = 101.
        if expires_during_generation:
            study.legacy.generate('partial', 'test', 101)
        else:
            study.write(tmp_path / 'SCORES.json', {'retained': True})
    monkeypatch.setattr(study.legacy, 'evaluate', scoring)
    with pytest.raises(RuntimeError, match='budget'):
        study.evaluate()
    assert not generated and study.legacy.generate is original
    if not expires_during_generation:
        assert study.read(tmp_path / 'SCORES.json')['retained']
        assert study.read(tmp_path / 'SCORING_BUDGET_OVERRUN.json')['overrun_seconds'] == 1.


def test_finish_runs_scoring_then_verification_within_remaining_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'HERE', tmp_path)
    monkeypatch.setattr(study.time, 'time', lambda: 99.)
    study.write(tmp_path / 'BUDGET.json', {'start_unix': 0., 'limit_seconds': 100.})
    study.write(tmp_path / 'FREEZE.json', {'files': {}})
    calls = []
    class Finished:
        returncode = 0
        def poll(self):
            return 0
        def wait(self):
            return 0
    def launch(args):
        calls.append(args)
        return Finished()
    monkeypatch.setattr(study.subprocess, 'Popen', launch)
    study.finish()
    assert [call[-1] for call in calls] == ['evaluate', 'verify']
    record = study.read(tmp_path / 'FINISH_SUPERVISION.json')
    assert record['status'] == 'complete' and record['total_wall_seconds'] == 99.
    monkeypatch.setattr(study.time, 'time', lambda: 101.)
    with pytest.raises(RuntimeError, match='budget'):
        study.finish()
    assert len(calls) == 2
