"""Run pre-seal guard tests independently of published experiment results."""
import pytest


@pytest.fixture(autouse=True)
def isolated_stage_preseal_directory(request):
    if request.node.originalname not in {
        'test_new_thin_entries_and_isolated_bindings',
        'test_new_cases_and_verifiers_are_bound_without_old_data',
    }:
        return
    monkeypatch = request.getfixturevalue('monkeypatch')
    directory = request.getfixturevalue('tmp_path')
    stage = request.module.study
    verifier = request.module.evaluate.verifier
    for owner in (stage, stage.cases, stage.supervisor, verifier.legacy, verifier.base):
        monkeypatch.setattr(owner, 'HERE', directory)
