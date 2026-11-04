"""Isolate the frozen pre-seal test from completed research artifacts."""
import pytest


@pytest.fixture(autouse=True)
def isolated_completion_preseal_directory(request, monkeypatch, tmp_path):
    # The archived test predates its experiment's all-model seal. Keep its bytes
    # frozen and supply the empty directory its test-label guard requires.
    if request.node.originalname != 'test_matrix_and_isolated_verifiers':
        return
    stage = request.getfixturevalue('stage')
    module = request.module
    verifier = module.he.verifier if stage is module.historical else module.de.verifier
    for owner in (stage, stage.cases, verifier.legacy, verifier.base):
        monkeypatch.setattr(owner, 'HERE', tmp_path)
