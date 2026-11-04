"""Call the adapter on real module objects without generating any data."""
import hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_historical_20261031 import study as historical,evaluate as he
from experiments.completion_development_20261031 import study as development,evaluate as de

path=ROOT/'tests/conftest.py'
spec=importlib.util.spec_from_file_location('_review_root_test_adapter',path)
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
function=adapter.isolated_completion_preseal_directory.__wrapped__
fake_module=SimpleNamespace(historical=historical,he=he,de=de)
checks={}
class Request:
    node=SimpleNamespace(originalname='test_matrix_and_isolated_verifiers')
    module=fake_module
    def __init__(self,stage):self.stage=stage
    def getfixturevalue(self,name):assert name=='stage';return self.stage

for stage,verifier in [(historical,he.verifier),(development,de.verifier)]:
    owners=(stage,stage.cases,verifier.legacy,verifier.base)
    old=[o.HERE for o in owners]
    generate=stage.cases.generate;gate=stage.check_stage_gate
    folder=HERE/'harness_review_validation'/'adapter_scope'/stage.STAGE
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'ALL_MODELS_FROZEN.json').exists()
    with pytest.MonkeyPatch.context() as patch:
        function(Request(stage),patch,folder)
        assert all(o.HERE==folder for o in owners)
        assert stage.cases.generate is generate and stage.check_stage_gate is gate
        def no_rng(*a,**kw):raise AssertionError('Guard must reject before RNG or arrays.')
        patch.setattr(np.random,'default_rng',no_rng)
        try:stage.cases.generate(stage.cases.CASES[0],'test',stage.cases.seeds(stage.cases.CASES[0])[0])
        except RuntimeError as error:assert 'seal' in str(error)
        else:raise AssertionError('Actual guard did not run.')
    assert [o.HERE for o in owners]==old
    checks[stage.STAGE]=dict(same_empty_here=True,guard_function_unchanged=True,guard_rejected_before_rng=True,paths_restored=True)

class OtherRequest:
    node=SimpleNamespace(originalname='test_other_existing_function')
    def getfixturevalue(self,name):raise AssertionError('Unrelated test must not resolve stage.')
with pytest.MonkeyPatch.context() as patch:function(OtherRequest(),patch,folder)
result=dict(status='passed',checks=checks,unrelated_name_no_stage_access=True,
            adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frozen_test_sha256=hashlib.sha256((ROOT/'tests/test_completion_historical_study.py').read_bytes()).hexdigest(),
            scope='Direct fixture wiring and real early guard only; no formal arrays, no fit, no preflight.')
(HERE/'harness_review_validation/adapter_scope_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
