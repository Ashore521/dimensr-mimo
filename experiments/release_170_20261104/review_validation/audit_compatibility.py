"""Bounded release pre-review: no fitting, initialization, data generation or scoring."""
import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.abc
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = '652b05884269c7563b101e0a14a9b15f749bbc6f'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))


class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments' or fullname.startswith('experiments.'):
            raise ImportError('Review forbids runtime experiment dependencies.')


sys.meta_path.insert(0, NoExperiments())
import numpy as np
import scipy
import torch
torch.set_num_threads(1)
import dimensr_mimo
from dimensr_mimo import DimenSRMIMORegressor, PhysicalModel, SearchConfig
from dimensr_mimo.graph import System
from dimensr_mimo import completion, repair, sharing


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


lock_path = ROOT/'experiments/compositional_completion_20261031/CANDIDATE_LOCK.json'
lock = read(lock_path)
assert sha(lock_path) == '51103c4e7085da90695f7845df5a312ca6ff693dc686bbc651772bf63c74b452'
source = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (ROOT/'src').rglob('*.py')}
assert source == lock['source_files'] and len(source) == 41
prior_review = read(ROOT/'experiments/compositional_completion_20261031/CORE_REVIEW_CHECKS.json')
assert source == prior_review['current_source_files']
unchanged_boundaries = ['evaluation.py','fitting.py','graph.py','grammar.py','pareto.py','selection.py','physical.py','physical_selection.py','audit.py']
for name in unchanged_boundaries:
    rel = 'src/dimensr_mimo/'+name
    old = subprocess.check_output(['git','show',BASE+':'+rel], cwd=ROOT)
    assert hashlib.sha256(old).hexdigest() == source[rel]

legacy_source = subprocess.check_output(['git','show',BASE+':src/dimensr_mimo/config.py'],cwd=ROOT).decode()
legacy_module = types.ModuleType('_review_legacy_config')
legacy_module.__package__ = 'dimensr_mimo'
sys.modules[legacy_module.__name__] = legacy_module
exec(compile(legacy_source, 'git:'+BASE+':config.py', 'exec'), legacy_module.__dict__)
defaults = asdict(SearchConfig())
legacy_defaults = asdict(legacy_module.SearchConfig())
assert {k:defaults[k] for k in legacy_defaults} == legacy_defaults
assert set(defaults)-set(legacy_defaults) == {'precision_repair','compositional_completion'}
assert defaults['precision_repair'] is defaults['compositional_completion'] is False
limits = dict(nrmse=[1e-4,1e-4],shared_graph_size=24,independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6)
config_count = valid_count = 0
for initialization,strategy,selection,pr,cc,units,constraints,use_sharing in itertools.product(
        ('legacy','dimensional','compound','implicit'),('random','matched','closure'),
        ('weighted','guarded','precision'),(False,True),(False,True),(False,True),(False,True),(False,True)):
    config_count += 1
    kwargs = dict(initialization=initialization,sharing_strategy=strategy,final_selection=selection,
        precision_repair=pr,compositional_completion=cc,use_units=units,use_constraints=constraints,use_sharing=use_sharing,
        acceptance=deepcopy(limits) if selection=='precision' else None)
    expected = ((strategy!='closure' or (selection=='precision' and use_sharing)) and
        (selection!='precision' or (units and constraints)) and not (pr and cc) and
        (not(pr or cc) or (initialization=='compound' and strategy=='closure' and selection=='precision')))
    try:
        SearchConfig(**kwargs).validate()
        actual = True
    except ValueError:
        actual = False
    assert actual == expected, kwargs
    valid_count += actual
strict_rejections = 0
for key,value in itertools.product(('precision_repair','compositional_completion'),(0,1,None,'true',np.bool_(True))):
    try:
        SearchConfig(**{key:value}).validate()
    except ValueError:
        strict_rejections += 1
    else:
        raise AssertionError((key,value))

models = [
    'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json',
    'experiments/implicit_rational_20261028/runs/rational_n0_s702_compound/model.json',
    'experiments/precision_repair_20261028/runs/rational_n0_s702_repair/model.json',
    'experiments/completion_development_verified_20261031/runs/algebraic_norm_n0_s9131_completion/model.json',
]
roundtrips=[]
for i,relative in enumerate(models):
    path=ROOT/relative
    saved=read(path)
    model=DimenSRMIMORegressor.load(path)
    assert model.best_.to_dict()==saved['best']
    bounds=np.asarray(model.diagnostics_['domain']['input_bounds'])
    X=bounds[:,0]+np.linspace(.1,.9,7)[:,None]*(bounds[:,1]-bounds[:,0])
    expected=System.from_dict(saved['best']['system']).evaluate(X,np.asarray(saved['best']['constants']))
    np.testing.assert_array_equal(model.predict(X),expected)
    target=OUT/f'native_roundtrip_{i}.json'
    model.save(target)
    reloaded=DimenSRMIMORegressor.load(target)
    assert reloaded.to_dict()==model.to_dict()
    np.testing.assert_array_equal(reloaded.predict(X),expected)
    rejected=[]
    if 'post_selection' in saved:
        tampered=deepcopy(saved)
        tampered['post_selection']['source_model_sha256']='tampered'
        target=OUT/f'tampered_provenance_{i}.json'
        target.write_text(json.dumps(tampered),encoding='utf-8')
        try:DimenSRMIMORegressor.load(target)
        except ValueError:rejected.append('selection_provenance')
        else:raise AssertionError('Tampered provenance accepted.')
    if saved['config'].get('compositional_completion'):
        tampered=deepcopy(saved)
        tampered['diagnostics']['sharing_closure']['base_candidate_sha256']='tampered'
        target=OUT/f'tampered_base_{i}.json'
        target.write_text(json.dumps(tampered),encoding='utf-8')
        try:DimenSRMIMORegressor.load(target)
        except ValueError:rejected.append('base_pool_sha256')
        else:raise AssertionError('Tampered base accepted.')
    roundtrips.append(dict(path=relative,sha256=sha(path),saved_version=saved['package_version'],
        missing_default_fields=[k for k in ('precision_repair','compositional_completion') if k not in saved['config']],
        exact_prediction_rows=7,tamper_rejections=rejected))

physical_rel='experiments/completion_development_verified_20261031/runs/algebraic_norm_n0_s9131_completion/physical.json'
physical=PhysicalModel.load(ROOT/physical_rel)
physical.save(OUT/'physical_roundtrip.json')
assert PhysicalModel.load(OUT/'physical_roundtrip.json').to_dict()==physical.to_dict()
np.testing.assert_array_equal(physical.predict(X),model.predict(X))
freeze=read(ROOT/'experiments/completion_development_verified_20261031/FREEZE.json')
frozen_checks={}
for relative in ('conftest.py','tests/conftest.py','tests/test_completion_historical_study.py'):
    assert sha(ROOT/relative)==freeze['files'][relative]
    frozen_checks[relative]=freeze['files'][relative]
assert not any(n=='experiments' or n.startswith('experiments.') for n in sys.modules)
assert all(Path(m.__file__).resolve().is_relative_to(ROOT/'src') for m in (dimensr_mimo,completion,repair,sharing))
assert source=={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'src').rglob('*.py')}
record=dict(status='passed',scope='source pre-review; no training or experiment scoring',
    source_count=len(source),source_files=source,candidate_lock_sha256=sha(lock_path),
    source_matches_independent_core_review=True,unchanged_boundary_modules=unchanged_boundaries,
    old_defaults_unchanged=True,config_combinations=config_count,valid_configurations=valid_count,
    strict_bool_rejections=strict_rejections,native_roundtrips=roundtrips,
    physical=dict(path=physical_rel,sha256=sha(ROOT/physical_rel),roundtrip_exact=True,native_predictions_exact=True),
    frozen_test_checks=frozen_checks,experiment_imports_blocked=True,
    environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
        executable=sys.executable,package_path=str(Path(dimensr_mimo.__file__).resolve())),
    script_sha256=sha(Path(__file__)))
(OUT/'audit_results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ('source_files','native_roundtrips')}))
