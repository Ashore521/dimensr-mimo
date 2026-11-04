"""Small independent installed-package probe; no fitting or test-label access."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
installed=ROOT/'outputs/r170installed'
sys.dont_write_bytecode=True
sys.path.insert(0,str(installed))


class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='experiments' or fullname.startswith('experiments.'):
            raise ImportError('Independent installed probe forbids experiment imports.')


sys.meta_path.insert(0,NoExperiments())
import numpy as np
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel,SearchConfig,__version__
from dimensr_mimo.graph import System


def forbidden(*a,**k):raise AssertionError('No fit is permitted in release review.')
DimenSRMIMORegressor.fit=forbidden
cfg=SearchConfig();cfg.validate()
assert __version__=='1.7.0' and cfg.initialization=='legacy' and cfg.sharing_strategy=='random'
assert cfg.precision_repair is cfg.compositional_completion is False
relative='experiments/completion_release_validation_20261104/runs/log_affine_ratio_n0_s10241_completion/model.json'
path=ROOT/relative
saved=json.loads(path.read_text(encoding='utf-8'))
model=DimenSRMIMORegressor.load(path)
assert model.best_.to_dict()==saved['best']
physical=PhysicalModel.load(path.with_name('physical.json'))
bounds=np.asarray(physical.input_bounds)
X=np.random.default_rng(1701).uniform(bounds[:,0],bounds[:,1],(7,len(bounds)))
expected=System.from_dict(saved['best']['system']).evaluate(X,np.asarray(saved['best']['constants']))
np.testing.assert_array_equal(model.predict(X),expected)
np.testing.assert_array_equal(physical.predict(X),expected)
model.save(OUT/'installed_native_roundtrip.json')
physical.save(OUT/'installed_physical_roundtrip.json')
assert DimenSRMIMORegressor.load(OUT/'installed_native_roundtrip.json').to_dict()==model.to_dict()
assert PhysicalModel.load(OUT/'installed_physical_roundtrip.json').to_dict()==physical.to_dict()
legacy_relative='experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json'
legacy=DimenSRMIMORegressor.load(ROOT/legacy_relative)
legacy.save(OUT/'installed_legacy_roundtrip.json')
assert DimenSRMIMORegressor.load(OUT/'installed_legacy_roundtrip.json').to_dict()==legacy.to_dict()
assert legacy.config_.precision_repair is legacy.config_.compositional_completion is False
loaded={name:str(Path(m.__file__).resolve()) for name,m in sys.modules.items()
    if (name=='dimensr_mimo' or name.startswith('dimensr_mimo.')) and getattr(m,'__file__',None)}
assert all(Path(p).is_relative_to(installed) for p in loaded.values())
assert not any(name=='experiments' or name.startswith('experiments.') for name in sys.modules)
record=dict(status='passed',version=__version__,installed_modules=loaded,experiment_imports_blocked=True,
    new_native=relative,new_native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    new_native_and_physical_roundtrips=True,prediction_rows=7,probe_seed=1701,predictions_exact=True,
    legacy=legacy_relative,legacy_roundtrip=True,default_configuration_passed=True,fit_forbidden=True,
    scientific_scoring_performed=False)
(OUT/'installed_probe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='installed_modules'}))
